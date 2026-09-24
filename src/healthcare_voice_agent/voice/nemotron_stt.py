"""Streaming adapter for NeMo-Speech.cpp's transcription WebSocket, NOT OpenAI.

Wire contract inspected at upstream commit 07003daa7eefea542076310722ccaa89709ee3c3:
https://github.com/NVIDIA/NeMo-Speech.cpp/blob/07003daa7eefea542076310722ccaa89709ee3c3/server/http/http_server.cpp

The server has no item IDs. It emits zero or more completed phrases followed by
an ordered ``input_audio_buffer.committed`` acknowledgement per explicit commit.
Only that acknowledgement releases the assembled segment. Partials are ignored:
the server can emit replacement text as a delta without identifying replacements.
Binary PCM is sent DURING speech; this is not batch transcription. Local VAD
controls commits. The server's loaded ASR model, not a request field, selects the
checkpoint. ``model`` is the application/run-record label.
"""
from __future__ import annotations

import asyncio
import json
import math
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    CancelFrame, EndFrame, InputAudioRawFrame, StartFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601
from websockets.asyncio.client import connect


@dataclass
class _Segment:
    sequence: int
    generation: int
    user_id: str = ""
    audio_bytes: int = 0
    commit_sent: bool = False
    deadline: float | None = None
    parts: list[str] = field(default_factory=list)
    text_chars: int = 0


class NemotronSTTService(STTService):
    """Bounded FIFO streaming with no retries, fallback, or implicit downloads."""

    def __init__(self, *, model: str, base_url: str, language: str | None = None,
                 api_key: str | None = None, max_segment_seconds: float = 60,
                 max_pending_segments: int = 8, inference_timeout_seconds: float = 30,
                 ws_close_timeout_seconds: float = 2, finalize_transcripts: bool = True,
                 **kwargs):
        for value in (max_segment_seconds, inference_timeout_seconds, ws_close_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Nemotron time limits must be positive and finite")
        if not 1 <= max_pending_segments <= 32:
            raise ValueError("Nemotron pending segment limit must be 1 to 32")
        super().__init__(sample_rate=16000, settings=STTSettings(model=model, language=language), **kwargs)
        self._endpoint = base_url
        self._api_key = api_key
        self._language = language or "auto"
        self._max_bytes = int(max_segment_seconds * 32000)
        self._max_pending = max_pending_segments
        self._timeout = inference_timeout_seconds
        self._close_timeout = ws_close_timeout_seconds
        self._finalize_transcripts = finalize_transcripts
        self._active = True
        self._socket = None
        self._workers = []
        self._outgoing = asyncio.Queue(maxsize=max_pending_segments * 4096)
        self._queued_bytes = 0
        self._segments: deque[_Segment] = deque()
        self._open_segment: _Segment | None = None
        self._preroll = bytearray()
        self._generation = 0
        self._sequence = 0
        self._recent_events: set[str] = set()
        self._event_order = deque()
        self._stop_generation: ContextVar[int | None] = ContextVar("nemotron_stop_generation", default=None)

    def can_generate_metrics(self):
        return True

    async def start(self, frame: StartFrame):
        if not self._active:
            return
        await super().start(frame)
        try:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            self._socket = await connect(
                self._endpoint, additional_headers=headers, open_timeout=self._timeout,
                close_timeout=self._close_timeout, max_size=262144, max_queue=16,
                proxy=None,  # Do not inherit an ambient proxy for private audio.
            )
            if not self._active:
                await self._shutdown()
                return
            async with asyncio.timeout(self._timeout):
                created = self._decode(await self._socket.recv())
                if created.get("type") != "session.created":
                    raise ValueError("Missing session creation")
                await self._socket.send(json.dumps({
                    "type": "session.update", "session": {
                        "sample_rate": 16000, "language": self._language,
                        "automatic_punctuation": True, "word_timestamps": False,
                        "speaker_diarization": False,
                    },
                }))
                updated = self._decode(await self._socket.recv())
                if updated.get("type") != "session.updated":
                    raise ValueError("Missing session acknowledgement")
            if not self._active:
                await self._shutdown()
                return
            self._workers = [self.create_task(coro, name=name) for coro, name in (
                (self._send_loop(), "nemotron_sender"),
                (self._receive_loop(), "nemotron_receiver"),
                (self._deadline_loop(), "nemotron_deadlines"),
            )]
        except asyncio.CancelledError:
            await self._shutdown()
            raise
        except Exception:
            await self._fail("Nemotron connection or session setup failed")

    @staticmethod
    def _decode(message):
        event = json.loads(message)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("Malformed event")
        return event

    async def process_frame(self, frame, direction):
        # Reserve generation/commit identities before the framework can await.
        if isinstance(frame, (CancelFrame, EndFrame)):
            self._active = False
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._active:
                return
            if self._open_segment is not None:
                return
            if len(self._segments) >= self._max_pending:
                await self._fail("Nemotron transcription queue is full")
                return
            self._generation += 1
            self._sequence += 1
            segment = _Segment(self._sequence, self._generation, self._user_id)
            self._segments.append(segment)
            self._open_segment = segment
            if self._preroll:
                data = bytes(self._preroll)
                self._preroll.clear()
                if not self._enqueue_audio(data):
                    await self._fail("Nemotron audio queue or segment limit exceeded")
                    return
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if not self._active or self._open_segment is None:
                return
            segment = self._open_segment
            self._open_segment = None
            segment.deadline = asyncio.get_running_loop().time() + self._timeout
            try:
                self._outgoing.put_nowait((segment, None))
            except asyncio.QueueFull:
                await self._fail("Nemotron transcription queue is full")
                return
            token = self._stop_generation.set(segment.generation)
            try:
                await super().process_frame(frame, direction)
            finally:
                self._stop_generation.reset(token)
            return
        await super().process_frame(frame, direction)

    async def _handle_vad_user_stopped_speaking(self, frame):
        # No framework last-transcript timeout: only commit ACKs finalize input.
        if self._active and self._stop_generation.get() == self._generation:
            self._user_speaking = self._open_segment is not None
            if frame.stop_secs:
                await self.start_ttfb_metrics(start_time=frame.timestamp - frame.stop_secs)

    def _enqueue_audio(self, audio: bytes) -> bool:
        segment = self._open_segment
        if segment is None or segment.audio_bytes + len(audio) > self._max_bytes:
            return False
        if self._queued_bytes + len(audio) > self._max_bytes * self._max_pending:
            return False
        try:
            self._outgoing.put_nowait((segment, audio))
        except asyncio.QueueFull:
            return False
        segment.audio_bytes += len(audio)
        self._queued_bytes += len(audio)
        return True

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction):
        if not self._active or self._muted or not self.is_usable:
            return
        if frame.sample_rate != 16000 or frame.num_channels != 1 or len(frame.audio) % 2:
            await self._fail("Nemotron requires 16000 Hz mono PCM16")
            return
        self._user_id = getattr(frame, "user_id", "")
        if self._open_segment is None:
            self._preroll.extend(frame.audio)
            del self._preroll[:-32000]
        elif frame.audio:
            self._open_segment.user_id = self._user_id
            if not self._enqueue_audio(frame.audio):
                await self._fail("Nemotron audio queue or segment limit exceeded")

    async def run_stt(self, audio: bytes):
        raise RuntimeError("Nemotron audio requires a reserved acoustic segment")
        yield  # pragma: no cover

    async def _send_loop(self):
        try:
            while self._active:
                segment, audio = await self._outgoing.get()
                if not self._active:
                    return
                if audio is None:
                    segment.commit_sent = True  # ACK can arrive during send's await.
                    message = json.dumps({"type": "input_audio_buffer.commit"})
                else:
                    self._queued_bytes -= len(audio)
                    message = audio
                await asyncio.wait_for(self._socket.send(message), self._timeout)
                if audio is not None:
                    self._record_stt_audio_usage(audio)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._fail("Nemotron audio transmission failed")

    async def _receive_loop(self):
        try:
            async for message in self._socket:
                if not self._active:
                    return
                await self._handle_event(self._decode(message))
                if not self._active:
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._fail("Nemotron transcription stream failed")
        finally:
            if self._active:
                await self._fail("Nemotron transcription stream closed")

    async def _handle_event(self, event):
        if not self._active:
            return
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            await self._fail("Nemotron event has no identity")
            return
        if event_id in self._recent_events:
            return
        self._recent_events.add(event_id)
        self._event_order.append(event_id)
        if len(self._event_order) > 512:
            self._recent_events.discard(self._event_order.popleft())
        kind = event.get("type")
        if kind not in {
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
            "input_audio_buffer.committed",
        } or not self._segments:
            await self._fail("Nemotron returned an unexpected event")
            return
        segment = self._segments[0]
        if segment.deadline is not None and asyncio.get_running_loop().time() >= segment.deadline:
            await self._fail("Nemotron transcription exceeded its completion deadline")
            return
        if kind.endswith(".delta"):
            if not isinstance(event.get("delta"), str):
                await self._fail("Nemotron returned an invalid partial")
            return  # Not safe to concatenate upstream replacement-style deltas.
        if kind.endswith(".completed"):
            text = event.get("transcript")
            if not isinstance(text, str) or segment.text_chars + len(text) > 65536:
                await self._fail("Nemotron returned an invalid transcript")
                return
            segment.text_chars += len(text)
            if text.strip():
                segment.parts.append(text.strip())
            return
        if not segment.commit_sent:
            await self._fail("Nemotron acknowledged an unsent audio commit")
            return
        self._segments.popleft()
        text = " ".join(segment.parts)
        if text:
            frame = TranscriptionFrame(text, segment.user_id, time_now_iso8601(),
                                       language=self._settings.language)
            frame.metadata.update(stt_speech_generation=segment.generation,
                                  stt_segment_sequence=segment.sequence)
            await self.push_frame(frame)
        await self.emit_stt_usage_metrics()

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, TranscriptionFrame):
            if not self._active:
                return
            frame.finalized = bool(self._finalize_transcripts and
                frame.metadata.get("stt_speech_generation") == self._generation and
                self._open_segment is None and not self._segments)
            self._finalize_pending = False
        await super().push_frame(frame, direction)

    async def _deadline_loop(self):
        while self._active:
            await asyncio.sleep(min(0.1, self._timeout / 4))
            now = asyncio.get_running_loop().time()
            if any(s.deadline is not None and now >= s.deadline for s in self._segments):
                await self._fail("Nemotron transcription exceeded its completion deadline")
                return

    async def _fail(self, reason):
        if not self._active:
            return
        await self._shutdown()
        await self.push_error(error_msg=f"{reason}. Reconnect to start a fresh session.",
                              force_treat_as_permanent=True)

    async def _shutdown(self):
        self._active = False
        self._preroll.clear()
        self._segments.clear()
        self._open_segment = None
        self._queued_bytes = 0
        while not self._outgoing.empty():
            self._outgoing.get_nowait()
        workers, self._workers = self._workers, []
        for task in workers:
            if task is asyncio.current_task():
                # Keep the current worker tracked until outer cleanup can join
                # it, including an awaited downstream error publication.
                self._workers.append(task)
            else:
                await self.cancel_task(task)
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                await asyncio.wait_for(socket.close(), self._close_timeout)
            except (Exception, asyncio.CancelledError):
                # Task cancellation still propagates from the caller's scope.
                pass

    async def stop(self, frame: EndFrame):
        await self._shutdown()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._shutdown()
        await super().cancel(frame)

    async def cleanup(self):
        await self._shutdown()
        await super().cleanup()
