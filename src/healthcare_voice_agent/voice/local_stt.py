"""Bounded, locally transcribed VAD segments with acoustic-generation guards.

Only providers.py loads the model. The injected synchronous callable receives
16 kHz mono float32 samples. It must serialize access to shared model state
across sessions: cancelling an asyncio waiter cannot stop a native inference
thread. This adapter discards its result after cancellation/timeout.

Pipecat 1.11's SegmentedSTTService supplies pre-roll buffering and the background
worker lifecycle. We reserve segment identities before awaits and bypass its
unconditional finalized=True assignment so old segments cannot end newer speech.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from contextvars import ContextVar
from typing import Callable

import numpy as np
from pipecat.frames.frames import (
    CancelFrame, EndFrame, InputAudioRawFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.utils.time import time_now_iso8601


@dataclass(frozen=True)
class _Segment:
    sequence: int
    generation: int
    pcm: bytes
    user_id: str


class LocalWhisperSTTService(SegmentedSTTService):
    """Transcribe complete VAD segments without blocking microphone processing.

    There is deliberately no automatic retry or remote fallback. A failed,
    timed-out, oversized or dropped segment makes the service unusable instead
    of allowing the remaining words to become a partial user instruction.
    """

    def __init__(
        self, *, transcribe: Callable[[np.ndarray], dict], model: str,
        language: str | None = None, max_segment_seconds: float = 60,
        max_pending_segments: int = 8, inference_timeout_seconds: float = 30,
        sample_rate: int = 16000, **kwargs,
    ):
        if sample_rate != 16000:
            raise ValueError("Local Whisper requires 16000 Hz mono PCM")
        if max_segment_seconds <= 0 or max_pending_segments < 1 or inference_timeout_seconds <= 0:
            raise ValueError("Local Whisper limits must be positive")
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model=model, language=language), **kwargs,
        )
        self._transcribe = transcribe
        self._max_segment_bytes = int(max_segment_seconds * sample_rate) * 2
        self._max_pending_segments = max_pending_segments
        self._inference_timeout_seconds = inference_timeout_seconds
        self._segment_queue: asyncio.Queue[_Segment | None] = asyncio.Queue()
        self._pending_count = 0
        self._speech_generation = 0
        self._segment_sequence = 0
        self._segment_open = False
        self._active = True
        self._stop_generation: ContextVar[int | None] = ContextVar("local_stt_stop_generation", default=None)

    @property
    def wants_wav_segments(self) -> bool:
        return False

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame, direction):
        # Reserve identities synchronously, before the native VAD handler can
        # suspend while reporting metrics or forwarding its control frame.
        if isinstance(frame, (CancelFrame, EndFrame)):
            self._active = False
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._active:
                return
            self._speech_generation += 1
            self._segment_open = True
            self._user_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if not self._active:
                return
            if not self._segment_open:
                return  # Duplicate stop must not enqueue pre-roll/silence again.
            self._segment_open = False
            self._user_speaking = False
            if self._pending_count >= self._max_pending_segments:
                await self._fail("Local transcription queue is full")
                return
            pcm = bytes(self._audio_buffer)
            self._audio_buffer.clear()
            self._segment_sequence += 1
            self._pending_count += 1
            self._segment_queue.put_nowait(_Segment(
                self._segment_sequence, self._speech_generation,
                pcm + self._trailing_silence(), self._user_id,
            ))
        # Call the shared STT processor, not Segmented.process_frame, because
        # the segment was already captured above. Audio still uses the inherited
        # pre-roll implementation through our process_audio_frame override.
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            token = self._stop_generation.set(self._speech_generation)
            try:
                await STTService.process_frame(self, frame, direction)
            finally:
                self._stop_generation.reset(token)
        else:
            await STTService.process_frame(self, frame, direction)

    async def _handle_vad_user_stopped_speaking(self, frame):
        # A local job always has an explicit completion/deadline. Do not start
        # the streaming-STT last-transcript timeout: a late old segment must
        # not be measured as completion of newer speech, and overlapping stop
        # handlers must not overwrite each other's timer task references.
        if self._stop_generation.get() != self._speech_generation or not self._active:
            return
        self._user_speaking = self._segment_open
        if frame.stop_secs:
            await self.start_ttfb_metrics(start_time=frame.timestamp - frame.stop_secs)

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction):
        if not self._active or self._muted or not self.is_usable:
            return
        if frame.sample_rate != 16000 or frame.num_channels != 1 or len(frame.audio) % 2:
            await self._fail("Local transcription received an unsupported audio format")
            return
        if self._segment_open and len(self._audio_buffer) + len(frame.audio) > self._max_segment_bytes:
            await self._fail("Local transcription speech segment exceeded its duration limit")
            return
        self._user_id = getattr(frame, "user_id", "")
        self._audio_buffer += frame.audio
        # Use our synchronously reserved acoustic state, not the base
        # _user_speaking flag: an older VAD handler may resume after a newer
        # start frame and update that base flag. Never truncate newer speech.
        if not self._segment_open and len(self._audio_buffer) > self._audio_buffer_size_1s:
            self._audio_buffer = self._audio_buffer[-self._audio_buffer_size_1s:]

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, TranscriptionFrame):
            if not self._active:
                return
            generation = frame.metadata.get("stt_speech_generation")
            frame.finalized = bool(
                generation == self._speech_generation and not self._segment_open
            )
            # Do not call Segmented.push_frame: it blindly finalizes old frames.
            self._finalize_pending = False
            await STTService.push_frame(self, frame, direction)
        else:
            await super().push_frame(frame, direction)

    async def _segment_task_handler(self):
        while self._active:
            segment = await self._segment_queue.get()
            if segment is None:
                return
            try:
                if not self._active:
                    return
                self._record_stt_audio_usage(segment.pcm)
                await self.emit_stt_usage_metrics()
                await self.start_processing_metrics()
                samples = np.frombuffer(segment.pcm, dtype="<i2").astype(np.float32) / 32768.0
                # Never ask a generative recognizer to invent text for an empty
                # all-zero segment, including the synthetic trailing silence.
                if not np.any(samples):
                    continue
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._transcribe, samples),
                    timeout=self._inference_timeout_seconds,
                )
                if not self._active:
                    return
                if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                    await self._fail("Local transcription returned an invalid result")
                    return
                text = result["text"].strip()
                if not text:
                    continue  # No final flag from an empty/no-speech recognition.
                language = result.get("language")
                if not isinstance(language, str):
                    language = self._settings.language
                frame = TranscriptionFrame(text, segment.user_id, time_now_iso8601(), language)
                frame.metadata.update({
                    "stt_speech_generation": segment.generation,
                    "stt_segment_sequence": segment.sequence,
                })
                decode_path = result.get("local_decode_path")
                if isinstance(decode_path, str) and decode_path in {"single_encode", "stock_fallback"}:
                    frame.metadata["stt_decode_path"] = decode_path
                await self.push_frame(frame)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                await self._fail("Local transcription exceeded its inference deadline")
                return
            except Exception:
                # Do not expose model exceptions: they can contain paths,
                # transcript text, tokens, or audio representations.
                await self._fail("Local transcription failed")
                return
            finally:
                self._pending_count = max(0, self._pending_count - 1)
                await self.stop_processing_metrics()

    async def run_stt(self, audio: bytes):
        # Segments must carry a reserved generation; inference only runs in the
        # ordered worker above, never on a raw streaming audio frame.
        raise RuntimeError("Local Whisper requires a reserved VAD segment")
        yield  # pragma: no cover

    def _discard_pending(self):
        self._audio_buffer.clear()
        self._segment_open = False
        while not self._segment_queue.empty():
            self._segment_queue.get_nowait()
        self._pending_count = 0

    async def _fail(self, reason: str):
        if not self._active:
            return
        self._active = False
        self._discard_pending()
        if self._segment_task is not asyncio.current_task():
            await self._cancel_segment_task()
        await self.push_error(
            error_msg=f"{reason}. Reconnect to start a fresh session.",
            force_treat_as_permanent=True,
        )

    async def _close_local(self):
        self._active = False
        self._discard_pending()
        await self._cancel_segment_task()

    async def stop(self, frame: EndFrame):
        # A disconnected session must not emit buffered transcripts or trigger
        # an extra LLM turn, even when Pipecat requests a graceful end.
        await self._close_local()
        await STTService.stop(self, frame)

    async def cancel(self, frame: CancelFrame):
        await self._close_local()
        await STTService.cancel(self, frame)

    async def cleanup(self):
        await self._close_local()
        await super().cleanup()
