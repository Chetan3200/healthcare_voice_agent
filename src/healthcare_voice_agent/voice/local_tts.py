"""Sentence/chunk Kokoro synthesis with native Pipecat audio-context handling.

The factory injects a process-wide serialized synchronous callable. Cancelling
an asyncio waiter cannot stop native inference already running in its thread;
this adapter instead invalidates its result and never emits it after teardown or
interruption. Model loading and all Kokoro-specific imports stay in providers.py.

For bounded first-audio latency, the first long request in a Pipecat context may
be losslessly split into complete Kokoro requests. This can alter prosody at a
split boundary and is not sample-level synthesis streaming. Later requests in
the same context remain whole to avoid needless fragmentation.

Auto language routing is deliberately small and auditable: a Devanagari letter
selects Hindi for the whole original request; otherwise English is selected.
This is not language identification, Romanized Hindi handling, or a guarantee of
mixed-language pronunciation.
"""

from __future__ import annotations

import asyncio
import math
import time
import unicodedata
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from pipecat.frames.frames import CancelFrame, Frame, InterruptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from healthcare_voice_agent.voice.speech_chunks import split_for_first_audio

if TYPE_CHECKING:
    from healthcare_voice_agent.voice.tracing import SessionTrace

SAMPLE_RATE = 24000


def _has_devanagari_letter(text: str) -> bool:
    return any(
        ("\u0900" <= char <= "\u097f" or "\ua8e0" <= char <= "\ua8ff")
        and unicodedata.category(char).startswith("L")
        for char in text
    )


def _pcm16(audio: Any, sample_rate: int) -> bytes:
    """Validate the native waveform, then saturate to little-endian signed PCM."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Unexpected output sample rate")
    samples = np.asarray(audio)
    if samples.ndim != 1 or not samples.size or samples.dtype.kind != "f":
        raise ValueError("Expected a nonempty mono floating-point waveform")
    if not np.isfinite(samples).all():
        raise ValueError("Nonfinite waveform")
    # Clip before scaling to avoid overflow even for huge but finite floats.
    return np.clip(np.rint(np.clip(samples, -1.0, 1.0) * 32768.0), -32768, 32767).astype("<i2").tobytes()


class LocalKokoroTTSService(TTSService):
    """Local CPU TTS: synthesize one text chunk off-loop, then yield bounded PCM.

    ``synthesize`` must return ``(float_mono_waveform, 24000)`` and must serialize
    native engine access across sessions, including work whose waiter timed out.
    There is no implication of sample-level synthesis streaming: the complete
    input text chunk is synthesized before the first audio frame is available.
    """

    def __init__(
        self,
        *,
        synthesize: Callable[..., tuple[Any, int]],
        voice: str = "af_heart",
        hindi_voice: str = "hf_alpha",
        language: str = "auto",
        model: str = "kokoro-v1.0",
        audio_chunk_ms: int = 100,
        first_chunk_chars: int = 60,
        speed: float = 1.0,
        inference_timeout_seconds: float = 30,
        trace: SessionTrace | None = None,
        **kwargs,
    ):
        if language not in {"auto", "en", "hi"}:
            raise ValueError("Kokoro language must be auto, en, or hi")
        if isinstance(audio_chunk_ms, bool) or not isinstance(audio_chunk_ms, int) or not 20 <= audio_chunk_ms <= 1000:
            raise ValueError("Kokoro audio chunk duration must be 20 to 1000 ms")
        if isinstance(first_chunk_chars, bool) or not isinstance(first_chunk_chars, int) or (
            first_chunk_chars and not 32 <= first_chunk_chars <= 256
        ):
            raise ValueError("Kokoro first chunk characters must be 0 or 32 to 256")
        if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
            raise ValueError("Kokoro speed must be between 0.5 and 2.0")
        if not math.isfinite(inference_timeout_seconds) or inference_timeout_seconds <= 0:
            raise ValueError("Kokoro inference timeout must be positive and finite")
        if not voice.strip() or not hindi_voice.strip():
            raise ValueError("Kokoro voices must be nonempty")
        super().__init__(
            settings=TTSSettings(model=model, voice=voice, language=language),
            sample_rate=SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            append_trailing_space=False,
            **kwargs,
        )
        self._synthesize = synthesize
        self._english_voice = voice
        self._hindi_voice = hindi_voice
        self._local_language = language
        self._local_speed = speed
        self._audio_chunk_ms = audio_chunk_ms
        self._first_chunk_chars = first_chunk_chars
        self._inference_timeout_seconds = inference_timeout_seconds
        self._trace = trace
        self._local_epoch = 0
        self._local_closed = False
        self._first_request_contexts: set[str] = set()

    @property
    def chunk_size(self) -> int:
        return SAMPLE_RATE * self._audio_chunk_ms // 1000 * 2

    def can_generate_metrics(self) -> bool:
        return True

    def _voice_and_language(self, text: str) -> tuple[str, str]:
        hindi = self._local_language == "hi" or (
            self._local_language == "auto" and _has_devanagari_letter(text)
        )
        return (self._hindi_voice, "hi") if hindi else (self._english_voice, "en-us")

    def _invalidate(self, *, close: bool = False) -> None:
        self._local_epoch += 1
        self._first_request_contexts.clear()
        if close:
            self._local_closed = True

    async def on_audio_context_completed(self, context_id: str):
        self._first_request_contexts.discard(context_id)
        await super().on_audio_context_completed(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        self._first_request_contexts.discard(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # System frames may overtake queued synthesis, so invalidate before the
        # base handler awaits task cancellation or audio-context cleanup.
        if isinstance(frame, InterruptionFrame):
            self._invalidate()
            # The base interruption handler stops all metrics. Cancel first
            # so unfinished inference is not reported as time-to-first-audio.
            await self.cancel_ttfb_metrics()
        elif isinstance(frame, CancelFrame):
            self._invalidate(close=True)
            await self.cancel_ttfb_metrics()
        await super().process_frame(frame, direction)

    async def cancel(self, frame: CancelFrame):
        self._invalidate(close=True)
        await self.cancel_ttfb_metrics()
        await super().cancel(frame)

    async def cleanup(self):
        self._invalidate(close=True)
        await self.cancel_ttfb_metrics()
        await super().cleanup()

    async def _await_synthesis(self, text: str, context_id: str, epoch: int,
                               voice: str, lang: str, deadline: float) -> tuple[Any, int] | None:
        """Keep only this existing context alive until inference settles.

        Pipecat 1.11's private refresh hook queues a non-audio sentinel: it resets
        its idle watchdog without recreating a removed context, emitting silence,
        or stopping TTFB. This compatibility boundary is exercised against the
        real native consumer in test_kokoro_watchdog.py.
        """
        if deadline <= time.monotonic():
            raise TimeoutError
        inference = asyncio.create_task(asyncio.to_thread(
            self._synthesize, text, voice=voice, lang=lang, speed=self._local_speed,
        ))
        # Normally 1 second, safely below Pipecat's unchanged 3-second idle limit.
        refresh_seconds = min(1.0, self._stop_frame_timeout_s / 3)
        try:
            # The deadline belongs to the original request, so pieces never get
            # independent 30-second budgets. It includes time queued on the
            # process-wide native engine lock.
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                done, _ = await asyncio.wait(
                    {inference}, timeout=min(refresh_seconds, remaining),
                )
                if self._local_closed or epoch != self._local_epoch:
                    return None
                if deadline <= time.monotonic():
                    raise TimeoutError
                if done:
                    return inference.result()
                # Do not enqueue a keepalive after the original deadline has
                # elapsed; only an actual bounded pending wait may refresh it.
                if deadline - time.monotonic() <= 0:
                    raise TimeoutError
                self._refresh_audio_context(context_id)
        finally:
            # Cancels only the asyncio waiter, not the native thread. The factory
            # lock still serializes late native work across interrupted sessions.
            if not inference.done():
                inference.cancel()
            await asyncio.gather(inference, return_exceptions=True)

    def _synthesis_event(self, event: str, context_id: str, started: float, **fields):
        if self._trace:
            self._trace.emit(event, context_id=context_id,
                             elapsed_seconds=round(time.monotonic() - started, 6), **fields)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if self._local_closed or not self.is_usable or not text.strip():
            return
        epoch = self._local_epoch
        emitted_audio = False
        first_request = context_id not in self._first_request_contexts
        if first_request:
            self._first_request_contexts.add(context_id)
        pieces = split_for_first_audio(text, self._first_chunk_chars) if first_request else (text,)
        voice, lang = self._voice_and_language(text)
        started = time.monotonic()
        deadline = started + self._inference_timeout_seconds
        total_audio_bytes = 0
        self._synthesis_event("local_tts_synthesis_started", context_id, started,
                              text_characters=len(text), language=lang,
                              piece_count=len(pieces),
                              inference_timeout_seconds=self._inference_timeout_seconds)
        try:
            # Usage and language routing remain attached to the caller's single,
            # original request, never to a split synthesis piece.
            await self.start_tts_usage_metrics(text)
            for piece in pieces:
                # The generator can be resumed after a prior piece was yielded.
                # Check before creating another thread task so interruption never
                # starts a later native synthesis request.
                if self._local_closed or epoch != self._local_epoch:
                    self._synthesis_event("local_tts_synthesis_discarded", context_id, started)
                    return
                result = await self._await_synthesis(
                    piece, context_id, epoch, voice, lang, deadline,
                )
                if self._local_closed or epoch != self._local_epoch:
                    self._synthesis_event("local_tts_synthesis_discarded", context_id, started)
                    return
                audio, sample_rate = result
                pcm = _pcm16(audio, sample_rate)
                if deadline <= time.monotonic():
                    raise TimeoutError
                total_audio_bytes += len(pcm)
                for offset in range(0, len(pcm), self.chunk_size):
                    if self._local_closed or epoch != self._local_epoch:
                        return
                    # The native consumer stops TTFB on the first real audio frame.
                    # Stopping it here would introduce an await before the first
                    # yield, where interruption could suppress audio after reporting
                    # a first-audio metric. Keep the validity check and yield atomic.
                    emitted_audio = True
                    yield TTSAudioRawFrame(
                        audio=pcm[offset:offset + self.chunk_size],
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                        context_id=context_id,
                    )
            self._synthesis_event("local_tts_synthesis_finished", context_id, started,
                                  piece_count=len(pieces),
                                  audio_seconds=total_audio_bytes / (SAMPLE_RATE * 2))
        except asyncio.CancelledError:
            self._synthesis_event("local_tts_synthesis_cancelled", context_id, started)
            raise
        except Exception as exc:
            if not self._local_closed and epoch == self._local_epoch:
                timed_out = isinstance(exc, TimeoutError)
                self._synthesis_event("local_tts_synthesis_failed", context_id, started,
                                      reason="inference_timeout" if timed_out else "synthesis_or_audio_error")
                await self.cancel_ttfb_metrics()
                self._invalidate(close=True)
                message = "Local Kokoro synthesis timed out" if timed_out else "Local Kokoro synthesis failed"
                await self.push_error(
                    error_msg=f"{message}. Reconnect to start a fresh session.",
                    force_treat_as_permanent=True,
                )
        finally:
            # A cancelled/late result belongs to an earlier turn. Its finally
            # block must not cancel a newer turn's TTFB measurement.
            if not emitted_audio and not self._local_closed and epoch == self._local_epoch:
                await self.cancel_ttfb_metrics()
