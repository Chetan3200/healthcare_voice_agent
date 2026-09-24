"""Breeze TTS 2 streaming HTTP client, not an in-process CUDA dependency.

Protocol inspected at breezeblue-ai/breeze-tts commit
008f769016b0a24711becd7a4925030bc93f608c, breeze_infer/api.py:
POST /v1/audio/speech accepts multipart text/instruction/cfg_scale/seed and
returns audio/pcm with X-Sample-Rate and X-Sample-Format=s16le. The upstream
server owns the checkpoint and allows only one concurrent request (409 otherwise).
A cancelled client cannot guarantee that already-running GPU work has stopped.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncGenerator, Awaitable
from typing import TypeVar
from urllib.parse import urlsplit

import aiohttp
from pipecat.frames.frames import (
    CancelFrame, EndFrame, Frame, InterruptionFrame, TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

SAMPLE_RATE = 24000
UPSTREAM_REVISION = "008f769016b0a24711becd7a4925030bc93f608c"
_T = TypeVar("_T")


class _StaleRequest(Exception):
    """An interrupted request no longer owns its audio context."""


class BreezeTTSService(TTSService):
    """Stream whole-text voice-design requests into native Pipecat contexts.

    No model/voice field is sent: the server loads one checkpoint, and Breeze
    chooses its voice from ``instructions`` rather than OpenAI voice names.
    There are no retries, redirects, reference uploads, or text rewrites.
    ``inference_timeout_seconds`` bounds the complete HTTP generation, including
    connect, header wait and all audio reads; ``max_audio_seconds`` bounds bytes.
    A final even-byte EOF is all the upstream protocol provides, so a semantically
    incomplete but correctly terminated PCM body cannot be detected here.
    """

    def __init__(
        self, *, base_url: str,
        model: str = "BreezeBlue/Breeze-TTS-2",
        instructions: str = "A warm, calm conversational voice.",
        audio_chunk_ms: int = 100,
        inference_timeout_seconds: float = 30,
        connect_timeout_seconds: float = 5,
        api_key: str | None = None,
        cfg_scale: float = 4,
        seed: int = 42,
        max_audio_seconds: float = 120,
        max_text_chars: int = 4000,
        **kwargs,
    ):
        endpoint = urlsplit(base_url)
        if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
            raise ValueError("Breeze requires an HTTP root URL without credentials, query or fragment")
        if endpoint.scheme == "http" and endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Unencrypted Breeze endpoints are allowed only on loopback")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("Breeze voice design requires nonempty instructions")
        if isinstance(audio_chunk_ms, bool) or not isinstance(audio_chunk_ms, int) or not 20 <= audio_chunk_ms <= 1000:
            raise ValueError("Breeze audio chunk duration must be 20 to 1000 ms")
        for value in (inference_timeout_seconds, connect_timeout_seconds, cfg_scale, max_audio_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Breeze timing, guidance and audio limits must be positive and finite")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Breeze seed must be an unsigned 32-bit integer")
        if isinstance(max_text_chars, bool) or not isinstance(max_text_chars, int) or max_text_chars < 1:
            raise ValueError("Breeze text limit must be a positive integer")
        super().__init__(
            settings=TTSSettings(model=model, voice="S0"),
            sample_rate=SAMPLE_RATE, push_start_frame=True, push_stop_frames=True,
            append_trailing_space=False, **kwargs,
        )
        self._endpoint = base_url.rstrip("/") + "/v1/audio/speech"
        self._instructions = instructions
        self._api_key = api_key
        self._cfg_scale = cfg_scale
        self._seed = seed
        self._audio_chunk_ms = audio_chunk_ms
        self._inference_timeout_seconds = inference_timeout_seconds
        self._connect_timeout_seconds = connect_timeout_seconds
        self._max_audio_bytes = int(max_audio_seconds * SAMPLE_RATE) * 2
        self._max_text_chars = max_text_chars
        self._session: aiohttp.ClientSession | None = None
        self._epoch = 0
        self._closed = False
        self._pending_io: set[asyncio.Task] = set()
        self._responses: set[aiohttp.ClientResponse] = set()

    @property
    def chunk_size(self) -> int:
        return SAMPLE_RATE * self._audio_chunk_ms // 1000 * 2

    def can_generate_metrics(self) -> bool:
        return True

    def _invalidate(self, *, close: bool = False) -> None:
        self._epoch += 1
        self._closed = self._closed or close
        for response in tuple(self._responses):
            response.close()
        for task in tuple(self._pending_io):
            task.cancel()

    async def _close_http(self) -> None:
        pending = tuple(self._pending_io)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._session is not None:
            await self._session.close()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (InterruptionFrame, CancelFrame, EndFrame)):
            self._invalidate(close=isinstance(frame, (CancelFrame, EndFrame)))
            # Native interruption cleanup stops metrics. Cancel TTFB first so
            # an unfinished header wait is not reported as first audio.
            await self.cancel_ttfb_metrics()
        await super().process_frame(frame, direction)

    async def stop(self, frame: EndFrame):
        self._invalidate(close=True)
        await self.cancel_ttfb_metrics()
        await self._close_http()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._invalidate(close=True)
        await self.cancel_ttfb_metrics()
        await self._close_http()
        await super().cancel(frame)

    async def cleanup(self):
        self._invalidate(close=True)
        await self.cancel_ttfb_metrics()
        await self._close_http()
        await super().cleanup()

    def _check_current(self, epoch: int, deadline: float) -> None:
        if self._closed or epoch != self._epoch:
            raise _StaleRequest
        if time.monotonic() >= deadline:
            raise TimeoutError

    async def _await_io(self, awaitable: Awaitable[_T], context_id: str,
                        epoch: int, deadline: float) -> _T:
        # Create only inside a currently valid request. The task is always owned
        # and drained below, including header requests interrupted before return.
        task = asyncio.ensure_future(awaitable)
        self._pending_io.add(task)
        refresh_seconds = min(1.0, self._stop_frame_timeout_s / 3)
        handed_off = False
        try:
            while True:
                self._check_current(epoch, deadline)
                done, _ = await asyncio.wait(
                    {task}, timeout=min(refresh_seconds, deadline - time.monotonic()),
                )
                self._check_current(epoch, deadline)
                if done:
                    result = task.result()
                    handed_off = True
                    return result
                # Pipecat 1.11's native non-audio sentinel only refreshes an
                # existing context. It never emits silence or completes TTFB.
                self._refresh_audio_context(context_id)
        finally:
            result = None
            try:
                if not task.done():
                    task.cancel()
                    result = (await asyncio.gather(task, return_exceptions=True))[0]
                else:
                    # Do not introduce a cancellation point between returning
                    # HTTP headers and handing response ownership to run_tts.
                    try:
                        result = task.result()
                    except BaseException:
                        pass  # Retrieve the exception; the original still propagates.
            finally:
                # If interruption/timeout wins while POST headers arrive, the
                # response may not have entered run_tts's ownership set yet.
                if (not handed_off or self._closed or epoch != self._epoch):
                    if isinstance(result, aiohttp.ClientResponse):
                        result.close()
                self._pending_io.discard(task)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if self._closed or not self.is_usable or not text.strip():
            return
        epoch = self._epoch
        deadline = time.monotonic() + self._inference_timeout_seconds
        response = None
        emitted_audio = False
        try:
            if len(text) > self._max_text_chars:
                raise ValueError("text limit")
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(
                        total=None, connect=self._connect_timeout_seconds,
                    ),
                    auto_decompress=False,
                    trust_env=False,
                )
            form = aiohttp.FormData()
            # A content type forces actual multipart rather than aiohttp's
            # urlencoded optimisation, matching the upstream documented wire API.
            form.add_field("text", text, content_type="text/plain; charset=utf-8")
            form.add_field("instruction", self._instructions)
            form.add_field("cfg_scale", str(self._cfg_scale))
            form.add_field("seed", str(self._seed))
            headers = {"Accept": "audio/pcm"}
            if self._api_key:
                headers["Authorization"] = "Bearer " + self._api_key
            await self.start_tts_usage_metrics(text)
            self._check_current(epoch, deadline)
            response = await self._await_io(
                self._session.post(self._endpoint, data=form, headers=headers,
                                   allow_redirects=False),
                context_id, epoch, deadline,
            )
            self._responses.add(response)
            if (response.status != 200 or response.content_type != "audio/pcm"
                    or response.headers.get("X-Sample-Rate") != "24000"
                    or response.headers.get("X-Sample-Format") != "s16le"
                    or response.headers.get("Content-Encoding", "identity").lower() != "identity"):
                raise ValueError("invalid streaming response")
            if response.content_length is not None and response.content_length > self._max_audio_bytes:
                raise ValueError("audio limit")
            remainder = b""
            byte_count = 0
            while True:
                self._check_current(epoch, deadline)
                chunk = await self._await_io(
                    response.content.read(self.chunk_size), context_id, epoch, deadline,
                )
                self._check_current(epoch, deadline)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > self._max_audio_bytes:
                    raise ValueError("audio limit")
                data = remainder + chunk
                aligned = len(data) - len(data) % 2
                remainder = data[aligned:]
                if aligned:
                    # No await between the epoch/deadline check and yield. The
                    # native context consumer owns true time-to-first-audio.
                    self._check_current(epoch, deadline)
                    emitted_audio = True
                    yield TTSAudioRawFrame(
                        audio=data[:aligned], sample_rate=SAMPLE_RATE,
                        num_channels=1, context_id=context_id,
                    )
            if not byte_count or remainder:
                raise ValueError("empty or unaligned PCM")
        except _StaleRequest:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed and epoch == self._epoch:
                await self.cancel_ttfb_metrics()
                self._invalidate(close=True)
                await self._close_http()
                reason = "timed out" if isinstance(exc, TimeoutError) else "failed"
                await self.push_error(
                    error_msg=f"Breeze synthesis {reason}. Reconnect to start a fresh session.",
                    force_treat_as_permanent=True,
                )
        finally:
            if response is not None:
                response.close()
                self._responses.discard(response)
            if not emitted_audio and not self._closed and epoch == self._epoch:
                await self.cancel_ttfb_metrics()
