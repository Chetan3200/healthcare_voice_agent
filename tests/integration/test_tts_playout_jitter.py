"""Deterministic synthetic coverage for TTS playout under bursty delivery.

This uses the configured OpenAI TTS chunk-size override, Pipecat's real output
buffering, and SmallWebRTC RawAudioTrack. The provider response and PCM are
fabricated; no network, model, browser, microphone, or speaker is used. This
establishes only that a controlled delivery schedule can expose different
playout-silence behavior. It is not evidence about any recorded live session.
"""

import asyncio
import socket
import time

import numpy as np
import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import OutputAudioRawFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import RawAudioTrack

from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice.providers import build_tts


SAMPLE_RATE = 24_000
BYTES_PER_10_MS = SAMPLE_RATE // 100 * 2
PCM_100_MS = b"\x01\x00" * (SAMPLE_RATE // 10)

# Twenty 100 ms source fragments, hence exactly two seconds of nonzero PCM.
# The first fragment is available immediately, followed by a controlled 300 ms
# stall and a compensating burst. Later delivery is steady. With identical raw
# arrivals, a 100 ms SDK read starts playout before that stall; a 500 ms read
# waits for the first five fragments and starts with a larger contiguous block.
RAW_ARRIVAL_MS = (
    0,
    300,
    310,
    320,
    330,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
)
SOURCE_PCM = PCM_100_MS * len(RAW_ARRIVAL_MS)


def _block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic TTS playout test must not contact a provider")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


class _ScheduledResponse:
    """Aggregate fixed raw arrivals exactly as an HTTP byte chunker would."""

    status_code = 200

    def __init__(self):
        self.yield_arrival_ms = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def iter_bytes(self, chunk_size):
        buffered = bytearray()
        for arrival_ms in RAW_ARRIVAL_MS:
            buffered.extend(PCM_100_MS)
            while len(buffered) >= chunk_size:
                self.yield_arrival_ms.append(arrival_ms)
                yield bytes(buffered[:chunk_size])
                del buffered[:chunk_size]
        if buffered:
            self.yield_arrival_ms.append(RAW_ARRIVAL_MS[-1])
            yield bytes(buffered)


class _UnusedTransport:
    """Only MediaSender.handle_audio_frame is used; no transport task is started."""


async def _configured_tts_frames(audio_chunk_ms, monkeypatch):
    config = load_config(None, environ={
        "LIVE_API_ENABLED": "true",
        "OPENAI_API_KEY": "offline-placeholder",
        "TTS_AUDIO_CHUNK_MS": str(audio_chunk_ms),
    })
    service = build_tts(config)
    response = _ScheduledResponse()

    def create(**kwargs):
        assert kwargs["response_format"] == "pcm"
        return response

    try:
        service._sample_rate = SAMPLE_RATE
        monkeypatch.setattr(
            service._client.audio.speech.with_streaming_response, "create", create,
        )
        frames = [frame async for frame in service.run_tts("Synthetic test.", "test-context")]
    finally:
        await service.cleanup()

    assert service.chunk_size == SAMPLE_RATE * audio_chunk_ms // 1000 * 2
    assert all(isinstance(frame, TTSAudioRawFrame) for frame in frames)
    assert all(
        frame.sample_rate == SAMPLE_RATE
        and frame.num_channels == 1
        and len(frame.audio) % 2 == 0
        for frame in frames
    )
    assert b"".join(frame.audio for frame in frames) == SOURCE_PCM
    return frames, response.yield_arrival_ms


async def _virtual_playout(frames, arrivals_ms):
    params = TransportParams(
        audio_out_enabled=True,
        audio_out_sample_rate=SAMPLE_RATE,
        audio_out_end_silence_secs=0,
    )
    sender = BaseOutputTransport.MediaSender(
        _UnusedTransport(),
        destination=None,
        sample_rate=SAMPLE_RATE,
        audio_chunk_size=SAMPLE_RATE // 100 * params.audio_out_10ms_chunks * 2,
        params=params,
    )
    sender._audio_queue = asyncio.Queue()

    track = RawAudioTrack(SAMPLE_RATE, auto_silence=True)
    # recv() normally paces against wall time. Make every virtual read immediately
    # due while preserving RawAudioTrack's actual queue and silence behavior.
    track._start = time.time() - 60

    next_frame = 0
    stopped = False
    first_arrival_ms = arrivals_ms[0]
    track_frames = []
    output_chunks = []

    # Two seconds of signal plus one second for the controlled gap and trailing
    # observation. This loop contains no real-time sleeps.
    for playout_tick in range(300):
        now_ms = first_arrival_ms + playout_tick * 10
        while next_frame < len(frames) and arrivals_ms[next_frame] <= now_ms:
            await sender.handle_audio_frame(frames[next_frame])
            next_frame += 1

        if next_frame == len(frames) and not stopped:
            await sender.handle_tts_stopped(TTSStoppedFrame())
            stopped = True

        while not sender._audio_queue.empty():
            output = sender._audio_queue.get_nowait()
            sender._audio_queue.task_done()
            if isinstance(output, OutputAudioRawFrame):
                assert output.sample_rate == SAMPLE_RATE
                assert output.num_channels == 1
                assert len(output.audio) == SAMPLE_RATE // 100 * 4 * 2
                assert len(output.audio) % 2 == 0
                output_chunks.append(output.audio)
                track.add_audio_bytes(output.audio)

        frame = await track.recv()
        assert frame.sample_rate == SAMPLE_RATE
        samples = frame.to_ndarray().reshape(-1)
        assert samples.dtype == np.int16
        assert samples.size == SAMPLE_RATE // 100
        track_frames.append(samples.astype("<i2", copy=False).tobytes())

    assert next_frame == len(frames)
    assert b"".join(output_chunks) == SOURCE_PCM

    signal = [chunk != bytes(BYTES_PER_10_MS) for chunk in track_frames]
    assert sum(signal) == len(SOURCE_PCM) // BYTES_PER_10_MS
    first_signal = signal.index(True)
    last_signal = len(signal) - 1 - signal[::-1].index(True)
    internal_zero_frames = sum(not present for present in signal[first_signal : last_signal + 1])

    # Removing auto-silence must recover the exact original PCM bytes in order.
    assert b"".join(chunk for chunk, present in zip(track_frames, signal) if present) == SOURCE_PCM
    await sender.cleanup()
    return internal_zero_frames


def test_configured_chunking_changes_synthetic_bursty_playout_resilience(monkeypatch):
    _block_network(monkeypatch)

    async def exercise():
        results = {}
        for chunk_ms in (100, 500):
            frames, arrivals = await _configured_tts_frames(chunk_ms, monkeypatch)
            results[chunk_ms] = await _virtual_playout(frames, arrivals)
        return results

    zero_frames = asyncio.run(exercise())

    # Exact deterministic result for the fixed virtual schedule: the 100 ms mode
    # begins before the early provider stall. Its first 100 ms SDK frame supplies
    # two complete 40 ms output chunks while 20 ms remains buffered, so playout
    # inserts 220 ms of internal silence. The 500 ms mode begins after the
    # compensating burst and remains continuous. This is a synthetic mechanism
    # test, not a live-session claim.
    assert zero_frames == {100: 22, 500: 0}
