"""Offline first-audio regressions with injected synthetic Kokoro callbacks."""

import asyncio
import threading
import time

import numpy as np
import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import InterruptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager

from healthcare_voice_agent.voice.local_tts import LocalKokoroTTSService
from healthcare_voice_agent.voice.speech_chunks import split_for_first_audio


def test_first_piece_pcm_arrives_before_later_native_piece():
    text = "First sentence is ready now. Second sentence waits for its turn."
    pieces = split_for_first_audio(text, 32)
    calls, release = [], threading.Event()

    def synthesize(piece, **kwargs):
        calls.append((piece, kwargs))
        if len(calls) == 2:
            assert release.wait(2)
        return np.ones(100, np.float32), 24000

    async def scenario():
        service = LocalKokoroTTSService(synthesize=synthesize, first_chunk_chars=32)
        generator = service.run_tts(text, "context")
        try:
            frame = await asyncio.wait_for(anext(generator), 1)
            assert isinstance(frame, TTSAudioRawFrame)
            assert calls == [(pieces[0], {"voice": "af_heart", "lang": "en-us", "speed": 1.0})]
            release.set()
            await generator.aclose()
        finally:
            release.set()
            await service.cleanup()
    asyncio.run(scenario())


def test_pieces_share_original_voice_language_and_usage(monkeypatch):
    text = "Hello डॉक्टर, your appointment is confirmed. Please arrive ten minutes early."
    calls, usage = [], []

    def synthesize(piece, **kwargs):
        calls.append((piece, kwargs))
        return np.ones(8, np.float32), 24000

    async def usage_metric(original):
        usage.append(original)

    async def scenario():
        service = LocalKokoroTTSService(synthesize=synthesize, first_chunk_chars=32)
        monkeypatch.setattr(service, "start_tts_usage_metrics", usage_metric)
        try:
            frames = [frame async for frame in service.run_tts(text, "context")]
            assert frames
            assert "".join(piece for piece, _ in calls) == text
            assert usage == [text]
            assert {tuple(sorted(kwargs.items())) for _, kwargs in calls} == {
                (("lang", "hi"), ("speed", 1.0), ("voice", "hf_alpha")),
            }
        finally:
            await service.cleanup()
    asyncio.run(scenario())


def test_total_deadline_is_not_restarted_for_later_pieces(monkeypatch):
    text = "First request has enough words to split. Second request should exceed total time."
    calls = []

    def synthesize(piece, **kwargs):
        calls.append(piece)
        time.sleep(.045)
        return np.ones(8, np.float32), 24000

    async def no_error(**kwargs):
        pass

    async def scenario():
        service = LocalKokoroTTSService(
            synthesize=synthesize, first_chunk_chars=32, inference_timeout_seconds=.07,
        )
        monkeypatch.setattr(service, "push_error", no_error)
        try:
            started = time.monotonic()
            frames = [frame async for frame in service.run_tts(text, "context")]
            elapsed = time.monotonic() - started
            assert frames  # The first piece is valid and yielded before deadline.
            assert len(calls) == 2
            assert service._local_closed
            assert .05 <= elapsed < .12
        finally:
            await service.cleanup()
    asyncio.run(scenario())


def test_interruption_prevents_later_pieces():
    text = "First sentence can be emitted. Second sentence must not be synthesized after interruption."
    pieces = split_for_first_audio(text, 32)
    calls = []

    def synthesize(piece, **kwargs):
        calls.append(piece)
        return np.ones(8, np.float32), 24000

    async def scenario():
        service = LocalKokoroTTSService(synthesize=synthesize, first_chunk_chars=32)
        manager = TaskManager()
        service._task_manager = manager
        generator = service.run_tts(text, "context")
        try:
            assert isinstance(await anext(generator), TTSAudioRawFrame)
            await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            assert await anext(generator, None) is None
            assert calls == [pieces[0]]
        finally:
            await generator.aclose()
            await service.cleanup()
            assert not manager.current_tasks()
    asyncio.run(scenario())


def test_later_request_in_same_context_is_not_split():
    first = "First request can split at this comma, while these remaining words stay together."
    later = "Later request is also long enough but must stay exactly intact in this same context."
    calls = []

    def synthesize(piece, **kwargs):
        calls.append(piece)
        return np.ones(8, np.float32), 24000

    async def scenario():
        service = LocalKokoroTTSService(synthesize=synthesize, first_chunk_chars=32)
        try:
            [frame async for frame in service.run_tts(first, "context")]
            first_count = len(calls)
            [frame async for frame in service.run_tts(later, "context")]
            assert first_count > 1
            assert calls[first_count:] == [later]
        finally:
            await service.cleanup()
    asyncio.run(scenario())


@pytest.mark.parametrize("value", [True, False, 1, 31, 257, 60.0])
def test_invalid_first_chunk_chars_is_rejected(value):
    with pytest.raises(ValueError):
        LocalKokoroTTSService(synthesize=lambda *args, **kwargs: None, first_chunk_chars=value)


def test_lakshadweep_response_uses_two_complete_phrase_requests():
    text = (
        "Yes, Lakshadweep is a group of beautiful islands and a union territory of India, "
        "known for its beaches and marine life."
    )
    expected = (
        "Yes, Lakshadweep is a group of beautiful islands and a union territory of India, ",
        "known for its beaches and marine life.",
    )
    calls = []
    def synthesize(piece, **kwargs):
        calls.append(piece)
        return np.ones(2401, np.float32), 24000
    async def scenario():
        service = LocalKokoroTTSService(synthesize=synthesize, first_chunk_chars=60)
        generator = service.run_tts(text, "lakshadweep")
        try:
            first = await anext(generator)
            assert isinstance(first, TTSAudioRawFrame)
            assert calls == [expected[0]]  # Actual PCM before the second phrase starts.
            rest = [frame async for frame in generator]
            assert calls == list(expected)
            assert [len(frame.audio) for frame in [first, *rest]] == [4800, 2, 4800, 2]
            assert "".join(calls) == text
        finally:
            await generator.aclose()
            await service.cleanup()
    asyncio.run(scenario())


def test_unsplittable_long_sentence_remains_one_native_request():
    text = "The islands are known for their beautiful beaches and colourful coral reefs and abundant marine life."
    calls = []
    def synthesize(piece, **kwargs):
        calls.append(piece)
        return np.ones(8, np.float32), 24000
    async def scenario():
        service = LocalKokoroTTSService(synthesize=synthesize)
        try:
            frames = [frame async for frame in service.run_tts(text, "intact")]
            assert len(frames) == 1
            assert calls == [text]
        finally:
            await service.cleanup()
    asyncio.run(scenario())
