"""Real local TTS adapter/context tests with injected CPU-only fake synthesis.

No model loading, inference, network, microphone, playback, or measured latency.
The fake callback returns numeric arrays, never recorded voice samples.
"""

import asyncio
import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    AggregatedTextFrame, CancelFrame, InterruptionFrame, TTSAudioRawFrame,
    TTSStartedFrame, TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.text.base_text_aggregator import AggregationType

from healthcare_voice_agent.voice.local_tts import LocalKokoroTTSService


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Local TTS unit inference tests must stay offline")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def observe(service, monkeypatch):
    observed = SimpleNamespace(errors=[], usage=[], ttfb=0, cancelled_ttfb=0)

    async def error(**kwargs):
        observed.errors.append(kwargs)

    async def usage(text):
        observed.usage.append(text)

    async def ttfb():
        observed.ttfb += 1

    async def cancel_ttfb():
        observed.cancelled_ttfb += 1

    monkeypatch.setattr(service, "push_error", error)
    monkeypatch.setattr(service, "start_tts_usage_metrics", usage)
    monkeypatch.setattr(service, "stop_ttfb_metrics", ttfb)
    monkeypatch.setattr(service, "cancel_ttfb_metrics", cancel_ttfb)
    return observed


async def collect(service, text="Hello", context="test-context"):
    return [frame async for frame in service.run_tts(text, context)]


@pytest.mark.parametrize("chunk_ms", [20, 100, 500, 1000])
def test_pcm_chunks_preserve_remainder_and_first_audio_metrics(chunk_ms, monkeypatch):
    calls = []
    count = 24000 * chunk_ms // 1000
    samples = np.full(count * 2 + 7, 0.5, dtype=np.float32)

    def synthesize(text, **kwargs):
        calls.append((text, kwargs))
        return samples, 24000

    async def exercise():
        service = LocalKokoroTTSService(synthesize=synthesize, audio_chunk_ms=chunk_ms)
        observed = observe(service, monkeypatch)
        try:
            text = "Hello,  please keep  these spaces. "
            frames = await collect(service, text)
            assert all(isinstance(frame, TTSAudioRawFrame) for frame in frames)
            assert [len(frame.audio) for frame in frames] == [count * 2, count * 2, 14]
            assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in frames)
            assert all(frame.context_id == "test-context" for frame in frames)
            pcm = np.frombuffer(b"".join(frame.audio for frame in frames), dtype="<i2")
            np.testing.assert_array_equal(pcm, np.full(len(samples), 16384, dtype=np.int16))
            assert calls == [(text, {"voice": "af_heart", "lang": "en-us", "speed": 1.0})]
            assert observed.usage == [text]
            # Direct generator calls do not consume an audio context. Native
            # consumer metrics are covered in test_kokoro_watchdog.py instead.
            assert observed.ttfb == 0
            assert not observed.errors
        finally:
            await service.cleanup()
    asyncio.run(exercise())


@pytest.mark.parametrize("language,text,voice,lang", [
    ("auto", "Hello डॉक्टर, appointment is tomorrow.", "hf_alpha", "hi"),
    ("auto", "नमस्ते", "hf_alpha", "hi"),
    ("auto", "Appointment १२३ ।", "af_heart", "en-us"),
    ("auto", "Mera appointment kal hai", "af_heart", "en-us"),
    ("auto", "English", "af_heart", "en-us"),
    ("en", "नमस्ते", "af_heart", "en-us"),
    ("hi", "Hello", "hf_alpha", "hi"),
])
def test_language_routing_never_rewrites_text(language, text, voice, lang):
    calls = []

    def synthesize(actual_text, **kwargs):
        calls.append((actual_text, kwargs))
        return np.array([0.1], np.float32), 24000

    async def exercise():
        service = LocalKokoroTTSService(synthesize=synthesize, language=language)
        try:
            assert len(await collect(service, text)) == 1
            assert calls == [(text, {"voice": voice, "lang": lang, "speed": 1.0})]
        finally:
            await service.cleanup()
    asyncio.run(exercise())


def test_custom_voices_speed_and_saturation():
    calls = []

    def synthesize(text, **kwargs):
        calls.append(kwargs)
        return np.array([-10, -1, -.5, 0, .5, 1, 10], np.float32), 24000

    async def exercise():
        service = LocalKokoroTTSService(
            synthesize=synthesize, language="hi", voice="am_michael", hindi_voice="hm_omega", speed=1.25,
        )
        try:
            frame, = await collect(service)
            assert calls == [{"voice": "hm_omega", "lang": "hi", "speed": 1.25}]
            assert np.frombuffer(frame.audio, dtype="<i2").tolist() == [
                -32768, -32768, -16384, 0, 16384, 32767, 32767,
            ]
        finally:
            await service.cleanup()
    asyncio.run(exercise())


@pytest.mark.parametrize("result", [
    (np.array([], np.float32), 24000),
    (np.array([np.nan], np.float32), 24000),
    (np.array([np.inf], np.float32), 24000),
    (np.zeros((2, 2), np.float32), 24000),
    (np.array([1, 2], np.int16), 24000),
    (np.array([0.1], np.float32), 16000),
    None,
])
def test_invalid_native_output_fails_closed_without_false_first_audio(result, monkeypatch):
    async def exercise():
        service = LocalKokoroTTSService(synthesize=lambda *a, **k: result)
        observed = observe(service, monkeypatch)
        try:
            assert await collect(service) == []
            assert len(observed.errors) == 1
            assert observed.errors[0]["force_treat_as_permanent"] is True
            assert observed.ttfb == 0
            assert observed.cancelled_ttfb == 1
            assert await collect(service, "do not retry") == []
            assert len(observed.errors) == 1
        finally:
            await service.cleanup()
    asyncio.run(exercise())


def test_exception_message_is_not_exposed(monkeypatch):
    def synthesize(*args, **kwargs):
        raise RuntimeError("private-utterance-marker /private/patient.wav")

    async def exercise():
        service = LocalKokoroTTSService(synthesize=synthesize)
        observed = observe(service, monkeypatch)
        try:
            assert await collect(service) == []
            assert "private" not in repr(observed.errors)
            assert "exception" not in observed.errors[0]
            assert observed.errors[0]["force_treat_as_permanent"] is True
        finally:
            await service.cleanup()
    asyncio.run(exercise())


@pytest.mark.parametrize("shutdown", ["cancel", "cleanup", "interrupt"])
def test_slow_synthesis_is_off_loop_and_late_audio_is_discarded(shutdown, monkeypatch):
    release = threading.Event()
    callback_thread = []

    async def exercise():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        main_thread = threading.get_ident()

        def synthesize(*args, **kwargs):
            callback_thread.append(threading.get_ident())
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise AssertionError("test callback was not released")
            return np.ones(2500, np.float32), 24000

        service = LocalKokoroTTSService(synthesize=synthesize)
        service._task_manager = manager = TaskManager()
        observed = observe(service, monkeypatch)

        async def push(*args, **kwargs):
            pass
        monkeypatch.setattr(service, "push_frame", push)
        pending = asyncio.create_task(collect(service))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert callback_thread == [callback_thread[0]]
            assert callback_thread[0] != main_thread
            # Getting here before release proves event-loop responsiveness.
            if shutdown == "cancel":
                await service.cancel(CancelFrame())
            elif shutdown == "cleanup":
                await service.cleanup()
            else:
                await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            # The base interruption handler calls stop_all_metrics (whose
            # TTFB stop is now a no-op because this adapter cancelled it first).
            stops_after_control = observed.ttfb
            cancellations_after_control = observed.cancelled_ttfb
            release.set()
            assert await asyncio.wait_for(pending, 1) == []
            assert observed.ttfb == stops_after_control
            assert observed.cancelled_ttfb == cancellations_after_control
            assert not observed.errors
            if shutdown == "interrupt":
                # Interrupting one utterance does not close the next turn.
                assert len(await collect(service, "new turn")) == 2
            else:
                assert await collect(service, "new turn") == []
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
            await service.cleanup()
            assert not manager.current_tasks()
    asyncio.run(exercise())


def test_cancelled_async_waiter_never_returns_native_result(monkeypatch):
    release = threading.Event()

    async def exercise():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        exited = asyncio.Event()

        def synthesize(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            release.wait(5)
            loop.call_soon_threadsafe(exited.set)
            return np.ones(100, np.float32), 24000

        service = LocalKokoroTTSService(synthesize=synthesize)
        observed = observe(service, monkeypatch)
        pending = asyncio.create_task(collect(service))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            release.set()
            await asyncio.wait_for(exited.wait(), 1)
            assert observed.ttfb == 0
            assert not observed.errors
        finally:
            release.set()
            await service.cleanup()
    asyncio.run(exercise())


def test_timeout_fails_closed_but_does_not_claim_to_stop_native_thread(monkeypatch):
    release = threading.Event()

    async def exercise():
        loop = asyncio.get_running_loop()
        exited = asyncio.Event()

        def synthesize(*args, **kwargs):
            release.wait(5)
            loop.call_soon_threadsafe(exited.set)
            return np.ones(100, np.float32), 24000

        service = LocalKokoroTTSService(synthesize=synthesize, inference_timeout_seconds=0.03)
        observed = observe(service, monkeypatch)
        try:
            assert await asyncio.wait_for(collect(service), 1) == []
            assert len(observed.errors) == 1
            assert observed.ttfb == 0
            assert not exited.is_set()
            release.set()
            await asyncio.wait_for(exited.wait(), 1)
            assert await collect(service, "after timeout") == []
        finally:
            release.set()
            await service.cleanup()
    asyncio.run(exercise())


def test_native_tts_context_has_request_event_start_audio_and_text(monkeypatch):
    requests = []
    text = "Hello डॉक्टर, appointment tomorrow."

    async def exercise():
        service = LocalKokoroTTSService(
            synthesize=lambda *a, **k: (np.ones(2400, np.float32), 24000),
        )
        observe(service, monkeypatch)
        service._sample_rate = 24000

        @service.event_handler("on_tts_request")
        async def request(tts, context_id, actual_text):
            requests.append((context_id, actual_text))

        try:
            await service._push_tts_frames(AggregatedTextFrame(text, AggregationType.SENTENCE))
            await asyncio.sleep(0)  # Let the event callback run; no inference delay.
            assert len(requests) == 1
            context_id, supplied_text = requests[0]
            assert supplied_text == text
            queue = service._audio_contexts[context_id]
            queued = []
            while not queue.empty():
                queued.append(queue.get_nowait())
            assert isinstance(queued[0], TTSStartedFrame)
            assert any(isinstance(frame, TTSAudioRawFrame) for frame in queued)
            assert any(isinstance(frame, TTSTextFrame) and frame.text == text for frame in queued)
            assert service._push_stop_frames is True
            assert all(frame.context_id == context_id for frame in queued)
        finally:
            await service.cleanup()
    asyncio.run(exercise())


@pytest.mark.parametrize("kwargs", [
    {"language": "fr"}, {"audio_chunk_ms": 0}, {"audio_chunk_ms": 100.5},
    {"speed": float("nan")}, {"speed": 3}, {"voice": " "},
    {"inference_timeout_seconds": 0}, {"inference_timeout_seconds": float("inf")},
])
def test_bad_constructor_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LocalKokoroTTSService(synthesize=lambda *a, **k: None, **kwargs)
