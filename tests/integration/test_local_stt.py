"""Actual segmented Pipecat paths with synchronous fake inference, no network."""

import asyncio
import socket
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pipecat")

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame, EndFrame, InputAudioRawFrame, StartFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager

from healthcare_voice_agent.voice.local_stt import LocalWhisperSTTService


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Local STT tests must never connect or download models")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@asynccontextmanager
async def adapter(monkeypatch, transcribe=lambda _: {"text": "hello", "language": "en"}, **kwargs):
    service = LocalWhisperSTTService(transcribe=transcribe, model="offline-model", **kwargs)
    frames, errors = [], []
    manager = TaskManager()
    await service.setup(FrameProcessorSetup(
        clock=SystemClock(), task_manager=manager,
        pipeline_worker=SimpleNamespace(app_resources={}), audio_in_sample_rate=16000,
    ))

    async def push(self, frame, direction=FrameDirection.DOWNSTREAM):
        frames.append(frame)

    async def error(**kwargs):
        errors.append(kwargs)

    monkeypatch.setattr(FrameProcessor, "push_frame", push)
    monkeypatch.setattr(service, "push_error", error)
    await service.start(StartFrame())
    try:
        yield SimpleNamespace(service=service, frames=frames, errors=errors, manager=manager)
    finally:
        await service.cleanup()
        assert not manager.current_tasks()


async def speech_start(service):
    await service.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2), FrameDirection.UPSTREAM)


async def speech_stop(service):
    await service.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), FrameDirection.UPSTREAM)


async def pcm(service, values=(1, 2, 3)):
    audio = np.asarray(values, dtype="<i2").tobytes()
    await service.process_frame(InputAudioRawFrame(audio, 16000, 1), FrameDirection.DOWNSTREAM)


async def segment(service, values=(1, 2, 3)):
    await speech_start(service)
    await pcm(service, values)
    await speech_stop(service)


def transcripts(c):
    return [f for f in c.frames if isinstance(f, TranscriptionFrame)]


async def until(condition):
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0.001)


def test_pcm_conversion_generation_finalization_and_trailing_silence(monkeypatch):
    seen = []
    def infer(audio):
        seen.append(audio)
        return {"text": " नमस्ते world ", "language": "hi", "segments": []}

    async def run():
        async with adapter(monkeypatch, infer) as c:
            await segment(c.service, (-32768, 0, 32767))
            await until(lambda: transcripts(c))
            f, = transcripts(c)
            assert f.text == "नमस्ते world" and f.language == "hi"
            assert f.finalized is True
            assert f.metadata == {"stt_speech_generation": 1, "stt_segment_sequence": 1}
            assert seen[0].dtype == np.float32
            np.testing.assert_array_equal(seen[0][:3], [-1, 0, 32767 / 32768])
            assert len(seen[0]) == 8003  # Native 0.5 s trailing silence.
            assert not np.any(seen[0][3:])
            assert not c.errors
    asyncio.run(run())


def test_slow_inference_is_off_loop_and_resumed_segments_remain_fifo(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def infer(audio):
        calls.append(int(audio[0] * 32768))
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        return {"text": "earlier phrase" if len(calls) == 1 else "corrected ending"}

    async def run():
        async with adapter(monkeypatch, infer) as c:
            try:
                await segment(c.service, (1,))
                await until(entered.is_set)
                # A blocking decoder would deadlock the event loop here.
                await asyncio.wait_for(segment(c.service, (2,)), timeout=0.2)
                assert calls == [1]
                release.set()
                await until(lambda: len(transcripts(c)) == 2)
                first, second = transcripts(c)
                assert [f.text for f in transcripts(c)] == ["earlier phrase", "corrected ending"]
                assert first.finalized is False and second.finalized is True
                assert first.metadata["stt_speech_generation"] == 1
                assert second.metadata["stt_speech_generation"] == 2
                assert calls == [1, 2]
            finally:
                release.set()
    asyncio.run(run())


def test_old_local_frame_cannot_satisfy_real_turn_strategy(monkeypatch):
    from tests.integration.test_finalized_turn_strategy import strategy_only, stopped_speech, expire

    async def run():
        async with adapter(monkeypatch) as c:
            await segment(c.service)
            await until(lambda: transcripts(c))
            queued = transcripts(c)[0]
            assert queued.finalized
            async with strategy_only() as (strategy, manager, stopped):
                await stopped_speech(strategy)
                await stopped_speech(strategy)  # VAD overtakes already queued data.
                await strategy.process_frame(queued)
                await expire(strategy, manager)
                assert stopped == [] and strategy._text == ""
    asyncio.run(run())


def test_duplicate_stops_do_not_transcribe_twice(monkeypatch):
    calls = []
    def infer(audio):
        calls.append(audio)
        return {"text": "one"}

    async def run():
        async with adapter(monkeypatch, infer) as c:
            await segment(c.service)
            await speech_stop(c.service)
            await until(lambda: transcripts(c))
            assert len(calls) == 1 and c.service._segment_sequence == 1
    asyncio.run(run())


def test_empty_and_zero_audio_are_not_sent_to_generative_decoder(monkeypatch):
    def never(audio):
        raise AssertionError("All-zero audio must not be decoded")

    async def run():
        async with adapter(monkeypatch, never) as c:
            await segment(c.service, ())
            await segment(c.service, (0, 0, 0))
            await until(lambda: c.service._pending_count == 0)
            assert transcripts(c) == [] and not c.errors
    asyncio.run(run())


def test_empty_recognition_does_not_emit_a_final_flag(monkeypatch):
    async def run():
        async with adapter(monkeypatch, lambda _: {"text": "   "}) as c:
            await segment(c.service)
            await until(lambda: c.service._pending_count == 0)
            assert transcripts(c) == [] and not c.errors
    asyncio.run(run())


@pytest.mark.parametrize("response", [None, {"text": None}, {}])
def test_invalid_results_fail_closed(monkeypatch, response):
    async def run():
        async with adapter(monkeypatch, lambda _: response) as c:
            await segment(c.service)
            await until(lambda: c.errors)
            assert c.errors[0]["force_treat_as_permanent"] is True
            assert not c.service._active and not transcripts(c)
    asyncio.run(run())


def test_inference_exception_is_sanitized_and_discards_following_segments(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def infer(audio):
        calls.append(audio)
        entered.set()
        assert release.wait(2)
        raise RuntimeError("private-transcript-marker secret-provider-token")

    async def run():
        async with adapter(monkeypatch, infer) as c:
            try:
                await segment(c.service)
                await until(entered.is_set)
                await segment(c.service)
                release.set()
                await until(lambda: c.errors)
                assert len(calls) == 1 and not transcripts(c)
                assert "private" not in repr(c.errors) and "secret" not in repr(c.errors)
                assert c.service._segment_queue.empty()
                assert c.errors[0]["force_treat_as_permanent"] is True
            finally:
                release.set()
    asyncio.run(run())


@pytest.mark.parametrize("method,frame", [("cancel", CancelFrame), ("stop", EndFrame), ("cleanup", None)])
def test_teardown_discards_inflight_result(monkeypatch, method, frame):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    def infer(audio):
        entered.set()
        assert release.wait(2)
        returned.set()
        return {"text": "must never reach the next session"}

    async def run():
        async with adapter(monkeypatch, infer) as c:
            try:
                await segment(c.service)
                await until(entered.is_set)
                await segment(c.service)
                if frame:
                    await getattr(c.service, method)(frame())
                else:
                    await c.service.cleanup()
                release.set()
                await until(returned.is_set)
                assert not transcripts(c) and c.service._segment_queue.empty()
                assert c.service._audio_buffer == b""
                await speech_start(c.service)
                await pcm(c.service)
                await speech_stop(c.service)
                assert not transcripts(c)
            finally:
                release.set()
    asyncio.run(run())


def test_timeout_is_permanent_and_late_thread_result_is_discarded(monkeypatch):
    release = threading.Event()
    def infer(audio):
        assert release.wait(2)
        return {"text": "too late"}

    async def run():
        async with adapter(monkeypatch, infer, inference_timeout_seconds=0.02) as c:
            try:
                await segment(c.service)
                await until(lambda: c.errors)
                assert "deadline" in c.errors[0]["error_msg"]
                assert c.errors[0]["force_treat_as_permanent"] is True
                release.set()
                assert not transcripts(c)
            finally:
                release.set()
    asyncio.run(run())


def test_segment_duration_limit_fails_instead_of_truncating(monkeypatch):
    async def run():
        async with adapter(monkeypatch, max_segment_seconds=0.001) as c:
            await speech_start(c.service)
            await pcm(c.service, range(17))  # 16 samples allowed.
            assert c.errors and not c.service._active
            assert not c.service._audio_buffer and not transcripts(c)
    asyncio.run(run())


def test_pending_limit_includes_inflight_job(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def infer(audio):
        entered.set()
        assert release.wait(2)
        return {"text": "old"}

    async def run():
        async with adapter(monkeypatch, infer, max_pending_segments=1) as c:
            try:
                await segment(c.service)
                await until(entered.is_set)
                await segment(c.service)
                assert c.errors and "queue" in c.errors[0]["error_msg"]
                assert not transcripts(c)
            finally:
                release.set()
    asyncio.run(run())


def test_wrong_audio_format_is_rejected(monkeypatch):
    async def run():
        async with adapter(monkeypatch) as c:
            await c.service.process_frame(InputAudioRawFrame(b"\x00\x00", 24000, 1), FrameDirection.DOWNSTREAM)
            assert c.errors and "format" in c.errors[0]["error_msg"]
    asyncio.run(run())


def test_actual_aggregator_preserves_ordered_local_segments(monkeypatch):
    from tests.integration.test_finalized_turn_strategy import aggregator, feed
    entered, release = threading.Event(), threading.Event()
    count = 0
    def infer(audio):
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert release.wait(2)
            return {"text": "Use the right side,"}
        return {"text": "not the left."}

    async def run():
        async with adapter(monkeypatch, infer) as c:
            try:
                await segment(c.service)
                await until(entered.is_set)
                await segment(c.service)
                release.set()
                await until(lambda: len(transcripts(c)) == 2)
                async with aggregator() as b:
                    for _ in range(2):
                        await feed(b, VADUserStartedSpeakingFrame())
                        await feed(b, VADUserStoppedSpeakingFrame(stop_secs=0.2))
                    await feed(b, transcripts(c)[0])
                    assert b.committed == []
                    await feed(b, transcripts(c)[1])
                    await asyncio.wait_for(b.ready.wait(), 1)
                    assert b.committed == ["Use the right side, not the left."]
            finally:
                release.set()
    asyncio.run(run())


def test_stale_base_vad_flag_cannot_truncate_active_audio(monkeypatch):
    async def run():
        async with adapter(monkeypatch) as c:
            await speech_start(c.service)
            c.service._user_speaking = False  # Simulate delayed old base stop.
            await pcm(c.service, [1] * 20000)
            assert len(c.service._audio_buffer) == 40000  # More than 1s pre-roll.
            await speech_stop(c.service)
            await until(lambda: transcripts(c))
            assert transcripts(c)[0].metadata["stt_speech_generation"] == 1
    asyncio.run(run())


def test_speech_identity_is_reserved_before_native_stop_suspends(monkeypatch):
    stop_entered, release_stop = asyncio.Event(), asyncio.Event()
    calls = []
    original = LocalWhisperSTTService._handle_vad_user_stopped_speaking
    async def delayed_stop(self, frame):
        if self._speech_generation == 1:
            stop_entered.set()
            await release_stop.wait()
        await original(self, frame)

    async def run():
        monkeypatch.setattr(LocalWhisperSTTService, "_handle_vad_user_stopped_speaking", delayed_stop)
        async with adapter(monkeypatch, lambda audio: calls.append(int(audio[0] * 32768)) or {"text": "piece"}) as c:
            await speech_start(c.service)
            await pcm(c.service, (1,))
            old_stop = asyncio.create_task(speech_stop(c.service))
            try:
                await stop_entered.wait()
                assert c.service._segment_sequence == 1
                await segment(c.service, (2,))
                release_stop.set()
                await old_stop
                await until(lambda: len(transcripts(c)) == 2)
                assert calls == [1, 2]
                assert [f.metadata["stt_speech_generation"] for f in transcripts(c)] == [1, 2]
            finally:
                release_stop.set()
                await old_stop
    asyncio.run(run())


@pytest.mark.parametrize("decode_path", ["single_encode", "stock_fallback", None, "private-marker", []])
def test_decode_path_metadata_is_allowlisted(monkeypatch, decode_path):
    async def run():
        async with adapter(monkeypatch, lambda _: {"text": "synthetic", "language": "en", "local_decode_path": decode_path}) as c:
            await segment(c.service)
            await until(lambda: transcripts(c))
            frame, = transcripts(c)
            if isinstance(decode_path, str) and decode_path in {"single_encode", "stock_fallback"}:
                assert frame.metadata["stt_decode_path"] == decode_path
            else:
                assert "stt_decode_path" not in frame.metadata
    asyncio.run(run())
