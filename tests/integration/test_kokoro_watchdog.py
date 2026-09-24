"""Real Pipecat context-consumer regressions, using synthetic callback arrays.

No native models, downloads, installs, API calls, microphone or playback.
"""

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
    AggregatedTextFrame, CancelFrame, InterruptionFrame, MetricsFrame, StartFrame,
    TTSAudioRawFrame, TTSStoppedFrame,
)
from pipecat.metrics.metrics import TTFAMetricsData, TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.text.base_text_aggregator import AggregationType

from healthcare_voice_agent.voice.local_tts import LocalKokoroTTSService


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Kokoro watchdog regressions must remain offline")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@asynccontextmanager
async def running(monkeypatch, synthesize, **kwargs):
    events = []
    trace = SimpleNamespace(emit=lambda event, **fields: events.append((event, fields)))
    service = LocalKokoroTTSService(synthesize=synthesize, trace=trace, **kwargs)
    frames, errors = [], []
    manager = TaskManager()
    clock = SystemClock()
    clock.start()
    await service.setup(FrameProcessorSetup(
        clock=clock, task_manager=manager, pipeline_worker=SimpleNamespace(app_resources={}),
        enable_metrics=True, enable_usage_metrics=True,
    ))
    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        frames.append(frame)
    async def error(*args, **kwargs):
        errors.append(kwargs.get("error_msg", args[0] if args else "error"))
    monkeypatch.setattr(service, "push_frame", push)
    monkeypatch.setattr(service, "push_error", error)
    await service.start(StartFrame())
    service._turn_context_id = "native-kokoro-test"
    try:
        yield SimpleNamespace(service=service, frames=frames, errors=errors, events=events, manager=manager)
    finally:
        await service.cleanup()
        assert not manager.current_tasks()


def first_audio_metrics(frames):
    return [data for frame in frames if isinstance(frame, MetricsFrame)
            for data in frame.data if isinstance(data, (TTFBMetricsData, TTFAMetricsData))]


@pytest.mark.parametrize("watchdog,delay", [(0.06, 0.22), (3.0, 3.15)])
def test_pending_synthesis_survives_native_idle_watchdog(monkeypatch, watchdog, delay):
    release = threading.Event()
    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        def synthesize(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(8)
            return np.full(2501, .25, np.float32), 24000
        async with running(monkeypatch, synthesize, stop_frame_timeout_s=watchdog) as c:
            pending = asyncio.create_task(c.service._push_tts_frames(
                AggregatedTextFrame("A synthetic slow sentence.", AggregationType.SENTENCE),
            ))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                await asyncio.sleep(delay)
                assert c.errors == []
                assert c.service.audio_context_available("native-kokoro-test")
                assert not any(isinstance(f, (TTSAudioRawFrame, TTSStoppedFrame)) for f in c.frames)
                assert first_audio_metrics(c.frames) == []
                assert [event for event, _ in c.events] == ["local_tts_synthesis_started"]
                release.set()
                await asyncio.wait_for(pending, 2)
                await c.service.on_turn_context_completed()
                await asyncio.wait_for(c.service._serialization_queue.join(), 2)
                audio = [f for f in c.frames if isinstance(f, TTSAudioRawFrame)]
                assert [len(f.audio) for f in audio] == [4800, 202]
                assert c.errors == []
                assert not c.service.get_audio_contexts()
                assert c.service._stop_frame_timeout_s == watchdog
                ttfb = [data for data in first_audio_metrics(c.frames) if isinstance(data, TTFBMetricsData)]
                assert len(ttfb) == 1
                assert ttfb[0].value >= delay
                assert [event for event, _ in c.events] == [
                    "local_tts_synthesis_started", "local_tts_synthesis_finished",
                ]
                assert c.events[-1][1]["elapsed_seconds"] >= delay
                assert c.events[-1][1]["audio_seconds"] == 2501 / 24000
                assert "synthetic slow sentence" not in repr(c.events)
            finally:
                release.set()
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(scenario())


def test_keepalives_do_not_extend_total_inference_deadline(monkeypatch):
    release = threading.Event()
    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        def synthesize(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(3)
            return np.ones(2400, np.float32), 24000
        async with running(monkeypatch, synthesize, stop_frame_timeout_s=.03,
                           inference_timeout_seconds=.18) as c:
            refreshes = []
            refresh = c.service._refresh_audio_context
            def record(context_id):
                refreshes.append(context_id)
                refresh(context_id)
            monkeypatch.setattr(c.service, "_refresh_audio_context", record)
            pending = asyncio.create_task(c.service._push_tts_frames(
                AggregatedTextFrame("Synthetic deadline check.", AggregationType.SENTENCE),
            ))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.wait_for(pending, .8)
                assert any("Local Kokoro synthesis timed out" in error for error in c.errors)
                assert len(refreshes) >= 3
                assert c.service._local_closed
                assert c.events[-1][0] == "local_tts_synthesis_failed"
                assert c.events[-1][1]["reason"] == "inference_timeout"
                assert .15 <= c.events[-1][1]["elapsed_seconds"] < .8
                count = len(refreshes)
                await asyncio.wait_for(c.service._serialization_queue.join(), 1)
                release.set()
                await asyncio.sleep(.06)
                assert len(refreshes) == count
                assert not any(isinstance(f, TTSAudioRawFrame) for f in c.frames)
                assert first_audio_metrics(c.frames) == []
            finally:
                release.set()
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize("shutdown", ["interrupt", "cancel", "cleanup", "cancel_waiter"])
def test_keepalives_stop_and_late_audio_is_discarded_on_shutdown(monkeypatch, shutdown):
    release = threading.Event()
    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        def synthesize(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(3)
            return np.ones(2400, np.float32), 24000
        async with running(monkeypatch, synthesize, stop_frame_timeout_s=.04) as c:
            refreshes = []
            refresh = c.service._refresh_audio_context
            def record(context_id):
                refreshes.append(context_id)
                refresh(context_id)
            monkeypatch.setattr(c.service, "_refresh_audio_context", record)
            pending = asyncio.create_task(c.service._push_tts_frames(
                AggregatedTextFrame("Synthetic shutdown check.", AggregationType.SENTENCE),
            ))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.sleep(.1)
                assert c.errors == []
                assert len(refreshes) >= 3
                if shutdown == "interrupt":
                    await c.service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
                elif shutdown == "cancel":
                    await c.service.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
                elif shutdown == "cleanup":
                    await c.service.cleanup()
                # These direct calls bypass the processor's input-task owner.
                # Mirror that owner's cancellation of the in-flight generator.
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                count = len(refreshes)
                release.set()
                await asyncio.sleep(.07)
                assert len(refreshes) == count
                assert not any(isinstance(f, TTSAudioRawFrame) for f in c.frames)
                assert first_audio_metrics(c.frames) == []
                assert len(c.events) == 2
                assert c.events[0][0] == "local_tts_synthesis_started"
                # Epoch invalidation can beat explicit cancellation while the
                # native interruption handler yields to clean up its context.
                assert c.events[1][0] in {
                    "local_tts_synthesis_cancelled", "local_tts_synthesis_discarded",
                }
                if shutdown == "interrupt":
                    assert not c.service._local_closed
                    assert not c.service.get_audio_contexts()
                    # The interrupted session can still speak a fresh turn.
                    c.service._turn_context_id = "fresh-after-interruption"
                    await c.service._push_tts_frames(AggregatedTextFrame(
                        "New synthetic turn.", AggregationType.SENTENCE,
                    ))
                    await c.service.on_turn_context_completed()
                    # An interrupted native consumer does not task_done its old
                    # queue item. Wait for this context, not queue.join's old count.
                    async with asyncio.timeout(1):
                        while c.service.audio_context_available("fresh-after-interruption"):
                            await asyncio.sleep(.005)
                    audio = [f for f in c.frames if isinstance(f, TTSAudioRawFrame)]
                    assert audio and all(f.context_id == "fresh-after-interruption" for f in audio)
                    assert c.errors == []
            finally:
                release.set()
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(scenario())


def test_unused_context_still_times_out_normally(monkeypatch):
    async def scenario():
        async with running(monkeypatch, lambda *a, **k: None, stop_frame_timeout_s=.03) as c:
            await c.service.create_audio_context("unused-context")
            await asyncio.wait_for(c.service._serialization_queue.join(), 1)
            assert any("completed with no audio" in error for error in c.errors)
            assert not c.service.get_audio_contexts()
            assert c.events == []
            assert first_audio_metrics(c.frames) == []
    asyncio.run(scenario())


def test_backend_failure_keeps_trace_and_errors_sanitized(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("private-marker /private/patient.wav")
    async def scenario():
        async with running(monkeypatch, fail, stop_frame_timeout_s=.03) as c:
            await c.service._push_tts_frames(AggregatedTextFrame(
                "Synthetic error check.", AggregationType.SENTENCE,
            ))
            await asyncio.wait_for(c.service._serialization_queue.join(), 1)
            assert any("Local Kokoro synthesis failed" in error for error in c.errors)
            assert c.events[-1][0] == "local_tts_synthesis_failed"
            assert c.events[-1][1]["reason"] == "synthesis_or_audio_error"
            assert "private-marker" not in repr(c.events) + repr(c.errors)
            assert "patient.wav" not in repr(c.events) + repr(c.errors)
            assert first_audio_metrics(c.frames) == []
    asyncio.run(scenario())
