"""Offline real-Pipecat lifecycle tests against the inspected C++ wire contract."""
import asyncio
import json
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
pytest.importorskip("pipecat")
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame, EndFrame, InputAudioRawFrame, StartFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from healthcare_voice_agent.voice import nemotron_stt as module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Nemotron protocol tests must never access a server")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


class FakeSocket:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.messages.put_nowait(json.dumps({"type": "session.created"}))
        self.messages.put_nowait(json.dumps({"type": "session.updated"}))
        self.sent = []
        self.closed = False
        self.number = 0

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        return await self.messages.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.recv()

    async def close(self):
        self.closed = True

    def event(self, kind, **kwargs):
        self.number += 1
        data = {"type": kind, "event_id": f"event_{self.number}", **kwargs}
        self.messages.put_nowait(json.dumps(data))
        return data


@asynccontextmanager
async def adapter(monkeypatch, **kwargs):
    ws, frames, errors, connections = FakeSocket(), [], [], []
    async def connect(url, **options):
        connections.append((url, options))
        return ws
    monkeypatch.setattr(module, "connect", connect)
    service = module.NemotronSTTService(
        model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        base_url="ws://127.0.0.1:8080/v1/audio/transcriptions/realtime", **kwargs,
    )
    manager = TaskManager()
    await service.setup(FrameProcessorSetup(
        clock=SystemClock(), task_manager=manager,
        pipeline_worker=SimpleNamespace(app_resources={}), audio_in_sample_rate=16000,
    ))
    async def push(self, frame, direction=FrameDirection.DOWNSTREAM):
        frames.append(frame)
    async def error(**kw):
        errors.append(kw)
    monkeypatch.setattr(FrameProcessor, "push_frame", push)
    monkeypatch.setattr(service, "push_error", error)
    await service.start(StartFrame())
    try:
        yield SimpleNamespace(service=service, ws=ws, frames=frames, errors=errors,
                              connections=connections, manager=manager)
    finally:
        await service.cleanup()
        assert ws.closed
        assert not manager.current_tasks()


async def until(condition):
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0.001)


async def start(c):
    await c.service.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2), FrameDirection.UPSTREAM)


async def stop(c):
    await c.service.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), FrameDirection.UPSTREAM)


async def audio(c, data=b"\x01\x00" * 320):
    await c.service.process_frame(InputAudioRawFrame(data, 16000, 1), FrameDirection.DOWNSTREAM)


def transcripts(c):
    return [f for f in c.frames if isinstance(f, TranscriptionFrame)]


def commits(c):
    return [x for x in c.ws.sent if isinstance(x, str) and json.loads(x)["type"] == "input_audio_buffer.commit"]


async def complete(c, text="hello"):
    c.ws.event("conversation.item.input_audio_transcription.completed", transcript=text)
    c.ws.event("input_audio_buffer.committed")
    await asyncio.sleep(0.005)


def test_binary_audio_streams_before_stop_and_completion_waits_for_ack(monkeypatch):
    async def run():
        async with adapter(monkeypatch, api_key="private-test-key") as c:
            assert c.connections[0][1]["additional_headers"] == {"Authorization": "Bearer private-test-key"}
            session = json.loads(c.ws.sent[0])["session"]
            assert session["sample_rate"] == 16000 and session["language"] == "auto"
            await audio(c, b"\x02\x00")  # retained pre-roll, not sent yet
            assert len(c.ws.sent) == 1
            await start(c)
            await audio(c)
            await until(lambda: len(c.ws.sent) == 3)
            assert c.ws.sent[1] == b"\x02\x00" and isinstance(c.ws.sent[2], bytes)
            assert not commits(c)
            c.ws.event("conversation.item.input_audio_transcription.delta", delta="partial")
            c.ws.event("conversation.item.input_audio_transcription.completed", transcript="first phrase")
            await asyncio.sleep(0.005)
            assert not transcripts(c)
            await stop(c)
            await until(lambda: len(commits(c)) == 1)
            await complete(c, "second phrase")
            frame, = transcripts(c)
            assert frame.text == "first phrase second phrase" and frame.finalized
            assert frame.metadata["stt_speech_generation"] == 1
            assert not c.errors
    asyncio.run(run())


def test_resumed_speech_preserves_fifo_and_old_completion_cannot_finalize(monkeypatch):
    async def run():
        async with adapter(monkeypatch) as c:
            await start(c); await audio(c); await stop(c)
            await start(c); await audio(c)
            await until(lambda: len(commits(c)) == 1)
            await complete(c, "old phrase")
            assert transcripts(c)[0].finalized is False
            await stop(c)
            await until(lambda: len(commits(c)) == 2)
            await complete(c, "new phrase")
            assert [f.text for f in transcripts(c)] == ["old phrase", "new phrase"]
            assert transcripts(c)[1].finalized is True
            assert [f.metadata["stt_speech_generation"] for f in transcripts(c)] == [1, 2]
    asyncio.run(run())


def test_duplicate_stop_and_event_do_not_duplicate_transcription(monkeypatch):
    async def run():
        async with adapter(monkeypatch) as c:
            await start(c); await audio(c); await stop(c); await stop(c)
            await until(lambda: len(commits(c)) == 1)
            event = c.ws.event("conversation.item.input_audio_transcription.completed", transcript="once")
            c.ws.messages.put_nowait(json.dumps(event))
            ack = c.ws.event("input_audio_buffer.committed")
            c.ws.messages.put_nowait(json.dumps(ack))
            await until(lambda: transcripts(c))
            await asyncio.sleep(0.005)
            assert len(transcripts(c)) == 1 and transcripts(c)[0].text == "once"
            assert not c.errors
    asyncio.run(run())


@pytest.mark.parametrize("kind,fields", [
    ("error", {"error": {"message": "private token transcript"}}),
    ("conversation.item.input_audio_transcription.completed", {"transcript": None}),
    ("conversation.item.input_audio_transcription.delta", {"delta": 12}),
    ("input_audio_buffer.cleared", {}),
    ("input_audio_buffer.committed", {}),  # no commit has been sent yet
])
def test_protocol_errors_fail_closed_and_sanitized(monkeypatch, kind, fields):
    async def run():
        async with adapter(monkeypatch) as c:
            await start(c); await audio(c)
            c.ws.event(kind, **fields)
            await until(lambda: c.errors)
            assert c.errors[0]["force_treat_as_permanent"] is True
            assert "private" not in repr(c.errors)
            assert not transcripts(c) and c.ws.closed and not c.service._active
    asyncio.run(run())


def test_empty_completion_never_emits_finalized_text(monkeypatch):
    async def run():
        async with adapter(monkeypatch) as c:
            await start(c); await audio(c); await stop(c)
            await until(lambda: commits(c))
            await complete(c, "  ")
            assert not transcripts(c) and not c.errors and not c.service._segments
    asyncio.run(run())


def test_completion_deadline_fails_closed(monkeypatch):
    async def run():
        async with adapter(monkeypatch, inference_timeout_seconds=0.025) as c:
            await start(c); await audio(c); await stop(c)
            await until(lambda: c.errors)
            assert "deadline" in c.errors[0]["error_msg"] and c.ws.closed
            await complete(c, "late")
            assert not transcripts(c)
    asyncio.run(run())


@pytest.mark.parametrize("method,frame", [("cancel", CancelFrame), ("stop", EndFrame), ("cleanup", None)])
def test_teardown_discards_pending_results(monkeypatch, method, frame):
    async def run():
        async with adapter(monkeypatch) as c:
            await start(c); await audio(c); await stop(c)
            if frame:
                await getattr(c.service, method)(frame())
            else:
                await c.service.cleanup()
            await complete(c, "late")
            assert not transcripts(c) and not c.errors and c.ws.closed
    asyncio.run(run())


def test_pending_and_duration_limits(monkeypatch):
    async def run():
        async with adapter(monkeypatch, max_pending_segments=1) as c:
            await start(c); await audio(c); await stop(c); await start(c)
            assert c.errors and not c.service._active
        async with adapter(monkeypatch, max_segment_seconds=0.005) as c:
            await start(c); await audio(c)
            assert c.errors and not c.service._active
    asyncio.run(run())


def test_disabled_finalization_still_retains_generation_guard(monkeypatch):
    async def run():
        async with adapter(monkeypatch, finalize_transcripts=False) as c:
            await start(c); await audio(c); await stop(c)
            await until(lambda: commits(c))
            await complete(c)
            assert not transcripts(c)[0].finalized
            assert transcripts(c)[0].metadata["stt_speech_generation"] == 1
    asyncio.run(run())


def test_malformed_event_is_sanitized(monkeypatch):
    async def run():
        async with adapter(monkeypatch) as c:
            c.ws.messages.put_nowait("{private invalid json")
            await until(lambda: c.errors)
            assert "private" not in repr(c.errors) and c.ws.closed
    asyncio.run(run())


def test_expired_ack_cannot_win_race_with_watchdog(monkeypatch):
    async def run():
        async with adapter(monkeypatch) as c:
            await start(c); await audio(c); await stop(c)
            await until(lambda: commits(c))
            c.service._segments[0].deadline = asyncio.get_running_loop().time() - 1
            await complete(c, "late")
            await until(lambda: c.errors)
            assert not transcripts(c)
    asyncio.run(run())


def test_send_failure_discards_following_audio(monkeypatch):
    async def run():
        async with adapter(monkeypatch) as c:
            async def fail(data):
                raise RuntimeError("private provider token")
            c.ws.send = fail
            await start(c); await audio(c)
            await until(lambda: c.errors)
            assert "private" not in repr(c.errors) and c.ws.closed
            assert c.service._outgoing.empty() and not transcripts(c)
    asyncio.run(run())


def test_cleanup_during_connection_cannot_resurrect_socket(monkeypatch):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        ws = FakeSocket()
        async def connect(url, **options):
            assert options["proxy"] is None
            entered.set()
            await release.wait()
            return ws
        monkeypatch.setattr(module, "connect", connect)
        service = module.NemotronSTTService(model="nemotron-3.5-asr-streaming-0.6b",
            base_url="ws://127.0.0.1:8080/v1/audio/transcriptions/realtime")
        manager = TaskManager()
        await service.setup(FrameProcessorSetup(clock=SystemClock(), task_manager=manager,
            pipeline_worker=SimpleNamespace(app_resources={}), audio_in_sample_rate=16000))
        pending = asyncio.create_task(service.start(StartFrame()))
        await asyncio.wait_for(entered.wait(), 1)
        await service.cleanup()
        release.set()
        await asyncio.wait_for(pending, 1)
        assert ws.closed
        assert service._socket is None
        assert not service._workers and not manager.current_tasks()
    asyncio.run(run())
