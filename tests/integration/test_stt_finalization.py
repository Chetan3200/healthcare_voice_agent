"""Real pinned STT adapter hooks, with sockets and provider sends replaced.

No audio or provider inference. VAD controls traverse the actual native
process_frame / commit path; only external I/O and trace sinks are mocked.
"""

import asyncio
import json
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    InterimTranscriptionFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager

from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice.providers import build_stt


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("STT finalization tests must remain offline")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@asynccontextmanager
async def adapter(monkeypatch, *, enabled=True, model="gpt-live-transcribe"):
    config = load_config(None, environ={
        "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "offline-placeholder",
        "STT_MODEL": model,
        "STT_FINALIZE_TRANSCRIPTS": str(enabled).lower(),
    })
    service = build_stt(config)
    service._task_manager = manager = TaskManager()
    captured = SimpleNamespace(service=service, frames=[], sends=[], errors=[], on_push=None)

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        if captured.on_push:
            await captured.on_push(frame)
        captured.frames.append(frame)

    async def send(message):
        captured.sends.append(message)

    async def error(**kwargs):
        captured.errors.append(kwargs)

    monkeypatch.setattr(service, "push_frame", push)
    monkeypatch.setattr(service, "_ws_send", send)
    monkeypatch.setattr(service, "push_error", error)
    try:
        yield captured
    finally:
        service._websocket = None
        await service.cleanup()
        assert not manager.current_tasks()


async def speech_start(service):
    await service.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2), FrameDirection.UPSTREAM)


async def speech_stop(service):
    await service.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2), FrameDirection.UPSTREAM)


async def ack(service, item_id, previous=None):
    await service._handle_commit_acknowledged({"item_id": item_id, "previous_item_id": previous})


async def complete(service, item_id, text):
    await service._handle_transcription_completed({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id, "content_index": 0, "transcript": text,
    })


def transcripts(captured):
    return [frame for frame in captured.frames if isinstance(frame, TranscriptionFrame)]


@pytest.mark.parametrize("enabled", [True, False])
def test_gpt_live_session_lifecycle_sends_plural_english_languages_only(enabled, monkeypatch):
    async def exercise():
        async with adapter(monkeypatch, enabled=enabled) as c:
            s = c.service
            messages = []

            class FakeSocket:
                async def send(self, payload):
                    messages.append(json.loads(payload))

            s._websocket = FakeSocket()
            monkeypatch.setattr(s, "_ws_send", type(s)._ws_send.__get__(s))

            # Initial connection and every reconnect receive session.created.
            await s._handle_session_created({})
            await s._handle_session_created({})

            # A runtime settings update while ready sends the same corrected shape.
            s._session_ready = True
            await s._update_settings(type(s._settings)(noise_reduction="near_field"))

            assert len(messages) == 3
            for request in messages:
                assert request["type"] == "session.update"
                input_audio = request["session"]["audio"]["input"]
                assert input_audio["transcription"] == {
                    "model": "gpt-live-transcribe",
                    "languages": ["en"],
                }
                assert "language" not in input_audio["transcription"]
                assert input_audio["turn_detection"] is None
            assert messages[-1]["session"]["audio"]["input"]["noise_reduction"] == {
                "type": "near_field",
            }
    asyncio.run(exercise())


def test_legacy_realtime_model_keeps_singular_language_field(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch, model="gpt-4o-transcribe") as c:
            s = c.service
            messages = []

            class FakeSocket:
                async def send(self, payload):
                    messages.append(json.loads(payload))

            s._websocket = FakeSocket()
            monkeypatch.setattr(s, "_ws_send", type(s)._ws_send.__get__(s))
            await s._handle_session_created({})

            request, = messages
            transcription = request["session"]["audio"]["input"]["transcription"]
            assert transcription == {"model": "gpt-4o-transcribe", "language": "en"}
            assert "languages" not in transcription
    asyncio.run(exercise())


def test_real_commit_path_marks_a_complete_current_transcript_final(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            await speech_start(s)
            await speech_stop(s)
            assert c.sends == [{"type": "input_audio_buffer.commit"}]
            await ack(s, "item-one")
            await complete(s, "item-one", "Tell me about Bangalore.")
            frame, = transcripts(c)
            assert frame.text == "Tell me about Bangalore."
            assert frame.finalized is True
            assert frame.metadata["stt_speech_generation"] == 1
            assert frame.metadata["stt_commit_sequences"] == [1]
            assert frame.metadata["stt_item_ids"] == ["item-one"]
            assert len(frame.metadata["stt_completion_received_at"]) == 1
            assert s._transcript_commits.pending_count == 0
            assert not c.errors
    asyncio.run(exercise())


def test_resumed_speech_and_reverse_completions_produce_one_ordered_frame(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            await speech_start(s)
            await speech_stop(s)
            await ack(s, "one")
            await speech_start(s)
            await speech_stop(s)
            await ack(s, "two", "one")
            await complete(s, "two", "not the left.")
            assert transcripts(c) == []
            await complete(s, "one", "I mean the right side,")
            frame, = transcripts(c)
            assert frame.text == "I mean the right side, not the left."
            assert frame.finalized is True
            assert frame.metadata["stt_item_ids"] == ["one", "two"]
            assert frame.metadata["stt_speech_generation"] == 2
            await complete(s, "one", "duplicate must not be appended")
            await s._handle_transcription_delta({"item_id": "one", "delta": "late partial"})
            assert len(transcripts(c)) == 1
            assert not any(isinstance(f, InterimTranscriptionFrame) for f in c.frames)
            assert not c.errors
    asyncio.run(exercise())


def test_new_commit_is_reserved_before_native_stop_handling_awaits(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            await speech_start(s)
            await speech_stop(s)
            await ack(s, "one")
            await speech_start(s)

            async def completion_during_stop(frame):
                if isinstance(frame, VADUserStoppedSpeakingFrame):
                    assert s._transcript_commits.pending_count == 2
                    await complete(s, "one", "first piece")
                    assert transcripts(c) == []

            c.on_push = completion_during_stop
            await speech_stop(s)
            c.on_push = None
            await ack(s, "two", "one")
            await complete(s, "two", "second piece")
            assert [f.text for f in transcripts(c)] == ["first piece second piece"]
            assert not c.errors
    asyncio.run(exercise())


def test_resumption_during_native_stop_does_not_relabel_the_old_commit(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            await speech_start(s)

            async def resume(frame):
                if isinstance(frame, VADUserStoppedSpeakingFrame):
                    c.on_push = None
                    await speech_start(s)

            c.on_push = resume
            await speech_stop(s)
            await ack(s, "one")
            assert s._transcript_commits.item_generation("one") == 1
            await complete(s, "one", "a paused sentence")
            assert transcripts(c) == []
            await speech_stop(s)
            await ack(s, "two", "one")
            await complete(s, "two", "with its ending")
            frame, = transcripts(c)
            assert frame.text == "a paused sentence with its ending"
            assert frame.metadata["stt_speech_generation"] == 2
    asyncio.run(exercise())


def test_generation_tag_survives_resumption_while_frame_is_being_pushed(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            await speech_start(s)
            await speech_stop(s)
            await ack(s, "one")

            async def resume(frame):
                if isinstance(frame, TranscriptionFrame):
                    c.on_push = None
                    await speech_start(s)

            c.on_push = resume
            await complete(s, "one", "earlier words")
            frame, = transcripts(c)
            assert frame.metadata["stt_speech_generation"] == 1
            assert s._transcript_commits.generation == 2
            # The real downstream strategy must ignore this older generation
            # for stopping, while the aggregator keeps its words in order.
    asyncio.run(exercise())


def test_completion_before_ack_is_held_and_empty_completion_does_not_invent_text(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            await speech_start(s)
            await speech_stop(s)
            await complete(s, "one", "")
            assert transcripts(c) == []
            await ack(s, "one")
            assert s._transcript_commits.pending_count == 0
            await speech_start(s)
            await speech_stop(s)
            await complete(s, "two", "real text")
            assert transcripts(c) == []
            await ack(s, "two", "one")
            assert [f.text for f in transcripts(c)] == ["real text"]
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["failed", "wrong_order", "unmatched", "malformed"])
def test_ambiguous_or_missing_segments_stop_instead_of_releasing_partial_text(failure, monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            await speech_start(s)
            await speech_stop(s)
            if failure == "failed":
                await ack(s, "one")
                await s._handle_transcription_failed({"item_id": "one", "error": {"message": "private-marker"}})
            elif failure == "wrong_order":
                await ack(s, "one", "not-a-known-predecessor")
            elif failure == "malformed":
                await s._handle_transcription_completed({"item_id": "one", "transcript": []})
            else:
                await ack(s, "one")
                await complete(s, "unknown-item", "must not escape")
            assert len(c.errors) == 1
            assert c.errors[0]["force_treat_as_permanent"] is True
            assert "private-marker" not in str(c.errors)
            assert not s._transcript_commits.active
            await complete(s, "one", "late completion after failure")
            assert transcripts(c) == []
    asyncio.run(exercise())


def test_disabled_fix_keeps_the_native_nonfinal_transcription(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch, enabled=False) as c:
            await speech_start(c.service)
            await speech_stop(c.service)
            await complete(c.service, "legacy-item", "original behavior")
            frame, = transcripts(c)
            assert frame.finalized is False
            assert "stt_speech_generation" not in frame.metadata
            assert c.sends == [{"type": "input_audio_buffer.commit"}]
            assert c.service._transcript_commits.pending_count == 0
            assert not c.errors
    asyncio.run(exercise())


def test_receive_dispatcher_processes_ack_then_completion_and_closes_state(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            await speech_start(s)
            await speech_stop(s)

            class FakeSocket:
                async def __aiter__(self):
                    for event in [
                        {"type": "session.updated"},
                        {"type": "conversation.item.created"},
                        {"type": "input_audio_buffer.committed", "item_id": "one", "previous_item_id": None},
                        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "one", "content_index": 0, "transcript": "complete sentence"},
                    ]:
                        yield json.dumps(event)
                    s._disconnecting = True  # Intentional peer shutdown, not a stream failure.

            s._websocket = FakeSocket()
            await s._receive_messages()
            assert s._session_ready is True
            assert [f.text for f in transcripts(c)] == ["complete sentence"]
            assert not s._transcript_commits.active
            await complete(s, "one", "must not append after shutdown")
            assert len(transcripts(c)) == 1
            assert not c.errors
    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["input_audio_buffer.commit", "input_audio_buffer.append"])
def test_send_failure_invalidates_pending_audio_instead_of_waiting_forever(kind, monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            class BrokenSocket:
                async def send(self, message):
                    raise OSError("private-network-marker")
            s._websocket = BrokenSocket()
            monkeypatch.setattr(s, "_ws_send", type(s)._ws_send.__get__(s))
            if kind == "input_audio_buffer.commit":
                await speech_start(s)
                await speech_stop(s)
            else:
                await s._ws_send({"type": kind, "audio": "synthetic-placeholder"})
            assert not s._transcript_commits.active
            assert s._transcript_commits.pending_count == 0
            assert len(c.errors) == 1
            assert c.errors[0]["force_treat_as_permanent"] is True
            assert "private-network-marker" not in str(c.errors)
            await complete(s, "late", "must not be emitted")
            assert transcripts(c) == []
    asyncio.run(exercise())


def test_new_live_item_downgrades_an_older_completed_item(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            await speech_start(s)
            await speech_stop(s)
            await ack(s, "old")
            await s._handle_transcription_delta({"item_id": "new", "delta": "another word"})
            await complete(s, "old", "old complete text")
            frame, = transcripts(c)
            assert frame.finalized is False
            interim = next(f for f in c.frames if isinstance(f, InterimTranscriptionFrame))
            assert interim.metadata["stt_item_id"] == "new"
    asyncio.run(exercise())


def test_overlapping_stops_preserve_wire_order_when_first_send_is_suspended(monkeypatch):
    async def exercise():
        async with adapter(monkeypatch) as c:
            s = c.service
            entered, release, second_reserved = asyncio.Event(), asyncio.Event(), asyncio.Event()
            calls, wire_order = [], []
            class SlowSocket:
                async def send(self, message):
                    number = len(calls) + 1
                    calls.append(json.loads(message))
                    if number == 1:
                        entered.set()
                        await release.wait()
                    wire_order.append(number)
            s._websocket = SlowSocket()
            monkeypatch.setattr(s, "_ws_send", type(s)._ws_send.__get__(s))
            await speech_start(s)
            first = asyncio.create_task(speech_stop(s))
            second = None
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                await speech_start(s)
                async def observe_reservation(frame):
                    if isinstance(frame, VADUserStoppedSpeakingFrame):
                        second_reserved.set()
                c.on_push = observe_reservation
                second = asyncio.create_task(speech_stop(s))
                await asyncio.wait_for(second_reserved.wait(), timeout=1)
                assert len(calls) == 1
                assert s._transcript_commits.pending_count == 2
            finally:
                release.set()
                await asyncio.gather(first, *([second] if second else []))
            assert wire_order == [1, 2]
            assert calls == [{"type": "input_audio_buffer.commit"}] * 2
            await ack(s, "one")
            await ack(s, "two", "one")
            await complete(s, "two", "ending")
            await complete(s, "one", "beginning")
            frame, = transcripts(c)
            assert frame.text == "beginning ending"
            assert frame.metadata["stt_commit_sequences"] == [1, 2]
            assert not c.errors
    asyncio.run(exercise())
