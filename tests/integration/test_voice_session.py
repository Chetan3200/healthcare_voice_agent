"""Offline lifecycle coverage for one WebRTC voice session."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pipecat", reason="Install the voice extra to test voice sessions")

from pipecat.runner.types import SmallWebRTCRunnerArguments

from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice import session


class FakeEvents:
    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def register(callback):
            self.handlers[name] = callback
            return callback
        return register

    async def fire(self, name, *args):
        callback = self.handlers.get(name)
        if callback is not None:
            return await callback(self, *args)


class FakePeer:
    def __init__(self):
        self.disconnects = 0

    async def disconnect(self):
        self.disconnects += 1


class FakeTransport(FakeEvents):
    pass


class FakeWorker(FakeEvents):
    pass


class FakeBundle:
    def __init__(self):
        self.worker = FakeWorker()
        self.context = object()
        self.cleanup_calls = 0
        self.cleanup_raises = False

    async def cleanup_unstarted(self):
        self.cleanup_calls += 1
        if self.cleanup_raises:
            raise RuntimeError("bundle cleanup failed")


class FakeRunner(FakeEvents):
    instances = []
    fail_registration = False
    disconnect_during_registration = False
    transport = None

    def __init__(self, **kwargs):
        super().__init__()
        self.cancelled = asyncio.Event()
        self.run_started = asyncio.Event()
        self.run_finished = asyncio.Event()
        self.cancel_reasons = []
        self.workers = []
        type(self).instances.append(self)

    async def add_workers(self, worker):
        self.workers.append(worker)
        if type(self).disconnect_during_registration:
            await type(self).transport.fire("on_client_disconnected", object())
        if type(self).fail_registration:
            raise RuntimeError("registration failed")

    async def cancel(self, *, reason):
        self.cancel_reasons.append(reason)
        self.cancelled.set()

    async def run(self):
        # Model the Pipecat runner clearing a pre-run cancellation request.
        self.cancelled.clear()
        self.run_started.set()
        await self.fire("on_ready")
        await self.cancelled.wait()
        self.run_finished.set()


@pytest.fixture
def offline(monkeypatch, tmp_path):
    config = load_config(None, environ={
        "LIVE_API_ENABLED": "true",
        "OPENAI_API_KEY": "dummy-offline-key",
        "VOICE_MAX_SESSION_SECONDS": "0.02",
    })
    created_transports = []
    created_transport_params = []
    created_bundles = []
    created_prompts = []
    FakeRunner.instances = []
    FakeRunner.fail_registration = False
    FakeRunner.disconnect_during_registration = False
    FakeRunner.transport = None

    async def create_fake_transport(*args, **_kwargs):
        created_transport_params.append(args[1]["webrtc"]())
        transport = FakeTransport()
        created_transports.append(transport)
        FakeRunner.transport = transport
        return transport

    async def build_fake_pipeline(*_args, **_kwargs):
        created_prompts.append(_kwargs["prompt"])
        bundle = FakeBundle()
        created_bundles.append(bundle)
        return bundle

    monkeypatch.setattr(session, "create_transport", create_fake_transport)
    monkeypatch.setattr(session, "build_pipeline", build_fake_pipeline)
    monkeypatch.setattr(session, "WorkerRunner", FakeRunner)

    def args(peer, *, body=None, supplied_session_id=None):
        runner_args = SmallWebRTCRunnerArguments(peer, body=body, session_id=supplied_session_id)
        runner_args.cli_args = SimpleNamespace(app_config=config, run_root=tmp_path)
        return runner_args

    return SimpleNamespace(
        root=tmp_path, args=args, transports=created_transports,
        transport_params=created_transport_params, bundles=created_bundles,
        prompts=created_prompts,
    )


def read_events(directory: Path):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]


def test_connect_disconnect_cancels_closes_peer_and_flushes_trace(offline):
    async def exercise():
        peer = FakePeer()
        task = asyncio.create_task(session.run_session(offline.args(peer)))
        while not FakeRunner.instances or not FakeRunner.instances[0].run_started.is_set():
            await asyncio.sleep(0)
        transport = offline.transports[0]
        await transport.fire("on_client_connected", object())
        await transport.fire("on_client_disconnected", object())
        await task
        return peer

    peer = asyncio.run(exercise())
    runner = FakeRunner.instances[0]
    directory = next((offline.root / "sessions").iterdir())
    events = read_events(directory)
    params = offline.transport_params[0]
    assert (params.audio_in_enabled, params.audio_out_enabled) == (True, True)
    assert (params.audio_in_sample_rate, params.audio_out_sample_rate) == (16000, 24000)
    assert runner.cancel_reasons == ["client_disconnected"]
    assert peer.disconnects == 1
    assert [event["event"] for event in events][-2:] == ["client_disconnected", "session_finished"]
    assert events[-1]["data"]["reason"] == "client_disconnected"


def test_successive_connections_have_private_fresh_run_state(offline):
    async def one_connection(body, session_id):
        peer = FakePeer()
        index = len(FakeRunner.instances)
        task = asyncio.create_task(session.run_session(offline.args(peer, body=body, supplied_session_id=session_id)))
        while len(FakeRunner.instances) <= index or not FakeRunner.instances[index].run_started.is_set():
            await asyncio.sleep(0)
        await offline.transports[index].fire("on_client_disconnected", object())
        await task
        return peer

    asyncio.run(one_connection({"prompt": "attacker prompt", "session_id": "../../escape"}, "client-one"))
    asyncio.run(one_connection({"config": {"voice": {"max_session_seconds": 999}}, "run_root": "/tmp"}, "client-two"))
    directories = sorted((offline.root / "sessions").iterdir())
    assert len(directories) == 2
    assert {directory.name for directory in directories}.isdisjoint({"client-one", "client-two", "escape"})
    assert all(len(directory.name) == 32 for directory in directories)
    expected_prompt = session.files("healthcare_voice_agent.agent").joinpath("prompts.md").read_text("utf-8")
    assert offline.prompts == [expected_prompt, expected_prompt]
    assert all((directory / "prompt.txt").read_text(encoding="utf-8") == expected_prompt for directory in directories)
    assert all((directory / "config.json").is_file() and (directory / "events.jsonl").is_file() for directory in directories)
    assert offline.bundles[0] is not offline.bundles[1]
    assert offline.bundles[0].context is not offline.bundles[1].context


def test_disconnect_during_registration_is_replayed_when_runner_is_ready(offline):
    FakeRunner.disconnect_during_registration = True

    peer = FakePeer()
    asyncio.run(session.run_session(offline.args(peer)))

    runner = FakeRunner.instances[0]
    assert runner.cancel_reasons == ["client_disconnected", "client_disconnected"]
    assert runner.run_finished.is_set()
    assert peer.disconnects == 1


def test_deadline_stops_idle_runner_and_leaves_no_deadline_task(offline):
    async def exercise():
        peer = FakePeer()
        await session.run_session(offline.args(peer))
        return peer, [task for task in asyncio.all_tasks() if task.get_name().startswith("deadline-")]

    peer, deadlines = asyncio.run(exercise())
    runner = FakeRunner.instances[0]
    directory = next((offline.root / "sessions").iterdir())
    assert runner.cancel_reasons == ["session_limit"]
    assert peer.disconnects == 1
    assert deadlines == []
    assert any(event["event"] == "session_limit" for event in read_events(directory))


def test_external_cancellation_stops_shielded_runner_then_reraises(offline):
    async def exercise():
        peer = FakePeer()
        task = asyncio.create_task(session.run_session(offline.args(peer)))
        while not FakeRunner.instances or not FakeRunner.instances[0].run_started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return peer

    peer = asyncio.run(exercise())
    runner = FakeRunner.instances[0]
    directory = next((offline.root / "sessions").iterdir())
    assert runner.cancel_reasons == ["session_cancelled"]
    assert runner.run_finished.is_set()
    assert peer.disconnects == 1
    assert [event["event"] for event in read_events(directory)][-2:] == ["session_cancelled", "session_finished"]


def test_registration_failure_releases_unstarted_bundle_and_closes_peer(offline):
    FakeRunner.fail_registration = True

    peer = FakePeer()
    asyncio.run(session.run_session(offline.args(peer)))

    assert offline.bundles[0].cleanup_calls == 1
    assert peer.disconnects == 1
    directory = next((offline.root / "sessions").iterdir())
    events = read_events(directory)
    assert any(event["event"] == "session_error" for event in events)
    assert events[-1]["event"] == "session_finished"
    assert events[-1]["data"]["reason"] == "session_error"


def test_cleanup_failure_still_closes_peer_and_flushes_trace(offline):
    FakeRunner.fail_registration = True

    peer = FakePeer()

    async def build_bundle_with_failing_cleanup(*_args, **_kwargs):
        bundle = FakeBundle()
        bundle.cleanup_raises = True
        offline.bundles.append(bundle)
        return bundle

    # Keep the test's fake transport and runner while making only cleanup fail.
    original = session.build_pipeline
    session.build_pipeline = build_bundle_with_failing_cleanup
    try:
        asyncio.run(session.run_session(offline.args(peer)))
    finally:
        session.build_pipeline = original

    directory = next((offline.root / "sessions").iterdir())
    events = read_events(directory)
    assert offline.bundles[0].cleanup_calls == 1
    assert peer.disconnects == 1
    assert any(event["event"] == "session_error" for event in events)
    assert events[-1] == {**events[-1], "event": "session_finished", "data": {"reason": "session_error"}}


def test_directory_creation_failure_closes_peer_before_runtime_objects(offline, monkeypatch):
    peer = FakePeer()

    def fail_mkdir(_path, *_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(session.Path, "mkdir", fail_mkdir)
    asyncio.run(session.run_session(offline.args(peer)))

    assert peer.disconnects == 1
    assert offline.transports == []
    assert offline.bundles == []
    assert FakeRunner.instances == []


def test_trace_creation_failure_closes_peer_before_runtime_objects(offline, monkeypatch):
    peer = FakePeer()

    def fail_trace(*_args, **_kwargs):
        raise OSError("trace unavailable")

    monkeypatch.setattr(session, "SessionTrace", fail_trace)
    asyncio.run(session.run_session(offline.args(peer)))

    assert peer.disconnects == 1
    assert offline.transports == []
    assert offline.bundles == []
    assert FakeRunner.instances == []


def test_unlimited_session_does_not_create_a_deadline_task(offline):
    async def exercise():
        peer = FakePeer()
        args = offline.args(peer)
        args.cli_args.app_config = load_config(None, environ={
            "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "dummy-offline-key",
        })
        task = asyncio.create_task(session.run_session(args))
        while not FakeRunner.instances or not FakeRunner.instances[0].run_started.is_set():
            await asyncio.sleep(0)
        try:
            assert not any(t.get_name().startswith("deadline-") for t in asyncio.all_tasks())
            assert not task.done()
        finally:
            await offline.transports[0].fire("on_client_disconnected", object())
            await task
        assert peer.disconnects == 1
        assert FakeRunner.instances[0].cancel_reasons == ["client_disconnected"]

    asyncio.run(exercise())


def test_new_connection_reloads_and_records_the_current_prompt(offline, monkeypatch, tmp_path):
    import hashlib

    prompt_directory = tmp_path / "prompt-package"
    prompt_directory.mkdir()
    prompt_path = prompt_directory / "prompts.md"
    monkeypatch.setattr(session, "files", lambda package: prompt_directory)

    async def connect_and_disconnect():
        index = len(FakeRunner.instances)
        peer = FakePeer()
        task = asyncio.create_task(session.run_session(offline.args(peer)))
        while len(FakeRunner.instances) <= index or not FakeRunner.instances[index].run_started.is_set():
            await asyncio.sleep(0)
        await offline.transports[index].fire("on_client_disconnected", object())
        await task
        assert peer.disconnects == 1

    prompt_path.write_text("First test instruction.\n", encoding="utf-8")
    asyncio.run(connect_and_disconnect())
    prompt_path.write_text("Updated test instruction.\n", encoding="utf-8")
    asyncio.run(connect_and_disconnect())

    assert offline.prompts == ["First test instruction.\n", "Updated test instruction.\n"]
    directories = list((offline.root / "sessions").iterdir())
    assert len(directories) == 2
    assert {(directory / "prompt.txt").read_text(encoding="utf-8") for directory in directories} == set(offline.prompts)
    for directory in directories:
        prompt_bytes = (directory / "prompt.txt").read_bytes()
        event = next(event for event in read_events(directory) if event["event"] == "prompt_loaded")
        assert event["data"]["sha256"] == hashlib.sha256(prompt_bytes).hexdigest()
