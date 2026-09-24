"""Profile lifecycle integration, with all DB/provider resources replaced."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
pytest.importorskip("pipecat")

from healthcare_voice_agent.demo import runtime
from healthcare_voice_agent.demo.database import DemoDatabaseError
from healthcare_voice_agent.voice import session
from tests.integration.test_voice_session import offline, FakePeer, FakeRunner, read_events


def demo_args(offline, peer):
    args = offline.args(peer)
    args.cli_args.app_config = args.cli_args.app_config.model_copy(update={"agent_mode": "frontdesk_demo"})
    return args


def fake_resource():
    epochs = []
    controller = SimpleNamespace(speech_started=lambda: epochs.append(True))
    return SimpleNamespace(controller=controller, prompt="Synthetic demo-only instruction", close=AsyncMock(), epochs=epochs)


def test_demo_uses_injected_prompt_closes_runtime_and_does_not_save_prompt(offline, monkeypatch):
    resource = fake_resource()
    monkeypatch.setattr(runtime, "open_demo_runtime", AsyncMock(return_value=resource))
    peer = FakePeer()
    asyncio.run(session.run_session(demo_args(offline, peer)))
    resource.close.assert_awaited_once()
    assert resource.epochs and peer.disconnects == 1
    assert offline.prompts == [resource.prompt]
    directory = next((offline.root / "sessions").iterdir())
    assert not (directory / "prompt.txt").exists()
    assert any(e["event"] == "prompt_loaded" for e in read_events(directory))
    assert FakeRunner.instances[0].run_finished.is_set()


def test_database_failure_never_constructs_transport_or_providers(offline, monkeypatch):
    monkeypatch.setattr(runtime, "open_demo_runtime", AsyncMock(side_effect=DemoDatabaseError("Synthetic demo unavailable")))
    peer = FakePeer()
    asyncio.run(session.run_session(demo_args(offline, peer)))
    assert not offline.transports and not offline.bundles
    assert peer.disconnects == 1


def test_pipeline_failure_still_closes_isolated_database_resource(offline, monkeypatch):
    resource = fake_resource()
    monkeypatch.setattr(runtime, "open_demo_runtime", AsyncMock(return_value=resource))
    monkeypatch.setattr(session, "build_pipeline", AsyncMock(side_effect=RuntimeError("Synthetic construction error")))
    peer = FakePeer()
    asyncio.run(session.run_session(demo_args(offline, peer)))
    resource.close.assert_awaited_once()
    assert peer.disconnects == 1
