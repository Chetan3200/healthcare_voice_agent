"""Bounded error-policy checks; database, transport and runner are offline fakes."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("pipecat")
from pipecat.frames.frames import ErrorFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService

from healthcare_voice_agent.clinician import runtime as clinician_runtime
from healthcare_voice_agent.demo import runtime as demo_runtime
from healthcare_voice_agent.voice import session
from tests.integration.test_voice_session import offline, FakePeer, FakeRunner, read_events


@pytest.mark.parametrize("mode, source, usable, fatal, keep_open", [
    ("clinician", "llm", True, False, True),
    ("clinician", "llm", False, False, False),
    ("clinician", "llm", True, True, False),
    ("clinician", "other", True, False, False),
    ("clinician", "unknown", True, False, False),
    ("frontdesk_demo", "llm", True, False, False),
    ("conversation", "llm", True, False, False),
])
def test_only_usable_nonfatal_clinician_llm_errors_keep_connection(
        offline, monkeypatch, mode, source, usable, fatal, keep_open):
    resource = SimpleNamespace(close=AsyncMock(), interrupt=Mock())
    monkeypatch.setattr(clinician_runtime, "open_clinician_runtime", AsyncMock(return_value=resource))
    monkeypatch.setattr(demo_runtime, "open_demo_runtime", AsyncMock(return_value=resource))
    processor = None if source == "unknown" else Mock(
        spec=LLMService if source == "llm" else FrameProcessor)
    if processor is not None:
        processor.is_usable = usable
    frame = ErrorFrame(error="Synthetic connection error", processor=processor)
    frame.fatal = fatal  # Exercise the supported legacy flag without constructing a deprecated frame.

    async def exercise():
        peer = FakePeer()
        args = offline.args(peer)
        config = args.cli_args.app_config
        args.cli_args.app_config = config.model_copy(update={
            "agent_mode": mode,
            "voice": config.voice.model_copy(update={"max_session_seconds": None}),
        })
        task = asyncio.create_task(session.run_session(args))
        async def ready():
            while not FakeRunner.instances or not FakeRunner.instances[0].run_started.is_set():
                if task.done():
                    pytest.fail("Offline session failed before registering its error handler")
                await asyncio.sleep(0)
        try:
            await asyncio.wait_for(ready(), timeout=2)
            runner = FakeRunner.instances[0]
            await offline.bundles[0].worker.fire("on_pipeline_error", frame)
            if keep_open:
                assert not task.done() and not runner.cancelled.is_set()
                assert runner.cancel_reasons == [] and peer.disconnects == 0
                resource.close.assert_not_awaited()
                resource.interrupt.assert_not_called()
                # Another failed request must not trigger a hidden retry or shutdown policy.
                await offline.bundles[0].worker.fire("on_pipeline_error", frame)
                assert runner.cancel_reasons == [] and peer.disconnects == 0
                await offline.transports[0].fire("on_client_disconnected", object())
            await asyncio.wait_for(task, timeout=2)
            assert runner.cancel_reasons == ["client_disconnected" if keep_open else "service_error"]
            assert peer.disconnects == 1
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(exercise())

    events = read_events(next((offline.root / "sessions").iterdir()))
    retained = [event for event in events if event["event"] == "recoverable_llm_error"]
    assert len(retained) == (2 if keep_open else 0)
    assert all(event["data"] == {"action": "keep_session_open"} for event in retained)
    assert events[-1]["data"]["reason"] == ("client_disconnected" if keep_open else "service_error")
    if mode != "conversation":
        resource.close.assert_awaited_once()
