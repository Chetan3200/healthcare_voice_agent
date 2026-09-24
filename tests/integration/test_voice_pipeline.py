"""Actual Pipecat assembly with fake providers and bundled local turn models.

No pipeline is started. Network connections are blocked, including during
construction of the local analyzers. This is not an audio/ASR quality test.
"""

import asyncio
import socket

import pytest

pytest.importorskip("pipecat")

from pipecat.processors.frame_processor import FrameProcessor
from healthcare_voice_agent.config import ConfigurationError, load_config
from healthcare_voice_agent.voice import pipeline as wiring
from healthcare_voice_agent.voice.rtvi import SafeRTVIProcessor
from healthcare_voice_agent.voice.tracing import SessionTrace


class FakeService(FrameProcessor):
    def __init__(self, name):
        super().__init__(name=name)
        self.cleanups = 0
        self._register_event_handler("on_tts_request")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def cleanup(self):
        self.cleanups += 1
        await super().cleanup()


class FakeTransport:
    def __init__(self):
        self.incoming = FakeService("microphone")
        self.outgoing = FakeService("speaker")

    def input(self):
        return self.incoming

    def output(self):
        return self.outgoing


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline pipeline tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv("PIPECAT_SMART_TURN_LOG_DATA", raising=False)


def live_test_config():
    return load_config(None, environ={
        "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "dummy-offline-key",
    })


@pytest.mark.parametrize("stop_seconds", [1.5, 3.0])
def test_real_pipeline_order_local_models_and_fresh_contexts(stop_seconds, tmp_path, monkeypatch):
    created = []
    instructions = []
    network_traces = []
    analyzers = []
    real_analyzer = wiring.LocalSmartTurnAnalyzerV3

    def capture_analyzer(**kwargs):
        analyzer = real_analyzer(**kwargs)
        analyzers.append(analyzer)
        return analyzer

    monkeypatch.setattr(wiring, "LocalSmartTurnAnalyzerV3", capture_analyzer)

    def stt(config):
        service = FakeService("fake-stt")
        created.append(service)
        return service

    def llm(config, *, system_instruction, trace):
        instructions.append(system_instruction)
        network_traces.append(trace)
        service = FakeService("fake-llm")
        created.append(service)
        return service

    def tts(config, *, trace):
        network_traces.append(trace)
        service = FakeService("fake-tts")
        created.append(service)
        return service

    monkeypatch.setattr(wiring.providers, "build_stt", stt)
    monkeypatch.setattr(wiring.providers, "build_llm", llm)
    monkeypatch.setattr(wiring.providers, "build_tts", tts)

    async def scenario():
        bundles = []
        traces = []
        try:
            for number in range(2):
                trace = SessionTrace(tmp_path / f"{number}.jsonl", session_id=str(number))
                traces.append(trace)
                transport = FakeTransport()
                config = load_config(None, environ={
                    "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "dummy-offline-key",
                    "VOICE_SMART_TURN_STOP_SECONDS": str(stop_seconds),
                })
                bundle = await wiring.build_pipeline(
                    config, transport, trace, prompt="Simple offline test prompt",
                )
                bundles.append(bundle)
                chain = bundle.pipeline.processors[1:-1]
                assert chain[0] is transport.incoming
                assert chain[1] is bundle.audio_gate
                assert bundle.audio_gate.ready is False
                assert chain[2].name == "fake-stt"
                assert type(chain[3]).__name__ == "LLMUserAggregator"
                assert chain[4].name == "fake-llm"
                assert chain[5].name == "fake-tts"
                assert chain[6] is transport.outgoing
                assert type(chain[7]).__name__ == "LLMAssistantAggregator"
                assert isinstance(bundle.worker.rtvi, SafeRTVIProcessor)
            assert bundles[0].context is not bundles[1].context
            assert instructions == ["Simple offline test prompt"] * 2
            assert network_traces == [traces[0], traces[0], traces[1], traces[1]]
            assert len(analyzers) == 2
            defaults = wiring.SmartTurnParams().model_dump()
            expected = {**defaults, "stop_secs": stop_seconds}
            assert all(analyzer.params.model_dump() == expected for analyzer in analyzers)
            assert all(analyzer._stop_ms == stop_seconds * 1000 for analyzer in analyzers)
        finally:
            for bundle in bundles:
                await bundle.cleanup_unstarted()
            for trace in traces:
                trace.close()
        assert len(created) == 6
        assert all(service.cleanups == 1 for service in created)

    asyncio.run(scenario())


def test_partial_construction_releases_already_created_clients(tmp_path, monkeypatch):
    stt, llm = FakeService("stt"), FakeService("llm")
    monkeypatch.setattr(wiring.providers, "build_stt", lambda config: stt)
    monkeypatch.setattr(wiring.providers, "build_llm", lambda *a, **k: llm)

    def broken_tts(config, *, trace):
        raise RuntimeError("forced-constructor-failure")
    monkeypatch.setattr(wiring.providers, "build_tts", broken_tts)
    trace = SessionTrace(tmp_path / "failure.jsonl", session_id="failure")
    try:
        with pytest.raises(RuntimeError, match="forced-constructor-failure"):
            asyncio.run(wiring.build_pipeline(live_test_config(), FakeTransport(), trace, prompt="test"))
    finally:
        trace.close()
    assert stt.cleanups == llm.cleanups == 1


def test_disabled_live_mode_does_not_construct_services(tmp_path, monkeypatch):
    def unexpected(config):
        pytest.fail("The live guard must run before constructing services")
    monkeypatch.setattr(wiring.providers, "build_stt", unexpected)
    trace = SessionTrace(tmp_path / "disabled.jsonl", session_id="disabled")
    try:
        with pytest.raises(ConfigurationError, match="disabled"):
            asyncio.run(wiring.build_pipeline(load_config(None, environ={}), None, trace, prompt="test"))
    finally:
        trace.close()


def test_hidden_smart_turn_audio_recording_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPECAT_SMART_TURN_LOG_DATA", "true")
    trace = SessionTrace(tmp_path / "recording.jsonl", session_id="recording")
    try:
        with pytest.raises(ConfigurationError, match="raw-audio"):
            asyncio.run(wiring.build_pipeline(live_test_config(), None, trace, prompt="test"))
    finally:
        trace.close()


def test_real_worker_starts_and_cancels_with_fake_services(tmp_path, monkeypatch):
    from pipecat.workers.runner import WorkerRunner
    services = []

    def fake_service(name):
        service = FakeService(name)
        services.append(service)
        return service

    monkeypatch.setattr(wiring.providers, "build_stt", lambda config: fake_service("stt"))
    monkeypatch.setattr(wiring.providers, "build_llm", lambda *a, **k: fake_service("llm"))
    monkeypatch.setattr(wiring.providers, "build_tts", lambda config, **kw: fake_service("tts"))

    async def scenario():
        trace = SessionTrace(tmp_path / "worker.jsonl", session_id="worker")
        bundle = await wiring.build_pipeline(live_test_config(), FakeTransport(), trace, prompt="test")
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        started = []
        client_ready = asyncio.Event()
        ui_messages = []

        async def capture_message(message, **kwargs):
            ui_messages.append(message)
        monkeypatch.setattr(bundle.worker.rtvi, "push_transport_message", capture_message)

        @bundle.worker.rtvi.event_handler("on_client_ready")
        async def ready_observed(processor):
            client_ready.set()

        @bundle.worker.event_handler("on_pipeline_started")
        async def pipeline_started(worker, frame):
            import pipecat.processors.frameworks.rtvi.models as RTVI
            started.append(True)
            await worker.rtvi._handle_client_ready("offline-ready", RTVI.ClientReadyData(
                version=RTVI.PROTOCOL_VERSION, about=RTVI.AboutClientData(library="offline-test"),
            ))
            await asyncio.wait_for(client_ready.wait(), timeout=1)
            await runner.cancel(reason="offline lifecycle smoke test")

        try:
            await runner.add_workers(bundle.worker)
            await asyncio.wait_for(runner.run(), timeout=10)
            assert started == [True]
            assert bundle.audio_gate.ready is True
            assert any(message.type == "bot-ready" for message in ui_messages)
            assert bundle.worker.has_finished()
            assert all(service.cleanups == 1 for service in services)
        finally:
            trace.close()

    asyncio.run(scenario())


def test_readiness_gate_drops_pre_ready_audio_without_buffering(monkeypatch):
    from pipecat.frames.frames import InputAudioRawFrame, TextFrame
    from pipecat.processors.frame_processor import FrameDirection
    from healthcare_voice_agent.voice.readiness import ClientReadyAudioGate

    async def scenario():
        gate = ClientReadyAudioGate()
        forwarded = []

        async def capture(frame, direction):
            forwarded.append(frame)
        monkeypatch.setattr(gate, "push_frame", capture)
        early = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        control = TextFrame("control")
        await gate.process_frame(early, FrameDirection.DOWNSTREAM)
        await gate.process_frame(control, FrameDirection.DOWNSTREAM)
        assert forwarded == [control]
        gate.mark_ready()
        current = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        await gate.process_frame(current, FrameDirection.DOWNSTREAM)
        assert forwarded == [control, current]  # Earlier speech is not replayed.
        await gate.cleanup()

    asyncio.run(scenario())
