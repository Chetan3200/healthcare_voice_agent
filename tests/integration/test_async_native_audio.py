"""Offline native-audio coverage for deferred clinician tool delivery."""
import asyncio
import copy
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("pipecat")

from openai.types.chat import ChatCompletionChunk
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import CancelFrame, ErrorFrame, LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from healthcare_voice_agent.clinician.flow import functions
from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice.clinician import ClinicianFlowManager
from healthcare_voice_agent.voice.deferred_tools import DeferredToolDelivery, PROCESSING_NOTICE
from healthcare_voice_agent.voice.providers import build_llm, build_tts
from healthcare_voice_agent.voice.reply_tts import FullReplyTTSBuffer


class _MemoryOutput(BaseOutputTransport):
    """Real MediaSender drain, with no device or network output."""
    def __init__(self):
        super().__init__(TransportParams(audio_out_enabled=True, audio_out_sample_rate=24000))
        self.writes = 0

    async def start(self, frame):
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        self.writes += 1
        await asyncio.sleep(.001)
        return True


def test_deferred_tool_delivery_uses_real_tts_and_media_sender(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native-audio integration test must stay offline")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    async def scenario():
        config = load_config(None, environ={"AGENT_MODE": "clinician", "LIVE_API_ENABLED": "true",
            "OPENAI_API_KEY": "offline-placeholder", "TTS_AUDIO_CHUNK_MS": "20"})
        llm, tts, requests, tts_requests = (build_llm(config, system_instruction="Offline test."),
            build_tts(config), [], [])
        scripts = iter((
            {"tool_calls": [{"index": 0, "id": "search-1", "type": "function", "function":
                {"name": "search_clinic_instructions", "arguments": '{"query":"cast care"}'}}]},
            {"content": "Keep the cast dry."},
        ))

        async def chat_create(**kwargs):
            requests.append(copy.deepcopy(kwargs["messages"]))
            delta = next(scripts)
            async def stream():
                yield ChatCompletionChunk(id="offline", object="chat.completion.chunk", created=0,
                    model="offline", choices=[{"index": 0, "delta": delta}])
            return stream()

        class Response:
            status_code = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def iter_bytes(self, chunk_size):
                yield b"\0" * 480  # synthetic zero PCM, retained only in memory

        monkeypatch.setattr(llm._client.chat.completions, "create", chat_create)
        def tts_create(**kwargs):
            tts_requests.append(kwargs["input"])
            return Response()
        monkeypatch.setattr(tts._client.audio.speech.with_streaming_response, "create", tts_create)
        context = LLMContext([{"role": "user", "content": "Find cast guidance."}], tools=ToolsSchema(
            standard_tools=[FunctionSchema(name="search_clinic_instructions", description="Search",
                properties={"query": {"type": "string"}}, required=["query"])]))
        assistant, output = LLMAssistantAggregator(context), _MemoryOutput()
        gate = DeferredToolDelivery(SimpleNamespace(trace=None, case_revision=0), notice_seconds=.02)
        gate.bind(llm, tts, output, context)
        manager = ClinicianFlowManager(llm=llm, worker=Mock(spec=PipelineWorker),
            context_aggregator=SimpleNamespace(assistant=lambda: assistant))
        manager.state.update(session=SimpleNamespace(trace=None, case_revision=0), delivery=gate)
        completed = asyncio.Event()

        async def search(args, flow):
            await asyncio.sleep(.1)
            completed.set()
            return {"status": "ok", "data": {"source": "synthetic"}, "meta": {"request_id": "offline"}}, None

        llm.register_function("search_clinic_instructions", await manager._create_transition_func(
            "search_clinic_instructions", search), cancel_on_interruption=False, timeout_secs=1,
            cancellable_by_llm=True)
        try:
            down, up = await asyncio.wait_for(run_test(Pipeline([
                gate, llm, FullReplyTTSBuffer(), tts, output, assistant]), frames_to_send=[
                LLMContextFrame(context), SleepFrame(sleep=.22), CancelFrame()], send_end_frame=False,
                observers=[gate.observer], start_timeout=5), timeout=5)
        finally:
            await tts.cleanup()
        assert completed.is_set() and output.writes >= 2
        assert not [f for f in [*down, *up] if isinstance(f, ErrorFrame)]
        assert len(requests) == 2
        assert len([m for m in context.get_messages() if m["role"] == "user"]) == 1
        assert tts_requests == [PROCESSING_NOTICE, "Keep the cast dry."]
        assert 'BACKGROUND_DELIVERY' in requests[-1][-1]['content']
        assert not any('BACKGROUND_DELIVERY' in m.get('content', '') for m in context.get_messages())
        assert not gate.speech_busy
    asyncio.run(scenario())
