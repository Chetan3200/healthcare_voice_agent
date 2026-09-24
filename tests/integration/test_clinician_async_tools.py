"""Offline native Pipecat coverage for deferred clinician read tools.

The OpenAI stream is scripted locally.  These tests deliberately use the real
OpenAILLMService, Pipecat assistant aggregator, function task registry, and
DeferredToolDelivery observer; no provider, database, or audio persistence is
used.
"""
import asyncio
import copy
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    ErrorFrame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMTextFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSSpeakFrame,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.tests.utils import SleepFrame, run_test

from healthcare_voice_agent.clinician.flow import ASYNC_READ_TOOLS, functions
from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice.clinician import ClinicianFlowManager
from healthcare_voice_agent.voice.deferred_tools import DeferredToolDelivery, PROCESSING_NOTICE
from healthcare_voice_agent.voice.providers import build_llm


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access is forbidden in clinician async-tool tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    logger.disable("pipecat")
    yield
    logger.enable("pipecat")


def _llm():
    config = load_config(None, environ={
        "AGENT_MODE": "clinician", "LIVE_API_ENABLED": "true",
        "OPENAI_API_KEY": "offline-placeholder",
    })
    return build_llm(config, system_instruction="Offline native async-tool test.")


def _stream_script(monkeypatch, llm, scripts, requests, advertised_tools=None):
    from openai.types.chat import ChatCompletionChunk

    iterator = iter(scripts)

    async def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        if advertised_tools is not None:
            advertised_tools.append([t.get('function', {}).get('name') for t in kwargs.get('tools', [])])
        delta = next(iterator)

        async def stream():
            chunks = ([{"tool_calls": [call]} for call in delta["tool_calls"]]
                      if "tool_calls" in delta else [delta])
            for chunk in chunks:
                yield ChatCompletionChunk(
                    id="offline", object="chat.completion.chunk", created=0,
                    model="offline", choices=[{"index": 0, "delta": chunk}],
                )
        return stream()

    monkeypatch.setattr(llm._client.chat.completions, "create", create)


class _ScriptedUserTurn(FrameProcessor):
    """Makes a second queued context a completed intervening user turn."""
    def __init__(self, gate, context, content="New question."):
        super().__init__()
        self.gate, self.context, self.contexts = gate, context, 0
        self.content = content

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self.gate.user_started()
            self.context.add_message({"role": "user", "content": self.content})
        elif isinstance(frame, LLMContextFrame):
            self.contexts += 1
            if self.contexts == 2:
                self.gate.user_stopped()
        await self.push_frame(frame, direction)


class _SyntheticTTS(FrameProcessor):
    """Frame-only TTS: no synthesis and no audio payloads."""
    def __init__(self):
        super().__init__()
        self.context_number = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (LLMTextFrame, TTSSpeakFrame)):
            self.context_number += 1
            context_id = "notice" if isinstance(frame, TTSSpeakFrame) else f"reply-{self.context_number}"
            await self.push_frame(TTSStartedFrame(context_id), direction)
            await self.push_frame(frame, direction)
            return
        await self.push_frame(frame, direction)


class _DrainOutput(FrameProcessor):
    """Transport-shaped drain: only the new-answer reply waits for release."""
    def __init__(self, release):
        super().__init__()
        self.release = release
        self.drained = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSStartedFrame):
            await self.push_frame(BotStartedSpeakingFrame(), direction)
            if frame.context_id != "notice":
                await self.release.wait()
            await self.push_frame(BotStoppedSpeakingFrame(), direction)
            self.drained.append(frame.context_id)
            await self.push_frame(TTSStoppedFrame(frame.context_id), direction)
            return
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSStoppedFrame):
            # The transport emits the ordered stop above; do not forward an
            # upstream service stop with the wrong observer source.
            self.drained.append(frame.context_id)
            return
        await self.push_frame(frame, direction)


def test_native_background_read_survives_barge_in_and_delivers_once_after_drain(monkeypatch):
    async def scenario():
        llm = _llm()
        requests, advertised = [], []
        _stream_script(monkeypatch, llm, [
            {"tool_calls": [{"index": 0, "id": "search-1", "type": "function",
                "function": {"name": "search_clinic_instructions", "arguments": "{\"query\": \"cast care\"}"}}]},
            {"content": "Here is the answer to the new question."},
            {"content": "The completed lookup says to keep the cast dry."},
        ], requests, advertised)
        context = LLMContext([{"role": "user", "content": "Find cast guidance."}],
            tools=ToolsSchema(standard_tools=[FunctionSchema(name="search_clinic_instructions",
                description="Public search", properties={"query": {"type": "string"}}, required=["query"])]))
        assistant = LLMAssistantAggregator(context)
        release = asyncio.Event()
        session = SimpleNamespace(trace=None, case_revision=0)
        gate = DeferredToolDelivery(session, notice_seconds=0.02)
        output = _DrainOutput(release)
        tts = _SyntheticTTS()
        gate.bind(llm, tts, output, context)
        manager = ClinicianFlowManager(llm=llm, worker=Mock(spec=PipelineWorker),
            context_aggregator=SimpleNamespace(assistant=lambda: assistant))
        manager.state.update(session=session, delivery=gate)
        started, completed = asyncio.Event(), asyncio.Event()

        async def search(args, flow):
            started.set()
            await asyncio.sleep(0.15)
            completed.set()
            return {"status": "ok", "data": {"source": "synthetic"},
                    "meta": {"request_id": "offline-search"}}, None

        llm.register_function("search_clinic_instructions",
            await manager._create_transition_func("search_clinic_instructions", search),
            cancel_on_interruption=False, timeout_secs=1, cancellable_by_llm=True)

        state_before_close = {}
        async def release_after_background_result():
            await completed.wait()
            await asyncio.sleep(0.03)
            release.set()
            await asyncio.sleep(0.04)
            state_before_close.update(ready=dict(gate._ready), presenting=dict(gate._presenting),
                                      calls=tuple(gate._calls), busy=(gate.llm_busy, gate.speech_busy, gate.user_pending),
                                      check_queued=gate._check_queued)
        releaser = asyncio.create_task(release_after_background_result())
        try:
            down, up = await asyncio.wait_for(run_test(
                Pipeline([_ScriptedUserTurn(gate, context), gate, llm, tts, output, assistant]),
                frames_to_send=[LLMContextFrame(context), SleepFrame(sleep=0.04),
                    InterruptionFrame(), LLMContextFrame(context), SleepFrame(sleep=0.26), CancelFrame()],
                send_end_frame=False, observers=[gate.observer], start_timeout=5), timeout=5)
        finally:
            release.set()
            await releaser
        errors = [frame for frame in [*down, *up] if isinstance(frame, ErrorFrame)]
        assert not errors
        assert started.is_set() and completed.is_set()
        assert 'cancel_search_clinic_instructions' in advertised[0]
        assert state_before_close["check_queued"] is False, state_before_close
        assert len(requests) == 3, {"requests": len(requests), "before_close": state_before_close,
                                   "drained": output.drained}
        assert requests[1][-1] == {"role": "user", "content": "New question."}
        assert "BACKGROUND_DELIVERY" in requests[2][-1]["content"]
        assert output.drained.count("notice") == 1
        assert output.drained.count("reply-2") == 1
        assert output.drained.count("reply-3") == 1
        assert len([f for f in down if isinstance(f, TTSSpeakFrame) and f.text == PROCESSING_NOTICE]) == 1
        assert not any(m["role"] == "developer" and "BACKGROUND_DELIVERY" in m["content"]
                       for m in context.get_messages())
    asyncio.run(scenario())


def test_flow_schema_restamps_async_read_cancellation_and_cancel_tool_eligibility():
    async def scenario():
        llm = _llm()
        manager = ClinicianFlowManager(llm=llm, worker=Mock(spec=PipelineWorker),
            context_aggregator=SimpleNamespace(assistant=lambda: LLMAssistantAggregator(LLMContext())))
        schemas = {tool.name: await manager._create_function_schema(tool) for tool in functions()}
        for name in ASYNC_READ_TOOLS:
            handler = schemas[name].handler
            assert handler._pipecat_cancel_on_interruption is False
            assert handler._pipecat_cancellable_by_llm is True
        assert schemas["open_case"].handler._pipecat_cancel_on_interruption is True
    asyncio.run(scenario())


def test_async_read_uses_own_deadline_not_native_thirty_second_default(monkeypatch):
    async def scenario():
        llm = _llm()
        assistant = LLMAssistantAggregator(LLMContext())
        manager = ClinicianFlowManager(llm=llm, worker=Mock(spec=PipelineWorker),
            context_aggregator=SimpleNamespace(assistant=lambda: assistant))
        manager.background_timeout_seconds = 0.01
        manager.state.update(session=SimpleNamespace(trace=None, case_revision=7), delivery=DeferredToolDelivery(SimpleNamespace(trace=None)))
        callback = AsyncMock()
        params = FunctionCallParams(function_name="search_clinic_instructions", tool_call_id="offline-timeout",
            arguments={}, llm=llm, pipeline_worker=manager._worker, context=LLMContext(), result_callback=callback)
        async def never_returns(args, flow):
            await asyncio.Event().wait()
        invoke = await manager._create_transition_func("search_clinic_instructions", never_returns)
        await invoke(params)
        result = callback.await_args.args[0]
        assert result["error"]["code"] == "TIMEOUT"
        assert 10 <= result["meta"]["duration_ms"] < 500
    asyncio.run(scenario())


def test_case_change_invalidates_delivery_and_uses_native_read_task_settlement(monkeypatch):
    async def scenario():
        llm = _llm()
        assistant = LLMAssistantAggregator(LLMContext())
        manager = ClinicianFlowManager(llm=llm, worker=Mock(spec=PipelineWorker),
            context_aggregator=SimpleNamespace(assistant=lambda: assistant))
        delivery = DeferredToolDelivery(SimpleNamespace(trace=None))
        manager.state.update(session=SimpleNamespace(trace=None, case_revision=3), delivery=delivery)
        cancel = AsyncMock()
        monkeypatch.setattr(llm, "_cancel_function_call_tasks", cancel)
        callback = AsyncMock()
        params = FunctionCallParams(function_name="open_case", tool_call_id="case-change",
            arguments={"case_id": "1042"}, llm=llm, pipeline_worker=manager._worker,
            context=LLMContext(), result_callback=callback)

        async def open_case(args, flow):
            return {"status": "ok", "data": {"state": "selected"}, "meta": {}}, None

        await (await manager._create_transition_func("open_case", open_case))(params)
        assert delivery.generation == 1
        predicate = cancel.await_args.args[0]
        assert predicate(SimpleNamespace(function_name="get_study"))
        assert not predicate(SimpleNamespace(function_name="open_case"))
        assert cancel.await_args.kwargs == {"reason": "case change", "run_llm": False}
    asyncio.run(scenario())


def test_explicit_native_cancel_stops_pending_search_without_late_delivery(monkeypatch):
    async def scenario():
        llm, requests, advertised = _llm(), [], []
        _stream_script(monkeypatch, llm, [
            {'tool_calls': [{'index': 0, 'id': 'search-1', 'type': 'function',
                'function': {'name': 'search_clinic_instructions', 'arguments': '{}'}}]},
            {'tool_calls': [{'index': 0, 'id': 'cancel-1', 'type': 'function',
                'function': {'name': 'cancel_search_clinic_instructions', 'arguments': '{}'}}]},
            {'content': 'I cancelled the search.'},
        ], requests, advertised)
        context = LLMContext([{'role': 'user', 'content': 'Find public guidance.'}],
            tools=ToolsSchema(standard_tools=[FunctionSchema(name='search_clinic_instructions',
                description='Public search', properties={}, required=[])]))
        assistant = LLMAssistantAggregator(context)
        release = asyncio.Event()
        release.set()
        output, tts = _DrainOutput(release), _SyntheticTTS()
        session = SimpleNamespace(trace=None, case_revision=0)
        gate = DeferredToolDelivery(session, notice_seconds=0.02)
        gate.bind(llm, tts, output, context)
        manager = ClinicianFlowManager(llm=llm, worker=Mock(spec=PipelineWorker),
            context_aggregator=SimpleNamespace(assistant=lambda: assistant))
        manager.state.update(session=session, delivery=gate)
        cancelled = asyncio.Event()
        async def search(args, flow):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        llm.register_function('search_clinic_instructions',
            await manager._create_transition_func('search_clinic_instructions', search),
            cancel_on_interruption=False, timeout_secs=1, cancellable_by_llm=True)
        @llm.event_handler('on_function_calls_cancelled')
        async def on_cancel(service, calls):
            for call in calls:
                gate.finish(call.tool_call_id)
                gate.discard(call.tool_call_id)
        down, up = await asyncio.wait_for(run_test(
            Pipeline([_ScriptedUserTurn(gate, context, 'Cancel that search.'),
                      gate, llm, tts, output, assistant]),
            frames_to_send=[LLMContextFrame(context), SleepFrame(sleep=0.04),
                InterruptionFrame(), LLMContextFrame(context), SleepFrame(sleep=0.15), CancelFrame()],
            send_end_frame=False, observers=[gate.observer], start_timeout=5), timeout=5)
        assert not any(isinstance(f, ErrorFrame) for f in [*down, *up])
        assert cancelled.is_set() and len(requests) == 3
        assert 'cancel_search_clinic_instructions' in advertised[1]
        assert not gate._ready and not gate._calls
        assert not any('BACKGROUND_DELIVERY' in str(m) for m in context.messages)
    asyncio.run(scenario())


def test_search_ready_before_imaging_turn_is_delivered_after_imaging_audio(monkeypatch):
    """Reproduce session 88be96b7: ready search, imaging tool round, imaging-only answer."""
    async def scenario():
        llm, requests, events, spoken = _llm(), [], [], []
        _stream_script(monkeypatch, llm, [
            {'tool_calls': [{'index': 0, 'id': 'search-1', 'type': 'function', 'function':
                {'name': 'search_clinic_instructions', 'arguments': '{"query":"cast care"}'}}]},
            {'tool_calls': [{'index': 0, 'id': 'study-1', 'type': 'function', 'function':
                {'name': 'get_study', 'arguments': '{"case_id":"1042","study_id":null}'}}]},
            # Deliberately omit every search fact, as the live model did.
            {'content': 'The imaging was an X-ray of the left wrist, AP and lateral.'},
            {'content': 'Coming back to the public cast guidance, keep the cast dry.'},
        ], requests)
        context = LLMContext([{'role': 'user', 'content': 'Find public wrist cast guidance.'}],
            tools=ToolsSchema(standard_tools=[
                FunctionSchema(name=name, description=name, properties={}, required=[])
                for name in ('search_clinic_instructions', 'get_study')]))
        assistant = LLMAssistantAggregator(context)
        release_search, release_audio, imaging_audio_started = (asyncio.Event() for _ in range(3))
        session = SimpleNamespace(case_revision=1, trace=SimpleNamespace(
            emit=lambda event, **fields: events.append((event, fields))))
        gate = DeferredToolDelivery(session, notice_seconds=.01)

        class Output(_DrainOutput):
            async def process_frame(self, frame, direction):
                if isinstance(frame, TTSStartedFrame) and frame.context_id != 'notice':
                    imaging_audio_started.set()
                if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
                    spoken.append(frame.text)
                await super().process_frame(frame, direction)

        class UserTurn(_ScriptedUserTurn):
            async def process_frame(self, frame, direction):
                if isinstance(frame, LLMContextFrame) and self.contexts == 1:
                    release_search.set()
                    # Complete the search while the new user turn is still pending,
                    # immediately BEFORE the imaging request enters inference.
                    while 'search-1' not in gate._ready:
                        await asyncio.sleep(0)
                await super().process_frame(frame, direction)

        output, tts = Output(release_audio), _SyntheticTTS()
        gate.bind(llm, tts, output, context)
        manager = ClinicianFlowManager(llm=llm, worker=Mock(spec=PipelineWorker),
            context_aggregator=SimpleNamespace(assistant=lambda: assistant))
        manager.state.update(session=session, delivery=gate)
        executed = []

        async def search(args, flow):
            executed.append('search')
            await release_search.wait()
            return {'status': 'ok', 'data': {'passage': 'Keep the cast dry.'},
                    'meta': {'request_id': 'offline-search'}}, None

        async def study(args, flow):
            executed.append('study')
            return {'status': 'ok', 'data': {'modality': 'XR', 'body_part': 'wrist',
                    'laterality': 'left', 'views': ['AP', 'lateral']},
                    'meta': {'request_id': 'offline-study'}}, None

        for name, handler in [('search_clinic_instructions', search), ('get_study', study)]:
            llm.register_function(name, await manager._create_transition_func(name, handler),
                cancel_on_interruption=False, timeout_secs=1, cancellable_by_llm=True)

        async def inspect_before_audio_drain():
            try:
                await asyncio.wait_for(imaging_audio_started.wait(), timeout=3)
                assert set(gate._ready) == {'search-1'}
                assert set(gate._presenting) == {'study-1'}
                assert not any(e == 'background_result_response_finished' for e, _ in events)
                assert len(requests) == 3  # Search must not speak over the imaging answer.
            finally:
                release_audio.set()

        inspector = asyncio.create_task(inspect_before_audio_drain())
        try:
            down, up = await asyncio.wait_for(run_test(Pipeline([
                UserTurn(gate, context, 'What imaging was done for this case?'),
                gate, llm, tts, output, assistant]), frames_to_send=[
                LLMContextFrame(context), SleepFrame(sleep=.04), InterruptionFrame(),
                LLMContextFrame(context), SleepFrame(sleep=.25), CancelFrame()],
                send_end_frame=False, observers=[gate.observer], start_timeout=5), timeout=5)
        finally:
            release_search.set()
            release_audio.set()
            await inspector
        assert not any(isinstance(f, ErrorFrame) for f in [*down, *up])
        assert executed == ['search', 'study']  # No repeated lookup or second agent.
        assert len(requests) == 4
        scopes = [r[-1]['content'] for r in requests[2:]]
        assert 'study-1' in scopes[0] and 'search-1' not in scopes[0]
        assert 'search-1' in scopes[1] and 'study-1' not in scopes[1]
        finished = [d['tool_call_ids'] for e, d in events if e == 'background_result_response_finished']
        assert finished == [['study-1'], ['search-1']]
        assert not gate._ready and not gate._presenting
        assert not any('BACKGROUND_DELIVERY' in m.get('content', '') for m in context.messages)
        assert spoken == [
            'The imaging was an X-ray of the left wrist, AP and lateral.',
            'Coming back to the public cast guidance, keep the cast dry.',
        ]
    asyncio.run(scenario())
