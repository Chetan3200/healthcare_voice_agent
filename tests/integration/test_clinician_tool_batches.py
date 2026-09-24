"""Native Flows/aggregator completion policy, offline; not live model behavior."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger
from pipecat.flows import ContextStrategy, ContextStrategyConfig, FlowManager, NO_RESPONSE
from pipecat.frames.frames import (
    FunctionCallInProgressFrame, FunctionCallResultFrame, LLMMessagesUpdateFrame, LLMRunFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.services.llm_service import FunctionCallParams

from healthcare_voice_agent.voice.clinician import ClinicianFlowManager, _trace_record_ids


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError('Network forbidden in batch tests')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    logger.disable('pipecat')
    yield
    logger.enable('pipecat')


def node():
    return {'name': 'clinician', 'role_message': 'Synthetic offline role.',
            'task_messages': [{'role': 'developer', 'content': 'Selected case updated.'}],
            'functions': [], 'respond_immediately': True,
            'context_strategy': ContextStrategyConfig(strategy=ContextStrategy.RESET)}


class Batch:
    def __init__(self, manager_type=ClinicianFlowManager):
        self.context = LLMContext()
        self.aggregator = LLMAssistantAggregator(self.context)
        self.aggregator._maybe_push_context_after_function_result = AsyncMock()
        # Exercise native result/transition logic without starting a worker or any providers.
        self.tasks = []
        def create_task(coro, name):
            task = asyncio.create_task(coro, name=name)
            self.tasks.append(task)
            return task
        self.aggregator.create_task = create_task
        self.worker = Mock(spec=PipelineWorker)
        self.worker.queue_frames = AsyncMock()
        self.manager = manager_type(llm=SimpleNamespace(_function_call_tasks={}), worker=self.worker,
            context_aggregator=SimpleNamespace(assistant=lambda: self.aggregator))
        self.properties = []

    async def start(self, call_id, group='batch'):
        self.manager._llm._function_call_tasks[call_id] = SimpleNamespace(
            tool_call_id=call_id, group_id=group, settled=False)
        await self.aggregator._handle_function_call_in_progress(FunctionCallInProgressFrame(
            function_name=call_id, tool_call_id=call_id, arguments={},
            group_id=group, cancel_on_interruption=True))

    async def finish(self, call_id, *, next_node=None, error=False, arguments=None):
        result = {'status': 'error' if error else 'ok', 'data': None if error else {},
                  'meta': {'request_id': 'backend-' + call_id}}
        async def handler(args, manager):
            return result, next_node
        async def callback(value, *, properties=None):
            self.properties.append(properties)
            self.manager._llm._function_call_tasks[call_id].settled = True
            await self.aggregator._handle_function_call_result(FunctionCallResultFrame(
                function_name=call_id, tool_call_id=call_id, arguments=arguments or {},
                result=value, properties=properties))
        invoke = await self.manager._create_transition_func(call_id, handler)
        params = FunctionCallParams(function_name=call_id, tool_call_id=call_id,
            arguments=arguments or {}, llm=self.manager._llm, pipeline_worker=self.worker,
            context=self.context, result_callback=callback)
        await invoke(params)
        assert params.result_callback is callback  # Wrapper must not mutate shared params.
        if self.tasks:
            await asyncio.gather(*self.tasks)

    @property
    def inferences(self):
        return self.aggregator._maybe_push_context_after_function_result.await_count


def test_single_tool_continues_once():
    async def check():
        batch = Batch()
        await batch.start('read')
        await batch.finish('read')
        assert batch.inferences == 1
        assert batch.properties[-1].run_llm is None
    asyncio.run(check())


def test_fast_error_waits_for_sibling_study_result():
    async def check():
        batch = Batch()
        await batch.start('model')
        await batch.start('study')
        await batch.finish('model', error=True)
        assert batch.inferences == 0
        await batch.finish('study')
        assert batch.inferences == 1
        contents = [m['content'] for m in batch.context.get_messages() if m['role'] == 'tool']
        assert len(contents) == 2 and all('IN_PROGRESS' not in text for text in contents)
    asyncio.run(check())


def test_unrelated_group_is_not_blocked_by_slow_batch():
    async def check():
        batch = Batch()
        await batch.start('slow', group='earlier-batch')
        await batch.start('independent', group='new-batch')
        await batch.finish('independent')
        assert batch.inferences == 1
        assert batch.aggregator.has_function_calls_in_progress
        await batch.finish('slow')
        assert batch.inferences == 2
    asyncio.run(check())


@pytest.mark.parametrize('edge_first', [True, False])
def test_case_transition_runs_once_after_batch_without_old_node_inference(edge_first):
    async def check():
        batch = Batch()
        await batch.manager.initialize({**node(), 'respond_immediately': False})
        batch.worker.queue_frames.reset_mock()
        await batch.start('select')
        await batch.start('read')
        order = ['select', 'read'] if edge_first else ['read', 'select']
        for index, name in enumerate(order):
            await batch.finish(name, next_node=node() if name == 'select' else None)
            assert batch.inferences == 0
            if index == 0:
                assert batch.worker.queue_frames.await_count == 0
        await batch.manager._check_and_execute_transition()
        await batch.manager._check_and_execute_transition()
        frames = [f for call in batch.worker.queue_frames.await_args_list for f in call.args[0]]
        assert sum(isinstance(f, LLMMessagesUpdateFrame) for f in frames) == 1
        assert sum(isinstance(f, LLMRunFrame) for f in frames) == 1
        assert batch.manager._pending_transition is None
    asyncio.run(check())


def test_no_response_and_unmodified_native_manager_remain_unchanged():
    async def check():
        batch = Batch()
        await batch.start('silent')
        await batch.finish('silent', next_node=NO_RESPONSE)
        assert batch.inferences == 0 and batch.properties[-1].run_llm is False
        native = Batch(FlowManager)
        await native.start('read')
        await native.finish('read')
        assert native.properties[-1].run_llm is True  # No global Pipecat patch.
    asyncio.run(check())


def test_diagnostics_link_call_and_backend_result_without_free_text():
    async def check():
        batch = Batch()
        trace = SimpleNamespace(emit=Mock())
        batch.manager.state['session'] = SimpleNamespace(trace=trace)
        await batch.start('lookup')
        await batch.finish('lookup', arguments={'case_id': '<corrected_case_id>',
            'study_id': None, 'model_result_id': 'MR-1055-01-V1',
            'report_id': 'private free text', 'patient_name': 'Private Person',
            'query': 'private medical text'})
        name, = trace.emit.call_args.args
        fields = trace.emit.call_args.kwargs
        assert name == 'clinician_tool_call'
        assert fields['tool_call_id'] == 'lookup' and fields['request_id'] == 'backend-lookup'
        assert fields['record_arguments'] == {'case_id': '<corrected_case_id>',
            'study_id': None, 'model_result_id': 'MR-1055-01-V1', 'report_id': '<redacted>'}
        assert 'Private Person' not in str(fields) and 'medical text' not in str(fields)
        assert _trace_record_ids({'case_id': {'unexpected': 'private'}}) == {'case_id': '<redacted>'}
    asyncio.run(check())


def test_native_llm_fast_error_and_slow_result_produce_one_continuation(monkeypatch):
    import copy
    from openai.types.chat import ChatCompletionChunk
    from pipecat.frames.frames import CancelFrame, ErrorFrame, LLMContextFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.tests.utils import SleepFrame, run_test
    from healthcare_voice_agent.config import load_config
    from healthcare_voice_agent.voice.providers import build_llm

    async def check():
        config = load_config(None, environ={'AGENT_MODE': 'clinician',
            'LIVE_API_ENABLED': 'true', 'OPENAI_API_KEY': 'offline-placeholder'})
        llm = build_llm(config, system_instruction='Offline batch test.')
        requests = []
        scripts = iter([
            {'tool_calls': [
                {'index': 0, 'id': 'fast-id', 'type': 'function',
                 'function': {'name': 'fast_error', 'arguments': '{}'}},
                {'index': 1, 'id': 'slow-id', 'type': 'function',
                 'function': {'name': 'slow_study', 'arguments': '{}'}},
            ]},
            {'content': 'Both results are available.'},
        ])
        async def create(**kwargs):
            requests.append(copy.deepcopy(kwargs['messages']))
            delta = next(scripts)
            async def stream():
                # Match OpenAI's streamed per-tool deltas; Pipecat reads the first
                # tool delta in each chunk rather than a bundled final message.
                chunks = ([{'tool_calls': [call]} for call in delta['tool_calls']]
                          if 'tool_calls' in delta else [delta])
                for chunk in chunks:
                    yield ChatCompletionChunk(id='offline', object='chat.completion.chunk',
                        created=0, model='offline', choices=[{'index': 0, 'delta': chunk}])
            return stream()
        monkeypatch.setattr(llm._client.chat.completions, 'create', create)
        context = LLMContext([{'role': 'user', 'content': 'Get the study and model result.'}])
        aggregator = LLMAssistantAggregator(context)
        arrivals = []
        native_result = aggregator._handle_function_call_result
        async def observe_result(frame):
            arrivals.append((frame.function_name, frame.run_llm, frame.properties.run_llm,
                [(key, value.group_id if value else None)
                 for key, value in aggregator._function_calls_in_progress.items()]))
            await native_result(frame)
        monkeypatch.setattr(aggregator, '_handle_function_call_result', observe_result)
        worker = Mock(spec=PipelineWorker)
        worker.queue_frames = AsyncMock()
        manager = ClinicianFlowManager(llm=llm, worker=worker,
            context_aggregator=SimpleNamespace(assistant=lambda: aggregator))
        async def fast(args, flow):
            return {'status': 'error', 'error': 'INVALID_ARGUMENT'}, None
        async def slow(args, flow):
            await asyncio.sleep(0.04)
            return {'status': 'ok', 'study_id': 'ST-1042-01'}, None
        for name, handler in [('fast_error', fast), ('slow_study', slow)]:
            llm.register_function(name, await manager._create_transition_func(name, handler),
                                  cancel_on_interruption=True)
        down, up = await asyncio.wait_for(run_test(
            Pipeline([llm, aggregator]),
            frames_to_send=[LLMContextFrame(context), SleepFrame(sleep=0.2), CancelFrame()],
            send_end_frame=False, start_timeout=5), timeout=5)
        errors = [frame for frame in [*down, *up] if isinstance(frame, ErrorFrame)]
        assert not errors, '\n'.join(str(item) for item in arrivals)
        assert len(requests) == 2
        results = [message['content'] for message in requests[1] if message['role'] == 'tool']
        assert len(results) == 2 and all('IN_PROGRESS' not in value for value in results)
        assert any('INVALID_ARGUMENT' in value for value in results)
        assert any('ST-1042-01' in value for value in results)
    asyncio.run(check())


def test_unrelated_group_can_continue_while_case_transition_has_pending_sibling():
    async def check():
        batch = Batch()
        await batch.manager.initialize({**node(), 'respond_immediately': False})
        batch.worker.queue_frames.reset_mock()
        await batch.start('select', group='case-batch')
        await batch.start('slow', group='case-batch')
        await batch.finish('select', next_node=node())
        assert batch.manager._pending_transition is not None
        await batch.start('independent', group='different-batch')
        await batch.finish('independent')
        assert batch.inferences == 1  # No global hold on unrelated tool groups.
        assert batch.manager._pending_transition is not None
        await batch.finish('slow')
        assert batch.inferences == 1
        frames = [f for call in batch.worker.queue_frames.await_args_list for f in call.args[0]]
        assert sum(isinstance(f, LLMRunFrame) for f in frames) == 1
        assert batch.manager._pending_transition is None
    asyncio.run(check())
