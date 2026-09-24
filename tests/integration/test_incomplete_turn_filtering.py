"""Native Pipecat flow with scripted LLM streams; no network/audio/model downloads.

These test protocol/wiring, NOT the LLM's judgement of real hesitant speech.
"""
import asyncio
import copy
import socket
from types import SimpleNamespace

import pytest
from loguru import logger
from openai.types.chat import ChatCompletionChunk
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import (
    AggregatedTextFrame, CancelFrame, ErrorFrame, LLMUpdateSettingsFrame, STTMetadataFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_start import VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy
from pipecat.turns.user_stop.deferred_user_turn_stop_strategy import DeferredUserTurnStopStrategy
from pipecat.turns.user_stop.llm_turn_completion_user_turn_stop_strategy import LLMTurnCompletionUserTurnStopStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies

from healthcare_voice_agent.config import load_config, write_run_config
from healthcare_voice_agent.voice import pipeline as wiring
from healthcare_voice_agent.voice.providers import build_llm
from healthcare_voice_agent.voice.reply_tts import FullReplyTTSBuffer
from healthcare_voice_agent.voice.tracing import SessionTrace
from healthcare_voice_agent.voice.turns import SpeechGenerationTurnStopStrategy
from tests.integration.test_finalized_turn_strategy import Analyzer, transcript, strategy_only, stopped_speech
from tests.integration.test_voice_pipeline import FakeService, FakeTransport


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Network forbidden in incomplete-turn tests')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)
    monkeypatch.delenv('PIPECAT_SMART_TURN_LOG_DATA', raising=False)
    logger.disable('pipecat')  # Do not dump composed prompts in a failing test.
    yield
    logger.enable('pipecat')


def configuration(mode='clinician'):
    return load_config(None, environ={
        'AGENT_MODE': mode, 'LIVE_API_ENABLED': 'true', 'OPENAI_API_KEY': 'offline-placeholder',
    })


def utterance(text, generation=1, wait=0.12):
    return [VADUserStartedSpeakingFrame(), SleepFrame(sleep=0.02),
            VADUserStoppedSpeakingFrame(stop_secs=0.2), transcript(text, generation),
            SleepFrame(sleep=wait)]


async def exchange(monkeypatch, responses, frames, *, completion_config=None, tool=False,
                   verdict=EndOfTurnState.COMPLETE):
    llm = build_llm(configuration(), system_instruction='Synthetic offline role.')
    requests, stopped, tool_calls = [], [], []
    scripts = iter(responses)

    async def create(**kwargs):
        requests.append(copy.deepcopy(kwargs['messages']))
        script = next(scripts)  # Unexpected extra inference fails the assertions below.
        async def stream():
            for delta in script:
                if isinstance(delta, tuple):
                    delay, delta = delta
                    await asyncio.sleep(delay)
                if isinstance(delta, str):
                    delta = {'content': delta}
                yield ChatCompletionChunk(id='offline', object='chat.completion.chunk',
                    created=0, model='offline', choices=[{'index': 0, 'delta': delta}])
        return stream()
    monkeypatch.setattr(llm._client.chat.completions, 'create', create)
    context = LLMContext()
    strategy = SpeechGenerationTurnStopStrategy(turn_analyzer=Analyzer(verdict))
    pair = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        user_turn_stop_timeout=15.0,
        user_turn_strategies=FilterIncompleteUserTurnStrategies(
            start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy(use_interim=False)],
            stop=[strategy], config=completion_config)))

    async def on_stop(aggregator, strategy, message):
        stopped.append(message.content)
    pair.user().add_event_handler('on_user_turn_stopped', on_stop)
    if tool:
        async def lookup(params):
            tool_calls.append(params.arguments)
            await params.result_callback({'status': 'ok', 'data': {'appointments': []}})
        llm.register_function('get_appointments', lookup, cancel_on_interruption=True)

    spoken = []
    class CaptureTTSInput(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, AggregatedTextFrame):
                spoken.append(frame.text)
            await self.push_frame(frame, direction)

    down, up = await asyncio.wait_for(run_test(
        Pipeline([pair.user(), llm, FullReplyTTSBuffer(), CaptureTTSInput(), pair.assistant()]),
        # Match a disconnected voice session. Native aggregator shutdown can
        # flush unsubmitted text; test the detector-before-shutdown separately.
        frames_to_send=[STTMetadataFrame(service_name='offline-stt', ttfs_p99_latency=1.0),
                        *frames, CancelFrame()], send_end_frame=False,
        start_timeout=5), timeout=5)
    assert not any(isinstance(f, ErrorFrame) for f in [*down, *up])
    assert llm._owned_client_closed
    return SimpleNamespace(
        requests=requests, stopped=stopped, tools=tool_calls, llm=llm,
        spoken=spoken,
        context=context.get_messages())


@pytest.mark.parametrize('mode', ['clinician', 'frontdesk_demo', 'conversation'])
def test_profile_wiring_keeps_smart_turn_and_isolates_filter(mode, monkeypatch, tmp_path):
    pairs, analyzers = [], []
    native_pair = wiring.LLMContextAggregatorPair
    def capture_pair(*args, **kwargs):
        pairs.append(kwargs['user_params'])
        return native_pair(*args, **kwargs)
    def analyzer(**kwargs):
        value = Analyzer()
        analyzers.append(value)
        return value
    monkeypatch.setattr(wiring, 'LLMContextAggregatorPair', capture_pair)
    monkeypatch.setattr(wiring, 'LocalSmartTurnAnalyzerV3', analyzer)
    monkeypatch.setattr(wiring.providers, 'build_stt', lambda *a, **kw: FakeService('stt'))
    monkeypatch.setattr(wiring.providers, 'build_llm', lambda *a, **kw: FakeService('llm'))
    monkeypatch.setattr(wiring.providers, 'build_tts', lambda *a, **kw: FakeService('tts'))
    adapter = SimpleNamespace(processors=[], input_gate=FakeService('input-gate'),
        bind=lambda *a: None)
    options = {'clinician_adapter': adapter} if mode == 'clinician' else (
        {'frontdesk_adapter': adapter} if mode == 'frontdesk_demo' else {})
    async def check():
        trace = SessionTrace(tmp_path / 'events.jsonl', session_id='offline')
        bundle = None
        try:
            bundle = await wiring.build_pipeline(configuration(mode), FakeTransport(), trace,
                                                 prompt='Synthetic offline role.', **options)
            strategies = pairs[0].user_turn_strategies
            if mode == 'clinician':
                assert isinstance(strategies, FilterIncompleteUserTurnStrategies)
                assert isinstance(strategies.stop[0], DeferredUserTurnStopStrategy)
                assert isinstance(strategies.stop[1], LLMTurnCompletionUserTurnStopStrategy)
                assert strategies.stop[1].config.incomplete_short_timeout == 5
                assert strategies.stop[1].config.incomplete_long_timeout == 10
                assert pairs[0].user_turn_stop_timeout > 10
                detector = strategies.stop[0].inner
            else:
                assert not isinstance(strategies, FilterIncompleteUserTurnStrategies)
                assert len(strategies.stop) == 1 and pairs[0].user_turn_stop_timeout == 5
                detector = strategies.stop[0]
            assert isinstance(detector, SpeechGenerationTurnStopStrategy)
            assert detector._turn_analyzer is analyzers[0] and detector._finalize_transcripts
            record = write_run_config(configuration(mode), tmp_path / 'run').read_text()
            assert ('native conversational LLM gate' in record) == (mode == 'clinician')
        finally:
            if bundle:
                await bundle.cleanup_unstarted()
            trace.close()
    asyncio.run(check())


def test_complete_response_is_one_request_without_spoken_markers(monkeypatch):
    async def check():
        config = UserTurnCompletionConfig()
        result = await exchange(monkeypatch, [[config.complete_marker, ' The report is reviewed.']],
            utterance('Show the reviewed report.'))
        # A space arriving in the next chunk is preserved; no marker reaches TTS.
        assert result.spoken == [' The report is reviewed.']
        assert result.stopped == ['Show the reviewed report.']
        assert len(result.requests) == 1
        assert result.llm._filter_incomplete_user_turns
    asyncio.run(check())


@pytest.mark.parametrize('kind', ['short', 'long'])
def test_incomplete_is_silent_and_continuation_retains_both_fragments(kind, monkeypatch):
    async def check():
        config = UserTurnCompletionConfig()
        marker = getattr(config, f'incomplete_{kind}_marker')
        result = await exchange(monkeypatch, [[marker, ' This must never reach TTS.'],
            [config.complete_marker + ' I will compare those reports.']], [
            *utterance('Compare January with'),
            *utterance('the March report.', generation=2)])
        assert result.spoken == ['I will compare those reports.']
        assert result.stopped == ['Compare January with the March report.']
        assert len(result.requests) == 2  # Initial verdict, then the continued request.
        second = str(result.requests[1])
        assert 'Compare January with' in second and 'the March report.' in second
        assert any(m.get('role') == 'assistant' and m.get('content') == marker
                   for m in result.requests[1])
        assert result.llm._incomplete_timeout_task is None
    asyncio.run(check())


def test_native_timeout_reprompts_once_and_role_update_keeps_filter(monkeypatch):
    async def check():
        config = UserTurnCompletionConfig(incomplete_short_timeout=0.05, incomplete_long_timeout=0.08)
        result = await exchange(monkeypatch, [[config.incomplete_long_marker],
            [config.complete_marker + ' Take your time.']], [
            LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction='Replacement synthetic role.')),
            *utterance('Let me think.', wait=0.3)], completion_config=config)
        assert result.spoken == ['Take your time.'] and len(result.requests) == 2
        assert result.stopped == ['Let me think.']
        system = next(m['content'] for m in result.requests[0] if m['role'] in ('system', 'developer'))
        assert 'Replacement synthetic role.' in system
        assert config.completion_instructions in system
        assert result.llm._incomplete_timeout_task is None
    asyncio.run(check())


def test_markerless_tool_call_completes_turn_and_post_tool_answer_is_spoken(monkeypatch):
    async def check():
        args = '{"case_id":"1042","time_scope":"upcoming","appointment_type":null,"statuses":null}'
        call = {'tool_calls': [{'index': 0, 'id': 'offline-call', 'type': 'function',
            'function': {'name': 'get_appointments', 'arguments': args}}]}
        complete = UserTurnCompletionConfig().complete_marker
        result = await exchange(monkeypatch, [[call], [complete + ' No matching appointments.']],
            utterance('List my upcoming appointments.', wait=0.25), tool=True)
        assert len(result.tools) == 1 and result.tools[0]['case_id'] == '1042'
        assert result.spoken == ['No matching appointments.']
        assert result.stopped == ['List my upcoming appointments.']
        assert len(result.requests) == 2
    asyncio.run(check())


def test_smart_turn_incomplete_does_not_trigger_inference_while_listening():
    async def check():
        async with strategy_only(EndOfTurnState.INCOMPLETE) as (detector, manager, stopped):
            wrapped = FilterIncompleteUserTurnStrategies(stop=[detector]).stop[0]
            triggered = []
            async def inference(*args):
                triggered.append(True)
            wrapped.add_event_handler('on_user_turn_inference_triggered', inference)
            await stopped_speech(wrapped)
            await wrapped.process_frame(transcript('Compare January with'))
            assert triggered == [] and stopped == []
    asyncio.run(check())


def test_resumed_speech_suppresses_an_old_complete_verdict(monkeypatch):
    async def check():
        complete = UserTurnCompletionConfig().complete_marker
        result = await exchange(monkeypatch, [[(0.08, complete + ' Obsolete answer.')],
            [complete + ' Answer to the continued request.']], [
            *utterance('Compare January with', wait=0.03),
            VADUserStartedSpeakingFrame(), SleepFrame(sleep=0.15),
            VADUserStoppedSpeakingFrame(stop_secs=0.2), transcript('March.', generation=2),
            SleepFrame(sleep=0.15)])
        assert len(result.requests) == 2
        assert result.spoken == ['Answer to the continued request.']
        assert result.stopped == ['Compare January with March.']
    asyncio.run(check())


def test_resuming_cancels_reprompt_even_while_still_speaking(monkeypatch):
    async def check():
        config = UserTurnCompletionConfig(incomplete_short_timeout=0.08, incomplete_long_timeout=0.08)
        result = await exchange(monkeypatch, [[config.incomplete_long_marker],
            [config.complete_marker + ' Here is the report.']], [
            *utterance('Let me think.', wait=0.03),
            VADUserStartedSpeakingFrame(), SleepFrame(sleep=0.18),
            VADUserStoppedSpeakingFrame(stop_secs=0.2), transcript('Show the report.', generation=2),
            SleepFrame(sleep=0.15)], completion_config=config)
        assert len(result.requests) == 2 and result.spoken == ['Here is the report.']
        assert result.stopped == ['Let me think. Show the report.']
    asyncio.run(check())
