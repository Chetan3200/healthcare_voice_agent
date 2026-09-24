"""Native LLM/buffer/TTS recovery with scripted streams; no provider/network calls."""
import asyncio
import copy
import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import httpx2
import pytest
from loguru import logger
from openai import APIConnectionError, AuthenticationError
from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import (
    CancelFrame, ErrorFrame, InterruptionFrame, LLMContextFrame, TTSTextFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator, LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies

from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice.providers import build_llm, build_tts
from healthcare_voice_agent.voice.reply_tts import FullReplyTTSBuffer, LLMErrorFallbackFrame


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden in fallback tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    logger.disable("pipecat")
    yield
    logger.enable("pipecat")


def connection_error():
    return APIConnectionError(request=httpx2.Request("POST", "https://offline.invalid/v1"))


async def exercise(monkeypatch, scripts, *, mode="clinician", interrupt=False,
                   incomplete_gate=False, next_request=True, sdk_outcomes=None, tool=False,
                   completion_config=None):
    config = load_config(None, environ={
        "AGENT_MODE": mode, "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "offline-placeholder",
    })
    trace = SimpleNamespace(emit=Mock())
    llm = build_llm(config, trace=trace)
    tts = build_tts(config)
    requests, speech, closed_streams = [], [], []
    scripts = iter(scripts)

    async def create(**kwargs):
        # This replaces the SDK boundary after retry exhaustion. Separate tests
        # use the real SDK HTTP loop to prove the retry count.
        requests.append(copy.deepcopy(kwargs["messages"]))
        script = next(scripts)
        if isinstance(script, Exception):
            raise script

        async def stream():
            try:
                for item in script:
                    if isinstance(item, Exception):
                        raise item
                    if isinstance(item, float):
                        await asyncio.sleep(item)
                        continue
                    yield ChatCompletionChunk(id="offline", object="chat.completion.chunk",
                        created=0, model="offline", choices=[{"index": 0, "delta": {"content": item}}])
            finally:
                closed_streams.append(True)
        return stream()

    class Audio:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def iter_bytes(self, chunk_size):
            yield bytes(480)  # Synthetic silence; no actual playback.

    def synthesize(**kwargs):
        speech.append(kwargs["input"])
        return Audio()

    http_requests = []
    if sdk_outcomes is None:
        monkeypatch.setattr(llm._client.chat.completions, "create", create)
    else:
        outcomes = iter(sdk_outcomes)
        async def send(request, **kwargs):
            http_requests.append(json.loads(request.content))
            outcome = next(outcomes)
            if outcome is None:
                raise httpx2.ConnectError("Offline transport failure", request=request)
            delta = {"content": outcome} if isinstance(outcome, str) else outcome
            chunk = {"id": "offline", "object": "chat.completion.chunk", "created": 0,
                "model": "offline", "choices": [{"index": 0, "delta": delta}]}
            return httpx2.Response(200, request=request,
                headers={"content-type": "text/event-stream"},
                content=("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
        monkeypatch.setattr(llm._client._client, "send", send)
        monkeypatch.setattr(llm._client, "_calculate_retry_timeout", lambda *a, **kw: 0)
    monkeypatch.setattr(tts._client.audio.speech.with_streaming_response, "create", synthesize)
    context = LLMContext([{"role": "user", "content": "Read the selected synthetic report."}])
    stages = [llm, FullReplyTTSBuffer(), tts]
    tool_calls = []
    if tool:
        async def read_record(params):
            tool_calls.append(params.arguments)
            await params.result_callback({"status": "ok", "data": {"reviewed": True}})
        llm.register_function("read_record", read_record, cancel_on_interruption=True)
        stages.append(LLMAssistantAggregator(context))
    if incomplete_gate:
        from healthcare_voice_agent.voice.turns import SpeechGenerationTurnStopStrategy
        from tests.integration.test_finalized_turn_strategy import Analyzer
        pair = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
            user_turn_strategies=FilterIncompleteUserTurnStrategies(
                stop=[SpeechGenerationTurnStopStrategy(turn_analyzer=Analyzer())],
                config=completion_config)))
        stages = [pair.user(), *stages, pair.assistant()]
    frames = [LLMContextFrame(context), SleepFrame(sleep=0.15)]
    if interrupt:
        frames.extend([InterruptionFrame(), SleepFrame(sleep=0.05)])
    if next_request:
        # A later user request, not an automatic application retry.
        frames.extend([UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(),
            LLMContextFrame(LLMContext([
                {"role": "user", "content": "Try again."},
            ])), SleepFrame(sleep=0.15)])
    frames.append(CancelFrame())
    down, up = await asyncio.wait_for(run_test(
        Pipeline(stages), frames_to_send=frames, send_end_frame=False, start_timeout=5,
    ), timeout=5)
    assert llm._owned_client_closed and tts._owned_client_closed
    return SimpleNamespace(llm=llm, speech=speech, requests=requests,
        down=down, up=up, closed_streams=closed_streams, trace=trace,
        http_requests=http_requests, tool_calls=tool_calls)


@pytest.mark.parametrize("incomplete_gate", [False, True])
def test_terminal_error_speaks_once_and_later_request_can_succeed(monkeypatch, incomplete_gate):
    async def check():
        from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
        prefix = UserTurnCompletionConfig().complete_marker if incomplete_gate else ""
        result = await exercise(monkeypatch, [connection_error(), [prefix + "The report is available."]],
                                incomplete_gate=incomplete_gate)
        assert result.speech == [LLMErrorFallbackFrame().text, "The report is available."]
        assert len(result.requests) == 2  # No hidden round replay from the fallback.
        assert sum(isinstance(frame, ErrorFrame) for frame in result.up) == 1
        assert result.llm.is_usable
        if not incomplete_gate:  # The assistant aggregator consumes TTSTextFrame.
            assert [frame.text for frame in result.down if isinstance(frame, TTSTextFrame)] == result.speech
        else:
            assert result.llm._filter_incomplete_user_turns
        result.trace.emit.assert_called_once_with("llm_error_fallback", action="queued_for_tts")
    asyncio.run(check())


def test_real_sdk_exhaustion_speaks_once_without_replaying_completed_tool(monkeypatch):
    async def check():
        result = await exercise(monkeypatch, [], next_request=False, tool=True, sdk_outcomes=[
            {"tool_calls": [{"index": 0, "id": "offline-read", "type": "function",
                "function": {"name": "read_record", "arguments": "{}"}}]},
            None, None,
        ])
        assert len(result.http_requests) == 3  # Tool round, then exactly two HTTP attempts.
        assert result.http_requests[1] == result.http_requests[2]
        assert any(message["role"] == "tool" and '"reviewed": true' in message["content"]
                   for message in result.http_requests[2]["messages"])
        assert result.tool_calls == [{}]
        assert result.speech == [LLMErrorFallbackFrame().text]
        assert sum(isinstance(frame, ErrorFrame) for frame in result.up) == 1
    asyncio.run(check())


def test_real_sdk_retry_success_does_not_speak_fallback(monkeypatch):
    async def check():
        result = await exercise(monkeypatch, [], next_request=False,
            sdk_outcomes=[None, "The report is available."])
        assert len(result.http_requests) == 2
        assert result.http_requests[0] == result.http_requests[1]
        assert result.speech == ["The report is available."]
        assert not any(isinstance(frame, ErrorFrame) for frame in result.up)
        result.trace.emit.assert_not_called()
    asyncio.run(check())


@pytest.mark.parametrize("gate_prefix", [None, "complete", "partial_marker", "no_marker"])
def test_failed_partial_reply_is_discarded_before_fallback(monkeypatch, gate_prefix):
    async def check():
        from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
        marker = UserTurnCompletionConfig().complete_marker
        partial = "UNFINISHED CLINICAL TEXT MUST NOT BE SPOKEN"
        if gate_prefix == "complete":
            partial = marker + partial
        elif gate_prefix == "partial_marker":
            partial = marker[:max(1, len(marker) // 2)]
        next_reply = (marker if gate_prefix is not None else "") + "The next reply."
        result = await exercise(monkeypatch, [
            [partial, connection_error()], [next_reply],
        ], incomplete_gate=gate_prefix is not None)
        assert result.speech == [LLMErrorFallbackFrame().text, "The next reply."]
        assert len(result.requests) == 2
        assert len(result.closed_streams) == 2
    asyncio.run(check())


@pytest.mark.parametrize("incomplete_gate", [False, True])
def test_interruption_cancels_stale_stream_without_fallback(monkeypatch, incomplete_gate):
    async def check():
        from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
        prefix = UserTurnCompletionConfig().complete_marker if incomplete_gate else ""
        result = await exercise(monkeypatch, [
            ["Stale unfinished text.", 30.0, connection_error()], [prefix + "The corrected request."],
        ], interrupt=True, incomplete_gate=incomplete_gate)
        assert result.speech == ["The corrected request."]
        assert not any(isinstance(frame, ErrorFrame) for frame in result.up)
        assert len(result.closed_streams) == 2
        result.trace.emit.assert_not_called()
    asyncio.run(check())


@pytest.mark.parametrize("kind", ["short", "long"])
def test_failed_incomplete_reply_does_not_schedule_hidden_reengagement(monkeypatch, kind):
    async def check():
        from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
        config = UserTurnCompletionConfig(incomplete_short_timeout=0.03, incomplete_long_timeout=0.03)
        marker = getattr(config, f"incomplete_{kind}_marker")
        result = await exercise(monkeypatch, [[marker, connection_error()]],
            incomplete_gate=True, completion_config=config, next_request=False)
        assert result.speech == [LLMErrorFallbackFrame().text]
        assert len(result.requests) == 1
        assert result.llm._incomplete_timeout_task is None
    asyncio.run(check())


@pytest.mark.parametrize("mode", ["frontdesk_demo", "conversation"])
def test_other_profiles_do_not_get_clinician_fallback(monkeypatch, mode):
    async def check():
        # Their session-level stop policy is covered in test_voice_session_errors.
        result = await exercise(monkeypatch, [connection_error(), ["Later reply."]], mode=mode)
        assert LLMErrorFallbackFrame().text not in result.speech
        result.trace.emit.assert_not_called()
    asyncio.run(check())


def test_permanent_authentication_error_does_not_offer_retry(monkeypatch):
    async def check():
        request = httpx2.Request("POST", "https://offline.invalid/v1")
        error = AuthenticationError("Offline authentication failure",
            response=httpx2.Response(401, request=request), body=None)
        # The real session stops unusable processors; this pipeline-only test
        # checks classification/fallback, not the separately tested stop handler.
        result = await exercise(monkeypatch, [error], next_request=False)
        assert not result.llm.is_usable
        assert result.speech == []
        result.trace.emit.assert_not_called()
    asyncio.run(check())
