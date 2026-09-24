"""Real Pipecat/OpenAI SSE client against in-memory HTTPX2, no models/network."""

import asyncio
import json
import socket

import pytest

pytest.importorskip("pipecat")
httpx2 = pytest.importorskip("httpx2")
from pipecat.processors.aggregators.llm_context import LLMContext
from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice import providers


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No network allowed")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


def config(**overrides):
    return load_config(None, environ={
        "STT_PROVIDER": "nemotron", "LLM_PROVIDER": "hybrid_diffusion", "TTS_PROVIDER": "breeze",
        "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "private-openai-never-forward",
        **overrides,
    })


def chunk(content):
    payload = {"id": "test", "object": "chat.completion.chunk", "created": 1,
               "model": "yuchen-zhu-zyc/HybridDiffusion-2B",
               "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
    return ("data: " + json.dumps(payload) + "\n\n").encode()


class Body(httpx2.AsyncByteStream):
    def __init__(self, *, stall=False):
        self.closed = False
        self.stall = stall
        self.started = asyncio.Event()

    async def __aiter__(self):
        yield chunk("Hello ")
        self.started.set()
        if self.stall:
            await asyncio.Event().wait()
        yield chunk("from the test.")
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


def build(monkeypatch, handler, **env):
    monkeypatch.setattr(providers, "_http_client", lambda *a, **kw: httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), trust_env=False,
    ))
    return providers.build_llm(config(**env), system_instruction="Synthetic private prompt marker")


def test_sse_request_and_owned_cleanup(monkeypatch):
    async def run():
        requests = []
        body = Body()
        def handler(request):
            requests.append(request)
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
        service = build(monkeypatch, handler, LLM_API_KEY="private-self-hosted-key", LLM_MAX_OUTPUT_TOKENS="123")
        try:
            assert service._client.max_retries == 0
            stream = await service.get_chat_completions(LLMContext(messages=[{"role": "user", "content": "Synthetic hello"}]))
            values = [part.choices[0].delta.content async for part in stream]
            await stream.close()
            assert values == ["Hello ", "from the test."]
            payload = json.loads(requests[0].content)
            assert str(requests[0].url) == "http://127.0.0.1:30000/v1/chat/completions"
            assert payload["stream"] is True
            assert payload["max_completion_tokens"] == 123
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            assert payload["messages"][0] == {"role": "system", "content": "Synthetic private prompt marker"}
            assert requests[0].headers["authorization"] == "Bearer private-self-hosted-key"
            assert "private-openai-never-forward" not in str(requests[0].headers)
            assert body.closed
        finally:
            await service.cleanup()
            await service.cleanup()
        assert service._client.is_closed()
    asyncio.run(run())


@pytest.mark.parametrize("cancel", [True, False])
def test_native_consumer_closes_stream_on_cancel_or_total_deadline(monkeypatch, cancel):
    async def run():
        body = Body(stall=True)
        service = build(monkeypatch, lambda request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body,
        ), LLM_TIMEOUT_SECONDS="0.05")
        text = []
        async def collect(value):
            text.append(value)
        monkeypatch.setattr(service, "_push_llm_text", collect)
        try:
            task = asyncio.create_task(service._process_context(LLMContext(messages=[{"role": "user", "content": "Hello"}])))
            await asyncio.wait_for(body.started.wait(), 1)
            if cancel:
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
                await task
            assert body.closed
            assert text == ["Hello "]
        finally:
            await service.cleanup()
    asyncio.run(run())


def test_tools_and_tool_choice_survive_request_adaptation(monkeypatch):
    async def run():
        service = build(monkeypatch, lambda request: None)
        try:
            tool = {"type": "function", "function": {"name": "synthetic_tool", "parameters": {"type": "object", "properties": {}}}}
            params = service.build_chat_completion_params({"messages": [], "tools": [tool], "tool_choice": "auto"})
            assert params["tools"] == [tool]
            assert params["tool_choice"] == "auto"
        finally:
            await service.cleanup()
    asyncio.run(run())


def test_backend_error_frames_do_not_include_private_exception(monkeypatch):
    async def run():
        service = build(monkeypatch, lambda request: None)
        frames = []
        async def collect(error, **kwargs):
            frames.append(error)
        monkeypatch.setattr(service, "push_error_frame", collect)
        try:
            await service.push_error("private-backend-body", exception=RuntimeError("private-backend-body"))
            assert len(frames) == 1
            assert "private-backend-body" not in frames[0].error
            assert frames[0].exception is None
        finally:
            await service.cleanup()
    asyncio.run(run())


def test_all_three_factories_construct_without_keys_network_or_native_models():
    async def run():
        cfg = config(OPENAI_API_KEY="")
        services = [providers.build_stt(cfg), providers.build_llm(cfg), providers.build_tts(cfg)]
        try:
            stt, llm, tts = services
            assert type(stt).__name__ == "NemotronSTTService"
            assert type(tts).__name__ == "BreezeTTSService"
            assert stt._socket is None and tts._session is None
            assert stt._endpoint.endswith("/v1/audio/transcriptions/realtime")
            assert tts._endpoint.endswith("/v1/audio/speech")
            assert llm._client.api_key == "not-required"
            assert llm._client._client.follow_redirects is False
            assert llm._client._client.trust_env is False
        finally:
            for service in reversed(services):
                await service.cleanup()
    asyncio.run(run())
