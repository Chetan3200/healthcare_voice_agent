"""Offline native OpenAI SDK retry coverage for the clinician LLM factory."""

import asyncio
import socket
from collections.abc import Callable

import anyio
import httpx2
import openai
import pytest

from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice import providers


@pytest.fixture(autouse=True)
def block_socket_connections(monkeypatch):
    """Keep these SDK retry tests offline even if transport wiring changes."""
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline retry tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def clinician_config():
    return load_config(None, environ={
        "OPENAI_API_KEY": "offline-placeholder",
        "LIVE_API_ENABLED": "true",
        "AGENT_MODE": "clinician",
    })


def _completion_response(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200,
        request=request,
        json={
            "id": "offline-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "offline-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "offline success"},
                "finish_reason": "stop",
            }],
        },
    )


def _failure_response(status_code: int) -> Callable[[httpx2.Request], httpx2.Response]:
    def response(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status_code,
            request=request,
            json={"error": {"message": "offline failure", "type": "test_error"}},
        )
    return response


def _connection_failure(request: httpx2.Request) -> None:
    raise httpx2.ConnectError("offline connection failure", request=request)


def _timeout_failure(request: httpx2.Request) -> None:
    raise httpx2.ReadTimeout("offline timeout", request=request)


def _build_llm_with_outcomes(monkeypatch, config, outcomes):
    attempts = []
    queued = list(outcomes)

    async def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(request)
        outcome = queued.pop(0)
        return outcome(request)

    def mock_http_client(*args, **kwargs):
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    monkeypatch.setattr(providers, "_http_client", mock_http_client)
    return providers.build_llm(config), attempts


async def _create_completion(service):
    return await service._client.chat.completions.create(
        model="offline-model",
        messages=[{"role": "user", "content": "offline request"}],
    )


@pytest.mark.parametrize("transient_failure", [
    _connection_failure,
    _timeout_failure,
    _failure_response(429),
    _failure_response(500),
    _failure_response(503),
])
def test_clinician_sdk_retries_one_transient_failure_then_succeeds(
    monkeypatch, clinician_config, transient_failure,
):
    service, attempts = _build_llm_with_outcomes(
        monkeypatch, clinician_config, [transient_failure, _completion_response],
    )
    try:
        completion = asyncio.run(_create_completion(service))
    finally:
        asyncio.run(service.cleanup())

    assert completion.choices[0].message.content == "offline success"
    assert len(attempts) == 2


@pytest.mark.parametrize("persistent_failure", [
    _connection_failure,
    _failure_response(503),
])
def test_clinician_sdk_stops_after_one_retry_for_persistent_transient_failure(
    monkeypatch, clinician_config, persistent_failure,
):
    service, attempts = _build_llm_with_outcomes(
        monkeypatch, clinician_config, [persistent_failure, persistent_failure],
    )
    try:
        with pytest.raises(openai.APIError):
            asyncio.run(_create_completion(service))
    finally:
        asyncio.run(service.cleanup())

    assert len(attempts) == 2


@pytest.mark.parametrize("status_code", [400, 401])
def test_clinician_sdk_does_not_retry_non_retryable_client_errors(
    monkeypatch, clinician_config, status_code,
):
    service, attempts = _build_llm_with_outcomes(
        monkeypatch, clinician_config, [_failure_response(status_code)],
    )
    try:
        with pytest.raises(openai.APIStatusError):
            asyncio.run(_create_completion(service))
    finally:
        asyncio.run(service.cleanup())

    assert len(attempts) == 1


def test_cancelling_clinician_sdk_backoff_prevents_retry(monkeypatch, clinician_config):
    service, attempts = _build_llm_with_outcomes(
        monkeypatch, clinician_config, [_failure_response(503), _completion_response],
    )
    entered_backoff = asyncio.Event()

    async def paused_backoff(delay):
        entered_backoff.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(anyio, "sleep", paused_backoff)

    async def exercise():
        task = asyncio.create_task(_create_completion(service))
        await entered_backoff.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise())
    finally:
        asyncio.run(service.cleanup())

    assert len(attempts) == 1
