"""Offline integration coverage for constructed OpenAI HTTP clients and pooling.

The fake backend feeds HTTP/1.1 bytes into the real httpcore2 connection pool.
It never opens a socket and the http11 module alone receives a controllable clock.
"""

import asyncio
import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("pipecat")
httpcore2 = pytest.importorskip("httpcore2")
httpx2 = pytest.importorskip("httpx2")
openai = pytest.importorskip("openai")
from httpcore2._async import http11
from httpcore2._backends.mock import AsyncMockBackend, AsyncMockStream

from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice.providers import build_llm, build_tts
from healthcare_voice_agent.voice.tracing import SessionTrace


_RESPONSE = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Length: 2\r\n"
    b"x-request-id: in-memory-request\r\n"
    b"\r\n"
    b"ok"
)


class _Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class _CountingBackend(AsyncMockBackend):
    """AsyncMockBackend with a fresh, in-memory HTTP response per socket."""

    def __init__(self):
        super().__init__([])
        self.connects = 0

    async def connect_tcp(self, *args, **kwargs):
        self.connects += 1
        # Two responses let the same HTTP/1.1 connection serve two requests.
        return AsyncMockStream([_RESPONSE, _RESPONSE])


class _FailingBackend(AsyncMockBackend):
    async def connect_tcp(self, *args, **kwargs):
        raise httpcore2.ConnectError("in-memory connection failure")


def _config(expiry: int, scheme: str = "http"):
    return load_config(None, environ={
        "OPENAI_API_KEY": "sk-test-keepalive-secret",
        "LIVE_API_ENABLED": "true",
        "LLM_BASE_URL": f"{scheme}://127.0.0.1:8765/v1",
        "TTS_BASE_URL": f"{scheme}://127.0.0.1:8765/v1",
        "LLM_KEEPALIVE_EXPIRY_SECONDS": str(expiry),
        "TTS_KEEPALIVE_EXPIRY_SECONDS": str(expiry),
    })


def _http_client(service):
    """The actual HTTPX2 client owned by the real SDK client."""
    return service._client._client


def test_keepalive_expiry_defaults_to_sixty_seconds():
    config = load_config(None, environ={
        "OPENAI_API_KEY": "offline-placeholder", "LIVE_API_ENABLED": "true",
    })
    assert config.llm.keepalive_expiry_seconds == config.tts.keepalive_expiry_seconds == 60


def _pool(service):
    return _http_client(service)._transport._pool


def _url(service):
    return str(service._client.base_url).rstrip("/") + "/offline"


def _records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _block_real_sockets_and_proxies(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("this offline test must use AsyncMockBackend, not a socket")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.mark.parametrize(("factory", "stage"), [(build_llm, "llm"), (build_tts, "tts")])
@pytest.mark.parametrize("expiry, expected_connects", [(60, 1), (5, 2)])
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_constructed_client_applies_keepalive_without_mutating_sdk_defaults(
    monkeypatch, tmp_path, factory, stage, expiry, expected_connects, scheme,
):
    """A 10-second idle connection is reused at 60s and replaced at 5s."""
    _block_real_sockets_and_proxies(monkeypatch)
    clock = _Clock()
    # httpcore2's HTTP/1.1 expiry implementation calls this module reference.
    # Do not replace global time.monotonic, which would affect asyncio/pytest.
    monkeypatch.setattr(http11, "time", clock)
    trace_path = tmp_path / f"{stage}-{expiry}.jsonl"
    trace = SessionTrace(trace_path, session_id="offline", secrets=("sk-test-keepalive-secret",))
    service = factory(_config(expiry, scheme), trace=trace)
    backend = _CountingBackend()
    pool = _pool(service)
    defaults = openai.DEFAULT_CONNECTION_LIMITS
    default_shape = (
        defaults.max_connections,
        defaults.max_keepalive_connections,
        defaults.keepalive_expiry,
    )
    try:
        assert defaults.keepalive_expiry == 5.0
        assert pool._keepalive_expiry == expiry
        assert pool._max_connections == defaults.max_connections
        assert pool._max_keepalive_connections == defaults.max_keepalive_connections
        assert service._client.max_retries == 0
        import ssl
        assert pool._ssl_context.verify_mode == ssl.CERT_REQUIRED
        assert pool._ssl_context.check_hostname is True
        pool._network_backend = backend

        async def exercise():
            # Constructing the streaming manager must not connect before entry.
            stream = _http_client(service).stream("POST", _url(service), content=b"private-request-body")
            assert backend.connects == 0
            async with stream as first:
                assert first.is_stream_consumed is False
                assert await first.aread() == b"ok"
            clock.now += 10
            second = await _http_client(service).post(_url(service), content=b"private-request-body")
            assert second.content == b"ok"

        asyncio.run(exercise())
        assert backend.connects == expected_connects
        assert (
            defaults.max_connections,
            defaults.max_keepalive_connections,
            defaults.keepalive_expiry,
        ) == default_shape
    finally:
        asyncio.run(service.cleanup())
        closed = _http_client(service).is_closed
        trace.close()
    assert closed

    records = _records(trace_path)
    starts = [record for record in records if record["event"] == "http_request_start"]
    headers = [record for record in records if record["event"] == "http_response_headers"]
    phases = [record for record in records if record["event"] == "http_connection_phase"]
    assert len(starts) == len(headers) == 2
    assert [record["data"]["connection_reuse"] for record in headers] == ["new", "reused" if expiry == 60 else "new"]
    assert all(record["data"]["stage"] == stage for record in starts + headers + phases)
    assert all(record["data"]["request_id"] for record in starts + headers)
    assert all(record["data"]["status_code"] == 200 for record in headers)
    assert all("response_headers_seconds" in record["data"] for record in headers)
    assert all("tcp_connect_seconds" in record["data"] for record in headers)
    expected_phases = {"tcp_connect", "tls_handshake"} if scheme == "https" else {"tcp_connect"}
    assert {record["data"]["phase"] for record in phases} == expected_phases
    assert len(phases) == expected_connects * len(expected_phases)
    assert all(record["data"]["outcome"] == "complete" for record in phases)
    encoded = trace_path.read_text(encoding="utf-8")
    for forbidden in ("sk-test-keepalive-secret", "private-request-body", "127.0.0.1:8765", "Authorization"):
        assert forbidden not in encoded


@pytest.mark.parametrize(("factory", "stage"), [(build_llm, "llm"), (build_tts, "tts")])
def test_constructed_client_propagates_core_failure_and_traces_it(monkeypatch, tmp_path, factory, stage):
    _block_real_sockets_and_proxies(monkeypatch)
    trace_path = tmp_path / f"{stage}-failure.jsonl"
    trace = SessionTrace(trace_path, session_id="offline")
    service = factory(_config(60), trace=trace)
    pool = _pool(service)
    pool._network_backend = _FailingBackend([])
    try:
        async def exercise():
            with pytest.raises(httpx2.ConnectError):
                await _http_client(service).get(_url(service))

        asyncio.run(exercise())
    finally:
        asyncio.run(service.cleanup())
        closed = _http_client(service).is_closed
        trace.close()
    assert closed

    records = _records(trace_path)
    failure = next(record for record in records if record["event"] == "http_request_failed")
    phase = next(record for record in records if record["event"] == "http_connection_phase")
    assert failure["data"] == {
        "stage": stage,
        "request_id": failure["data"]["request_id"],
        "phase": "tcp_connect",
        "headers_received": False,
        "error_kind": "transport",
    }
    assert phase["data"]["phase"] == "tcp_connect"
    assert phase["data"]["outcome"] == "failed"
