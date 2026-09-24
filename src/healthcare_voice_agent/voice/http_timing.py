"""Selected HTTPX/httpcore timing hooks, without request/response contents.

No provider imports, body reads, warm-up calls, retries, or network configuration.
Reuse is inferred only when the pinned transport emits a request-header event
without a connection-setup event. Uninstrumented transports remain unknown.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from healthcare_voice_agent.voice.tracing import SessionTrace


try:  # The production transport is optional for unit-only installs.
    import httpcore2 as _httpcore
except ImportError:  # pragma: no cover - depends on the installed transport
    try:
        import httpcore as _httpcore
    except ImportError:  # pragma: no cover - unit-only installs
        _httpcore = None

try:  # StreamClosed is exposed by the HTTPX wrapper, when installed.
    import httpx2 as _httpx
except ImportError:  # pragma: no cover - depends on the installed wrapper
    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover - unit-only installs
        _httpx = None


def _exception_classes(module: Any, *names: str) -> tuple[type[BaseException], ...]:
    """Return only known exception classes from an optional transport module."""
    return tuple(
        candidate for name in names
        if isinstance(candidate := getattr(module, name, None), type)
        and issubclass(candidate, BaseException)
    )


_TIMEOUT_ERRORS = (asyncio.TimeoutError,) + _exception_classes(
    _httpcore, "TimeoutException",
)
_TRANSPORT_ERRORS = _exception_classes(
    _httpcore,
    "NetworkError", "ProtocolError", "ProxyError", "UnsupportedProtocol",
    "ConnectionNotAvailable",
)
_STREAM_CLOSED_ERRORS = _exception_classes(_httpx, "StreamClosed")


_STATE_KEY = "healthcare_http_timing"
_CONNECTION_PHASES = {
    "connection.connect_tcp": "tcp_connect",
    "connection.connect_unix_socket": "unix_connect",
    "connection.start_tls": "tls_handshake",
    "proxy.start_tls": "tls_handshake",
    "socks_proxy.connect_tcp": "tcp_connect",
    "socks_proxy.start_tls": "tls_handshake",
}
_HTTP_PHASES = {
    f"{protocol}.{operation}": label
    for protocol in ("http11", "http2")
    for operation, label in (
        ("send_request_headers", "send_headers"),
        ("send_request_body", "send_body"),
        ("receive_response_headers", "receive_headers"),
        ("receive_response_body", "receive_body"),
        ("response_closed", "response_close"),
    )
}


class HTTPConnectionTiming:
    """One hook pair per session-owned LLM/TTS client, with per-request state."""

    def __init__(self, trace: SessionTrace, *, stage: str,
                 clock: Callable[[], float] = time.perf_counter):
        if stage not in {"llm", "tts"}:
            raise ValueError("HTTP timing stage must be llm or tts")
        self._trace = trace
        self._stage = stage
        self._clock = clock
        self._sequence = 0

    def _emit(self, event: str, **fields: Any) -> None:
        try:
            self._trace.emit(event, stage=self._stage, **fields)
        except Exception:
            # Optional diagnostics must not fail a provider request
            # when their output sink fails. Never log the failed sink/object.
            pass

    async def on_request(self, request: Any) -> None:
        self._sequence += 1
        previous = request.extensions.get("trace")
        # Redirects or resending a Request can retain our old extension. Do not
        # chain stale timing states, but preserve an unrelated caller's hook.
        owner = getattr(previous, "__self__", None)
        if isinstance(owner, _RequestTiming):
            previous = owner.previous
        state = _RequestTiming(self, f"{self._stage}-{self._sequence}", previous)
        request.extensions[_STATE_KEY] = state
        request.extensions["trace"] = state.on_event
        state.emit("http_request_start")

    async def on_response(self, response: Any) -> None:
        state = response.request.extensions.get(_STATE_KEY)
        if isinstance(state, _RequestTiming) and state.owner is self:
            state.headers_received = True
            reuse = "new" if state.setup_observed else (
                "reused" if state.headers_started else "unknown"
            )
            state.emit(
                "http_response_headers", status_code=response.status_code,
                response_headers_seconds=round(self._clock() - state.started, 6),
                connection_reuse=reuse,
                tcp_connect_seconds=(None if reuse == "unknown" else
                                     round(state.durations["tcp_connect"], 6)),
                tls_handshake_seconds=(None if reuse == "unknown" else
                                       round(state.durations["tls_handshake"], 6)),
            )


class _RequestTiming:
    def __init__(self, owner: HTTPConnectionTiming, request_id: str, previous: Any):
        self.owner = owner
        self.request_id = request_id
        self.previous = previous
        self.started = owner._clock()
        self.setup_observed = False
        self.headers_started = False
        self.headers_received = False
        self._failure_recorded = False
        self._phase_starts: dict[str, float] = {}
        self.durations = {"tcp_connect": 0.0, "tls_handshake": 0.0, "unix_connect": 0.0}

    def emit(self, event: str, **fields: Any) -> None:
        self.owner._emit(event, request_id=self.request_id, **fields)

    def _failure_event(self, info: dict) -> tuple[str, str]:
        """Classify only a known exception object, without reading its contents."""
        try:
            exception = info.get("exception")
        except Exception:
            # The trace callback is optional. An unexpected info mapping must
            # not change the provider request's behavior.
            return "http_request_failed", "unknown"
        if isinstance(exception, asyncio.CancelledError):
            return "http_request_cancelled", "cancelled"
        if isinstance(exception, GeneratorExit):
            return "http_stream_closed", "generator_exit"
        if _STREAM_CLOSED_ERRORS and isinstance(exception, _STREAM_CLOSED_ERRORS):
            return "http_stream_closed", "stream_closed"
        if isinstance(exception, _TIMEOUT_ERRORS):
            return "http_request_failed", "timeout"
        if _TRANSPORT_ERRORS and isinstance(exception, _TRANSPORT_ERRORS):
            return "http_request_failed", "transport"
        return "http_request_failed", "unknown"

    async def on_event(self, name: str, info: dict) -> None:
        # `info` can contain Authorization headers, full request bodies, SSL
        # objects and raw exceptions. For failed known phases, inspect only the
        # exception object's class with isinstance; never store or serialize it.
        prefix, _, outcome = name.rpartition(".")
        phase = _CONNECTION_PHASES.get(prefix)
        if phase is not None and outcome in {"started", "complete", "failed"}:
            self.setup_observed = True
            now = self.owner._clock()
            if outcome == "started":
                self._phase_starts[prefix] = now
            elif outcome in {"complete", "failed"}:
                start = self._phase_starts.pop(prefix, None)
                elapsed = None if start is None else max(0.0, now - start)
                if elapsed is not None:
                    self.durations[phase] += elapsed
                self.emit("http_connection_phase", phase=phase, outcome=outcome,
                          duration_seconds=None if elapsed is None else round(elapsed, 6))
        http_phase = _HTTP_PHASES.get(prefix)
        if http_phase == "send_headers" and outcome == "started":
            self.headers_started = True
        if http_phase == "receive_body" and outcome == "complete":
            # Transport body completion is a lifecycle fact, not provider-turn success.
            self.emit("http_response_body_complete", headers_received=self.headers_received)
        if http_phase == "response_close" and outcome == "complete":
            # Closing an HTTP response is distinct from consuming a provider turn.
            self.emit("http_response_closed", headers_received=self.headers_received)
        if outcome == "failed" and (phase or http_phase) and not self._failure_recorded:
            self._failure_recorded = True
            event, error_kind = self._failure_event(info)
            self.emit(event, phase=phase or http_phase,
                      headers_received=self.headers_received, error_kind=error_kind)
        if self.previous is not None:
            # Keep the existing httpcore async callback contract and behavior.
            await self.previous(name, info)
