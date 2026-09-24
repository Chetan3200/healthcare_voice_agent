"""Deterministic diagnostics checks, without HTTP clients or network access."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from healthcare_voice_agent.voice.http_timing import HTTPConnectionTiming
from healthcare_voice_agent.voice.tracing import SessionTrace


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class Sink:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append({"event": event, **fields})


def request():
    return SimpleNamespace(extensions={})


def response(req):
    return SimpleNamespace(request=req, status_code=200)


def test_exact_phase_timings_and_contents_allowlist(tmp_path):
    async def scenario():
        clock = Clock()
        target = tmp_path / "http.jsonl"
        sink = SessionTrace(target, session_id="offline")
        timing = HTTPConnectionTiming(sink, stage="llm", clock=clock)
        req = request()
        try:
            await timing.on_request(req)
            callback = req.extensions["trace"]
            private_info = {
                "host": "private-host-marker", "headers": {"authorization": "private-header-marker"},
                "body": "private-body-marker", "exception": RuntimeError("private-error-marker"),
            }
            await callback("connection.connect_tcp.started", private_info)
            clock.value = 0.2
            await callback("connection.connect_tcp.complete", private_info)
            await callback("connection.start_tls.started", private_info)
            clock.value = 0.5
            await callback("connection.start_tls.complete", private_info)
            await callback("http11.send_request_headers.started", private_info)
            await callback("untrusted-name-private-marker.complete", private_info)
            clock.value = 0.8
            await timing.on_response(response(req))
        finally:
            sink.close()
        text = target.read_text()
        assert "private-" not in text
        events = [json.loads(line) for line in text.splitlines()]
        assert [e["event"] for e in events] == [
            "http_request_start", "http_connection_phase", "http_connection_phase",
            "http_response_headers",
        ]
        assert events[1]["data"]["duration_seconds"] == 0.2
        assert events[2]["data"]["duration_seconds"] == 0.3
        assert events[-1]["data"] == {
            "stage": "llm", "request_id": "llm-1", "status_code": 200,
            "response_headers_seconds": 0.8, "connection_reuse": "new",
            "tcp_connect_seconds": 0.2, "tls_handshake_seconds": 0.3,
        }
    asyncio.run(scenario())


@pytest.mark.parametrize("header_event,expected", [
    ("http11.send_request_headers.started", "reused"),
    ("http2.send_request_headers.started", "reused"),
    (None, "unknown"),
])
def test_no_connect_event_requires_transport_evidence_before_claiming_reuse(header_event, expected):
    async def scenario():
        sink = Sink()
        timing = HTTPConnectionTiming(sink, stage="tts")
        req = request()
        await timing.on_request(req)
        if header_event:
            await req.extensions["trace"](header_event, {})
        await timing.on_response(response(req))
        assert sink.events[-1]["connection_reuse"] == expected
        if expected == "unknown":
            assert sink.events[-1]["tcp_connect_seconds"] is None
            assert sink.events[-1]["tls_handshake_seconds"] is None
    asyncio.run(scenario())


def test_failure_is_recorded_once_without_exception_contents():
    async def scenario():
        sink = Sink()
        clock = Clock()
        timing = HTTPConnectionTiming(sink, stage="tts", clock=clock)
        req = request()
        await timing.on_request(req)
        callback = req.extensions["trace"]
        await callback("connection.connect_tcp.started", {})
        clock.value = 0.25
        info = {"exception": RuntimeError("must-not-appear")}
        await callback("connection.connect_tcp.failed", info)
        await callback("http11.send_request_headers.failed", info)
        failures = [e for e in sink.events if e["event"] == "http_request_failed"]
        assert failures == [{"event": "http_request_failed", "stage": "tts",
                             "request_id": "tts-1", "phase": "tcp_connect",
                             "headers_received": False, "error_kind": "unknown"}]
        assert "must-not-appear" not in json.dumps(sink.events)
    asyncio.run(scenario())


def test_cancel_before_headers_is_not_recorded_as_transport_failure():
    async def scenario():
        sink = Sink()
        timing = HTTPConnectionTiming(sink, stage="llm")
        req = request()
        await timing.on_request(req)
        await req.extensions["trace"](
            "http11.receive_response_headers.failed",
            {"exception": asyncio.CancelledError()},
        )
        terminal = [event for event in sink.events if event["event"].startswith("http_request_")]
        assert terminal[-1] == {
            "event": "http_request_cancelled", "stage": "llm", "request_id": "llm-1",
            "phase": "receive_headers", "headers_received": False, "error_kind": "cancelled",
        }
    asyncio.run(scenario())


def test_generator_close_after_headers_is_not_recorded_as_transport_failure():
    async def scenario():
        sink = Sink()
        timing = HTTPConnectionTiming(sink, stage="llm")
        req = request()
        await timing.on_request(req)
        await timing.on_response(response(req))
        await req.extensions["trace"](
            "http11.receive_response_body.failed", {"exception": GeneratorExit()},
        )
        assert sink.events[-1] == {
            "event": "http_stream_closed", "stage": "llm", "request_id": "llm-1",
            "phase": "receive_body", "headers_received": True, "error_kind": "generator_exit",
        }
    asyncio.run(scenario())


def test_timeout_after_headers_remains_a_transport_failure():
    async def scenario():
        sink = Sink()
        timing = HTTPConnectionTiming(sink, stage="tts")
        req = request()
        await timing.on_request(req)
        await timing.on_response(response(req))
        await req.extensions["trace"](
            "http11.receive_response_body.failed", {"exception": asyncio.TimeoutError()},
        )
        assert sink.events[-1] == {
            "event": "http_request_failed", "stage": "tts", "request_id": "tts-1",
            "phase": "receive_body", "headers_received": True, "error_kind": "timeout",
        }
    asyncio.run(scenario())


def test_unknown_exception_is_safe_and_does_not_call_string_conversion():
    class ExplosiveUnknown(Exception):
        def __str__(self):
            raise AssertionError("diagnostics must not stringify exceptions")

        def __repr__(self):
            raise AssertionError("diagnostics must not repr exceptions")

    async def scenario():
        sink = Sink()
        timing = HTTPConnectionTiming(sink, stage="tts")
        req = request()
        await timing.on_request(req)
        await req.extensions["trace"](
            "http11.receive_response_body.failed", {"exception": ExplosiveUnknown()},
        )
        assert sink.events[-1] == {
            "event": "http_request_failed", "stage": "tts", "request_id": "tts-1",
            "phase": "receive_body", "headers_received": False, "error_kind": "unknown",
        }
    asyncio.run(scenario())


def test_body_and_response_close_complete_are_lifecycle_events_not_request_success():
    async def scenario():
        sink = Sink()
        timing = HTTPConnectionTiming(sink, stage="llm")
        req = request()
        await timing.on_request(req)
        await timing.on_response(response(req))
        callback = req.extensions["trace"]
        await callback("http11.receive_response_body.complete", {})
        await callback("http11.response_closed.complete", {})
        assert [event["event"] for event in sink.events[-2:]] == [
            "http_response_body_complete", "http_response_closed",
        ]
        assert all(event["headers_received"] is True for event in sink.events[-2:])
    asyncio.run(scenario())


def test_concurrent_requests_have_independent_timing_state():
    async def scenario():
        sink = Sink()
        timing = HTTPConnectionTiming(sink, stage="llm")
        first, second = request(), request()
        await timing.on_request(first)
        await timing.on_request(second)
        await first.extensions["trace"]("connection.connect_tcp.started", {})
        await second.extensions["trace"]("http11.send_request_headers.started", {})
        await timing.on_response(response(second))
        await timing.on_response(response(first))
        records = [e for e in sink.events if e["event"] == "http_response_headers"]
        assert [(r["request_id"], r["connection_reuse"]) for r in records] == [
            ("llm-2", "reused"), ("llm-1", "new"),
        ]
    asyncio.run(scenario())


def test_existing_callback_preserved_without_chaining_stale_request_state():
    async def scenario():
        sink = Sink()
        timing = HTTPConnectionTiming(sink, stage="llm")
        req = request()
        calls = []
        async def previous(name, info):
            calls.append((name, info))
        req.extensions["trace"] = previous
        info = {"marker": object()}
        for _ in range(2):
            await timing.on_request(req)
            await req.extensions["trace"]("http11.send_request_headers.started", info)
            await timing.on_response(response(req))
        assert calls == [("http11.send_request_headers.started", info)] * 2
        assert [e["request_id"] for e in sink.events if e["event"] == "http_response_headers"] == [
            "llm-1", "llm-2",
        ]
    asyncio.run(scenario())


def test_existing_callback_errors_keep_their_original_behavior():
    async def scenario():
        timing = HTTPConnectionTiming(Sink(), stage="llm")
        req = request()
        async def broken(name, info):
            raise RuntimeError("caller-owned callback failed")
        req.extensions["trace"] = broken
        await timing.on_request(req)
        with pytest.raises(RuntimeError, match="caller-owned"):
            await req.extensions["trace"]("http11.send_request_headers.started", {})
    asyncio.run(scenario())


def test_failed_diagnostic_sink_does_not_break_requests():
    class BrokenSink:
        def emit(self, *args, **kwargs):
            raise OSError("trace sink unavailable")

    async def scenario():
        timing = HTTPConnectionTiming(BrokenSink(), stage="tts")
        req = request()
        await timing.on_request(req)
        await req.extensions["trace"]("http11.send_request_headers.started", {})
        await timing.on_response(response(req))
    asyncio.run(scenario())
