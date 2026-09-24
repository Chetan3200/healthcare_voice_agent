"""Loopback HTTP protocol tests; no model, CUDA, downloads or external APIs."""

import asyncio
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("pipecat")
from aiohttp import web
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    AggregatedTextFrame, CancelFrame, EndFrame, InterruptionFrame, MetricsFrame,
    StartFrame, TTSAudioRawFrame, TTSStoppedFrame,
)
from pipecat.metrics.metrics import TTFAMetricsData, TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.text.base_text_aggregator import AggregationType

from healthcare_voice_agent.voice.breeze_tts import BreezeTTSService

HEADERS = {"Content-Type": "audio/pcm", "X-Sample-Rate": "24000", "X-Sample-Format": "s16le"}


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch):
    original = socket.socket.connect
    original_ex = socket.socket.connect_ex
    def guard(method):
        def connect(self, address):
            if not isinstance(address, tuple) or address[0] != "127.0.0.1":
                raise AssertionError("Breeze tests allow only loopback HTTP")
            return method(self, address)
        return connect
    monkeypatch.setattr(socket.socket, "connect", guard(original))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(original_ex))


@asynccontextmanager
async def server(handler):
    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    runner = web.AppRunner(app, shutdown_timeout=.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def observe(service, monkeypatch):
    observed = SimpleNamespace(errors=[], usage=[], cancelled=0, ttfb=0)
    async def error(**kwargs):
        observed.errors.append(kwargs)
    async def usage(text):
        observed.usage.append(text)
    async def cancel():
        observed.cancelled += 1
    async def ttfb():
        observed.ttfb += 1
    monkeypatch.setattr(service, "push_error", error)
    monkeypatch.setattr(service, "start_tts_usage_metrics", usage)
    monkeypatch.setattr(service, "cancel_ttfb_metrics", cancel)
    monkeypatch.setattr(service, "stop_ttfb_metrics", ttfb)
    return observed


async def collect(service, text="A synthetic sentence.", context="test-context"):
    return [frame async for frame in service.run_tts(text, context)]


def test_real_http_multipart_streaming_alignment_and_exact_text(monkeypatch):
    async def scenario():
        requested = []
        release = asyncio.Event()
        text = "Hello,  preserve  these spaces. नमस्ते "
        async def handler(request):
            requested.append((request.content_type, dict(await request.post()),
                              request.headers.get("Authorization")))
            response = web.StreamResponse(headers=HEADERS)
            await response.prepare(request)
            await response.write(b"\x01\x02\x03")
            await release.wait()
            await response.write(b"\x04" + b"\x00\x01" * 3000)
            await response.write_eof()
            return response
        async with server(handler) as url:
            service = BreezeTTSService(base_url=url, api_key="synthetic-test-key")
            observed = observe(service, monkeypatch)
            stream = service.run_tts(text, "actual-context")
            try:
                first = await asyncio.wait_for(anext(stream), 1)
                assert first.audio == b"\x01\x02"  # first audio before full generation
                assert first.context_id == "actual-context"
                release.set()
                frames = [first] + [frame async for frame in stream]
                assert b"".join(f.audio for f in frames) == b"\x01\x02\x03\x04" + b"\x00\x01" * 3000
                assert all(len(f.audio) % 2 == 0 and len(f.audio) <= 4800 for f in frames)
                assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in frames)
                assert requested == [("multipart/form-data", {
                    "text": text, "instruction": "A warm, calm conversational voice.",
                    "cfg_scale": "4", "seed": "42",
                }, "Bearer synthetic-test-key")]
                assert observed.usage == [text]
                assert not observed.errors
                assert observed.ttfb == 0  # native context consumer, not generator, owns TTFB
                assert not service._pending_io and not service._responses
            finally:
                release.set()
                await stream.aclose()
                await service.cleanup()
                assert service._session.closed
    asyncio.run(scenario())


@pytest.mark.parametrize("status,headers,body", [
    (409, HEADERS, b"private backend exception"),
    (500, HEADERS, b"private patient text"),
    (302, {**HEADERS, "Location": "https://example.invalid/"}, b"private redirect"),
    (200, {**HEADERS, "Content-Type": "application/json"}, b"private invalid audio"),
    (200, {**HEADERS, "X-Sample-Rate": "16000"}, b"\x01\x02"),
    (200, {**HEADERS, "X-Sample-Format": "f32le"}, b"\x01\x02"),
    (200, {"Content-Type": "audio/pcm"}, b"\x01\x02"),
    (200, {**HEADERS, "Content-Encoding": "gzip"}, b"\x01\x02"),
    (200, HEADERS, b""),
    (200, HEADERS, b"\x01"),
])
def test_invalid_responses_fail_closed_without_leaking_body(status, headers, body, monkeypatch):
    async def scenario():
        calls = 0
        async def handler(request):
            nonlocal calls
            calls += 1
            return web.Response(status=status, headers=headers, body=body)
        async with server(handler) as url:
            service = BreezeTTSService(base_url=url)
            observed = observe(service, monkeypatch)
            try:
                assert await collect(service) == []
                assert len(observed.errors) == 1
                assert observed.errors[0]["force_treat_as_permanent"] is True
                assert "private" not in repr(observed.errors)
                assert "exception" not in observed.errors[0]
                assert observed.ttfb == 0
                assert await collect(service, "Never retry") == []
                assert calls == 1
                assert service._session.closed
            finally:
                await service.cleanup()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["headers", "body", "disconnect", "oversized", "odd_tail"])
def test_bounded_failures_and_no_retries(mode, monkeypatch):
    async def scenario():
        release = asyncio.Event()
        async def handler(request):
            if mode == "headers":
                await release.wait()
                return web.Response(headers=HEADERS, body=b"\x00\x01")
            response = web.StreamResponse(headers=HEADERS)
            await response.prepare(request)
            if mode == "body":
                await release.wait()
            elif mode == "disconnect":
                request.transport.close()  # incomplete HTTP chunked body
                return response
            elif mode == "oversized":
                await response.write(b"\x00\x01" * 2401)
            else:
                await response.write(b"\x00\x01\x02")
            try:
                await response.write_eof()
            except ConnectionError:
                pass
            return response
        async with server(handler) as url:
            service = BreezeTTSService(base_url=url, inference_timeout_seconds=.08,
                                       max_audio_seconds=.1)
            observed = observe(service, monkeypatch)
            try:
                frames = await asyncio.wait_for(collect(service), 1)
                assert len(observed.errors) == 1
                assert service._closed
                if mode in {"headers", "body"}:
                    assert "timed out" in observed.errors[0]["error_msg"]
                    assert not frames
                assert await collect(service) == []
                assert not service._pending_io and not service._responses
            finally:
                release.set()
                await service.cleanup()
    asyncio.run(scenario())


@asynccontextmanager
async def native_service(monkeypatch, url, **kwargs):
    service = BreezeTTSService(base_url=url, **kwargs)
    manager = TaskManager()
    clock = SystemClock()
    clock.start()
    frames, errors = [], []
    await service.setup(FrameProcessorSetup(
        clock=clock, task_manager=manager, pipeline_worker=SimpleNamespace(app_resources={}),
        enable_metrics=True, enable_usage_metrics=True,
    ))
    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        frames.append(frame)
    async def error(**kwargs):
        errors.append(kwargs)
    monkeypatch.setattr(service, "push_frame", push)
    monkeypatch.setattr(service, "push_error", error)
    await service.start(StartFrame())
    service._turn_context_id = "native-breeze"
    try:
        yield SimpleNamespace(service=service, frames=frames, errors=errors)
    finally:
        await service.cleanup()
        assert not manager.current_tasks()


def audio_metrics(frames):
    return [data for frame in frames if isinstance(frame, MetricsFrame)
            for data in frame.data if isinstance(data, (TTFBMetricsData, TTFAMetricsData))]


def test_native_watchdog_survives_header_wait_without_fake_audio_or_ttfb(monkeypatch):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        async def handler(request):
            entered.set()
            await release.wait()
            return web.Response(headers=HEADERS, body=b"\x00\x01" * 50)
        async with server(handler) as url:
            async with native_service(monkeypatch, url, stop_frame_timeout_s=.04) as c:
                pending = asyncio.create_task(c.service._push_tts_frames(
                    AggregatedTextFrame("Synthetic delayed voice.", AggregationType.SENTENCE),
                ))
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    await asyncio.sleep(.18)
                    assert c.service.audio_context_available("native-breeze")
                    assert not any(isinstance(f, (TTSAudioRawFrame, TTSStoppedFrame)) for f in c.frames)
                    assert not audio_metrics(c.frames)
                    release.set()
                    await asyncio.wait_for(pending, 1)
                    await c.service.on_turn_context_completed()
                    await asyncio.wait_for(c.service._serialization_queue.join(), 1)
                    assert sum(isinstance(f, TTSAudioRawFrame) for f in c.frames) == 1
                    ttfb = [m for m in audio_metrics(c.frames) if isinstance(m, TTFBMetricsData)]
                    assert len(ttfb) == 1 and ttfb[0].value >= .18
                    assert not c.errors
                finally:
                    release.set()
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize("shutdown", ["interrupt", "cancel", "end", "cleanup", "cancel_waiter"])
def test_pending_http_is_cancelled_and_late_audio_never_emitted(shutdown, monkeypatch):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0
        async def handler(request):
            nonlocal calls
            calls += 1
            entered.set()
            if calls == 1:
                await release.wait()
            return web.Response(headers=HEADERS, body=b"\x00\x01" * 10)
        async with server(handler) as url:
            async with native_service(monkeypatch, url, stop_frame_timeout_s=.03) as c:
                pending = asyncio.create_task(c.service._push_tts_frames(
                    AggregatedTextFrame("Synthetic cancelled voice.", AggregationType.SENTENCE),
                ))
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    await asyncio.sleep(.07)
                    if shutdown == "interrupt":
                        await c.service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
                    elif shutdown == "cancel":
                        await c.service.cancel(CancelFrame())
                    elif shutdown == "end":
                        await c.service.stop(EndFrame())
                    elif shutdown == "cleanup":
                        await c.service.cleanup()
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                    release.set()
                    await asyncio.sleep(.03)
                    assert not any(isinstance(f, TTSAudioRawFrame) for f in c.frames)
                    assert not audio_metrics(c.frames)
                    assert not c.errors
                    assert not c.service._pending_io and not c.service._responses
                    if shutdown == "interrupt":
                        assert not c.service._closed
                        assert len(await collect(c.service, context="new-context")) == 1
                    elif shutdown != "cancel_waiter":
                        assert c.service._closed and c.service._session.closed
                finally:
                    release.set()
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize("kwargs", [
    {"base_url": "http://example.com"}, {"base_url": "https://user:password@example.com"},
    {"base_url": "https://example.com?token=private"}, {"audio_chunk_ms": 1},
    {"inference_timeout_seconds": float("inf")}, {"cfg_scale": 0}, {"seed": -1},
    {"instructions": ""}, {"max_audio_seconds": 0}, {"max_text_chars": 0},
])
def test_invalid_configuration_is_rejected_without_http(kwargs):
    options = {"base_url": "http://127.0.0.1:7861", **kwargs}
    with pytest.raises(ValueError):
        BreezeTTSService(**options)


def test_construction_and_oversized_text_do_not_contact_server(monkeypatch):
    async def scenario():
        service = BreezeTTSService(base_url="http://127.0.0.1:1", max_text_chars=4)
        observed = observe(service, monkeypatch)
        assert service._session is None
        try:
            assert await collect(service, "too long") == []
            assert len(observed.errors) == 1
            assert service._session is None
        finally:
            await service.cleanup()
    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_real_queue_interruption_drains_client_before_next_post(phase, monkeypatch):
    """Use actual Pipecat input/process tasks, not manual run_tts cancellation.

    Delayed cancellation cleanup is an injected CLIENT task barrier. The old
    fake server handler deliberately remains active when the next POST arrives:
    client task completion says nothing about upstream GPU completion/readiness.
    """
    async def scenario():
        old_server_entered = asyncio.Event()
        old_io_entered = asyncio.Event()
        cancellation_entered = asyncio.Event()
        release_client_cleanup = asyncio.Event()
        release_server = asyncio.Event()
        old_client_finished = asyncio.Event()
        old_generator_finished = asyncio.Event()
        interruption_finished = asyncio.Event()
        new_frame_finished = asyncio.Event()
        starts = []

        async def handler(request):
            fields = await request.post()
            starts.append(fields["text"])
            if len(starts) == 1:
                response = web.StreamResponse(headers=HEADERS)
                if phase == "body":
                    await response.prepare(request)
                old_server_entered.set()
                await release_server.wait()
                if phase == "headers":
                    return web.Response(headers=HEADERS, body=b"\x01\x00")
                try:
                    await response.write_eof()
                except ConnectionError:
                    pass
                return response
            assert old_client_finished.is_set()
            assert old_generator_finished.is_set()
            # Deliberately demonstrate that waiting for client cleanup has NOT
            # waited for the previous server job. Real Breeze could return 409.
            assert not release_server.is_set()
            return web.Response(headers=HEADERS, body=b"\x02\x00" * 10)

        async with server(handler) as url:
            async with native_service(monkeypatch, url) as c:
                original_io = c.service._await_io
                original_run = c.service.run_tts
                old_io_count = 0

                async def monitored_io(awaitable, context_id, epoch, deadline):
                    nonlocal old_io_count
                    is_old = context_id == "native-breeze"
                    if is_old:
                        old_io_count += 1
                    monitored_wait = is_old and old_io_count == (1 if phase == "headers" else 2)
                    if not monitored_wait:
                        return await original_io(awaitable, context_id, epoch, deadline)

                    async def delayed_cancellation():
                        old_io_entered.set()
                        try:
                            return await awaitable
                        except asyncio.CancelledError:
                            cancellation_entered.set()
                            # _invalidate and native process cancellation can
                            # both cancel this child. Hold cleanup deterministically
                            # until released, while still ultimately propagating it.
                            while not release_client_cleanup.is_set():
                                try:
                                    await release_client_cleanup.wait()
                                except asyncio.CancelledError:
                                    continue
                            old_client_finished.set()
                            raise
                    return await original_io(delayed_cancellation(), context_id, epoch, deadline)

                async def monitored_run(text, context_id):
                    try:
                        async for frame in original_run(text, context_id):
                            yield frame
                    finally:
                        if text == "Old synthetic turn.":
                            old_generator_finished.set()

                monkeypatch.setattr(c.service, "_await_io", monitored_io)
                monkeypatch.setattr(c.service, "run_tts", monitored_run)
                started = asyncio.Event()
                async def started_callback(*args):
                    started.set()
                async def interrupted_callback(*args):
                    interruption_finished.set()
                async def new_callback(*args):
                    new_frame_finished.set()
                try:
                    # native_service starts the TTS consumer directly. A queued
                    # StartFrame also creates the real processor input/process
                    # tasks, whose interruption ownership is under regression.
                    await c.service.queue_frame(StartFrame(), callback=started_callback)
                    await asyncio.wait_for(started.wait(), 1)
                    await c.service.queue_frame(AggregatedTextFrame(
                        "Old synthetic turn.", AggregationType.SENTENCE,
                    ))
                    await asyncio.wait_for(old_server_entered.wait(), 1)
                    await asyncio.wait_for(old_io_entered.wait(), 1)
                    await c.service.queue_frame(InterruptionFrame(), callback=interrupted_callback)
                    await c.service.queue_frame(AggregatedTextFrame(
                        "Fresh synthetic turn.", AggregationType.SENTENCE,
                    ), callback=new_callback)
                    await asyncio.wait_for(cancellation_entered.wait(), 1)
                    await asyncio.sleep(.03)
                    assert starts == ["Old synthetic turn."]
                    assert not old_client_finished.is_set()
                    assert not old_generator_finished.is_set()
                    assert not interruption_finished.is_set()
                    assert not new_frame_finished.is_set()
                    assert not any(isinstance(f, TTSAudioRawFrame) for f in c.frames)

                    release_client_cleanup.set()
                    await asyncio.wait_for(interruption_finished.wait(), 1)
                    await asyncio.wait_for(new_frame_finished.wait(), 1)
                    assert old_client_finished.is_set()
                    assert old_generator_finished.is_set()
                    assert starts == ["Old synthetic turn.", "Fresh synthetic turn."]
                    assert not c.service._pending_io and not c.service._responses
                    assert not c.errors
                    audio = [f.audio for f in c.frames if isinstance(f, TTSAudioRawFrame)]
                    assert b"".join(audio) == b"\x02\x00" * 10
                finally:
                    release_client_cleanup.set()
                    release_server.set()
    asyncio.run(scenario())
