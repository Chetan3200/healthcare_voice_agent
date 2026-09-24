"""One connection, one context, one worker runner, and one private trace file."""

from __future__ import annotations

import asyncio
import hashlib
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

from loguru import logger
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import LLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from healthcare_voice_agent.config import write_run_config
from healthcare_voice_agent.voice.pipeline import build_pipeline
from healthcare_voice_agent.voice.flow_fingerprints import flow_prompt_fingerprint
from healthcare_voice_agent.voice.tracing import SessionTrace


async def run_session(runner_args: SmallWebRTCRunnerArguments) -> None:
    """Protect the negotiated peer even if directories/logs cannot be created.

    Browser request bodies do not select models, prompts, keys, or run paths.
    Only the frozen, server-side CLI configuration is used.
    """
    if not isinstance(runner_args, SmallWebRTCRunnerArguments) or runner_args.cli_args is None:
        raise ValueError("Use the local WebRTC development runner for this demo.")
    trace = None
    reason = "session_error"
    try:
        config = runner_args.cli_args.app_config
        config.require_live_ready()
        session_id = uuid4().hex  # Never use a client-supplied ID as a file path.
        directory = Path(runner_args.cli_args.run_root) / "sessions" / session_id
        directory.mkdir(parents=True, mode=0o700)
        trace = SessionTrace(directory / "events.jsonl", session_id=session_id,
                             secrets=config.redaction_secrets())
        trace.emit("session_created", clinic_tools_enabled=config.agent_mode == "clinician",
                   frontdesk_tools_enabled=config.agent_mode == "frontdesk_demo")
        reason = await _run_pipeline_session(runner_args, config, directory, trace)
    except asyncio.CancelledError:
        reason = "session_cancelled"
        raise
    except Exception as exc:
        if trace is not None:
            try:
                trace.emit("session_error", exception_type=type(exc).__name__, message=str(exc))
            except Exception:
                logger.error("Could not write the session error to its trace.")
        # No unredacted traceback through Starlette, including pre-trace errors.
        logger.error("Voice session failed ({})", type(exc).__name__)
    finally:
        try:
            # Idempotent after normal worker cleanup; also covers early failure.
            await runner_args.webrtc_connection.disconnect()
        except Exception as exc:
            logger.error("Peer cleanup failed ({})", type(exc).__name__)
            reason = "peer_cleanup_error"
        finally:
            if trace is not None:
                try:
                    if reason == "session_cancelled":
                        trace.emit("session_cancelled")
                    trace.emit("session_finished", reason=reason)
                finally:
                    trace.close()


async def _run_pipeline_session(runner_args, config, directory: Path, trace: SessionTrace) -> str:
    bundle = None
    demo_runtime = None
    frontdesk_adapter = None
    clinician_runtime = clinician_adapter = None
    runner_task = None
    deadline_task = None
    stop_reason = None
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)

    async def request_stop(reason: str) -> None:
        nonlocal stop_reason
        stop_reason = stop_reason or reason
        if demo_runtime is not None:
            demo_runtime.interrupt()
        if frontdesk_adapter is not None:
            frontdesk_adapter.abort_output()
        if clinician_adapter is not None:
            clinician_adapter.abort_output()
        await runner.cancel(reason=stop_reason)

    @runner.event_handler("on_ready")
    async def runner_ready(worker_runner):
        # run() clears its shutdown event. Preserve a disconnect/error that
        # arrived while workers were being registered, before run() started.
        if stop_reason:
            await runner.cancel(reason=stop_reason)

    @runner.event_handler("on_error")
    async def runner_error(worker_runner, *details):
        await request_stop("runner_error")
        trace.emit("runner_error")  # No arbitrary exception/request objects.

    async def expire_session() -> None:
        await asyncio.sleep(config.voice.max_session_seconds)
        await request_stop("session_limit")
        trace.emit("session_limit", seconds=config.voice.max_session_seconds)

    try:
        write_run_config(config, directory)
        if config.agent_mode == "frontdesk_demo":
            from healthcare_voice_agent.demo.runtime import open_demo_runtime
            from healthcare_voice_agent.voice.frontdesk_demo import FrontDeskVoiceAdapter
            demo_runtime = await open_demo_runtime(trace)
            frontdesk_adapter = FrontDeskVoiceAdapter(demo_runtime, trace)
            prompt = "You are an English-speaking synthetic appointment receptionist. Follow the active Pipecat flow."
        elif config.agent_mode == "clinician":
            from healthcare_voice_agent.clinician.runtime import open_clinician_runtime
            from healthcare_voice_agent.voice.clinician import ClinicianVoiceAdapter
            clinician_runtime = await open_clinician_runtime(config, trace)
            clinician_adapter = ClinicianVoiceAdapter(clinician_runtime)
            prompt = "You assist a clinician with synthetic records. Follow the active Pipecat flow."
        else:
            prompt = files("healthcare_voice_agent.agent").joinpath("prompts.md").read_text("utf-8")
            (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        trace.emit("prompt_loaded", sha256=hashlib.sha256(prompt.encode()).hexdigest())
        fingerprint = flow_prompt_fingerprint(config.agent_mode)
        if fingerprint is not None:
            trace.emit("flow_prompt_loaded", **fingerprint)
        transport = await create_transport(runner_args, {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True, audio_out_enabled=True,
                audio_in_sample_rate=config.voice.input_sample_rate,
                audio_out_sample_rate=config.voice.output_sample_rate,
            ),
        })
        pipeline_options = {"prompt": prompt}
        if frontdesk_adapter is not None:
            pipeline_options["frontdesk_adapter"] = frontdesk_adapter
        if clinician_adapter is not None:
            pipeline_options["clinician_adapter"] = clinician_adapter
        bundle = await build_pipeline(config, transport, trace, **pipeline_options)

        @transport.event_handler("on_client_connected")
        async def connected(transport, client):
            # Front-desk greeting waits for both client-ready and pipeline-start
            # inside build_pipeline; transport connection alone is not enough.
            trace.emit("client_connected")

        @transport.event_handler("on_client_disconnected")
        async def disconnected(transport, client):
            await request_stop("client_disconnected")
            trace.emit("client_disconnected")

        @bundle.worker.event_handler("on_pipeline_started")
        async def pipeline_started(worker, frame):
            trace.emit("pipeline_started")

        @bundle.worker.event_handler("on_pipeline_finished")
        async def pipeline_finished(worker, frame):
            trace.emit("pipeline_finished", final_frame=type(frame).__name__)

        @bundle.worker.event_handler("on_pipeline_error")
        async def pipeline_error(worker, frame):
            # A failed clinician LLM request need not end the connection.
            # Any eligible SDK retry is handled inside the provider, which queues
            # the fixed spoken fallback. Keep the native browser error notification;
            # never add a second retry layer or replay completed tool calls.
            if (config.agent_mode == "clinician" and not frame.fatal
                    and isinstance(frame.processor, LLMService) and frame.processor.is_usable):
                trace.emit("recoverable_llm_error", action="keep_session_open")
                return
            await request_stop("service_error")

        @bundle.worker.event_handler("on_pipeline_timeout")
        async def pipeline_timeout(worker, frame):
            await request_stop("pipeline_timeout")
            trace.emit("pipeline_timeout", frame_type=type(frame).__name__)

        @bundle.worker.event_handler("on_idle_timeout")
        async def idle_timeout(worker):
            await request_stop("idle_timeout")
            trace.emit("idle_timeout", seconds=config.voice.idle_timeout_seconds)

        await runner.add_workers(bundle.worker)
        if config.voice.max_session_seconds is not None:
            deadline_task = asyncio.create_task(expire_session(), name=f"deadline-{trace.session_id}")
        runner_task = asyncio.create_task(runner.run(), name=f"voice-{trace.session_id}")
        # ASGI cancellation must not abandon an in-flight provider session.
        try:
            await asyncio.shield(runner_task)
        except asyncio.CancelledError:
            await request_stop("session_cancelled")
            await runner_task
            raise
    finally:
        try:
            if deadline_task is not None:
                deadline_task.cancel()
                await asyncio.gather(deadline_task, return_exceptions=True)
            if bundle is not None and runner_task is None:
                await bundle.cleanup_unstarted()
        finally:
            try:
                if frontdesk_adapter is not None:
                    await frontdesk_adapter.close()
            finally:
                if demo_runtime is not None:
                    await demo_runtime.close()
                if clinician_adapter is not None:
                    await clinician_adapter.close()
                if clinician_runtime is not None:
                    await clinician_runtime.close()
    return stop_reason or "completed"
