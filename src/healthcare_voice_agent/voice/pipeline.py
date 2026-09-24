"""Provider-neutral assembly of the smallest conversational voice pipeline.

The worker owns processor cleanup after startup. The construction-failure path
below releases only resources that have not yet been handed to the worker.
Provider SDKs and database construction remain outside this assembly module.
An optional, injected front-desk adapter adds only synthetic-demo processors.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy

from healthcare_voice_agent.config import AppConfig, ConfigurationError
from healthcare_voice_agent.voice import providers
from healthcare_voice_agent.voice.observers import make_observers
from healthcare_voice_agent.voice.rtvi import SafeRTVIProcessor
from healthcare_voice_agent.voice.readiness import ClientReadyAudioGate
from healthcare_voice_agent.voice.reply_tts import FullReplyTTSBuffer
from healthcare_voice_agent.voice.tracing import SessionTrace
from healthcare_voice_agent.voice.turns import SpeechGenerationTurnStopStrategy


@dataclass
class VoicePipeline:
    pipeline: Pipeline
    worker: PipelineWorker
    context: LLMContext
    audio_gate: ClientReadyAudioGate

    async def cleanup_unstarted(self) -> None:
        """Only for construction/registration failure before runner.run()."""
        await self.pipeline.cleanup()
        await self.worker.rtvi.cleanup()


async def build_pipeline(config: AppConfig, transport, trace: SessionTrace,
                         *, prompt: str, frontdesk_adapter=None, clinician_adapter=None) -> VoicePipeline:
    config.require_live_ready()
    if (config.agent_mode == "frontdesk_demo") != (frontdesk_adapter is not None):
        raise ConfigurationError("The front-desk profile requires its isolated controller adapter.")
    if (config.agent_mode == "clinician") != (clinician_adapter is not None):
        raise ConfigurationError("The clinician profile requires its confirmed-case adapter.")
    flow_adapter = frontdesk_adapter or clinician_adapter
    if os.environ.get("PIPECAT_SMART_TURN_LOG_DATA", "").lower() == "true":
        raise ConfigurationError(
            "Unset PIPECAT_SMART_TURN_LOG_DATA: implicit raw-audio recording is disabled in this demo."
        )

    processors = list(frontdesk_adapter.processors) if frontdesk_adapter is not None else []
    unowned_turn_analyzer = None
    try:
        stt = providers.build_stt(config)
        processors.append(stt)
        llm = providers.build_llm(config, system_instruction=prompt, trace=trace)
        processors.append(llm)
        tts = providers.build_tts(config, trace=trace)
        processors.append(tts)

        # Both assets ship inside the pinned Pipecat package. No model hub call.
        vad = SileroVADAnalyzer(sample_rate=config.voice.input_sample_rate)
        unowned_turn_analyzer = LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams(stop_secs=config.voice.smart_turn_stop_seconds),
        )
        context = LLMContext()  # Never shared across connections.
        # Smart Turn finalizes user turns directly; no LLM completion verdict.
        pair = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=vad,
                user_turn_stop_timeout=5.0,
                user_turn_strategies=UserTurnStrategies(
                    # Late interim deltas (even a space) otherwise create a
                    # ghost turn and cancel an answer without a new final turn.
                    # Real VAD still permits immediate caller barge-in.
                    start=[VADUserTurnStartStrategy(),
                           TranscriptionUserTurnStartStrategy(use_interim=False)]
                          if flow_adapter is not None else None,
                    stop=[
                    SpeechGenerationTurnStopStrategy(
                        turn_analyzer=unowned_turn_analyzer,
                        finalize_transcripts=config.stt.finalize_transcripts,
                    ),
                ]),
            ),
        )
        processors.extend([pair.user(), pair.assistant()])
        unowned_turn_analyzer = None  # The user aggregator now owns its cleanup.

        delivery = getattr(clinician_adapter, "delivery", None)
        if delivery is not None:
            delivery.bind(llm, tts, transport.output(), context)
            processors.append(delivery)

            @pair.user().event_handler("on_user_turn_started")
            async def background_user_started(aggregator, strategy):
                delivery.user_started()

            @pair.user().event_handler("on_user_turn_stopped")
            async def background_user_stopped(aggregator, strategy, message):
                delivery.user_stopped()

            @llm.event_handler("on_function_calls_cancelled")
            async def background_calls_cancelled(service, calls):
                for call in calls:
                    trace.emit("background_call_cancelled", tool_name=call.function_name,
                               tool_call_id=call.tool_call_id)
                    delivery.finish(call.tool_call_id)
                delivery.check_ready()

        @pair.user().event_handler("on_user_turn_message_added")
        async def user_message_added(aggregator, message):
            if delivery is not None:
                delivery.user_committed()
            trace.emit("user_turn_committed", text=message.content,
                       transcript_timestamp=message.timestamp)

        @pair.assistant().event_handler("on_assistant_turn_stopped")
        async def assistant_turn_stopped(aggregator, message):
            # Output-linked context, NOT proof of what a human actually heard.
            trace.emit("assistant_context_turn", text=message.content,
                       interrupted=message.interrupted, transcript_timestamp=message.timestamp)

        @tts.event_handler("on_tts_request")
        async def tts_request(service, context_id, text):
            trace.emit("tts_input", context_id=context_id, text=text)

        audio_gate = ClientReadyAudioGate()
        processors.append(audio_gate)
        if flow_adapter is None:
            stages = [transport.input(), audio_gate, stt, pair.user(), llm, tts,
                      transport.output(), pair.assistant()]
        elif frontdesk_adapter is not None:
            reply_buffer = FullReplyTTSBuffer()
            processors.append(reply_buffer)
            stages = [transport.input(), audio_gate, stt, pair.user(),
                      frontdesk_adapter.input_gate, llm, reply_buffer, tts,
                      transport.output(), pair.assistant()]
        else:
            reply_buffer = FullReplyTTSBuffer()
            processors.append(reply_buffer)
            stages = [transport.input(), audio_gate, stt, pair.user(),
                      *([delivery] if delivery is not None else []), llm, reply_buffer,
                      tts, transport.output(), pair.assistant()]
        pipeline = Pipeline(stages)
        rtvi = SafeRTVIProcessor()
        processors.append(rtvi)
        voice = config.voice
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=voice.input_sample_rate,
                audio_out_sample_rate=voice.output_sample_rate,
                enable_metrics=True, enable_usage_metrics=True,
            ),
            observers=make_observers(trace) + ([delivery.observer] if delivery is not None else []),
            enable_rtvi=True,
            rtvi_processor=rtvi,
            rtvi_observer_params=RTVIObserverParams(
                system_logs_enabled=False,
                bot_llm_enabled=True,
            ),
            conversation_id=trace.session_id,
            idle_timeout_secs=voice.idle_timeout_seconds,
            setup_timeout_secs=voice.setup_timeout_seconds,
            start_timeout_secs=voice.start_timeout_seconds,
            cancel_timeout_secs=voice.cancel_timeout_seconds,
            processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
        )
        if flow_adapter is not None:
            flow_adapter.bind(llm, pair, worker, transport)

        @rtvi.event_handler("on_client_ready")
        async def client_ready(processor):
            # The worker also handles this event to send the normal bot-ready ACK.
            audio_gate.mark_ready()
            trace.emit("client_ready")
            if flow_adapter is not None:
                await flow_adapter.on_client_ready()

        if flow_adapter is not None:
            @worker.event_handler("on_pipeline_started")
            async def frontdesk_started(worker, frame):
                await flow_adapter.on_pipeline_started()

        return VoicePipeline(pipeline=pipeline, worker=worker, context=context, audio_gate=audio_gate)
    except BaseException:
        # Some clients may already exist when a later constructor fails.
        results = await asyncio.gather(
            *(processor.cleanup() for processor in processors), return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                trace.emit("construction_cleanup_error", exception_type=type(result).__name__)
        if unowned_turn_analyzer is not None:
            await unowned_turn_analyzer.cleanup()
        raise
