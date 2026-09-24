"""Selective Pipecat observers for the lightweight session trace."""

from __future__ import annotations

from collections import deque
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed, ProcessorSetUp
from pipecat.observers.error_observer import ErrorObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver

from .tracing import SessionTrace


_TRACE_FRAME_TYPES = (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    LLMTextFrame,
    TTSTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    MetricsFrame,
)


class _SelectiveTraceObserver(BaseObserver):
    """Trace only small text/control/metrics frames, once per frame ID."""

    def __init__(self, trace: SessionTrace, *, max_frames: int = 512):
        super().__init__()
        self._trace = trace
        self._seen: set[int] = set()
        self._history: deque[int] = deque(maxlen=max_frames)

    def _first_seen(self, frame_id: int) -> bool:
        if frame_id in self._seen:
            return False
        if len(self._history) == self._history.maxlen:
            self._seen.discard(self._history[0])
        self._history.append(frame_id)
        self._seen.add(frame_id)
        return True

    @staticmethod
    def _base(data: FramePushed) -> dict[str, Any]:
        return {"source": data.source.name, "frame_id": data.frame.id}

    async def on_processor_setup(self, data: ProcessorSetUp):
        self._trace.emit(
            "processor_setup",
            processor=data.processor.name,
            setup_seconds=round((data.finished_at_ns - data.started_at_ns) / 1_000_000_000, 6),
        )

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        # Filter first: raw audio and other unselected frames never consume
        # bounded dedup capacity or have their payloads inspected.
        if not isinstance(frame, _TRACE_FRAME_TYPES) or not self._first_seen(frame.id):
            return
        fields = self._base(data)
        if isinstance(frame, InterimTranscriptionFrame):
            self._trace.emit("stt_interim", **fields, text=frame.text)
        elif isinstance(frame, TranscriptionFrame):
            self._trace.emit(
                "stt_final", **fields, text=frame.text, finalized=frame.finalized,
                speech_generation=frame.metadata.get("stt_speech_generation"),
                decode_path=frame.metadata.get("stt_decode_path"),
                commit_sequences=frame.metadata.get("stt_commit_sequences"),
                item_ids=frame.metadata.get("stt_item_ids"),
                completion_received_at=frame.metadata.get("stt_completion_received_at"),
            )
        elif isinstance(frame, LLMTextFrame):
            self._trace.emit("assistant_generated_text", **fields, text=frame.text)
        elif isinstance(frame, TTSTextFrame):
            self._trace.emit(
                "tts_output_text", **fields, text=frame.text, context_id=frame.context_id
            )
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._trace.emit("llm_response_start", **fields)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._trace.emit("llm_response_end", **fields)
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._trace.emit("user_started_speaking", **fields)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._trace.emit("user_stopped_speaking", **fields)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._trace.emit("bot_started_speaking", **fields)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._trace.emit("bot_stopped_speaking", **fields)
        elif isinstance(frame, InterruptionFrame):
            self._trace.emit("interruption", **fields)
        elif isinstance(frame, MetricsFrame):
            self._trace.emit(
                "metrics",
                **fields,
                metrics=[
                    {"metric_type": type(metric).__name__, **metric.model_dump(mode="json")}
                    for metric in frame.data
                ],
            )


def make_observers(trace: SessionTrace) -> list[BaseObserver]:
    """Create observability hooks without recording raw audio or full contexts."""
    selective = _SelectiveTraceObserver(trace)
    errors = ErrorObserver()
    latency = UserBotLatencyObserver()

    @errors.event_handler("on_error")
    async def on_error(_observer, event) -> None:
        # ErrorObserver supplies a safe summary. Never serialize the exception.
        trace.emit(
            "pipeline_error",
            message=event.message,
            category=event.category.value,
            exception_type=event.exception_type,
            processor=event.processor,
            processor_usable=event.processor_usable,
        )

    @latency.event_handler("on_latency_measured")
    async def on_latency_measured(_observer, latency_seconds: float) -> None:
        trace.emit(
            "user_to_first_bot_speech_latency",
            seconds=float(latency_seconds),
            measurement="server-observed, not browser-heard",
        )

    @latency.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech_latency(_observer, latency_seconds: float) -> None:
        trace.emit(
            "first_bot_speech_latency",
            seconds=float(latency_seconds),
            measurement="server-observed, not browser-heard",
        )

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(_observer, breakdown) -> None:
        trace.emit("latency_breakdown", lines=list(breakdown.turn_contribution_lines()))

    return [selective, errors, latency]
