"""One pending-result gate around Pipecat's native async tool execution."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from pipecat.frames.frames import (
    CancelFrame, DataFrame, EndFrame, FunctionCallsStartedFrame, InterruptionFrame,
    LLMContextFrame, LLMMessagesAppendFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame,
    TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from healthcare_voice_agent.clinician.flow import ASYNC_READ_TOOLS
from healthcare_voice_agent.voice.reply_tts import LLMErrorFallbackFrame

PROCESSING_NOTICE = "Can I help you with anything else while the request is being processed?"


@dataclass
class _Check(DataFrame):
    pass


@dataclass
class _Call:
    group: str
    name: str
    request: str | list
    arguments: dict
    result: dict | None = None
    notice_due: bool = False
    notice_sent: bool = False
    timer: asyncio.Task | None = None


class _ActivityObserver(BaseObserver):
    def __init__(self, delivery):
        super().__init__()
        self.delivery = delivery

    async def on_push_frame(self, data):
        self.delivery.observe(data.source, data.frame)


class DeferredToolDelivery(FrameProcessor):
    def __init__(self, session, *, notice_seconds=3.0):
        super().__init__()
        self.session, self.notice_seconds = session, notice_seconds
        self.observer = _ActivityObserver(self)
        self.closed = self.user_pending = self.llm_busy = self.speech_busy = False
        self._wait_for_user = self._interrupted = self._check_queued = False
        self._audio_context = None
        self._calls: dict[str, _Call] = {}
        self._foreground: set[str] = set()
        self._reply: tuple[str, ...] = ()
        self.llm = self.tts = self.output = self.context = None

    def bind(self, llm, tts, output, context):
        self.llm, self.tts, self.output, self.context = llm, tts, output, context
        llm._clinician_delivery = self

    def _trace(self, event, **fields):
        if self.session.trace is not None:
            self.session.trace.emit(event, **fields)

    def begin_inference(self):
        self.llm_busy, self._interrupted = True, False

    def inference_context(self, context):
        selected = [(key, self._calls[key]) for key in self._reply if key in self._calls]
        self._reply = ()
        selected_ids = {key for key, _ in selected}
        messages = []
        for message in context.messages:
            payload = async_tool_messages.parse_message(message) if isinstance(message, dict) else None
            key = payload.tool_call_id if payload else message.get("tool_call_id") if isinstance(message, dict) else None
            pending = self._calls.get(key)
            if pending and pending.name == "search_clinic_instructions" and key not in selected_ids:
                # Unclaimed evidence is only handed off by this gate, not by a second reply path.
                message = {**message, "content": json.dumps({
                    "state": "ready" if pending.result is not None else "running",
                    "tool_name": pending.name, "arguments": pending.arguments,
                    "instruction": "pending_request(action=status) checks current work."})}
            elif payload and payload.kind == "final":
                content = json.loads(message["content"])
                content.pop("description", None)  # Remove the re-delivery directive, not evidence.
                message = {**message, "content": json.dumps(content)}
            messages.append(message)
        if not selected:
            return LLMContext(messages=messages, tools=context.tools, tool_choice=context.tool_choice)
        # One handoff for both automatic delivery and a caller's status request.
        # Native history retains evidence; there is no spoken/heard acknowledgement ledger.
        self._trace("background_result_handed_off", tool_call_ids=sorted(selected_ids))
        for key in selected_ids:
            self.finish(key)
        instruction = {"role": "developer", "content":
            "TOOL_RESULT_RESPONSE: Answer only the handed-off lookups. "
            "Do not ask permission, repeat unrelated answers, or restart cancelled lookups. "
            "Report errors/empty results honestly."}
        if all(call.name == "search_clinic_instructions" for _, call in selected):
            messages = [instruction, {"role": "user", "content": selected[0][1].request},
                        {"role": "assistant", "content": None, "tool_calls": [
                            {"id": key, "type": "function", "function": {
                                "name": call.name, "arguments": json.dumps(call.arguments)}}
                            for key, call in selected]}]
            messages.extend({"role": "tool", "tool_call_id": key, "content": json.dumps(call.result)}
                            for key, call in selected)
            return LLMContext(messages=messages)
        messages.append({**instruction, "content": instruction["content"] +
                         " Finish required dependent lookups for this request: " + json.dumps(selected[0][1].request)})
        return LLMContext(messages=messages, tools=context.tools, tool_choice=context.tool_choice)

    def user_started(self):
        self.user_pending = self._interrupted = True
        self._wait_for_user = False
        self._foreground.clear()
        self._reply = ()  # A not-yet-started handoff stays in _calls.

    def user_committed(self):
        self.llm_busy = True

    def user_stopped(self):
        self.user_pending = False
        self._kick()

    def begin(self, params):
        runner = next((r for r in params.llm._function_call_tasks.values()
                       if r.tool_call_id == params.tool_call_id), None)
        request = next((m.get("content") for m in reversed(params.context.messages)
                        if isinstance(m, dict) and m.get("role") == "user"), "")
        call = _Call((runner.group_id if runner else None) or params.tool_call_id,
                     params.function_name, request or "", dict(params.arguments))
        self._calls[params.tool_call_id] = call
        self._trace("background_call_started", tool_name=call.name, tool_call_id=params.tool_call_id)
        call.timer = asyncio.create_task(self._notice_after(params.tool_call_id, call))
        return call

    async def _notice_after(self, key, call):
        await asyncio.sleep(self.notice_seconds)
        if self._calls.get(key) is call and call.result is None:
            call.notice_due = True
            self._kick()

    def result_in_context(self, key, call, result):
        if not self.closed and self._calls.get(key) is call:
            call.result = result
            call.timer.cancel()
            self._trace("background_result_ready", tool_call_id=key)
            self._kick()

    def _ready_group(self, key):
        # Cancelled members are settled; independent surviving lookups still deliver.
        group_id = self._calls[key].group
        group = [(other, call) for other, call in self._calls.items() if call.group == group_id]
        return tuple(other for other, _ in group) if all(c.result is not None for _, c in group) else ()

    async def cancel(self, *, tool_name=None):
        key = next((key for key in reversed(self._calls)
                    if tool_name is None or self._calls[key].name == tool_name), None)
        if self.closed or key is None:
            return {"state": "not_pending"}
        call = self._calls[key]
        self.finish(key)  # Suppress a ready/late result before awaiting native cancellation.
        self._trace("background_call_retired", tool_name=call.name, tool_call_id=key)
        await self.llm._cancel_function_call_tasks(
            lambda item: item.tool_call_id == key, reason="caller cancelled lookup", run_llm=False)
        return {"state": "cancelled" if call.result is None else "result_discarded",
                "tool_name": call.name, "arguments": call.arguments}

    async def control(self, action):
        if action == "cancel":
            result = await self.cancel()
            text = ("There is no pending lookup to cancel." if result["state"] == "not_pending"
                    else "I've stopped that lookup.")
            self.speech_busy = True
            await self.push_frame(TTSSpeakFrame(text))
            return result
        key = next(reversed(self._calls), None)
        if self.closed or key is None:
            return {"state": "not_pending"}
        call = self._calls[key]
        self._reply = self._ready_group(key)
        if not self._reply:
            self.speech_busy = True
            await self.push_frame(TTSSpeakFrame("It's still running; I'll share the result when it's ready."))
        return {"state": "ready" if self._reply else "running",
                "tool_name": call.name, "arguments": call.arguments}

    def finish(self, key):
        call = self._calls.pop(key, None)
        if call and call.timer:
            call.timer.cancel()
        self._foreground.discard(key)
        self._reply = tuple(item for item in self._reply if item != key)

    def check_ready(self):
        self._kick()

    def _idle(self):
        control_running = any(not item.settled and item.function_name not in ASYNC_READ_TOOLS
                              for item in getattr(self.llm, "_function_call_tasks", {}).values())
        return not (self.closed or self._wait_for_user or self.user_pending or self.llm_busy
                    or self.speech_busy or control_running)

    def _kick(self):
        if self._calls and self._idle() and not self._check_queued:
            self._check_queued = True
            self.create_task(self.queue_frame(_Check()))

    def observe(self, source, frame):
        if self.closed:
            return
        if source is self.llm:
            if isinstance(frame, LLMFullResponseStartFrame):
                self.llm_busy = True
            elif isinstance(frame, FunctionCallsStartedFrame):
                self._foreground.update(c.tool_call_id for c in frame.function_calls if c.function_name in ASYNC_READ_TOOLS)
            elif isinstance(frame, LLMErrorFallbackFrame):
                if not self._interrupted:
                    self.user_pending, self._wait_for_user = False, True
                    self._reply = ()
                self.speech_busy = True
            elif isinstance(frame, LLMTextFrame) and not frame.skip_tts and frame.text.strip():
                self.speech_busy = True
            elif isinstance(frame, LLMFullResponseEndFrame):
                self.llm_busy = False
                self._kick()
        elif source is self.tts and isinstance(frame, TTSStartedFrame):
            self._audio_context, self.speech_busy = frame.context_id, True
        elif source is self.output and isinstance(frame, TTSStoppedFrame) and frame.context_id == self._audio_context:
            self._audio_context, self.speech_busy = None, False
            self._kick()

    def invalidate(self):
        for key in tuple(self._calls):
            self.finish(key)
        self._foreground.clear()
        self._reply = ()

    def stop(self):
        self.closed = True
        self.invalidate()

    async def cleanup(self):
        timers = [call.timer for call in self._calls.values() if call.timer]
        self.stop()
        await asyncio.gather(*timers, return_exceptions=True)
        await super().cleanup()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (EndFrame, CancelFrame)):
            self.stop()
        elif isinstance(frame, InterruptionFrame):
            self._reply = ()
            self.llm_busy = self.speech_busy = False
            self._audio_context = None
        elif isinstance(frame, _Check):
            self._check_queued = False
            if not self._idle():
                return
            ready = [key for key in self._calls if self._ready_group(key)]
            foreground_pending = any(key not in self._calls or self._calls[key].result is None
                                     for key in self._foreground)
            if ready and not foreground_pending:
                candidates = [key for key in ready if key in self._foreground] or ready
                self._reply = self._ready_group(candidates[-1])
                self._trace("background_delivery_requested", tool_call_ids=list(self._reply))
                self.llm_busy = True
                frame = LLMMessagesAppendFrame(messages=[], run_llm=True)
            else:
                due = [c for c in self._calls.values() if c.result is None and c.notice_due and not c.notice_sent]
                if due:
                    groups = {c.group for c in due}
                    for call in self._calls.values():
                        if call.group in groups:
                            call.notice_sent = True
                    self.speech_busy = True
                    self._trace("background_processing_notice")
                    await self.push_frame(TTSSpeakFrame(PROCESSING_NOTICE))
                return
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            self.llm_busy = True
        await self.push_frame(frame, direction)
