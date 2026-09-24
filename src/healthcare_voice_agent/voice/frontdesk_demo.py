"""Native Pipecat Flows integration for the isolated front-desk demo."""
from __future__ import annotations

from pipecat.flows import FlowManager
from pipecat.frames.frames import (
    CancelFrame, EndFrame, ErrorFrame, InterruptionFrame, LLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class _InputGate(FrameProcessor):
    """Advance the demo session once per final caller turn and refresh state."""

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.last_user_key = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            session = self.adapter.session
            if isinstance(frame, InterruptionFrame):
                session.interrupt()
            elif isinstance(frame, (CancelFrame, EndFrame, ErrorFrame)):
                self.adapter.abort_output()
            elif isinstance(frame, LLMContextFrame):
                if frame.speculation or self.adapter.stopped or session.closed:
                    return
                # Tool results trigger another inference on the same caller turn.
                # Advance the runtime only for a new final user-context message.
                users = [
                    (i, message)
                    for i, message in enumerate(frame.context.get_messages())
                    if isinstance(message, dict) and message.get("role") == "user"
                ]
                if users:
                    index, message = users[-1]
                    text = message.get("content")
                    key = (index, text) if isinstance(text, str) else None
                    if key and any(char.isalnum() for char in text) and key != self.last_user_key:
                        turn_epoch = await session.user_turn()
                        self.last_user_key = key
                        if turn_epoch != session.epoch or session.closed:
                            return

                # Keep exactly one trusted state message without changing caller
                # message indices on tool-result inferences.
                from healthcare_voice_agent.demo import flow

                prefix = "Current front-desk state (trusted):\n"
                state = {
                    "role": "developer",
                    "content": prefix + flow.current_state(session),
                }
                messages = frame.context.get_messages()
                for i, message in enumerate(messages):
                    if (
                        isinstance(message, dict)
                        and isinstance(message.get("content"), str)
                        and message["content"].startswith(prefix)
                    ):
                        messages[i] = state
                        break
                else:
                    messages.append(state)
        await self.push_frame(frame, direction)


class FrontDeskVoiceAdapter:
    def __init__(self, session, trace):
        self.session, self.trace = session, trace
        self.stopped = False
        self.client_ready = self.pipeline_started = self.initialized = False
        self.flow = None
        self.input_gate = _InputGate(self)
        self.processors = (self.input_gate,)

    def bind(self, llm, context_aggregator, worker, transport):
        from healthcare_voice_agent.demo.flow import global_functions

        self.flow = FlowManager(
            llm=llm,
            context_aggregator=context_aggregator,
            worker=worker,
            transport=transport,
            global_functions=global_functions(),
        )
        self.flow.state["session"] = self.session

    async def _initialize(self):
        if self.client_ready and self.pipeline_started and not self.initialized and not self.stopped:
            from healthcare_voice_agent.demo.flow import initial_node

            self.initialized = True
            await self.flow.initialize(initial_node())

    async def on_client_ready(self):
        self.client_ready = True
        await self._initialize()

    async def on_pipeline_started(self):
        self.pipeline_started = True
        await self._initialize()

    def abort_output(self):
        self.stopped = True
        self.session.interrupt()

    async def close(self):
        self.abort_output()
