"""Small native Flows adapter for the read-only clinician assistant."""
import asyncio
from dataclasses import replace
import re

from pydantic import ValidationError
from pipecat.flows import FlowManager
from pipecat.frames.frames import FunctionCallResultProperties
from healthcare_voice_agent.clinician.flow import functions, initial_node, ASYNC_READ_TOOLS
from healthcare_voice_agent.voice.deferred_tools import DeferredToolDelivery
from healthcare_voice_agent.tools.contracts import TOOLS


def _trace_record_ids(arguments):
    """Only synthetic record-ID shapes/placeholders, never names or free text."""
    ids = {}
    for key in ("case_id", "study_id", "model_result_id", "report_id"):
        if key not in arguments:
            continue
        value = arguments[key]
        safe = isinstance(value, str) and (
            re.fullmatch(r"[0-9]+|(?:ST|MR|RR)-[0-9]+-[0-9]+(?:-V[0-9]+)?", value)
            or value in {"<original_case_id>", "<corrected_case_id>", "<selected_case_id>"}
        )
        ids[key] = value if value is None or safe else "<redacted>"
    return ids


class ClinicianFlowManager(FlowManager):
    """Pipecat 1.11 adapter: keep native per-group batching, not per-result reruns."""

    async def _create_transition_func(self, name, handler):
        native_handler = await super()._create_transition_func(name, handler)

        async def invoke(params):
            delivery = self.state.get("delivery")
            if name == "search_clinic_instructions":
                arguments = dict(params.arguments)
                mode = arguments.pop("mode", "new")  # Voice control only, never sent to Exa.
                params = replace(params, arguments=arguments)
                if delivery is not None:
                    delivery._trace("background_search_requested", tool_call_id=params.tool_call_id,
                                    replaces_pending=mode == "replace")
                if mode == "replace" and delivery is not None:
                    try:
                        TOOLS[name].arguments_model.model_validate(arguments)
                    except ValidationError:
                        pass  # The normal handler returns the validation error; keep old work intact.
                    else:
                        await delivery.cancel(tool_name=name)
            if delivery is not None and name == "open_case":
                delivery.invalidate()
                # Use the pinned native cancellation path so task AND context
                # bookkeeping settle before Flows performs its case reset.
                await params.llm._cancel_function_call_tasks(
                    lambda item: item.function_name in ASYNC_READ_TOOLS,
                    reason="case change", run_llm=False)
            call = delivery.begin(params) if delivery is not None and name in ASYNC_READ_TOOLS else None
            reported = False

            async def complete(result, *, properties=None):
                nonlocal reported
                properties = properties or FunctionCallResultProperties()
                # Reuse native membership: a fast result can precede a sibling's
                # in-progress frame. Do not recreate native execution/grouping.
                runners = tuple(getattr(params.llm, "_function_call_tasks", {}).values())
                current = next((item for item in runners
                                if item.tool_call_id == params.tool_call_id), None)
                group_id = (current.group_id if current else None) or params.tool_call_id
                pending = self._pending_transition
                if pending and pending["result"] is result:
                    pending["group_id"] = group_id
                transition_in_group = pending and pending.get("group_id") == group_id
                run_llm = properties.run_llm
                if run_llm is not False:
                    sibling_pending = current is not None and current.group_id and any(
                        item.tool_call_id != params.tool_call_id
                        and item.group_id == current.group_id and not item.settled
                        for item in runners
                    )
                    run_llm = False if transition_in_group or sibling_pending else None
                context_updated = properties.on_context_updated or self._check_and_execute_transition
                if call is not None:
                    call.timer.cancel()
                    call.notice_due = False
                    native_updated = context_updated
                    async def context_updated():
                        await native_updated()
                        delivery.result_in_context(params.tool_call_id, call, result)
                    run_llm = False
                properties = replace(properties, run_llm=run_llm,
                                     on_context_updated=context_updated)
                trace = getattr(self.state.get("session"), "trace", None)
                if trace is not None:
                    trace.emit("clinician_tool_call", tool_name=name,
                               tool_call_id=params.tool_call_id,
                               request_id=result.get("meta", {}).get("request_id"),
                               record_arguments=_trace_record_ids(params.arguments))
                reported = True
                await params.result_callback(result, properties=properties)

            try:
                # Pipecat owns execution, cancellation and the existing 30s deadline.
                await native_handler(replace(params, result_callback=complete))
            finally:
                if call is not None:
                    # A reported sibling still counts as pending until its
                    # result actually reaches the context aggregator.
                    if not reported:
                        delivery.finish(params.tool_call_id)
                    await asyncio.gather(call.timer, return_exceptions=True)

        return invoke


class ClinicianVoiceAdapter:
    def __init__(self, session):
        self.session = session
        self.client_ready = self.pipeline_started = self.initialized = self.stopped = False
        self.flow = None
        self.delivery = DeferredToolDelivery(session)

    def bind(self, llm, context_aggregator, worker, transport):
        self.flow = ClinicianFlowManager(llm=llm, context_aggregator=context_aggregator,
                                worker=worker, transport=transport, global_functions=functions())
        self.flow.state["session"] = self.session
        self.flow.state["delivery"] = self.delivery

    async def _initialize(self):
        if self.client_ready and self.pipeline_started and not self.initialized and not self.stopped:
            self.initialized = True
            await self.flow.initialize(initial_node(self.session))

    async def on_client_ready(self):
        self.client_ready = True
        await self._initialize()

    async def on_pipeline_started(self):
        self.pipeline_started = True
        await self._initialize()

    def abort_output(self):
        self.stopped = True
        self.delivery.stop()

    async def close(self):
        self.abort_output()
