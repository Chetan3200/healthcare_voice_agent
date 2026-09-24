"""HybridDiffusion's pinned SGLang chat protocol, on Pipecat's OpenAI transport.

No model loading and no fallback. Server-side decoding mode is chosen by the
GPU launcher, not inferred from the model ID. The native Pipecat consumer owns
SSE parsing, tool assembly, interruption handling and stream/socket cleanup.
"""

from __future__ import annotations

import asyncio


class HybridDiffusionLLMMixin:
    """Adapt requests without replacing Pipecat's context/stream lifecycle."""

    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        # The checkpoint's chat template supports this explicitly. Reasoning is
        # not spoken aloud. Preserve native OpenAI tools and tool_choice fields.
        params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return params

    async def get_chat_completions(self, context):
        delivery = getattr(self, "_clinician_delivery", None)
        if delivery is not None:
            context = delivery.inference_context(context)
        # Equivalent to the pinned base adapter, minus its full-context DEBUG
        # log and timeout retry. Never print synthetic prompts or backend errors.
        from pipecat.utils.types import assert_given
        adapter = self.get_llm_adapter()
        invocation = adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )
        return await self._client.chat.completions.create(
            **self.build_chat_completion_params(invocation)
        )

    async def _process_context(self, context):
        # HTTP phase timeouts alone let a server trickle SSE forever. Bound the
        # whole turn as well; the base async context manager closes the stream.
        async with asyncio.timeout(self._hybrid_deadline_seconds):
            await super()._process_context(context)

    async def push_error(self, error_msg, exception=None, **kwargs):
        # Native SDK errors may contain response bodies, request text or tokens.
        # Do not attach them to frames, traces, browser messages or logs.
        await super().push_error(
            error_msg="HybridDiffusion request failed. Check the model server and reconnect.",
            force_treat_as_permanent=True,
        )
