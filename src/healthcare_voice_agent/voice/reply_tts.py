"""Buffer one LLM reply for one TTS request, without changing its wording."""
from dataclasses import dataclass

from pipecat.frames.frames import (
    AggregatedTextFrame, CancelFrame, DataFrame, EndFrame, ErrorFrame, InterruptionFrame,
    LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame, TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class DiscardLLMReplyFrame(DataFrame):
    """Clear buffered text in order, before a cancelled LLM emits its end frame."""


@dataclass
class LLMErrorFallbackFrame(TTSSpeakFrame):
    """Ordered discard marker plus fixed direct speech; never an LLM request."""

    text: str = "I'm still here, but I couldn't finish that request. Say 'try again' and I'll retry."


class FullReplyTTSBuffer(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._parts = []
        self._collecting = False
        self._discarding = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, DiscardLLMReplyFrame):
                self._parts.clear()
                self._collecting = False
                self._discarding = True
                return
            if isinstance(frame, LLMFullResponseStartFrame):
                self._parts.clear()
                self._collecting = not frame.skip_tts
                self._discarding = False
            elif isinstance(frame, (InterruptionFrame, CancelFrame, EndFrame, ErrorFrame,
                                    LLMErrorFallbackFrame)):
                self._parts.clear()
                self._collecting = False
                if isinstance(frame, LLMErrorFallbackFrame):
                    self._discarding = True
            elif isinstance(frame, LLMTextFrame) and self._discarding:
                # The native incomplete-turn filter can flush unmarked partial
                # text just before its end frame. Drop it until the next start.
                return
            elif isinstance(frame, LLMTextFrame) and self._collecting and not frame.skip_tts:
                self._parts.append(frame.text)
                return
            elif isinstance(frame, LLMFullResponseEndFrame):
                text = "".join(self._parts)
                self._parts.clear()
                self._collecting = False
                if text.strip() and not frame.skip_tts:
                    # Native pre-aggregated input bypasses TTS sentence splitting.
                    await self.push_frame(
                        AggregatedTextFrame(text, aggregated_by="full_response"), direction)
        await self.push_frame(frame, direction)
