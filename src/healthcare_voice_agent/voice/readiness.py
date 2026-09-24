"""Do not send microphone audio to STT before the browser is ready.

SmallWebRTCInputTransport in Pipecat 1.11 ignores audio_in_stream_on_start.
This small frame gate is explicit and tested. It is not a VAD/turn controller.
"""

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ClientReadyAudioGate(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.ready = False

    def mark_ready(self) -> None:
        self.ready = True

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and not self.ready:
            return  # Drop, rather than replay speech captured before readiness.
        await self.push_frame(frame, direction)
