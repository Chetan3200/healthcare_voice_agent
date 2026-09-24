"""RTVI processor with provider-error details kept server-side."""

from __future__ import annotations

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import ErrorFrame
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor

_GENERIC_ERROR = "The voice service encountered an error. Please try again."


class SafeRTVIProcessor(RTVIProcessor):
    """Sanitize Pipecat 1.11's private RTVI error send path.

    Pipecat 1.11's ``_send_error_frame`` at processor.py:550-553 forwards
    ``ErrorFrame.error`` to ``push_transport_message``. This narrow pinned
    override preserves the inherited RTVI handshake and public transport path
    while replacing provider text with a fixed client-safe message.
    """

    async def _send_error_frame(self, frame: ErrorFrame):
        message = RTVI.Error(data=RTVI.ErrorData(error=_GENERIC_ERROR, fatal=frame.fatal))
        await self.push_transport_message(message)

    async def _send_error_response(self, id: str, error: str):
        message = RTVI.ErrorResponse(id=id, data=RTVI.ErrorResponseData(error=_GENERIC_ERROR))
        await self.push_transport_message(message)
