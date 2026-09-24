"""Small, privacy-conscious JSONL tracing primitives for one voice session.

This module deliberately has no Pipecat imports so it can be used by server
startup and tested without the optional voice dependency.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_REDACTED = "[REDACTED]"
_OMITTED_BYTES = "[omitted bytes]"
_OMITTED_CONTEXT = "[omitted context]"
_CREDENTIAL_FIELD = re.compile(
    r"(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|"
    r"password|secret|credential|bearer|token|private[_-]?key)$",
    re.IGNORECASE,
)
_CONTEXT_FIELD = re.compile(r"(?:context|conversation|history)$", re.IGNORECASE)
_OPENAI_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_.*…-]+", re.IGNORECASE)
_BEARER_TOKEN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)


def _redact_text(value: str, secrets: tuple[str, ...]) -> str:
    """Mask configured literals and recognizable credentials in text."""
    for secret in secrets:
        if secret:
            value = value.replace(secret, _REDACTED)
    value = _OPENAI_TOKEN.sub(_REDACTED, value)
    return _BEARER_TOKEN.sub("Bearer " + _REDACTED, value)


def _safe_value(value: Any, secrets: tuple[str, ...], *, field_name: str | None = None) -> Any:
    """Convert a value to bounded, JSON-safe trace data without sensitive blobs."""
    if field_name and _CREDENTIAL_FIELD.search(field_name):
        return _REDACTED
    if field_name and _CONTEXT_FIELD.search(field_name):
        return _OMITTED_CONTEXT
    if (
        field_name and field_name.lower() == "messages"
        and isinstance(value, (Mapping, Sequence))
        and not isinstance(value, (str, bytes, bytearray, memoryview))
    ):
        return _OMITTED_CONTEXT
    if isinstance(value, BaseException):
        return f"[omitted exception: {type(value).__name__}]"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _OMITTED_BYTES
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item, secrets, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_safe_value(item, secrets) for item in value]
    # Do not call repr() on arbitrary provider/context objects, which can expose
    # credentials, transcripts, or exception details.
    return f"[omitted {type(value).__name__}]"


def install_log_redaction(secrets: tuple[str, ...]) -> None:
    """Install Loguru and stdlib logging redaction for later log records.

    The Loguru patcher survives Pipecat runner ``remove()/add()`` sink resets.
    The stdlib factory wraps, rather than replaces, the prior factory so SDK
    logging receives the same credential and traceback protection.
    """
    from loguru import logger

    frozen_secrets = tuple(secret for secret in secrets if secret)

    def exception_summary(exception: Any) -> str:
        exception_type = getattr(getattr(exception, "type", None), "__name__", None)
        exception_value = getattr(exception, "value", None)
        if isinstance(exception, tuple) and len(exception) >= 2:
            exception_type = getattr(exception[0], "__name__", "Exception")
            exception_value = exception[1]
        return f"[{exception_type or 'Exception'}: {_redact_text(str(exception_value or ''), frozen_secrets)}]"

    def patch(record: dict[str, Any]) -> None:
        message = _redact_text(str(record.get("message", "")), frozen_secrets)
        # Canonical LLM settings preserve the base prompt through tool sync,
        # but Pipecat DEBUG-logs it at construction/recomposition. Omit the
        # entire payload, not just credential-shaped substrings inside it.
        prompt_log = (
            record.get("name") == "pipecat.services.openai.base_llm"
            and (": Using system instruction:" in message or ": Generating chat from context " in message)
        ) or (
            record.get("name") == "pipecat.services.llm_service"
            and ": System instruction composed:" in message
        )
        if prompt_log:
            message = ("LLM chat context [omitted prompt]" if ": Generating chat from context " in message
                       else "LLM system instruction configured [omitted prompt]")
        if exception := record.get("exception"):
            message = f"{message} {exception_summary(exception)}"
            # Prevent Loguru from rendering traceback locals in diagnose mode.
            record["exception"] = None
        record["message"] = message
        record["extra"] = _safe_value(record.get("extra", {}), frozen_secrets)

    logger.configure(patcher=patch)

    previous_factory = logging.getLogRecordFactory()

    def redacting_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)
        if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            # Uvicorn's AccessFormatter unpacks these fields itself. Flattening
            # them makes it crash with "expected 5, got 0". Retain the template
            # and sanitized field types, particularly the integer status code.
            record.msg = _redact_text(str(record.msg), frozen_secrets)
            record.args = tuple(_safe_value(value, frozen_secrets) for value in record.args)
            if record.exc_info:
                # This remains a %-format template; exception text is literal.
                summary = exception_summary(record.exc_info).replace("%", "%%")
                record.msg = f"{record.msg} {summary}"
        else:
            try:
                message = record.getMessage()
            except Exception:
                message = str(record.msg)
            if record.exc_info:
                message = f"{message} {exception_summary(record.exc_info)}"
            record.msg = _redact_text(message, frozen_secrets)
            record.args = ()
        # ``exc_info`` can retain traceback locals, while these fields can hold
        # already-formatted traceback or stack text. Keep only the summary above.
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return record

    logging.setLogRecordFactory(redacting_factory)


class SessionTrace:
    """Append selected session events to one new owner-only JSONL file.

    Writes are synchronous and buffered by Python's file object. This is a
    minimal demo trace, not a zero-overhead telemetry path.
    """

    def __init__(self, path: Path, *, session_id: str, secrets: tuple[str, ...] = ()):
        self.path = Path(path)
        self.session_id = session_id
        self._secrets = tuple(secret for secret in secrets if secret)
        self._started_monotonic = time.monotonic()
        self._sequence = 0
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file = os.fdopen(fd, "w", encoding="utf-8", buffering=1)

    def emit(self, event: str, **fields: Any) -> None:
        """Record one sanitized event. Raw exceptions, audio, and contexts omit."""
        if self._closed:
            return
        self._sequence += 1
        payload = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": round(time.monotonic() - self._started_monotonic, 6),
            "sequence": self._sequence,
            "session_id": self.session_id,
            "event": _redact_text(str(event), self._secrets),
            "data": _safe_value(fields, self._secrets),
        }
        self._file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    def close(self) -> None:
        """Close the trace file. Repeated close calls are harmless."""
        if not self._closed:
            self._file.close()
            self._closed = True
