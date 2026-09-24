import json
import logging
import math
import os

import pytest

from healthcare_voice_agent.voice.tracing import SessionTrace, install_log_redaction


def test_trace_redacts_recursive_credentials_and_omits_sensitive_objects(tmp_path):
    secret = "top-secret-value"
    path = tmp_path / "trace.jsonl"
    trace = SessionTrace(path, session_id="session-1", secrets=(secret,))
    trace.emit(
        "provider_failure",
        message=f"failed with {secret}; Bearer abc.def-123; sk-proj-abc******xyz",
        api_key=secret,
        nested={"authorization": "Bearer should-not-appear", "note": secret},
        audio=b"raw-audio",
        error=RuntimeError(secret),
        context={"messages": [secret]},
        messages=[secret],
    )
    trace.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(record)
    assert secret not in encoded
    assert "abc.def-123" not in encoded
    assert "sk-proj-abc******xyz" not in encoded
    assert record["data"]["message"].startswith("failed with [REDACTED]")
    assert record["data"]["messages"] == "[omitted context]"
    assert record["data"]["audio"] == "[omitted bytes]"
    assert record["data"]["error"] == "[omitted exception: RuntimeError]"
    assert record["data"]["context"] == "[omitted context]"
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_trace_normalizes_nonfinite_floats_for_strict_json(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = SessionTrace(path, session_id="session-1")
    trace.emit("numbers", nan=math.nan, positive=math.inf, negative=-math.inf)
    trace.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["data"] == {"nan": None, "positive": None, "negative": None}


def test_trace_file_is_exclusive_and_close_is_idempotent(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = SessionTrace(path, session_id="one")
    with pytest.raises(FileExistsError):
        SessionTrace(path, session_id="two")
    trace.close()
    trace.close()


def test_loguru_global_patcher_redacts_after_sink_reset_and_clears_exception():
    loguru = pytest.importorskip("loguru")
    logger = loguru.logger
    secret = "test-log-secret"
    install_log_redaction((secret,))
    first, second = [], []
    first_sink = logger.add(lambda message: first.append(message.record), format="{message} {exception}")
    try:
        logger.info(f"first {secret} sk-proj-abc******xyz")
    finally:
        logger.remove(first_sink)

    second_sink = logger.add(lambda message: second.append(message.record), format="{message} {exception}")
    try:
        try:
            local_secret = secret
            raise RuntimeError(f"provider failed with {local_secret}")
        except RuntimeError:
            logger.bind(api_key=secret).exception(f"second {secret}")
    finally:
        logger.remove(second_sink)

    rendered = "\n".join(str(record) for record in first + second)
    assert secret not in rendered
    assert "abc******xyz" not in rendered
    assert first and second
    assert second[0]["exception"] is None
    assert "[REDACTED]" in second[0]["message"]


def test_stdlib_logging_factory_redacts_message_exception_and_stack_locals():
    pytest.importorskip("loguru")
    secret = "stdlib-log-secret"
    previous_factory = logging.getLogRecordFactory()
    logger = logging.getLogger("healthcare_voice_agent.tests.stdlib_redaction")
    previous_level, previous_propagate = logger.level, logger.propagate
    captured_records, captured_output = [], []

    class Capture(logging.Handler):
        def emit(self, record):
            captured_records.append(record)
            captured_output.append(self.format(record))

    handler = Capture()
    handler.setFormatter(logging.Formatter("%(message)s %(exc_text)s %(stack_info)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        install_log_redaction((secret,))
        try:
            local_secret = secret
            raise RuntimeError(f"provider rejected {local_secret}")
        except RuntimeError:
            logger.exception("request failed for %s", f"sk-proj-abc******{secret}", stack_info=True)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
        logging.setLogRecordFactory(previous_factory)

    assert len(captured_records) == 1
    record = captured_records[0]
    rendered = "\n".join(captured_output + [str(record.msg), str(record.args)])
    assert secret not in rendered
    assert "abc******" not in rendered
    assert record.exc_info is None
    assert record.exc_text is None
    assert record.stack_info is None
    assert "RuntimeError" in record.msg
    assert "[REDACTED]" in record.msg
