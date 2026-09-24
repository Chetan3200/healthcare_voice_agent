"""Real Uvicorn formatter regressions; no server, sockets, or providers."""

import io
import logging

import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("loguru")

from uvicorn.logging import AccessFormatter

from healthcare_voice_agent.voice.tracing import install_log_redaction


_ACCESS_TEMPLATE = '%s - "%s %s HTTP/%s" %d'
_SECRET = "private-uvicorn-test-value"


@pytest.fixture
def access_capture():
    previous_factory = logging.getLogRecordFactory()
    logger = logging.getLogger("uvicorn.access")
    previous_handlers = logger.handlers[:]
    previous_level, previous_propagate = logger.level, logger.propagate
    previous_disabled = logger.disabled
    output = io.StringIO()
    records = []

    class StrictHandler(logging.StreamHandler):
        def emit(self, record):
            records.append(record)
            super().emit(record)

        def handleError(self, record):
            raise AssertionError("Uvicorn access formatting failed")

    handler = StrictHandler(output)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.disabled = False
    try:
        install_log_redaction((_SECRET,))
        yield logger, handler, output, records
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
        logger.disabled = previous_disabled
        logging.setLogRecordFactory(previous_factory)
        handler.close()


@pytest.mark.parametrize("use_colors", [False, True])
@pytest.mark.parametrize("method,status", [("GET", 200), ("PATCH", 200), ("POST", 400), ("GET", 500)])
def test_uvicorn_access_args_survive_redaction_and_real_formatter(access_capture, use_colors, method, status):
    logger, handler, output, records = access_capture
    handler.setFormatter(AccessFormatter(
        '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        use_colors=use_colors,
    ))
    path = f"/sessions/test/api/offer?key={_SECRET}&token=sk-proj-test-marker&percent=50%25"
    logger.info(_ACCESS_TEMPLATE, "127.0.0.1:1234", method, path, "1.1", status)
    record, = records
    assert isinstance(record.args, tuple) and len(record.args) == 5
    assert record.args[4] == status
    rendered = output.getvalue()
    assert method in rendered
    assert str(status) in rendered
    assert "HTTP/1.1" in rendered
    assert "50%25" in rendered
    assert "[REDACTED]" in rendered
    for encoded in (rendered, record.getMessage(), str(record.args)):
        assert _SECRET not in encoded
        assert "sk-proj-test-marker" not in encoded


def test_multiple_redaction_installations_do_not_flatten_access_arguments(access_capture):
    logger, handler, output, records = access_capture
    handler.setFormatter(AccessFormatter('%(request_line)s %(status_code)s', use_colors=False))
    install_log_redaction(("another-private-marker",))
    logger.info(_ACCESS_TEMPLATE, "127.0.0.1:1234", "PATCH",
                f"/api/offer?first={_SECRET}&second=another-private-marker", "1.1", 200)
    assert len(records[0].args) == 5
    assert _SECRET not in output.getvalue()
    assert "another-private-marker" not in output.getvalue()
    assert "200 OK" in output.getvalue()


def test_access_exception_summary_preserves_formatting_and_omits_stack(access_capture):
    logger, handler, output, records = access_capture
    # A plain formatter also has to work on the retained template and arguments.
    handler.setFormatter(logging.Formatter("%(message)s %(exc_text)s %(stack_info)s"))
    try:
        local_secret = _SECRET
        raise RuntimeError(f"50% done; %s literal; {local_secret}")
    except RuntimeError:
        logger.error(_ACCESS_TEMPLATE, "127.0.0.1:1234", "GET", "/test", "1.1", 500,
                     exc_info=True, stack_info=True)
    record, = records
    assert len(record.args) == 5
    assert record.exc_info is record.exc_text is record.stack_info is None
    assert _SECRET not in output.getvalue()
    assert "Traceback" not in output.getvalue()
    assert "RuntimeError" in output.getvalue()
    assert "50% done; %s literal; [REDACTED]" in output.getvalue()
