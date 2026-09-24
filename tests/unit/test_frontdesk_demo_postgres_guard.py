"""Offline guard checks; these tests never connect to PostgreSQL or read .env."""
from types import SimpleNamespace

import pytest
from sqlalchemy import URL

from healthcare_voice_agent.demo.database import DemoDatabaseError
from scripts import test_frontdesk_demo_postgres as verification


def test_default_command_is_offline_and_requires_run(monkeypatch, capsys):
    monkeypatch.setattr(verification, "build_demo_engine", lambda: pytest.fail("engine constructed"))
    monkeypatch.setattr(verification, "run_verification", lambda: pytest.fail("test executed"))
    assert verification.main([]) == 0
    assert '"status": "not_run"' in capsys.readouterr().out


@pytest.mark.parametrize("field,value", [("port", 5433), ("database", "fracture_clinic"),
    ("host", "remote.example"), ("username", "clinic_owner")])
def test_wrong_engine_rejected_before_any_connection_or_write(field, value):
    values = {"drivername": "postgresql+psycopg", "host": "127.0.0.1", "port": 55432,
              "database": "frontdesk_synthetic_demo", "username": "frontdesk_demo_owner"}
    values[field] = value
    engine = SimpleNamespace(url=URL.create(**values),
        connect=lambda: pytest.fail("wrong database connection attempted"),
        begin=lambda: pytest.fail("write attempted"))
    with pytest.raises(DemoDatabaseError, match="fixed isolated"):
        verification.run_verification(engine)


def test_same_url_without_private_demo_settings_rejected_before_connection():
    engine = SimpleNamespace(url=URL.create("postgresql+psycopg", host="127.0.0.1", port=55432,
        database="frontdesk_synthetic_demo", username="frontdesk_demo_owner"),
        connect=lambda: pytest.fail("unverified connection attempted"))
    with pytest.raises(DemoDatabaseError, match="build_demo_engine"):
        verification.run_verification(engine)


def test_readiness_failure_precedes_qa_mutation(monkeypatch):
    events = []
    engine = SimpleNamespace(begin=lambda: pytest.fail("QA write attempted"))
    def reject(_):
        events.append("verify")
        raise DemoDatabaseError("identity marker mismatch")
    monkeypatch.setattr(verification, "verify_demo_database", reject)
    monkeypatch.setattr(verification, "BookingService", lambda _: pytest.fail("service constructed"))
    with pytest.raises(DemoDatabaseError):
        verification.run_verification(engine)
    assert events == ["verify"]


def test_cli_sanitizes_database_exception(monkeypatch, capsys):
    def fail():
        raise RuntimeError("private database credential or raw SQL")
    monkeypatch.setattr(verification, "run_verification", fail)
    assert verification.main(["--run"]) == 1
    output = capsys.readouterr().out
    assert "RuntimeError" in output
    assert "credential" not in output and "SQL" not in output
