"""Offline revision-004 rendering; no database connection or migration application."""
import io
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from healthcare_voice_agent.booking.tables import metadata

ROOT = Path(__file__).resolve().parents[2]


def render(command_name, range_name):
    output = io.StringIO()
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
    getattr(command, command_name)(config, range_name, sql=True)
    return output.getvalue()


def test_revision_004_is_single_head_extending_003():
    scripts = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))
    revision = scripts.get_revision("004")
    assert revision.down_revision == "003"
    assert revision.module.revision == "004"
    assert scripts.get_heads() == ["004"]


def test_upgrade_adds_only_persisted_cancellation_receipts():
    sql = render("upgrade", "003:004")
    assert "CREATE TABLE clinic.booking_cancellations" in sql
    assert "source_appointment_id" in sql and "source_record_version" in sql
    assert "source_snapshot JSONB NOT NULL" in sql and "result_snapshot JSONB" in sql
    assert "case_context_version" in sql and "prepared_conversation_version" in sql
    assert "confirmation_version = readback_version + 1" in sql
    assert "booking_cancellation_pending_session_once_idx" in sql
    assert "WHERE state = 'pending'" in sql
    assert "booking_cancellation_confirmation_event_once_idx" in sql
    assert "REFERENCES clinic.appointments(appointment_id, case_id)" in sql
    assert "UPDATE clinic.appointments" not in sql
    assert "UPDATE clinic.booking_allocations" not in sql
    assert "INSERT INTO clinic.booking_cancellations" not in sql
    assert "UPDATE public.alembic_version SET version_num='004'" in sql


def test_downgrade_refuses_to_erase_cancellation_history():
    sql = render("downgrade", "004:003")
    assert "RAISE EXCEPTION" in sql
    assert sql.index("RAISE EXCEPTION") < sql.index("DROP TABLE clinic.booking_cancellations")


def test_sqlalchemy_mapping_matches_revision_interface():
    table = metadata.tables["clinic.booking_cancellations"]
    assert set(table.columns.keys()) == {
        "cancellation_request_id", "session_id", "case_id", "clinic_id",
        "case_context_version", "prepared_conversation_version",
        "source_appointment_id", "source_record_version", "source_snapshot",
        "source_hash", "details_hash", "state", "expires_at", "readback_version",
        "readback_at", "confirmation_version", "confirmation_event_id", "confirmed_at",
        "result_snapshot", "cancelled_at", "created_at", "updated_at",
    }
    assert table.c.cancellation_request_id.primary_key
    assert table.c.result_snapshot.nullable
