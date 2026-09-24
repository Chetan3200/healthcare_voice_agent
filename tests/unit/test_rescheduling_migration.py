"""Offline revision rendering only; no database connection or applied migration."""
import io
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[2]


def test_new_revision_extends_existing_booking_schema():
    config = Config(str(ROOT / "alembic.ini"))
    revision = ScriptDirectory.from_config(config).get_revision("003")
    assert revision.down_revision == "002"
    assert revision.module.revision == "003"


def test_reschedule_upgrade_renders_safe_allocation_history_and_linkage(monkeypatch):
    # Revisions use offline SQL generation; no root dotenv or engine is needed.
    output = io.StringIO()
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
    command.upgrade(config, "002:003", sql=True)
    sql = output.getvalue()
    assert "ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE" in sql
    assert "PRIMARY KEY (draft_id)" in sql
    assert "booking_active_slot_once_idx" in sql and "WHERE active" in sql
    assert "CREATE TABLE clinic.booking_reschedules" in sql
    assert "source_record_version" in sql and "source_snapshot" in sql
    assert "REFERENCES clinic.booking_drafts(session_id, draft_id)" in sql
    assert "SET status" not in sql and "INSERT INTO clinic.appointments" not in sql
    assert "booking_allocation_matches_draft_fk" not in sql  # Existing reciprocal guards are not dropped.


def test_downgrade_refuses_to_discard_reschedule_history():
    output = io.StringIO()
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
    command.downgrade(config, "003:002", sql=True)
    sql = output.getvalue()
    assert "RAISE EXCEPTION" in sql
    assert sql.index("RAISE EXCEPTION") < sql.index("DROP TABLE clinic.booking_reschedules")
    assert "HAVING count(*) > 1" in sql
