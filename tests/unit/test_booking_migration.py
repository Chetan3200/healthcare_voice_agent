"""Offline revision-002 contract tests, not PostgreSQL execution tests.

Capture Alembic DDL and render offline SQL without credentials or a connection.
Actual PostgreSQL constraint/concurrency behavior needs its own live test host.
"""

from hashlib import sha256
import importlib.util
import io
from pathlib import Path
import re
from types import SimpleNamespace

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest

from healthcare_voice_agent.booking.tables import metadata

ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "migrations/versions/002_frontdesk_booking.py"
BASELINE = ROOT / "migrations/versions/001_initial_schema.py"
NEW_TABLES = {
    "booking_guard", "booking_clinics", "booking_sessions", "booking_slots",
    "booking_drafts", "booking_offers", "booking_holds", "booking_allocations",
}


def revision_module():
    spec = importlib.util.spec_from_file_location("frontdesk_migration_for_test", REVISION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def upgrade_statements(monkeypatch):
    module = revision_module()
    statements = []
    monkeypatch.setattr(module, "op", SimpleNamespace(execute=statements.append))
    module.upgrade()
    return statements


def table_statement(statements, name):
    return next(sql for sql in statements if f"CREATE TABLE clinic.{name} (" in sql)


def test_revision_extends_001_without_replacing_it():
    module = revision_module()
    assert (module.revision, module.down_revision) == ("002", "001")
    scripts = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))
    assert scripts.get_revision("002").down_revision == "001"
    assert scripts.get_revision("001").down_revision is None
    assert scripts.get_revision("003").down_revision == "002"
    assert scripts.get_revision("004").down_revision == "003"
    assert scripts.get_revision("005").down_revision == "004"
    assert scripts.get_heads() == ["005"]


def test_exactly_eight_new_tables_and_only_booking_schema_changes(upgrade_statements):
    sql = "\n".join(upgrade_statements)
    created = re.findall(r"CREATE TABLE clinic\.(\w+)", sql)
    assert set(created) == NEW_TABLES
    assert len(created) == 8
    altered = re.findall(r"ALTER TABLE clinic\.(\w+)", sql)
    assert set(altered) <= NEW_TABLES
    assert "DROP " not in sql
    assert "CREATE EXTENSION" not in sql
    assert "CASCADE" not in sql


@pytest.mark.parametrize("name", sorted(NEW_TABLES))
def test_new_table_column_interface_matches_query_mappings(upgrade_statements, name):
    sql = table_statement(upgrade_statements, name)
    declared = set(re.findall(
        r"^\s{12}([a-z][a-z0-9_]*)\s+(?:UUID|INTEGER|BOOLEAN|TEXT|JSONB|TIMESTAMPTZ|clinic\.record_id)\b",
        sql, re.MULTILINE,
    ))
    expected = set(metadata.tables[f"clinic.{name}"].columns.keys())
    if name == "booking_allocations":
        expected.remove("active")  # Added by 003, never back-edited into applied 002.
    assert declared == expected


def test_singleton_guard_is_created_and_seeded_once(upgrade_statements):
    assert "CHECK (id = 1)" in table_statement(upgrade_statements, "booking_guard")
    writes = [sql for sql in upgrade_statements if sql.startswith("INSERT")]
    assert writes == ["INSERT INTO clinic.booking_guard (id) VALUES (1)"]


def test_slot_and_clinic_constraints_are_authoritative(upgrade_statements):
    clinic = table_statement(upgrade_statements, "booking_clinics")
    slots = table_statement(upgrade_statements, "booking_slots")
    assert "clinic.valid_timezone(timezone)" in clinic
    assert "CHECK (ends_at > starts_at)" in slots
    assert "booking_slot_identity UNIQUE" in slots
    for kind in ("initial_consultation", "fracture_follow_up", "imaging", "physiotherapy", "other"):
        assert repr(kind) in slots
    assert "REFERENCES clinic.clinicians(clinician_id)" in slots
    assert "char_length(btrim(location)) > 0" in slots


def test_draft_state_and_null_groups_fail_closed(upgrade_statements):
    sql = table_statement(upgrade_statements, "booking_drafts")
    assert "state IN ('pending', 'booked', 'expired', 'failed')" in sql
    assert "state = 'failed' AND failure_code IS NOT NULL" in sql
    assert "state <> 'failed' AND failure_code IS NULL" in sql
    assert "state = 'booked' AND appointment_id IS NOT NULL AND confirmation_version IS NOT NULL" in sql
    assert "state <> 'booked' AND appointment_id IS NULL" in sql
    assert "readback_version IS NULL AND readback_at IS NULL" in sql
    assert "confirmation_version IS NULL AND confirmation_event_id IS NULL AND confirmed_at IS NULL" in sql
    assert "confirmation_version > readback_version" in sql
    assert "confirmed_at >= readback_at AND confirmed_at < hold_expires_at" in sql
    assert "CHECK (hold_expires_at > created_at)" in sql
    assert "CHECK (jsonb_typeof(details) = 'object')" in sql
    assert "CHECK (details_hash ~ '^[0-9a-f]{64}$')" in sql
    assert "REFERENCES clinic.appointments (appointment_id, case_id)" in sql


def test_hold_identity_expiry_and_allocation_identity_are_database_checked(upgrade_statements):
    holds = table_statement(upgrade_statements, "booking_holds")
    allocated = table_statement(upgrade_statements, "booking_allocations")
    assert "slot_id UUID PRIMARY KEY" in holds
    assert "draft_id UUID NOT NULL UNIQUE" in holds
    assert "FOREIGN KEY (draft_id, slot_id, expires_at)" in holds
    assert "REFERENCES clinic.booking_drafts (draft_id, slot_id, hold_expires_at)" in holds
    assert "slot_id UUID PRIMARY KEY" in allocated
    for field in ("draft_id", "booking_request_id"):
        assert f"{field} UUID NOT NULL UNIQUE" in allocated
    assert "appointment_id clinic.record_id NOT NULL UNIQUE" in allocated
    assert "FOREIGN KEY (draft_id, slot_id, booking_request_id, appointment_id)" in allocated
    assert "DEFERRABLE INITIALLY DEFERRED" in allocated


def test_current_pointer_and_atomic_committed_outcome_are_deferred(upgrade_statements):
    pointer = next(sql for sql in upgrade_statements if "ADD CONSTRAINT booking_session_current_draft_fk" in sql)
    outcome = next(sql for sql in upgrade_statements if "ADD CONSTRAINT booking_draft_committed_allocation_fk" in sql)
    assert "FOREIGN KEY (session_id, current_draft_id)" in pointer
    assert "REFERENCES clinic.booking_drafts (session_id, draft_id)" in pointer
    assert "REFERENCES clinic.booking_allocations (draft_id, slot_id, booking_request_id, appointment_id)" in outcome
    assert "DEFERRABLE INITIALLY DEFERRED" in pointer
    assert "DEFERRABLE INITIALLY DEFERRED" in outcome


def test_downgrade_only_removes_new_tables_and_breaks_cycles_first(monkeypatch):
    module = revision_module()
    statements = []
    monkeypatch.setattr(module, "op", SimpleNamespace(execute=statements.append))
    module.downgrade()
    assert statements[:2] == [
        "ALTER TABLE clinic.booking_sessions DROP CONSTRAINT booking_session_current_draft_fk",
        "ALTER TABLE clinic.booking_drafts DROP CONSTRAINT booking_draft_committed_allocation_fk",
    ]
    dropped = [sql.removeprefix("DROP TABLE clinic.") for sql in statements if sql.startswith("DROP TABLE")]
    assert set(dropped) == NEW_TABLES
    assert len(dropped) == 8
    assert dropped.index("booking_allocations") < dropped.index("booking_drafts")
    assert dropped.index("booking_holds") < dropped.index("booking_drafts")
    assert dropped.index("booking_drafts") < dropped.index("booking_sessions")
    assert all("CASCADE" not in sql for sql in statements)
    assert not any(sql.startswith("DROP TABLE clinic.appointments") for sql in statements)


def test_actual_alembic_offline_range_renders_without_database_or_dotenv(monkeypatch):
    import dotenv
    import psycopg
    import sqlalchemy
    def blocked(*args, **kwargs):
        raise AssertionError("Offline migration rendering must not load private settings or connect")
    monkeypatch.setattr(dotenv, "load_dotenv", blocked)
    monkeypatch.setattr(psycopg, "connect", blocked)
    monkeypatch.setattr(sqlalchemy, "create_engine", blocked)
    before = sha256(BASELINE.read_bytes()).digest()
    output = io.StringIO()
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
    command.upgrade(config, "001:002", sql=True)
    rendered = output.getvalue()
    assert "BEGIN;" in rendered and "COMMIT;" in rendered
    assert "UPDATE public.alembic_version SET version_num='002'" in rendered
    assert "CREATE TABLE clinic.booking_slots" in rendered
    assert "CREATE TABLE patients" not in rendered
    assert "CREATE EXTENSION" not in rendered
    assert sha256(BASELINE.read_bytes()).digest() == before


def test_draft_patient_identity_is_required_in_both_definitions(upgrade_statements):
    column = metadata.tables["clinic.booking_drafts"].c.patient_id
    assert column.nullable is False
    assert {fk.target_fullname for fk in column.foreign_keys} == {"clinic.patients.patient_id"}
    assert "patient_id clinic.record_id NOT NULL REFERENCES clinic.patients(patient_id)" in table_statement(upgrade_statements, "booking_drafts")


def test_portable_metadata_mirrors_all_composite_foreign_keys(upgrade_statements):
    expected = {
        "booking_session_current_draft_fk": ("booking_sessions", True),
        "booking_draft_appointment_case_fk": ("booking_drafts", None),
        "booking_draft_committed_allocation_fk": ("booking_drafts", True),
        "booking_hold_matches_draft_fk": ("booking_holds", True),
        "booking_allocation_matches_draft_fk": ("booking_allocations", True),
    }
    for name, (table_name, deferred) in expected.items():
        constraint = next(c for c in metadata.tables[f"clinic.{table_name}"].foreign_key_constraints if c.name == name)
        assert len(constraint.elements) > 1
        assert constraint.deferrable is deferred
        assert constraint.initially == ("DEFERRED" if deferred else None)
        assert name in "\n".join(upgrade_statements)
    from sqlalchemy import UniqueConstraint
    appointment_unique = [tuple(c.columns.keys()) for c in metadata.tables["clinic.appointments"].constraints if isinstance(c, UniqueConstraint)]
    assert ("appointment_id", "case_id") in appointment_unique


@pytest.fixture
def constrained_sqlite():
    """Ephemeral metadata fixture with FK enforcement, never the clinic DB."""
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4
    from sqlalchemy import create_engine, event
    from healthcare_voice_agent.booking import tables as t
    engine = create_engine("sqlite://", execution_options={"schema_translate_map": {"clinic": None}})
    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")
    metadata.create_all(engine)
    now = datetime(2030, 1, 1, 8, tzinfo=timezone.utc)
    session, other_session, slot, other_slot, draft, request = [uuid4() for _ in range(6)]
    draft_values = dict(draft_id=draft, booking_request_id=request, session_id=session,
        case_id="C1", patient_id="P1", case_context_version=1, slot_id=slot, state="pending",
        details={}, details_hash="0" * 64, hold_expires_at=now + timedelta(seconds=120),
        created_at=now, updated_at=now)
    with engine.begin() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        conn.execute(t.patients.insert(), [dict(patient_id="P1", display_name="Same synthetic name"),
                                        dict(patient_id="P2", display_name="Same synthetic name")])
        conn.execute(t.users.insert(), dict(user_id="U1", role="staff", is_active=True))
        conn.execute(t.clinicians.insert(), dict(clinician_id="CL1", display_name="Synthetic clinician"))
        conn.execute(t.cases.insert(), [dict(case_id="C1", patient_id="P1", status="open"),
                                      dict(case_id="C2", patient_id="P2", status="open")])
        conn.execute(t.clinics.insert(), dict(clinic_id="clinic", timezone="Asia/Kolkata"))
        conn.execute(t.sessions.insert(), [dict(session_id=session, clinic_id="clinic", user_id="U1",
            case_id="C1", case_context_version=1, conversation_version=1, active=True, created_at=now),
            dict(session_id=other_session, clinic_id="clinic", user_id="U1", case_id="C2",
                 case_context_version=1, conversation_version=1, active=True, created_at=now)])
        conn.execute(t.slots.insert(), [dict(slot_id=value, clinic_id="clinic", clinician_id="CL1",
            appointment_type="initial_consultation", starts_at=now + timedelta(days=1, hours=index),
            ends_at=now + timedelta(days=1, hours=index, minutes=30), location="Synthetic clinic", enabled=True)
            for index, value in enumerate((slot, other_slot))])
        conn.execute(t.drafts.insert(), draft_values)
    try:
        yield SimpleNamespace(engine=engine, t=t, now=now, session=session, other_session=other_session,
            slot=slot, other_slot=other_slot, draft=draft, request=request, draft_values=draft_values)
    finally:
        engine.dispose()


def _insert_fixture_appointment(conn, c):
    from datetime import timedelta
    conn.execute(c.t.appointments.insert(), dict(appointment_id="AP1", case_id="C1",
        appointment_type="initial_consultation", starts_at=c.now + timedelta(days=1),
        ends_at=c.now + timedelta(days=1, minutes=30), timezone="Asia/Kolkata", status="confirmed",
        clinician_id="CL1", location="Synthetic clinic", record_version=1, created_at=c.now, updated_at=c.now))


def _mark_fixture_booked(conn, c):
    from datetime import timedelta
    from uuid import uuid4
    conn.execute(c.t.drafts.update().where(c.t.drafts.c.draft_id == c.draft).values(
        state="booked", appointment_id="AP1", readback_version=1,
        readback_at=c.now + timedelta(seconds=10), confirmation_version=2,
        confirmation_event_id=uuid4(), confirmed_at=c.now + timedelta(seconds=11),
        updated_at=c.now + timedelta(seconds=12)))


@pytest.mark.parametrize("patient_id", [None, "missing-patient"])
def test_sqlite_fixture_rejects_missing_or_unknown_draft_patient(constrained_sqlite, patient_id):
    from uuid import uuid4
    from sqlalchemy.exc import IntegrityError
    c = constrained_sqlite
    values = {**c.draft_values, "draft_id": uuid4(), "booking_request_id": uuid4(), "patient_id": patient_id}
    with pytest.raises(IntegrityError):
        with c.engine.begin() as conn:
            conn.execute(c.t.drafts.insert(), values)


@pytest.mark.parametrize("mismatch", ["slot", "expiry"])
def test_sqlite_deferred_hold_must_match_draft_at_commit(constrained_sqlite, mismatch):
    from datetime import timedelta
    from sqlalchemy.exc import IntegrityError
    c = constrained_sqlite
    inserted = False
    with pytest.raises(IntegrityError):
        with c.engine.begin() as conn:
            conn.execute(c.t.holds.insert(), dict(draft_id=c.draft,
                slot_id=c.other_slot if mismatch == "slot" else c.slot,
                expires_at=c.draft_values["hold_expires_at"] + timedelta(seconds=1 if mismatch == "expiry" else 0)))
            inserted = True  # composite FK is deferred, not merely an insert check
    assert inserted


def test_sqlite_current_draft_cannot_belong_to_another_session(constrained_sqlite):
    from sqlalchemy.exc import IntegrityError
    c = constrained_sqlite
    with pytest.raises(IntegrityError):
        with c.engine.begin() as conn:
            conn.execute(c.t.sessions.update().where(c.t.sessions.c.session_id == c.other_session).values(current_draft_id=c.draft))


@pytest.mark.parametrize("missing_half", ["allocation", "booked_outcome"])
def test_sqlite_rejects_a_half_booking_at_commit(constrained_sqlite, missing_half):
    from sqlalchemy.exc import IntegrityError
    c = constrained_sqlite
    with pytest.raises(IntegrityError):
        with c.engine.begin() as conn:
            _insert_fixture_appointment(conn, c)
            if missing_half == "allocation":
                _mark_fixture_booked(conn, c)
            else:
                conn.execute(c.t.allocations.insert(), dict(slot_id=c.slot, draft_id=c.draft,
                    booking_request_id=c.request, appointment_id="AP1", created_at=c.now))


def test_sqlite_atomic_booking_succeeds_and_keeps_original_patient_identity(constrained_sqlite):
    c = constrained_sqlite
    with c.engine.begin() as conn:
        _insert_fixture_appointment(conn, c)
        # Forward and reciprocal composite FKs are deferred, permitting this order.
        conn.execute(c.t.allocations.insert(), dict(slot_id=c.slot, draft_id=c.draft,
            booking_request_id=c.request, appointment_id="AP1", created_at=c.now))
        _mark_fixture_booked(conn, c)
        conn.execute(c.t.sessions.update().where(c.t.sessions.c.session_id == c.session).values(current_draft_id=c.draft))
    with c.engine.begin() as conn:
        conn.execute(c.t.cases.update().where(c.t.cases.c.case_id == "C1").values(patient_id="P2"))
        persisted = conn.execute(c.t.drafts.select().where(c.t.drafts.c.draft_id == c.draft)).mappings().one()
        assert persisted["patient_id"] == "P1"
        assert persisted["state"] == "booked"
        assert conn.exec_driver_sql("PRAGMA foreign_key_check").all() == []
