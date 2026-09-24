"""Offline SQLite coverage for bounded canonical seeded-booking provenance."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, event, func, insert, select

from scripts import seed
from healthcare_voice_agent.booking import tables as t
from healthcare_voice_agent.booking.service import BookingService
from healthcare_voice_agent.demo.seeded_bookings import (
    SeededBookingLinkError, canonical_seeded_booking, link_existing_seeded_booking,
)

UTC = timezone.utc


@dataclass
class Harness:
    engine: object
    canonical: object
    service: BookingService
    context: object
    replacement_slot_id: UUID


def _canonical():
    dataset = seed.load_dataset(seed.DEFAULT_FIXTURE_DIR)
    seed.validate_dataset(dataset)
    return canonical_seeded_booking(dataset)


@pytest.fixture
def linked(tmp_path):
    canonical = _canonical()
    engine = create_engine(f"sqlite:///{tmp_path / 'seeded-booking.sqlite'}",
        connect_args={"check_same_thread": False, "timeout": 5}).execution_options(
            schema_translate_map={"clinic": None})

    @event.listens_for(engine, "connect")
    def foreign_keys(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    t.metadata.create_all(engine)
    replacement_slot_id = uuid4()
    with engine.begin() as connection:
        connection.execute(insert(t.guard).values(id=1))
        connection.execute(insert(t.patients).values(
            patient_id=canonical.patient_id, display_name=canonical.patient_display_name))
        connection.execute(insert(t.clinicians).values(
            clinician_id=canonical.clinician_id, display_name=canonical.clinician_display_name))
        connection.execute(insert(t.users).values(
            user_id=canonical.staff_id, is_active=True, role="staff"))
        connection.execute(insert(t.cases).values(
            case_id=canonical.case_id, patient_id=canonical.patient_id, status="open"))
        connection.execute(insert(t.clinics).values(
            clinic_id=canonical.clinic_id, timezone=canonical.timezone))
        connection.execute(insert(t.appointments).values(
            appointment_id=canonical.appointment_id, case_id=canonical.case_id,
            appointment_type=canonical.appointment_type, starts_at=canonical.starts_at,
            ends_at=canonical.ends_at, timezone=canonical.timezone, status=canonical.status,
            clinician_id=canonical.clinician_id, location=canonical.location,
            record_version=canonical.record_version, created_at=canonical.created_at,
            updated_at=canonical.updated_at))
        connection.execute(insert(t.slots).values(
            slot_id=replacement_slot_id, clinic_id=canonical.clinic_id,
            appointment_type=canonical.appointment_type,
            starts_at=canonical.starts_at + timedelta(days=1),
            ends_at=canonical.ends_at + timedelta(days=1),
            clinician_id=canonical.clinician_id, location=canonical.location, enabled=True))

    result = link_existing_seeded_booking(engine, canonical)
    assert result["status"] == "linked"
    clock = lambda: datetime(2026, 9, 23, 0, 0, tzinfo=UTC)
    service = BookingService(engine, test_clock=clock)
    context = service.set_confirmed_case(
        service.open_session(clinic_id=canonical.clinic_id, user_id=canonical.staff_id),
        canonical.case_id)
    yield Harness(engine, canonical, service, context, replacement_slot_id)
    engine.dispose()


def _count(engine, table):
    with engine.connect() as connection:
        return connection.execute(select(func.count()).select_from(table)).scalar_one()


def _appointment(engine, appointment_id):
    with engine.connect() as connection:
        return dict(connection.execute(select(t.appointments).where(
            t.appointments.c.appointment_id == appointment_id)).mappings().one())


def test_canonical_appointment_becomes_visible_without_changing_source_and_repeat_is_noop(linked):
    canonical = linked.canonical
    appointment = _appointment(linked.engine, canonical.appointment_id)
    assert appointment["status"] == "scheduled" and appointment["record_version"] == 1
    assert appointment["starts_at"] == canonical.starts_at.replace(tzinfo=None)
    assert appointment["ends_at"] == canonical.ends_at.replace(tzinfo=None)
    assert appointment["clinician_id"] == canonical.clinician_id
    assert appointment["created_at"] == canonical.created_at.replace(tzinfo=None)
    assert appointment["updated_at"] == canonical.updated_at.replace(tzinfo=None)
    assert _count(linked.engine, t.appointments) == 1

    listed = linked.service.list_bookings(linked.context)
    assert [row["appointment_id"] for row in listed] == [canonical.appointment_id]
    assert listed[0]["record_version"] == 1 and listed[0]["status"] == "scheduled"
    before = {table.name: _count(linked.engine, table) for table in (
        t.sessions, t.slots, t.drafts, t.allocations, t.offers, t.holds)}
    result = link_existing_seeded_booking(linked.engine, canonical)
    after = {table.name: _count(linked.engine, table) for table in (
        t.sessions, t.slots, t.drafts, t.allocations, t.offers, t.holds)}
    assert result["status"] == "already_linked" and before == after
    assert after[t.offers.name] == after[t.holds.name] == 0

    with linked.engine.connect() as connection:
        slot = connection.execute(select(t.slots).where(
            t.slots.c.slot_id == canonical.slot_id)).mappings().one()
        draft = connection.execute(select(t.drafts).where(
            t.drafts.c.draft_id == canonical.draft_id)).mappings().one()
    assert slot["enabled"] is False
    assert draft["details"]["provenance"] == {
        "kind": "bootstrap_existing_receipt",
        "source": "canonical_synthetic_fixture",
        "source_appointment_id": canonical.appointment_id,
        "created_new_appointment": False,
        "user_voice_booking": False,
        "audio_confirmation_recorded": False,
    }
    assert draft["readback_version"] is None and draft["readback_at"] is None


def test_existing_service_can_authorize_and_cancel_linked_booking(linked):
    source = linked.service.list_bookings(linked.context)[0]
    intent = linked.service.prepare_cancellation(linked.context,
        appointment_id=source["appointment_id"], expected_record_version=source["record_version"])
    details = {"appointment": intent["appointment"]}
    linked.service.authorize_direct_request(linked.context, operation="cancel",
        request_id=UUID(intent["cancellation_request_id"]), details=details,
        request_event_id=uuid4())
    result = linked.service.cancel_appointment(linked.context,
        cancellation_request_id=UUID(intent["cancellation_request_id"]))
    assert result["appointment"]["status"] == "cancelled"
    assert result["appointment"]["record_version"] == 2
    assert linked.service.list_bookings(linked.context) == []
    assert link_existing_seeded_booking(linked.engine, linked.canonical)["status"] == "already_linked"


def test_existing_service_can_authorize_and_reschedule_linked_booking(linked):
    source = linked.service.list_bookings(linked.context)[0]
    local_day = (linked.canonical.starts_at + timedelta(days=1)).astimezone(
        __import__("zoneinfo").ZoneInfo(linked.canonical.timezone)).date().isoformat()
    found = linked.service.invoke("find_available_slots", {
        "appointment_type": linked.canonical.appointment_type,
        "start_date": local_day, "end_date": local_day,
        "earliest_start_time": None, "latest_start_time": None,
        "clinician_id": linked.canonical.clinician_id, "max_results": 5,
    }, context=linked.context)
    assert found["status"] == "ok"
    assert str(linked.replacement_slot_id) in {row["slot_id"] for row in found["data"]["slots"]}
    draft = linked.service.prepare_reschedule(linked.context,
        appointment_id=source["appointment_id"], expected_record_version=source["record_version"],
        slot_id=linked.replacement_slot_id)
    details = {name: draft[name] for name in ("case_id", "patient_display_name", "slot")}
    linked.service.authorize_direct_request(linked.context, operation="reschedule",
        request_id=UUID(draft["draft_id"]), details=details, request_event_id=uuid4())
    result = linked.service.reschedule_appointment(linked.context, draft_id=UUID(draft["draft_id"]))
    assert result["rescheduled_from"] == linked.canonical.appointment_id
    assert result["appointment"]["appointment_id"] != linked.canonical.appointment_id
    assert link_existing_seeded_booking(linked.engine, linked.canonical)["status"] == "already_linked"


def test_conflicting_linkage_fails_closed_and_rolls_back(tmp_path):
    canonical = _canonical()
    engine = create_engine(f"sqlite:///{tmp_path / 'conflict.sqlite'}").execution_options(
        schema_translate_map={"clinic": None})
    event.listen(engine, "connect", lambda dbapi, _: dbapi.execute("PRAGMA foreign_keys=ON"))
    t.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(t.guard).values(id=1))
        connection.execute(insert(t.patients).values(
            patient_id=canonical.patient_id, display_name=canonical.patient_display_name))
        connection.execute(insert(t.clinicians).values(
            clinician_id=canonical.clinician_id, display_name=canonical.clinician_display_name))
        connection.execute(insert(t.users).values(
            user_id=canonical.staff_id, is_active=True, role="staff"))
        connection.execute(insert(t.cases).values(
            case_id=canonical.case_id, patient_id=canonical.patient_id, status="open"))
        connection.execute(insert(t.clinics).values(
            clinic_id=canonical.clinic_id, timezone=canonical.timezone))
        connection.execute(insert(t.appointments).values(
            appointment_id=canonical.appointment_id, case_id=canonical.case_id,
            appointment_type=canonical.appointment_type, starts_at=canonical.starts_at,
            ends_at=canonical.ends_at, timezone=canonical.timezone, status=canonical.status,
            clinician_id=canonical.clinician_id, location=canonical.location,
            record_version=canonical.record_version, created_at=canonical.created_at,
            updated_at=canonical.updated_at))
        connection.execute(insert(t.slots).values(
            slot_id=canonical.slot_id, clinic_id=canonical.clinic_id,
            appointment_type=canonical.appointment_type, starts_at=canonical.starts_at,
            ends_at=canonical.ends_at, clinician_id=canonical.clinician_id,
            location="Conflicting room", enabled=True))
    before = _appointment(engine, canonical.appointment_id)
    with pytest.raises(SeededBookingLinkError, match="Conflicting"):
        link_existing_seeded_booking(engine, canonical)
    assert _appointment(engine, canonical.appointment_id) == before
    assert _count(engine, t.sessions) == _count(engine, t.drafts) == _count(engine, t.allocations) == 0
    assert _count(engine, t.slots) == 1
    engine.dispose()


@pytest.mark.parametrize('target,canonical_rows', [(None, None), (lambda *a: None, None), (None, lambda *a: None)])
def test_postgres_core_cannot_connect_without_both_trusted_validators(target, canonical_rows):
    from types import SimpleNamespace
    def no_connection():
        pytest.fail('Missing PostgreSQL safety callbacks must fail before connecting')
    engine = SimpleNamespace(dialect=SimpleNamespace(name='postgresql'), connect=no_connection)
    with pytest.raises(SeededBookingLinkError, match='validators'):
        link_existing_seeded_booking(engine, _canonical(), validate_target=target,
                                     validate_canonical_rows=canonical_rows)
