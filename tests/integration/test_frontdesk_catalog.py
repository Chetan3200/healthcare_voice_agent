"""Offline SQLite coverage for the trusted read-only booking catalog.

No live database, provider, server, dependency installation, or .env is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, event, insert, select, update

from healthcare_voice_agent.booking import tables as t
from healthcare_voice_agent.booking.scheduling import ScheduleRule
from healthcare_voice_agent.booking.service import BookingError, BookingService
from healthcare_voice_agent.tools.frontdesk_contracts import FindAvailableSlotsArgs

UTC = timezone.utc
DAY = date(2026, 10, 5)


@dataclass
class Clock:
    now: datetime = datetime(2026, 10, 4, 12, tzinfo=UTC)

    def __call__(self):
        return self.now


def rule(clinician, opens, closes, *, kind="initial_consultation", location=None):
    return ScheduleRule(kind, clinician, (0, 1, 2, 3, 4, 5, 6), opens, closes, 30,
                        location or f"Room {clinician[-1]}")


def restrictions(**overrides):
    values = {"appointment_type": "initial_consultation", "start_date": DAY.isoformat(),
              "end_date": DAY.isoformat(), "earliest_start_time": None,
              "latest_start_time": None, "clinician_id": None, "max_results": 3}
    values.update(overrides)
    return FindAvailableSlotsArgs.model_validate(values)


def good(result):
    assert result["status"] == "ok", result
    return result["data"]


@dataclass
class CatalogHarness:
    engine: object
    service: BookingService
    clock: Clock
    context: object

    def rows(self, table):
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(table)).mappings()]

    def context_for(self, case_id):
        context = self.service.open_session(clinic_id="CLINIC", user_id="STAFF")
        return self.service.set_confirmed_case(context, case_id)

    def search(self, context, **overrides):
        return self.service.invoke("find_available_slots",
            restrictions(**overrides).model_dump(mode="json"), context=context)

    def prepare(self, context, clinician_id):
        slot = good(self.search(context, clinician_id=clinician_id, max_results=1))["slots"][0]
        return good(self.service.invoke("prepare_booking",
            {"case_id": context.case_id, "slot_id": slot["slot_id"]}, context=context))

    def book(self, context, clinician_id):
        draft = self.prepare(context, clinician_id)
        details = {name: draft[name] for name in ("case_id", "patient_display_name", "slot")}
        digest = self.service.mark_readback_presented(
            context, draft_id=UUID(draft["draft_id"]), details=details)
        context = self.service.advance_conversation(context)
        self.service.record_confirmation(context, draft_id=UUID(draft["draft_id"]),
            details_hash=digest, confirmation_event_id=uuid4())
        result = self.service.invoke("book_appointment",
            {"case_id": context.case_id, "draft_id": draft["draft_id"]}, context=context)
        assert result["status"] == "ok", result
        return context


@pytest.fixture
def clinic(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'catalog.sqlite'}",
        connect_args={"check_same_thread": False, "timeout": 5}).execution_options(
        schema_translate_map={"clinic": None})

    @event.listens_for(engine, "connect")
    def foreign_keys(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    t.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(t.guard).values(id=1))
        connection.execute(insert(t.patients), [
            {"patient_id": "PAT-A", "display_name": "Synthetic Patient A"},
            {"patient_id": "PAT-B", "display_name": "Synthetic Patient B"},
            {"patient_id": "PAT-C", "display_name": "Synthetic Patient C"},
        ])
        connection.execute(insert(t.cases), [
            {"case_id": "CASE-A", "patient_id": "PAT-A", "status": "open"},
            {"case_id": "CASE-B", "patient_id": "PAT-B", "status": "open"},
            {"case_id": "CASE-C", "patient_id": "PAT-C", "status": "open"},
        ])
        connection.execute(insert(t.clinicians), [
            {"clinician_id": "CLIN-A", "display_name": "Synthetic Dr A"},
            {"clinician_id": "CLIN-B", "display_name": "Synthetic Dr B"},
            {"clinician_id": "CLIN-C", "display_name": "Synthetic Dr C"},
            {"clinician_id": "CLIN-D", "display_name": "Synthetic Dr D"},
            {"clinician_id": "CLIN-PAST", "display_name": "Synthetic Dr Past"},
            {"clinician_id": "CLIN-OFF", "display_name": "Synthetic Dr Off"},
            {"clinician_id": "CLIN-FOREIGN", "display_name": "Synthetic Dr Foreign"},
        ])
        connection.execute(insert(t.users).values(user_id="STAFF", is_active=True, role="staff"))

    clock = Clock()
    service = BookingService(engine, test_clock=clock)
    service.register_clinic("CLINIC", "Asia/Kolkata")
    service.register_clinic("FOREIGN", "Asia/Kolkata")
    service.publish_schedule("CLINIC", (
        rule("CLIN-A", time(9), time(10, 30)),
        rule("CLIN-A", time(14), time(14, 30), kind="imaging"),
        rule("CLIN-B", time(11), time(11, 30)),
        rule("CLIN-C", time(12), time(12, 30)),
        rule("CLIN-D", time(13), time(13, 30)),
        rule("CLIN-OFF", time(15), time(15, 30), kind="other"),
    ), DAY, DAY)
    service.publish_schedule("CLINIC", (
        rule("CLIN-PAST", time(9), time(9, 30), kind="physiotherapy"),
    ), DAY - timedelta(days=2), DAY - timedelta(days=2))
    service.publish_schedule("FOREIGN", (
        rule("CLIN-FOREIGN", time(9), time(9, 30), kind="fracture_follow_up"),
    ), DAY, DAY)
    with engine.begin() as connection:
        connection.execute(update(t.slots).where(
            t.slots.c.clinic_id == "CLINIC", t.slots.c.clinician_id == "CLIN-OFF").values(enabled=False))
    context = service.set_confirmed_case(
        service.open_session(clinic_id="CLINIC", user_id="STAFF"), "CASE-A")
    yield CatalogHarness(engine, service, clock, context)
    engine.dispose()


def test_directory_shape_canonical_types_and_future_clinic_scope(clinic):
    catalog = clinic.service.booking_catalog(clinic.context)

    assert json.loads(json.dumps(catalog)) == catalog
    assert catalog == {
        "clinic_id": "CLINIC",
        "timezone": "Asia/Kolkata",
        "as_of": "2026-10-04T12:00:00+00:00",
        "appointment_types": ["initial_consultation", "imaging"],
        "clinicians": [
            {"clinician_id": "CLIN-A", "display_name": "Synthetic Dr A",
             "appointment_types": ["initial_consultation", "imaging"]},
            {"clinician_id": "CLIN-B", "display_name": "Synthetic Dr B",
             "appointment_types": ["initial_consultation"]},
            {"clinician_id": "CLIN-C", "display_name": "Synthetic Dr C",
             "appointment_types": ["initial_consultation"]},
            {"clinician_id": "CLIN-D", "display_name": "Synthetic Dr D",
             "appointment_types": ["initial_consultation"]},
        ],
        "availability_checked": False,
        "search_scope": None,
        "available_clinicians": None,
    }
    serialized = json.dumps(catalog)
    assert all(value not in serialized for value in (
        "CASE-A", "PAT-A", "Synthetic Patient A", "CLIN-PAST", "CLIN-OFF", "CLIN-FOREIGN"))


def test_availability_enumerates_fourth_clinician_beyond_finder_cutoff(clinic):
    scope = restrictions(max_results=3)
    finder = good(clinic.search(clinic.context, max_results=3))
    assert [slot["clinician"]["clinician_id"] for slot in finder["slots"]] == [
        "CLIN-A", "CLIN-A", "CLIN-A"]

    catalog = clinic.service.booking_catalog(clinic.context, restrictions=scope)
    assert catalog["availability_checked"] is True
    assert catalog["search_scope"] == scope.model_dump(mode="json")
    assert [item["clinician_id"] for item in catalog["available_clinicians"]] == [
        "CLIN-A", "CLIN-B", "CLIN-C", "CLIN-D"]
    assert all(item["appointment_types"] == ["initial_consultation"]
               for item in catalog["available_clinicians"])


def test_exact_location_clinician_and_daily_bounds_and_zero_preserve_directory(clinic):
    raw = restrictions(clinician_id="CLIN-D", earliest_start_time="13:00",
                       latest_start_time="13:00").model_dump(mode="json")
    raw.update(location="Room D", clinic_id="CLINIC", case_id="CASE-A")
    exact = clinic.service.booking_catalog(clinic.context, restrictions=raw)
    assert exact["search_scope"] == {
        **restrictions(clinician_id="CLIN-D", earliest_start_time="13:00",
                       latest_start_time="13:00").model_dump(mode="json"),
        "location": "Room D",
    }
    assert [item["clinician_id"] for item in exact["available_clinicians"]] == ["CLIN-D"]

    wrong_location = dict(raw, location="Room C")
    assert clinic.service.booking_catalog(
        clinic.context, restrictions=wrong_location)["available_clinicians"] == []
    wrong_time = dict(raw, earliest_start_time="12:59", latest_start_time="12:59")
    assert clinic.service.booking_catalog(
        clinic.context, restrictions=wrong_time)["available_clinicians"] == []
    imaging = clinic.service.booking_catalog(clinic.context, restrictions=restrictions(
        appointment_type="imaging", clinician_id="CLIN-A", max_results=1))
    assert [item["clinician_id"] for item in imaging["available_clinicians"]] == ["CLIN-A"]

    empty = clinic.service.booking_catalog(clinic.context, restrictions=restrictions(
        earliest_start_time="08:00", latest_start_time="08:00"))
    assert empty["availability_checked"] is True
    assert empty["available_clinicians"] == []
    assert empty["appointment_types"] == exact["appointment_types"]
    assert empty["clinicians"] == exact["clinicians"]


@pytest.mark.parametrize(("extra", "code"), [
    ({"clinic_id": "FOREIGN"}, "CLINIC_CONTEXT_MISMATCH"),
    ({"clinic_id": None}, "CLINIC_CONTEXT_MISMATCH"),
    ({"case_id": "CASE-B"}, "CASE_CONTEXT_MISMATCH"),
    ({"case_id": None}, "CASE_CONTEXT_MISMATCH"),
])
def test_restriction_scope_must_match_context(clinic, extra, code):
    raw = restrictions().model_dump(mode="json") | extra
    with pytest.raises(BookingError) as caught:
        clinic.service.booking_catalog(clinic.context, restrictions=raw)
    assert caught.value.code == code


def test_busy_hold_allocation_semantics_and_catalog_has_no_state_side_effects(clinic):
    now = clinic.clock.now
    with clinic.engine.begin() as connection:
        connection.execute(insert(t.appointments).values(
            appointment_id="LEGACY-B", case_id="CASE-B", appointment_type="initial_consultation",
            starts_at=datetime(2026, 10, 5, 5, 30, tzinfo=UTC),
            ends_at=datetime(2026, 10, 5, 6, 0, tzinfo=UTC), timezone="Asia/Kolkata",
            status="confirmed", clinician_id="CLIN-B", location="Room B", record_version=1,
            created_at=now, updated_at=now))

    hold_context = clinic.context_for("CASE-B")
    clinic.prepare(hold_context, "CLIN-C")
    booked_context = clinic.context_for("CASE-C")
    clinic.book(booked_context, "CLIN-D")

    watched = (t.offers, t.holds, t.drafts, t.sessions)
    before = {table.name: clinic.rows(table) for table in watched}
    catalog = clinic.service.booking_catalog(clinic.context, restrictions=restrictions())
    after = {table.name: clinic.rows(table) for table in watched}

    assert [item["clinician_id"] for item in catalog["available_clinicians"]] == ["CLIN-A"]
    assert after == before

    clinic.clock.now += timedelta(seconds=121)
    expired_before = {table.name: clinic.rows(table) for table in watched}
    later = clinic.service.booking_catalog(clinic.context, restrictions=restrictions())
    expired_after = {table.name: clinic.rows(table) for table in watched}
    assert "CLIN-C" in {item["clinician_id"] for item in later["available_clinicians"]}
    assert expired_after == expired_before
    assert expired_after[t.holds.name]
    assert any(row["state"] == "pending" for row in expired_after[t.drafts.name])


def test_catalog_matches_finder_by_not_adding_same_patient_conflicts(clinic):
    now = clinic.clock.now
    with clinic.engine.begin() as connection:
        connection.execute(insert(t.appointments).values(
            appointment_id="PATIENT-OVERLAP", case_id="CASE-A",
            appointment_type="initial_consultation",
            starts_at=datetime(2026, 10, 5, 5, 30, tzinfo=UTC),
            ends_at=datetime(2026, 10, 5, 6, 0, tzinfo=UTC), timezone="Asia/Kolkata",
            status="confirmed", clinician_id="CLIN-A", location="Legacy", record_version=1,
            created_at=now, updated_at=now))

    finder = good(clinic.search(clinic.context, clinician_id="CLIN-B", max_results=1))
    catalog = clinic.service.booking_catalog(clinic.context, restrictions=restrictions(
        clinician_id="CLIN-B", max_results=1))
    assert finder["slots"][0]["clinician"]["clinician_id"] == "CLIN-B"
    assert [item["clinician_id"] for item in catalog["available_clinicians"]] == ["CLIN-B"]
