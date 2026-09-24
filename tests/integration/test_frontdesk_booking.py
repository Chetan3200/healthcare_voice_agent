"""Isolated transactional tests, not live PostgreSQL/constraint validation.

SQLite uses BEGIN IMMEDIATE and test-only query-mapping DDL. Production Alembic
DDL, PostgreSQL locks and deferred constraints require their own live checkpoint.
No .env, real patient data, network, voice provider or model calls are used.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
import json
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, event, func, insert, select, update
from sqlalchemy.exc import OperationalError

from healthcare_voice_agent.booking import tables as t
from healthcare_voice_agent.booking.scheduling import ScheduleRule
from healthcare_voice_agent.booking.service import BookingError, BookingService

UTC = timezone.utc
DAY = date(2026, 10, 5)  # Monday


@dataclass
class Clock:
    now: datetime = datetime(2026, 10, 4, 12, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


@dataclass
class Harness:
    engine: object
    service: BookingService
    clock: Clock
    context: object
    slot_ids: list

    def new_context(self, case_id="CASE-B"):
        context = self.service.open_session(clinic_id="CLINIC", user_id="STAFF")
        return self.service.set_confirmed_case(context, case_id) if case_id else context

    def search(self, context=None, **kwargs):
        args = {"appointment_type": "initial_consultation", "start_date": DAY.isoformat(),
                "end_date": DAY.isoformat(), "earliest_start_time": None,
                "latest_start_time": None, "clinician_id": "CLIN-A", "max_results": 5, **kwargs}
        return self.service.invoke("find_available_slots", args, context=context or self.context)

    def prepare(self, slot_id=None, context=None):
        context = context or self.context
        if slot_id is None:
            slot_id = good(self.search(context))["slots"][0]["slot_id"]
        return self.service.invoke("prepare_booking", {"case_id": context.case_id, "slot_id": str(slot_id)}, context=context)

    def confirm(self, draft, context=None):
        context = context or self.context
        details = {name: draft[name] for name in ("case_id", "patient_display_name", "slot")}
        digest = self.service.mark_readback_presented(context, draft_id=UUID(draft["draft_id"]), details=details)
        context = self.service.advance_conversation(context)
        self.service.record_confirmation(context, draft_id=UUID(draft["draft_id"]),
                                         details_hash=digest, confirmation_event_id=uuid4())
        if context.session_id == self.context.session_id:
            self.context = context
        return context

    def book(self, draft, context=None):
        context = context or self.context
        return self.service.invoke("book_appointment", {"case_id": context.case_id, "draft_id": draft["draft_id"]}, context=context)

    def status(self, draft, context=None):
        context = context or self.context
        return self.service.invoke("get_booking_status", {"case_id": context.case_id,
            "booking_request_id": draft["booking_request_id"]}, context=context)

    def count(self, table):
        with self.engine.connect() as connection:
            return connection.execute(select(func.count()).select_from(table)).scalar_one()

    def rows(self, table):
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(table)).mappings()]


def good(result):
    assert result["status"] == "ok", result
    assert result["error"] is None
    return result["data"]


def error(result, code):
    assert result["status"] == "error", result
    assert result["data"] is None
    assert result["error"]["code"] == code, result
    assert result["error"]["candidates"] == []


def rule(clinician="CLIN-A", *, opens=time(9), closes=time(12), duration=30, kind="initial_consultation"):
    return ScheduleRule(kind, clinician, (0, 1, 2, 3, 4, 5, 6), opens, closes, duration, "Synthetic Front Desk")


@pytest.fixture
def clinic(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'isolated-booking.sqlite'}",
                           connect_args={"check_same_thread": False, "timeout": 5}).execution_options(
                               schema_translate_map={"clinic": None})
    @event.listens_for(engine, "connect")
    def foreign_keys(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    # Only test fixtures use create_all; never a production schema mechanism.
    t.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(t.guard).values(id=1))
        connection.execute(insert(t.patients), [
            {"patient_id": "PAT-A", "display_name": "Synthetic Alex"},
            {"patient_id": "PAT-B", "display_name": "Synthetic Blair"},
        ])
        connection.execute(insert(t.clinicians), [
            {"clinician_id": "CLIN-A", "display_name": "Synthetic Dr A"},
            {"clinician_id": "CLIN-B", "display_name": "Synthetic Dr B"},
        ])
        connection.execute(insert(t.users).values(user_id="STAFF", is_active=True, role="staff"))
        connection.execute(insert(t.cases), [
            {"case_id": "CASE-A", "patient_id": "PAT-A", "status": "open"},
            {"case_id": "CASE-A2", "patient_id": "PAT-A", "status": "open"},
            {"case_id": "CASE-B", "patient_id": "PAT-B", "status": "open"},
        ])
    clock = Clock()
    service = BookingService(engine, test_clock=clock)
    service.register_clinic("CLINIC", "Asia/Kolkata")
    ids = service.publish_schedule("CLINIC", (rule(), rule("CLIN-B")), DAY, DAY)
    context = service.open_session(clinic_id="CLINIC", user_id="STAFF")
    context = service.set_confirmed_case(context, "CASE-A")
    yield Harness(engine, service, clock, context, ids)
    engine.dispose()


def test_happy_path_confirmation_booking_idempotency_and_recovery(clinic):
    draft = good(clinic.prepare())
    assert clinic.count(t.appointments) == 0
    assert draft["draft_id"] != draft["booking_request_id"]
    clinic.confirm(draft)
    first = clinic.book(draft)
    committed = good(first)
    assert committed["appointment"]["status"] == "confirmed"
    assert committed["booking_request_id"] == draft["booking_request_id"]
    second = clinic.book(draft)
    assert good(second) == committed
    assert first["meta"]["request_id"] != second["meta"]["request_id"]
    assert first["meta"]["request_id"] != committed["booking_request_id"]
    assert clinic.count(t.appointments) == clinic.count(t.allocations) == 1
    assert clinic.count(t.holds) == 0
    clinic.clock.advance(121)
    status = good(clinic.status(draft))
    assert status["booking_state"] == "booked"
    assert status["appointment"] == committed["appointment"]
    assert good(clinic.book(draft)) == committed


def test_find_applies_daily_inclusive_time_window_order_and_has_more(clinic):
    clinic.service.publish_schedule("CLINIC", (rule(),), DAY + timedelta(days=1), DAY + timedelta(days=1))
    result = good(clinic.search(end_date=(DAY + timedelta(days=1)).isoformat(),
                               earliest_start_time="09:30", latest_start_time="10:00", max_results=3))
    assert [slot["starts_at"] for slot in result["slots"]] == [
        "2026-10-05T09:30:00+05:30", "2026-10-05T10:00:00+05:30", "2026-10-06T09:30:00+05:30"]
    assert result["has_more"] is True
    assert all(slot["timezone"] == "Asia/Kolkata" for slot in result["slots"])
    assert clinic.count(t.holds) == clinic.count(t.drafts) == clinic.count(t.appointments) == 0


def test_empty_search_is_success_and_search_needs_no_case(clinic):
    context = clinic.new_context(None)
    assert good(clinic.search(context))["slots"]
    result = clinic.search(context, appointment_type="imaging")
    assert good(result)["slots"] == []
    assert result["data"]["has_more"] is False
    assert result["meta"]["case_context_version"] is None


def test_schedule_publication_is_idempotent_and_rejects_silent_relocation(clinic):
    assert clinic.service.publish_schedule("CLINIC", (rule(), rule("CLIN-B")), DAY, DAY) == clinic.slot_ids
    assert clinic.count(t.slots) == 12
    with pytest.raises(ValueError, match="immutable"):
        clinic.service.publish_schedule("CLINIC", (replace(rule(), location="Changed Room"),), DAY, DAY)
    assert clinic.count(t.slots) == 12


@pytest.mark.parametrize("field", ["earliest_start_time", "latest_start_time", "clinician_id"])
def test_nullable_search_fields_are_required(clinic, field):
    args = {"appointment_type": "initial_consultation", "start_date": DAY.isoformat(), "end_date": DAY.isoformat(),
            "earliest_start_time": None, "latest_start_time": None, "clinician_id": None, "max_results": 5}
    del args[field]
    error(clinic.service.invoke("find_available_slots", args, context=clinic.context), "INVALID_ARGUMENT")
    assert clinic.count(t.offers) == 0


def test_prepare_retry_keeps_original_ids_and_expiry(clinic):
    slot = good(clinic.search())["slots"][0]
    draft = good(clinic.prepare(slot["slot_id"]))
    clinic.clock.advance(50)
    assert good(clinic.prepare(slot["slot_id"])) == draft
    assert clinic.count(t.drafts) == clinic.count(t.holds) == 1


def test_slot_must_have_been_offered_to_this_session(clinic):
    error(clinic.prepare(clinic.slot_ids[0]), "SLOT_NOT_OFFERED")
    assert clinic.count(t.drafts) == 0


def test_preparation_requires_confirmed_case(clinic):
    context = clinic.new_context(None)
    slot = good(clinic.search(context))["slots"][0]
    error(clinic.service.invoke("prepare_booking", {"case_id": "CASE-A", "slot_id": slot["slot_id"]}, context=context), "IDENTITY_UNRESOLVED")


def test_competing_cases_cannot_hold_same_slot(clinic):
    other = clinic.new_context()
    slot = good(clinic.search())["slots"][0]
    good(clinic.search(other))
    good(clinic.prepare(slot["slot_id"]))
    error(clinic.prepare(slot["slot_id"], other), "SLOT_UNAVAILABLE")
    assert clinic.count(t.holds) == 1


def test_new_selection_supersedes_old_and_releases_its_hold(clinic):
    slots = good(clinic.search())["slots"]
    first = good(clinic.prepare(slots[0]["slot_id"]))
    second = good(clinic.prepare(slots[1]["slot_id"]))
    assert second["draft_id"] != first["draft_id"]
    error(clinic.book(first), "DRAFT_SUPERSEDED")
    status = good(clinic.status(first))
    assert status["booking_state"] == "failed"
    assert status["failure"]["code"] == "DRAFT_SUPERSEDED"
    assert [str(row["draft_id"]) for row in clinic.rows(t.holds)] == [second["draft_id"]]
    assert slots[0]["slot_id"] in {slot["slot_id"] for slot in good(clinic.search())["slots"]}


def test_expiry_boundary_is_exact_and_status_does_not_mutate(clinic):
    draft = good(clinic.prepare())
    clinic.clock.advance(119)
    assert good(clinic.status(draft))["booking_state"] == "pending"
    before = clinic.rows(t.drafts), clinic.rows(t.holds), clinic.rows(t.appointments)
    clinic.clock.advance(1)
    assert good(clinic.status(draft))["booking_state"] == "expired"
    assert (clinic.rows(t.drafts), clinic.rows(t.holds), clinic.rows(t.appointments)) == before
    error(clinic.book(draft), "HOLD_EXPIRED")
    assert clinic.count(t.holds) == clinic.count(t.appointments) == 0
    new_draft = good(clinic.prepare(draft["slot"]["slot_id"]))
    assert new_draft["draft_id"] != draft["draft_id"]


def test_no_confirmation_argument_or_booking_without_consent(clinic):
    draft = good(clinic.prepare())
    error(clinic.book(draft), "CONFIRMATION_REQUIRED")
    error(clinic.service.invoke("book_appointment", {"case_id": "CASE-A", "draft_id": draft["draft_id"], "confirmed": True}, context=clinic.context), "INVALID_ARGUMENT")
    assert clinic.count(t.appointments) == clinic.count(t.allocations) == 0


def test_wrong_readback_and_confirmation_before_readback_are_rejected(clinic):
    draft = good(clinic.prepare())
    with pytest.raises(BookingError) as caught:
        clinic.service.mark_readback_presented(clinic.context, draft_id=UUID(draft["draft_id"]), details={"wrong": "details"})
    assert caught.value.code == "CONFIRMATION_STALE"
    with pytest.raises(BookingError) as caught:
        clinic.service.record_confirmation(clinic.context, draft_id=UUID(draft["draft_id"]), details_hash="0" * 64, confirmation_event_id=uuid4())
    assert caught.value.code == "CONFIRMATION_STALE"
    assert clinic.count(t.appointments) == 0


def test_new_user_utterance_invalidates_prior_confirmation(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    old_context = clinic.context
    clinic.context = clinic.service.advance_conversation(clinic.context)
    error(clinic.book(draft, old_context), "SESSION_CONTEXT_MISMATCH")
    error(clinic.book(draft), "CONFIRMATION_STALE")
    assert clinic.count(t.appointments) == 0


def test_selection_correction_releases_hold_and_supersedes_draft(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    clinic.context = clinic.service.advance_conversation(clinic.context, selection_changed=True)
    error(clinic.book(draft), "DRAFT_SUPERSEDED")
    assert clinic.count(t.holds) == clinic.count(t.appointments) == 0


def test_wrong_case_or_session_does_not_reveal_booking(clinic):
    draft = good(clinic.prepare())
    other = clinic.new_context("CASE-A")
    error(clinic.book(draft, other), "RECORD_NOT_FOUND")
    error(clinic.status(draft, other), "RECORD_NOT_FOUND")
    mismatched = clinic.service.invoke("book_appointment", {"case_id": "CASE-B", "draft_id": draft["draft_id"]}, context=clinic.context)
    error(mismatched, "CASE_CONTEXT_MISMATCH")
    assert "Synthetic Alex" not in json.dumps(mismatched)
    unknown = {**draft, "booking_request_id": str(uuid4())}
    assert clinic.status(unknown, other)["error"] == clinic.status(draft, other)["error"]


def test_confirmed_case_change_and_session_close_invalidate_pending_booking(clinic):
    draft = good(clinic.prepare())
    old = clinic.context
    clinic.context = clinic.service.set_confirmed_case(clinic.context, "CASE-B")
    error(clinic.book(draft, old), "SESSION_CONTEXT_MISMATCH")
    assert clinic.count(t.holds) == 0
    next_draft = good(clinic.prepare())
    clinic.service.close_session(clinic.context)
    error(clinic.book(next_draft), "SESSION_CONTEXT_MISMATCH")
    assert clinic.count(t.holds) == 0


def test_patient_conflict_across_cases_and_different_clinicians(clinic):
    other = clinic.new_context("CASE-A2")
    first = good(clinic.search())["slots"][0]
    second = good(clinic.search(other, clinician_id="CLIN-B"))["slots"][0]
    good(clinic.prepare(first["slot_id"]))
    error(clinic.prepare(second["slot_id"], other), "PATIENT_CONFLICT")
    assert clinic.count(t.holds) == 1


def add_legacy(clinic, *, clinician="CLIN-A", case_id="CASE-B", end=True, status="confirmed"):
    start = datetime(2026, 10, 5, 3, 30, tzinfo=UTC)  # 09:00 India
    with clinic.engine.begin() as connection:
        connection.execute(insert(t.appointments).values(appointment_id="LEGACY-" + uuid4().hex,
            case_id=case_id, appointment_type="imaging", starts_at=start,
            ends_at=start + timedelta(minutes=30) if end else None,
            timezone="Asia/Kolkata", status=status, clinician_id=clinician,
            location="Synthetic Legacy", record_version=1, created_at=clinic.clock(), updated_at=clinic.clock()))


def test_legacy_clinician_conflicts_and_adjacent_slots(clinic):
    add_legacy(clinic)
    slots = good(clinic.search())["slots"]
    assert slots[0]["starts_at"] == "2026-10-05T09:30:00+05:30"
    assert len(slots) == 5


def test_unknown_legacy_end_time_blocks_instead_of_guessing_duration(clinic):
    add_legacy(clinic, end=False)
    assert good(clinic.search())["slots"] == []


def test_legacy_patient_overlap_is_checked_during_prepare(clinic):
    add_legacy(clinic, clinician="CLIN-B", case_id="CASE-A2")
    slot = good(clinic.search())["slots"][0]
    error(clinic.prepare(slot["slot_id"]), "PATIENT_CONFLICT")


def test_backend_changes_after_confirmation_persist_definitive_failure(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    with clinic.engine.begin() as connection:
        connection.execute(update(t.slots).where(t.slots.c.slot_id == UUID(draft["slot"]["slot_id"])).values(enabled=False))
    error(clinic.book(draft), "SLOT_UNAVAILABLE")
    status = good(clinic.status(draft))
    assert status["booking_state"] == "failed"
    assert status["failure"]["code"] == "SLOT_UNAVAILABLE"
    assert clinic.count(t.holds) == clinic.count(t.appointments) == 0


def test_concurrent_book_retries_create_exactly_one_appointment(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: clinic.book(draft), range(8)))
    appointments = [good(result)["appointment"]["appointment_id"] for result in results]
    assert len(set(appointments)) == 1
    assert clinic.count(t.appointments) == clinic.count(t.allocations) == 1
    assert clinic.count(t.holds) == 0


def test_competing_concurrent_preparations_have_one_winner(clinic):
    other = clinic.new_context()
    slot = good(clinic.search())["slots"][0]
    good(clinic.search(other))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda ctx: clinic.prepare(slot["slot_id"], ctx), [clinic.context, other]))
    assert sorted(result["status"] for result in results) == ["error", "ok"]
    error(next(result for result in results if result["status"] == "error"), "SLOT_UNAVAILABLE")
    assert clinic.count(t.drafts) == clinic.count(t.holds) == 1


def test_failure_before_commit_rolls_back_all_booking_writes(clinic, monkeypatch):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    original = clinic.service._transaction
    @contextmanager
    def fail_before_commit():
        with original() as connection:
            yield connection
            raise TimeoutError("private-before-commit-marker")
    monkeypatch.setattr(clinic.service, "_transaction", fail_before_commit)
    result = clinic.book(draft)
    error(result, "TIMEOUT")
    assert "private-before-commit-marker" not in json.dumps(result)
    monkeypatch.setattr(clinic.service, "_transaction", original)
    assert clinic.count(t.appointments) == clinic.count(t.allocations) == 0
    assert clinic.count(t.holds) == 1
    assert good(clinic.status(draft))["booking_state"] == "pending"
    good(clinic.book(draft))


def test_lost_response_after_commit_is_recoverable_without_duplicate(clinic, monkeypatch):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    original = clinic.service._transaction
    @contextmanager
    def fail_after_commit():
        with original() as connection:
            yield connection
        raise TimeoutError("private-after-commit-marker")
    monkeypatch.setattr(clinic.service, "_transaction", fail_after_commit)
    result = clinic.book(draft)
    error(result, "TIMEOUT")
    monkeypatch.setattr(clinic.service, "_transaction", original)
    status = good(clinic.status(draft))
    assert status["booking_state"] == "booked"
    assert good(clinic.book(draft))["appointment"] == status["appointment"]
    assert clinic.count(t.appointments) == clinic.count(t.allocations) == 1
    assert "private-after-commit-marker" not in json.dumps(result)


def test_database_failure_is_sanitized_and_not_false_empty_search(clinic, monkeypatch):
    @contextmanager
    def unavailable():
        raise OperationalError("private-sql", {"password": "private-password"}, RuntimeError("private-stack"))
        yield
    monkeypatch.setattr(clinic.service, "_transaction", unavailable)
    result = clinic.search()
    error(result, "BACKEND_UNAVAILABLE")
    assert "private-" not in json.dumps(result)


def test_async_dispatch_uses_same_persisted_contract(clinic):
    draft = good(clinic.prepare())
    result = asyncio.run(clinic.service.invoke_async("get_booking_status", {
        "case_id": "CASE-A", "booking_request_id": draft["booking_request_id"]}, context=clinic.context))
    assert good(result)["booking_state"] == "pending"


def test_restart_can_resume_trusted_session_and_recover_request(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    good(clinic.book(draft))
    restarted = BookingService(clinic.engine, test_clock=clinic.clock)
    context = restarted.resume_context(session_id=clinic.context.session_id, user_id="STAFF")
    result = restarted.invoke("get_booking_status", {"case_id": "CASE-A", "booking_request_id": draft["booking_request_id"]}, context=context)
    assert good(result)["booking_state"] == "booked"
    with pytest.raises(BookingError):
        restarted.resume_context(session_id=context.session_id, user_id="OTHER")


def test_same_selection_prepare_retry_rechecks_patient_conflicts(clinic):
    draft = good(clinic.prepare())
    # A different calendar writer can add an appointment while this draft is
    # held. Even a preparation retry must recheck, not return an invalid draft.
    add_legacy(clinic, clinician="CLIN-B", case_id="CASE-A2")
    error(clinic.prepare(draft["slot"]["slot_id"]), "PATIENT_CONFLICT")
    assert clinic.count(t.allocations) == 0


def test_search_never_offers_a_slot_that_has_already_started(clinic):
    clinic.clock.now = datetime(2026, 10, 5, 3, 30, tzinfo=UTC)
    slots = good(clinic.search())["slots"]
    assert slots[0]["starts_at"] == "2026-10-05T09:30:00+05:30"


def test_overlapping_distinct_slots_share_clinician_capacity(clinic):
    clinic.service.publish_schedule("CLINIC", (rule(kind="imaging", duration=45),), DAY, DAY)
    other = clinic.new_context()
    consultation = good(clinic.search())["slots"][0]
    imaging = good(clinic.search(other, appointment_type="imaging"))["slots"][0]
    assert consultation["slot_id"] != imaging["slot_id"]
    good(clinic.prepare(consultation["slot_id"]))
    error(clinic.prepare(imaging["slot_id"], other), "SLOT_UNAVAILABLE")


def test_closed_case_is_rejected_before_preparation(clinic):
    slot = good(clinic.search())["slots"][0]
    with clinic.engine.begin() as connection:
        connection.execute(update(t.cases).where(t.cases.c.case_id == "CASE-A").values(status="closed"))
    error(clinic.prepare(slot["slot_id"]), "CASE_NOT_BOOKABLE")
    assert clinic.count(t.drafts) == clinic.count(t.holds) == 0


def test_closed_case_after_confirmation_records_failure(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    with clinic.engine.begin() as connection:
        connection.execute(update(t.cases).where(t.cases.c.case_id == "CASE-A").values(status="closed"))
    error(clinic.book(draft), "CASE_NOT_BOOKABLE")
    status = good(clinic.status(draft))
    assert status["booking_state"] == "failed"
    assert status["failure"]["code"] == "CASE_NOT_BOOKABLE"
    assert clinic.count(t.appointments) == 0


def test_inactive_staff_session_cannot_access_or_book(clinic):
    draft = good(clinic.prepare())
    with clinic.engine.begin() as connection:
        connection.execute(update(t.users).where(t.users.c.user_id == "STAFF").values(is_active=False))
    error(clinic.search(), "SESSION_CONTEXT_MISMATCH")
    error(clinic.book(draft), "SESSION_CONTEXT_MISMATCH")
    error(clinic.status(draft), "SESSION_CONTEXT_MISMATCH")


def test_intervening_utterance_requires_readback_again(clinic):
    draft = good(clinic.prepare())
    details = {name: draft[name] for name in ("case_id", "patient_display_name", "slot")}
    digest = clinic.service.mark_readback_presented(clinic.context, draft_id=UUID(draft["draft_id"]), details=details)
    clinic.context = clinic.service.advance_conversation(clinic.context)
    clinic.context = clinic.service.advance_conversation(clinic.context)
    with pytest.raises(BookingError) as caught:
        clinic.service.record_confirmation(clinic.context, draft_id=UUID(draft["draft_id"]), details_hash=digest, confirmation_event_id=uuid4())
    assert caught.value.code == "CONFIRMATION_STALE"
    error(clinic.book(draft), "CONFIRMATION_REQUIRED")
    clinic.confirm(draft)
    good(clinic.book(draft))


def test_snapshot_changes_after_readback_cannot_be_silently_booked(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    with clinic.engine.begin() as connection:
        connection.execute(update(t.clinicians).where(t.clinicians.c.clinician_id == "CLIN-A").values(display_name="Changed Synthetic Name"))
    error(clinic.book(draft), "SLOT_UNAVAILABLE")
    assert clinic.count(t.appointments) == clinic.count(t.allocations) == 0


def test_case_reassignment_even_with_same_name_cannot_retarget_a_draft(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    with clinic.engine.begin() as connection:
        connection.execute(update(t.patients).where(t.patients.c.patient_id == "PAT-B").values(display_name="Synthetic Alex"))
        connection.execute(update(t.cases).where(t.cases.c.case_id == "CASE-A").values(patient_id="PAT-B"))
    error(clinic.book(draft), "CASE_NOT_BOOKABLE")
    assert clinic.count(t.appointments) == 0
    assert good(clinic.status(draft))["booking_state"] == "failed"


def test_readback_uses_one_clock_snapshot_at_expiry_boundary(clinic):
    draft = good(clinic.prepare())
    expiry = datetime.fromisoformat(draft["hold_expires_at"])
    calls = []
    def clock():
        calls.append(True)
        return expiry - timedelta(microseconds=1) if len(calls) == 1 else expiry
    clinic.service._clock = clock
    details = {key: draft[key] for key in ("case_id", "patient_display_name", "slot")}
    clinic.service.mark_readback_presented(clinic.context, draft_id=UUID(draft["draft_id"]), details=details)
    assert len(calls) == 1


def test_selection_of_newly_unavailable_offered_slot_does_not_preserve_old_choice(clinic):
    choices = good(clinic.search())["slots"]
    old = good(clinic.prepare(choices[0]["slot_id"]))
    other = clinic.new_context()
    good(clinic.search(other))
    good(clinic.prepare(choices[1]["slot_id"], other))
    error(clinic.prepare(choices[1]["slot_id"]), "SLOT_UNAVAILABLE")
    outcome = good(clinic.status(old))
    assert outcome["booking_state"] == "failed"
    assert outcome["failure"]["code"] == "DRAFT_SUPERSEDED"
    assert not any(row["draft_id"] == UUID(old["draft_id"]) for row in clinic.rows(t.holds))


def test_maximum_calendar_date_can_return_an_empty_search(clinic):
    result = good(clinic.search(start_date="9999-12-31", end_date="9999-12-31"))
    assert result["slots"] == []


def test_correction_detected_after_speech_start_releases_selection_without_double_advance(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    clinic.context = clinic.service.advance_conversation(clinic.context)
    version = clinic.context.conversation_version
    clinic.service.invalidate_selection(clinic.context)
    resumed = clinic.service.resume_context(session_id=clinic.context.session_id, user_id=clinic.context.user_id)
    assert resumed.conversation_version == version
    assert clinic.count(t.holds) == 0
    error(clinic.book(draft), "DRAFT_SUPERSEDED")
