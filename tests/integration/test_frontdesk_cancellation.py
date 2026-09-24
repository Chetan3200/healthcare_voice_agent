"""Offline SQLite cancellation transactions; no live database, API, or server."""
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, update

from healthcare_voice_agent.booking import tables as t
from healthcare_voice_agent.booking.service import BookingError
from tests.integration.test_frontdesk_booking import clinic, good
from tests.integration.test_frontdesk_rescheduling import source_booking, replacement, commit


def prepare(clinic, source, context=None):
    return clinic.service.prepare_cancellation(context or clinic.context,
        appointment_id=source["appointment_id"],
        expected_record_version=source["record_version"])


def confirm(clinic, intent, context=None):
    context = context or clinic.context
    digest = clinic.service.mark_cancellation_presented(context,
        cancellation_request_id=UUID(intent["cancellation_request_id"]),
        details={"appointment": intent["appointment"]})
    context = clinic.service.advance_conversation(context)
    clinic.service.record_cancellation_confirmation(context,
        cancellation_request_id=UUID(intent["cancellation_request_id"]),
        details_hash=digest, confirmation_event_id=uuid4())
    if context.session_id == clinic.context.session_id:
        clinic.context = context
    return context


def cancel(clinic, intent, context=None):
    return clinic.service.cancel_appointment(context or clinic.context,
        cancellation_request_id=UUID(intent["cancellation_request_id"]))


def row_for(clinic, table, column, value):
    with clinic.engine.connect() as connection:
        return connection.execute(select(table).where(column == value)).mappings().one()


def test_confirmed_cancellation_is_scoped_idempotent_and_releases_only_source(clinic):
    original, old_receipt, source = source_booking(clinic)
    second_draft = good(clinic.prepare())
    clinic.confirm(second_draft)
    second_receipt = good(clinic.book(second_draft))
    second_before = row_for(clinic, t.appointments, t.appointments.c.appointment_id,
                            second_receipt["appointment"]["appointment_id"])

    intent = prepare(clinic, source)
    assert set(intent) == {"cancellation_request_id", "appointment", "expires_at", "cancellation_state"}
    assert intent["cancellation_state"] == "pending" and intent["appointment"] == source
    confirm(clinic, intent)
    result = cancel(clinic, intent)
    assert set(result) == {"cancellation_request_id", "appointment", "expires_at", "cancellation_state"}
    assert result["cancellation_state"] == "cancelled"
    assert result["appointment"] == {**source, "status": "cancelled", "record_version": 2}
    assert cancel(clinic, intent) == result
    assert clinic.service.get_cancellation_status(clinic.context,
        cancellation_request_id=UUID(intent["cancellation_request_id"])) == result

    source_row = row_for(clinic, t.appointments, t.appointments.c.appointment_id, source["appointment_id"])
    allocation = row_for(clinic, t.allocations, t.allocations.c.appointment_id, source["appointment_id"])
    second_after = row_for(clinic, t.appointments, t.appointments.c.appointment_id,
                           second_receipt["appointment"]["appointment_id"])
    assert source_row["status"] == "cancelled" and source_row["record_version"] == 2
    assert not allocation["active"]
    assert dict(second_after) == dict(second_before)
    assert good(clinic.book(original)) == old_receipt
    assert [item["appointment_id"] for item in clinic.service.list_bookings(clinic.context)] == [
        second_receipt["appointment"]["appointment_id"]]

    other = clinic.new_context()
    offered = good(clinic.search(other))["slots"]
    assert source["slot_id"] in {item["slot_id"] for item in offered}


def test_commit_requires_exact_presented_details_and_next_version_confirmation(clinic):
    _, _, source = source_booking(clinic)
    intent = prepare(clinic, source)
    with pytest.raises(BookingError) as missing:
        cancel(clinic, intent)
    assert missing.value.code == "CONFIRMATION_REQUIRED"
    with pytest.raises(BookingError) as wrong_details:
        clinic.service.mark_cancellation_presented(clinic.context,
            cancellation_request_id=UUID(intent["cancellation_request_id"]),
            details={"appointment": {**source, "location": "changed"}})
    assert wrong_details.value.code == "CONFIRMATION_STALE"

    digest = clinic.service.mark_cancellation_presented(clinic.context,
        cancellation_request_id=UUID(intent["cancellation_request_id"]),
        details={"appointment": source})
    clinic.context = clinic.service.advance_conversation(clinic.context)
    clinic.context = clinic.service.advance_conversation(clinic.context)
    with pytest.raises(BookingError) as stale:
        clinic.service.record_cancellation_confirmation(clinic.context,
            cancellation_request_id=UUID(intent["cancellation_request_id"]),
            details_hash=digest, confirmation_event_id=uuid4())
    assert stale.value.code == "CONFIRMATION_STALE"
    assert row_for(clinic, t.appointments, t.appointments.c.appointment_id,
                   source["appointment_id"])["status"] == "confirmed"


def test_wrong_session_case_and_stale_context_cannot_access_intent(clinic):
    _, _, source = source_booking(clinic)
    intent = prepare(clinic, source)
    other = clinic.new_context("CASE-A")
    wrong_case = clinic.new_context("CASE-B")
    for context in (other, wrong_case):
        with pytest.raises(BookingError):
            clinic.service.get_cancellation_status(context,
                cancellation_request_id=UUID(intent["cancellation_request_id"]))
    stale = clinic.context
    clinic.context = clinic.service.advance_conversation(clinic.context)
    with pytest.raises(BookingError) as mismatch:
        clinic.service.get_cancellation_status(stale,
            cancellation_request_id=UUID(intent["cancellation_request_id"]))
    assert mismatch.value.code == "SESSION_CONTEXT_MISMATCH"


def test_expiry_and_invalidation_are_terminal_without_cancelling(clinic):
    _, _, source = source_booking(clinic)
    intent = prepare(clinic, source)
    clinic.clock.advance(121)
    status = clinic.service.get_cancellation_status(clinic.context,
        cancellation_request_id=UUID(intent["cancellation_request_id"]))
    assert status["cancellation_state"] == "expired"
    with pytest.raises(BookingError) as expired:
        cancel(clinic, intent)
    assert expired.value.code == "HOLD_EXPIRED"

    fresh = prepare(clinic, source)
    clinic.service.invalidate_cancellation(clinic.context,
        cancellation_request_id=UUID(fresh["cancellation_request_id"]))
    abandoned = clinic.service.get_cancellation_status(clinic.context,
        cancellation_request_id=UUID(fresh["cancellation_request_id"]))
    assert abandoned["cancellation_state"] == "abandoned"
    clinic.service.invalidate_cancellation(clinic.context,
        cancellation_request_id=UUID(fresh["cancellation_request_id"]))
    assert row_for(clinic, t.appointments, t.appointments.c.appointment_id,
                   source["appointment_id"])["status"] == "confirmed"


def test_repeat_prepare_reuses_request_without_renewing_and_new_source_abandons_old(clinic):
    _, _, first = source_booking(clinic)
    second_draft = good(clinic.prepare())
    clinic.confirm(second_draft)
    good(clinic.book(second_draft))
    second = [item for item in clinic.service.list_bookings(clinic.context)
              if item["appointment_id"] != first["appointment_id"]][0]
    one = prepare(clinic, first)
    clinic.clock.advance(30)
    assert prepare(clinic, first) == one
    assert clinic.count(t.cancellations) == 1
    two = prepare(clinic, second)
    assert two["cancellation_request_id"] != one["cancellation_request_id"]
    assert clinic.service.get_cancellation_status(clinic.context,
        cancellation_request_id=UUID(one["cancellation_request_id"]))["cancellation_state"] == "abandoned"
    assert sum(row["state"] == "pending" for row in clinic.rows(t.cancellations)) == 1


def test_record_version_change_and_reschedule_race_fail_closed(clinic):
    _, _, source = source_booking(clinic)
    intent = prepare(clinic, source)
    confirm(clinic, intent)
    with clinic.engine.begin() as connection:
        connection.execute(update(t.appointments).where(
            t.appointments.c.appointment_id == source["appointment_id"]).values(record_version=2))
    with pytest.raises(BookingError) as stale:
        cancel(clinic, intent)
    assert stale.value.code == "CONFIRMATION_STALE"
    assert row_for(clinic, t.allocations, t.allocations.c.appointment_id,
                   source["appointment_id"])["active"]


def test_reschedule_winner_blocks_old_source_cancellation_but_replacement_can_cancel(clinic):
    _, _, source = source_booking(clinic)
    intent = prepare(clinic, source)
    change = replacement(clinic, source)
    clinic.confirm(change)
    result = commit(clinic, change)
    with pytest.raises(BookingError):
        confirm(clinic, intent)
    with pytest.raises(BookingError) as old_source:
        prepare(clinic, source)
    assert old_source.value.code == "RECORD_NOT_FOUND"
    replacement_source = clinic.service.list_bookings(clinic.context)[0]
    assert replacement_source["appointment_id"] == result["appointment"]["appointment_id"]
    replacement_intent = prepare(clinic, replacement_source)
    confirm(clinic, replacement_intent)
    assert cancel(clinic, replacement_intent)["appointment"]["status"] == "cancelled"


def test_concurrent_cancellation_and_reschedule_have_one_guard_locked_winner(clinic):
    _, _, source = source_booking(clinic)
    cancellation_context = clinic.new_context("CASE-A")
    reschedule_context = clinic.new_context("CASE-A")
    intent = prepare(clinic, source, cancellation_context)
    change = replacement(clinic, source, context=reschedule_context)
    cancellation_context = confirm(clinic, intent, cancellation_context)
    reschedule_context = clinic.confirm(change, reschedule_context)

    def attempt(kind):
        try:
            if kind == "cancel":
                return cancel(clinic, intent, cancellation_context)
            return commit(clinic, change, reschedule_context)
        except BookingError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, ("cancel", "reschedule")))
    assert sum(item is not None for item in outcomes) == 1
    source_row = row_for(clinic, t.appointments, t.appointments.c.appointment_id,
                         source["appointment_id"])
    assert source_row["status"] == "cancelled" and source_row["record_version"] == 2
    assert sum(row["active"] for row in clinic.rows(t.allocations)) in {0, 1}
    assert len(clinic.service.list_bookings(clinic.context)) in {0, 1}


def test_failure_after_calendar_updates_rolls_back_every_write(clinic, monkeypatch):
    _, _, source = source_booking(clinic)
    intent = prepare(clinic, source)
    confirm(clinic, intent)

    def fail_receipt(*args, **kwargs):
        raise RuntimeError("simulated receipt failure")
    monkeypatch.setattr(clinic.service, "_persist_cancellation_result", fail_receipt)
    with pytest.raises(RuntimeError):
        cancel(clinic, intent)
    appointment = row_for(clinic, t.appointments, t.appointments.c.appointment_id, source["appointment_id"])
    allocation = row_for(clinic, t.allocations, t.allocations.c.appointment_id, source["appointment_id"])
    persisted = row_for(clinic, t.cancellations, t.cancellations.c.cancellation_request_id,
                        UUID(intent["cancellation_request_id"]))
    assert appointment["status"] == "confirmed" and appointment["record_version"] == 1
    assert allocation["active"] and persisted["state"] == "pending"


def test_cancellation_lifecycle_does_not_touch_unrelated_booking_transaction(clinic):
    _, _, source = source_booking(clinic)
    draft = good(clinic.prepare())
    intent = prepare(clinic, source)
    before = (clinic.rows(t.drafts), clinic.rows(t.holds))
    clinic.service.invalidate_cancellation(clinic.context,
        cancellation_request_id=UUID(intent["cancellation_request_id"]))
    assert (clinic.rows(t.drafts), clinic.rows(t.holds)) == before

    next_intent = prepare(clinic, source)
    clinic.service.set_confirmed_case(clinic.context, "CASE-B")
    persisted = row_for(clinic, t.cancellations, t.cancellations.c.cancellation_request_id,
                        UUID(next_intent["cancellation_request_id"]))
    assert persisted["state"] == "abandoned"
