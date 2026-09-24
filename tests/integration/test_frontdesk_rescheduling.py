"""Offline SQLite transaction checks; not live PostgreSQL or model validation."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, update

from healthcare_voice_agent.booking import tables as t
from healthcare_voice_agent.booking.service import BookingError
from healthcare_voice_agent.tools.frontdesk_contracts import FRONTDESK_TOOLS
from tests.integration.test_frontdesk_booking import clinic, good


def source_booking(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    receipt = good(clinic.book(draft))
    return draft, receipt, clinic.service.list_bookings(clinic.context)[0]


def replacement(clinic, source, *, context=None, clinician_id="CLIN-A"):
    context = context or clinic.context
    slots = good(clinic.search(context, clinician_id=clinician_id))["slots"]
    return clinic.service.prepare_reschedule(context,
        appointment_id=source["appointment_id"], expected_record_version=source["record_version"],
        slot_id=UUID(slots[0]["slot_id"]))


def commit(clinic, draft, context=None):
    return clinic.service.reschedule_appointment(context or clinic.context, draft_id=UUID(draft["draft_id"]))


def assert_original_active(clinic, source):
    rows = {row["appointment_id"]: row for row in clinic.rows(t.appointments)}
    assert rows[source["appointment_id"]]["status"] == "confirmed"
    assert rows[source["appointment_id"]]["record_version"] == source["record_version"]
    allocation = next(row for row in clinic.rows(t.allocations) if row["appointment_id"] == source["appointment_id"])
    assert allocation["active"]


def test_atomic_reschedule_retains_receipts_and_frees_old_slot(clinic):
    original, old_receipt, source = source_booking(clinic)
    draft = replacement(clinic, source)
    assert draft["reschedule_from"] == source
    assert_original_active(clinic, source)
    assert clinic.count(t.appointments) == 1
    clinic.confirm(draft)
    result = commit(clinic, draft)
    assert result["rescheduled_from"] == source["appointment_id"]
    assert result["appointment"]["appointment_id"] != source["appointment_id"]
    rows = clinic.rows(t.appointments)
    assert sum(row["status"] == "confirmed" for row in rows) == 1
    assert sum(row["status"] == "cancelled" for row in rows) == 1
    assert sum(row["active"] for row in clinic.rows(t.allocations)) == 1
    assert clinic.count(t.holds) == 0
    assert commit(clinic, draft) == result
    assert good(clinic.status(draft))["appointment"] == result["appointment"]
    # Historical request receipts remain immutable, current listing is distinct.
    assert good(clinic.book(original)) == old_receipt
    current = clinic.service.list_bookings(clinic.context)
    assert [row["appointment_id"] for row in current] == [result["appointment"]["appointment_id"]]
    # Reuse original slot under another case: historical allocation does not lock it.
    other = clinic.new_context()
    offered = good(clinic.search(other))["slots"]
    assert original["slot"]["slot_id"] in [row["slot_id"] for row in offered]
    next_draft = good(clinic.prepare(original["slot"]["slot_id"], other))
    other = clinic.confirm(next_draft, other)
    assert good(clinic.book(next_draft, other))
    assert clinic.count(t.allocations) == 3
    with clinic.engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []


def test_prepare_retry_keeps_hold_and_source_unchanged(clinic):
    _, _, source = source_booking(clinic)
    draft = replacement(clinic, source)
    before = clinic.rows(t.holds)
    clinic.clock.advance(40)
    retry = clinic.service.prepare_reschedule(clinic.context,
        appointment_id=source["appointment_id"], expected_record_version=source["record_version"],
        slot_id=UUID(draft["slot"]["slot_id"]))
    assert retry == draft
    assert clinic.rows(t.holds) == before
    assert clinic.count(t.reschedules) == 1
    assert_original_active(clinic, source)


@pytest.mark.parametrize("action", ["no_confirmation", "expiry", "cancel", "source_version", "source_name", "disabled_slot"])
def test_rejected_change_never_cancels_original(clinic, action):
    _, _, source = source_booking(clinic)
    draft = replacement(clinic, source)
    if action != "no_confirmation":
        clinic.confirm(draft)
    if action == "expiry":
        clinic.clock.advance(121)
    elif action == "cancel":
        clinic.service.invalidate_selection(clinic.context)
    elif action == "source_version":
        with clinic.engine.begin() as connection:
            connection.execute(update(t.appointments).where(t.appointments.c.appointment_id == source["appointment_id"])
                .values(record_version=2))
        source["record_version"] = 2
    elif action == "source_name":
        with clinic.engine.begin() as connection:
            connection.execute(update(t.patients).where(t.patients.c.patient_id == "PAT-A").values(display_name="Changed"))
    elif action == "disabled_slot":
        with clinic.engine.begin() as connection:
            connection.execute(update(t.slots).where(t.slots.c.slot_id == UUID(draft["slot"]["slot_id"])).values(enabled=False))
    with pytest.raises(BookingError):
        commit(clinic, draft)
    assert_original_active(clinic, source)
    assert clinic.count(t.appointments) == 1
    assert clinic.rows(t.reschedules)[0]["replacement_appointment_id"] is None


def test_failure_after_source_cancellation_rolls_back_every_write(clinic, monkeypatch):
    _, _, source = source_booking(clinic)
    draft = replacement(clinic, source)
    clinic.confirm(draft)
    original = clinic.service._book_appointment
    def failed(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated uncertain transaction before commit")
    monkeypatch.setattr(clinic.service, "_book_appointment", failed)
    with pytest.raises(RuntimeError):
        commit(clinic, draft)
    assert_original_active(clinic, source)
    assert clinic.count(t.appointments) == 1 and clinic.count(t.holds) == 1
    assert clinic.rows(t.reschedules)[0]["replacement_appointment_id"] is None


def test_public_booking_tool_cannot_commit_a_reschedule(clinic):
    _, _, source = source_booking(clinic)
    draft = replacement(clinic, source)
    clinic.confirm(draft)
    result = clinic.book(draft)
    assert result["error"]["code"] == "INVALID_ARGUMENT"
    assert_original_active(clinic, source)
    assert clinic.count(t.appointments) == 1
    assert set(FRONTDESK_TOOLS) == {"find_available_slots", "prepare_booking", "book_appointment", "get_booking_status"}


def test_regular_draft_cannot_be_relabelled_as_reschedule(clinic):
    draft = good(clinic.prepare())
    clinic.confirm(draft)
    with pytest.raises(BookingError, match="Arguments"):
        commit(clinic, draft)
    assert clinic.count(t.appointments) == 0


def test_wrong_case_or_session_cannot_change_or_list_other_booking(clinic):
    _, _, source = source_booking(clinic)
    draft = replacement(clinic, source)
    other = clinic.new_context()
    assert clinic.service.list_bookings(other) == []
    with pytest.raises(BookingError):
        replacement(clinic, source, context=other)
    clinic.confirm(draft)
    with pytest.raises(BookingError):
        commit(clinic, draft, other)
    same_case_new_session = clinic.new_context("CASE-A")
    with pytest.raises(BookingError):
        commit(clinic, draft, same_case_new_session)
    assert_original_active(clinic, source)


def test_new_session_can_prepare_change_of_its_confirmed_case(clinic):
    _, _, source = source_booking(clinic)
    context = clinic.new_context("CASE-A")
    assert clinic.service.list_bookings(context)[0] == source
    draft = replacement(clinic, source, context=context)
    context = clinic.confirm(draft, context)
    assert commit(clinic, draft, context)["rescheduled_from"] == source["appointment_id"]


def test_switching_doctor_can_overlap_only_the_source_appointment(clinic):
    _, _, source = source_booking(clinic)
    draft = replacement(clinic, source, clinician_id="CLIN-B")
    assert draft["slot"]["starts_at"] == source["starts_at"]
    clinic.confirm(draft)
    result = commit(clinic, draft)
    assert result["appointment"]["clinician"]["clinician_id"] == "CLIN-B"
    assert len(clinic.service.list_bookings(clinic.context)) == 1


def test_clarification_refresh_preserves_source_overlap_exception(clinic):
    _, _, source = source_booking(clinic)
    draft = replacement(clinic, source, clinician_id="CLIN-B")
    details = {name: draft[name] for name in ("case_id", "patient_display_name", "slot")}
    previous = clinic.context.conversation_version
    digest = clinic.service.mark_readback_presented(clinic.context, draft_id=UUID(draft["draft_id"]), details=details)
    clinic.context = clinic.service.advance_conversation(clinic.context)
    assert clinic.service.refresh_confirmation_prompt(clinic.context, draft_id=UUID(draft["draft_id"]),
        details_hash=digest, previous_readback_version=previous) == digest
    clinic.context = clinic.service.advance_conversation(clinic.context)
    clinic.service.record_confirmation(clinic.context, draft_id=UUID(draft["draft_id"]),
        details_hash=digest, confirmation_event_id=uuid4())
    assert commit(clinic, draft)["appointment"]["starts_at"] == source["starts_at"]


def test_repeated_reschedule_chain_and_original_slot_reuse(clinic):
    _, _, source = source_booking(clinic)
    first = replacement(clinic, source)
    clinic.confirm(first)
    first_result = commit(clinic, first)
    source2 = clinic.service.list_bookings(clinic.context)[0]
    second = replacement(clinic, source2)
    clinic.confirm(second)
    second_result = commit(clinic, second)
    assert first_result["appointment"]["appointment_id"] != second_result["appointment"]["appointment_id"]
    assert len(clinic.service.list_bookings(clinic.context)) == 1
    assert clinic.count(t.reschedules) == 2


def test_concurrent_changes_of_same_original_have_one_winner(clinic):
    _, _, source = source_booking(clinic)
    ctx1, ctx2 = clinic.new_context("CASE-A"), clinic.new_context("CASE-A")
    options1 = good(clinic.search(ctx1))["slots"]
    options2 = good(clinic.search(ctx2))["slots"]
    one = clinic.service.prepare_reschedule(ctx1, appointment_id=source["appointment_id"], expected_record_version=1,
        slot_id=UUID(options1[0]["slot_id"]))
    two = clinic.service.prepare_reschedule(ctx2, appointment_id=source["appointment_id"], expected_record_version=1,
        slot_id=UUID(options2[1]["slot_id"]))
    ctx1, ctx2 = clinic.confirm(one, ctx1), clinic.confirm(two, ctx2)
    def attempt(item):
        draft, context = item
        try:
            return commit(clinic, draft, context)
        except BookingError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, [(one, ctx1), (two, ctx2)]))
    assert sum(item is not None for item in results) == 1
    assert len(clinic.service.list_bookings(clinic.context)) == 1


@pytest.mark.parametrize("version", [0, -1, True, "1", 2])
def test_version_validation(clinic, version):
    _, _, source = source_booking(clinic)
    slot = good(clinic.search())["slots"][0]
    with pytest.raises(BookingError):
        clinic.service.prepare_reschedule(clinic.context, appointment_id=source["appointment_id"],
            expected_record_version=version, slot_id=UUID(slot["slot_id"]))
    assert_original_active(clinic, source)


def test_other_patient_booking_and_other_patient_hold_still_block(clinic):
    _, _, source = source_booking(clinic)
    slots = good(clinic.search())["slots"]
    other = clinic.new_context()
    good(clinic.search(other))
    held = good(clinic.prepare(slots[0]["slot_id"], other))
    with pytest.raises(BookingError):
        clinic.service.prepare_reschedule(clinic.context, appointment_id=source["appointment_id"],
            expected_record_version=1, slot_id=UUID(slots[0]["slot_id"]))
    other = clinic.confirm(held, other)
    good(clinic.book(held, other))
    with pytest.raises(BookingError):
        clinic.service.prepare_reschedule(clinic.context, appointment_id=source["appointment_id"],
            expected_record_version=1, slot_id=UUID(slots[0]["slot_id"]))
    assert_original_active(clinic, source)


def test_reassigned_case_cannot_change_original_patients_booking(clinic):
    _, _, source = source_booking(clinic)
    slot = good(clinic.search())["slots"][0]
    with clinic.engine.begin() as connection:
        connection.execute(update(t.cases).where(t.cases.c.case_id == "CASE-A").values(patient_id="PAT-B"))
    with pytest.raises(BookingError):
        clinic.service.list_bookings(clinic.context)
    with pytest.raises(BookingError):
        clinic.service.prepare_reschedule(clinic.context, appointment_id=source["appointment_id"],
            expected_record_version=1, slot_id=UUID(slot["slot_id"]))
    assert_original_active(clinic, source)


def test_other_clinic_cannot_list_or_change_booking(clinic):
    _, _, source = source_booking(clinic)
    clinic.service.register_clinic("OTHER", "Asia/Kolkata")
    context = clinic.service.open_session(clinic_id="OTHER", user_id="STAFF")
    context = clinic.service.set_confirmed_case(context, "CASE-A")
    assert clinic.service.list_bookings(context) == []
    with pytest.raises(BookingError):
        clinic.service.prepare_reschedule(context, appointment_id=source["appointment_id"],
            expected_record_version=1, slot_id=UUID(source["slot_id"]))
    assert_original_active(clinic, source)


def test_reschedule_rejects_unoffered_same_slot_and_changed_type(clinic):
    from tests.integration.test_frontdesk_booking import rule, DAY
    _, _, source = source_booking(clinic)
    with pytest.raises(BookingError):
        clinic.service.prepare_reschedule(clinic.context, appointment_id=source["appointment_id"],
            expected_record_version=1, slot_id=UUID(source["slot_id"]))
    with pytest.raises(BookingError) as not_offered:
        clinic.service.prepare_reschedule(clinic.context, appointment_id=source["appointment_id"],
            expected_record_version=1, slot_id=clinic.slot_ids[-1])
    assert not_offered.value.code == "SLOT_NOT_OFFERED"
    clinic.service.publish_schedule("CLINIC", (rule("CLIN-B", kind="imaging"),), DAY, DAY)
    slot = good(clinic.search(appointment_type="imaging", clinician_id="CLIN-B"))["slots"][0]
    with pytest.raises(BookingError) as wrong_type:
        clinic.service.prepare_reschedule(clinic.context, appointment_id=source["appointment_id"],
            expected_record_version=1, slot_id=UUID(slot["slot_id"]))
    assert wrong_type.value.code == "INVALID_ARGUMENT"
    assert clinic.count(t.reschedules) == 0
    assert_original_active(clinic, source)


def test_source_change_blocks_clarification_receipt_refresh(clinic):
    _, _, source = source_booking(clinic)
    draft = replacement(clinic, source)
    details = {key: draft[key] for key in ("case_id", "patient_display_name", "slot")}
    previous = clinic.context.conversation_version
    digest = clinic.service.mark_readback_presented(clinic.context, draft_id=UUID(draft["draft_id"]), details=details)
    clinic.context = clinic.service.advance_conversation(clinic.context)
    with clinic.engine.begin() as connection:
        connection.execute(update(t.appointments).where(t.appointments.c.appointment_id == source["appointment_id"])
            .values(record_version=2))
    with pytest.raises(BookingError):
        clinic.service.refresh_confirmation_prompt(clinic.context, draft_id=UUID(draft["draft_id"]),
            details_hash=digest, previous_readback_version=previous)
    assert clinic.rows(t.reschedules)[0]["replacement_appointment_id"] is None


def test_expired_success_receipt_is_idempotent_and_recoverable(clinic):
    _, _, source = source_booking(clinic)
    draft = replacement(clinic, source)
    clinic.confirm(draft)
    result = commit(clinic, draft)
    clinic.clock.advance(121)
    assert commit(clinic, draft) == result
    assert good(clinic.status(draft))["appointment"] == result["appointment"]
    assert len(clinic.service.list_bookings(clinic.context)) == 1
