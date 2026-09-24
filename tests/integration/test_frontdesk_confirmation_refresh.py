"""Offline SQLite checks for the trusted short-clarification delivery hook.

No real database, model, network, browser or audio is used. The caller's audio
barrier remains a controller/adapter responsibility, not a tool argument.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, update

from healthcare_voice_agent.booking import tables as t
from healthcare_voice_agent.booking.service import BookingError
from healthcare_voice_agent.tools.frontdesk_contracts import FRONTDESK_TOOLS
from tests.integration.test_frontdesk_booking import clinic, error, good


def prepare_receipt(clinic, *, delivered=True):
    draft = good(clinic.prepare())
    version = clinic.context.conversation_version
    digest = clinic.rows(t.drafts)[0]["details_hash"]
    if delivered:
        details = {key: draft[key] for key in ("case_id", "patient_display_name", "slot")}
        assert clinic.service.mark_readback_presented(
            clinic.context, draft_id=UUID(draft["draft_id"]), details=details) == digest
    return draft, digest, version


def advance(clinic, count=1):
    for _ in range(count):
        clinic.context = clinic.service.advance_conversation(clinic.context)


def refresh(clinic, draft, digest, version, *, context=None):
    return clinic.service.refresh_confirmation_prompt(
        context or clinic.context, draft_id=UUID(draft["draft_id"]),
        details_hash=digest, previous_readback_version=version)


def state(clinic):
    return {table.name: clinic.rows(table) for table in
            (t.sessions, t.drafts, t.holds, t.appointments, t.allocations)}


def assert_rejected_without_writes(clinic, draft, digest, version, expected, *, context=None):
    before = state(clinic)
    with pytest.raises(BookingError) as caught:
        refresh(clinic, draft, digest, version, context=context)
    assert caught.value.code == expected
    assert state(clinic) == before


def test_full_receipt_clarification_then_new_affirmative_can_book(clinic):
    draft, digest, version = prepare_receipt(clinic)
    original = clinic.rows(t.drafts)[0]
    hold = clinic.rows(t.holds)[0]
    advance(clinic)  # An unclear final answer, not consent.
    clinic.clock.advance(10)
    assert refresh(clinic, draft, digest, version) == digest
    refreshed = clinic.rows(t.drafts)[0]
    assert refreshed["readback_version"] == clinic.context.conversation_version
    assert refreshed["readback_at"] == original["readback_at"]
    assert refreshed["updated_at"] > original["updated_at"]
    assert {key: value for key, value in refreshed.items() if key not in {"readback_version", "updated_at"}} == {
        key: value for key, value in original.items() if key not in {"readback_version", "updated_at"}}
    assert clinic.rows(t.holds)[0] == hold
    assert clinic.count(t.appointments) == clinic.count(t.allocations) == 0
    error(clinic.book(draft), "CONFIRMATION_REQUIRED")
    # Delivery of the question is neither consent nor the next user turn.
    with pytest.raises(BookingError) as caught:
        clinic.service.record_confirmation(clinic.context, draft_id=UUID(draft["draft_id"]),
            details_hash=digest, confirmation_event_id=uuid4())
    assert caught.value.code == "CONFIRMATION_STALE"
    advance(clinic)  # A new, complete affirmative answer after clarification.
    error(clinic.book(draft), "CONFIRMATION_REQUIRED")
    clinic.service.record_confirmation(clinic.context, draft_id=UUID(draft["draft_id"]),
        details_hash=digest, confirmation_event_id=uuid4())
    assert good(clinic.book(draft))["appointment"]["status"] == "confirmed"
    assert clinic.count(t.appointments) == clinic.count(t.allocations) == 1


def test_successive_clarifications_preserve_original_receipt_and_expiry(clinic):
    draft, digest, version = prepare_receipt(clinic)
    original = clinic.rows(t.drafts)[0]
    hold = clinic.rows(t.holds)[0]
    for _ in range(3):
        advance(clinic)
        clinic.clock.advance(10)
        assert refresh(clinic, draft, digest, version) == digest
        version = clinic.context.conversation_version
        current = clinic.rows(t.drafts)[0]
        assert current["readback_at"] == original["readback_at"]
        assert current["hold_expires_at"] == original["hold_expires_at"]
        assert current["confirmation_version"] is None
        assert clinic.rows(t.holds)[0] == hold
        assert clinic.count(t.drafts) == 1
        assert clinic.count(t.appointments) == 0


@pytest.mark.parametrize("scenario", ["missing_receipt", "fake_digest", "wrong_previous", "bool_previous", "no_turn", "gap"])
def test_clarification_cannot_manufacture_or_skip_receipts(clinic, scenario):
    draft, digest, version = prepare_receipt(clinic, delivered=scenario != "missing_receipt")
    if scenario != "no_turn":
        advance(clinic, 2 if scenario == "gap" else 1)
    if scenario == "fake_digest":
        digest = "0" * 64
    if scenario == "wrong_previous":
        version -= 1
    if scenario == "bool_previous":
        version = True
    assert_rejected_without_writes(clinic, draft, digest, version, "CONFIRMATION_STALE")


def test_duplicate_receipt_delivery_does_not_advance_or_reauthorize(clinic):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    refresh(clinic, draft, digest, version)
    assert_rejected_without_writes(clinic, draft, digest, version, "CONFIRMATION_STALE")
    error(clinic.book(draft), "CONFIRMATION_REQUIRED")


@pytest.mark.parametrize("field,value", [("case_id", "CASE-B"), ("clinic_id", "OTHER"),
    ("case_context_version", 999), ("conversation_version", 999), ("user_id", "OTHER")])
def test_forged_context_cannot_refresh(clinic, field, value):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    assert_rejected_without_writes(clinic, draft, digest, version, "SESSION_CONTEXT_MISMATCH",
                                  context=replace(clinic.context, **{field: value}))


@pytest.mark.parametrize("case_id", ["CASE-A", "CASE-B"])
def test_another_session_cannot_refresh_even_same_case(clinic, case_id):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    other = clinic.new_context(case_id)
    assert_rejected_without_writes(clinic, draft, digest, version, "RECORD_NOT_FOUND", context=other)


@pytest.mark.parametrize("elapsed", [120, 121])
def test_expired_hold_cannot_be_refreshed_or_renewed(clinic, elapsed):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    clinic.clock.advance(elapsed)
    assert_rejected_without_writes(clinic, draft, digest, version, "HOLD_EXPIRED")
    assert good(clinic.status(draft))["booking_state"] == "expired"
    assert clinic.count(t.appointments) == 0


def test_already_recorded_confirmation_cannot_be_refreshed(clinic):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    clinic.service.record_confirmation(clinic.context, draft_id=UUID(draft["draft_id"]),
                                      details_hash=digest, confirmation_event_id=uuid4())
    assert_rejected_without_writes(clinic, draft, digest, version, "CONFIRMATION_STALE")


def test_booked_draft_cannot_be_refreshed(clinic):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    clinic.service.record_confirmation(clinic.context, draft_id=UUID(draft["draft_id"]),
                                      details_hash=digest, confirmation_event_id=uuid4())
    good(clinic.book(draft))
    assert_rejected_without_writes(clinic, draft, digest, version, "DRAFT_SUPERSEDED")
    assert clinic.count(t.appointments) == 1


def test_superseded_selection_cannot_be_refreshed(clinic):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    clinic.service.invalidate_selection(clinic.context)
    assert_rejected_without_writes(clinic, draft, digest, version, "DRAFT_SUPERSEDED")


@pytest.mark.parametrize("mutation,code", [
    ("patient_name", "CONFIRMATION_STALE"), ("clinician_name", "CONFIRMATION_STALE"),
    ("clinician_id", "CONFIRMATION_STALE"), ("location", "CONFIRMATION_STALE"),
    ("starts_at", "CONFIRMATION_STALE"), ("ends_at", "CONFIRMATION_STALE"),
    ("timezone", "CONFIRMATION_STALE"), ("appointment_type", "CONFIRMATION_STALE"),
    ("reassigned_same_name", "CASE_NOT_BOOKABLE"), ("closed_case", "CASE_NOT_BOOKABLE"),
    ("disabled_slot", "SLOT_UNAVAILABLE"), ("missing_hold", "SLOT_UNAVAILABLE"),
])
def test_authoritative_changes_prevent_carrying_old_readback(clinic, mutation, code):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    slot_id = UUID(draft["slot"]["slot_id"])
    with clinic.engine.begin() as connection:
        if mutation == "patient_name":
            connection.execute(update(t.patients).where(t.patients.c.patient_id == "PAT-A").values(display_name="Changed Patient"))
        elif mutation == "clinician_name":
            connection.execute(update(t.clinicians).where(t.clinicians.c.clinician_id == "CLIN-A").values(display_name="Changed Doctor"))
        elif mutation == "clinician_id":
            connection.execute(t.clinicians.insert().values(clinician_id="CLIN-C", display_name="Another Doctor"))
            connection.execute(update(t.slots).where(t.slots.c.slot_id == slot_id).values(clinician_id="CLIN-C"))
        elif mutation == "timezone":
            connection.execute(update(t.clinics).where(t.clinics.c.clinic_id == "CLINIC").values(timezone="UTC"))
        elif mutation == "reassigned_same_name":
            connection.execute(update(t.patients).where(t.patients.c.patient_id == "PAT-B").values(display_name="Synthetic Alex"))
            connection.execute(update(t.cases).where(t.cases.c.case_id == "CASE-A").values(patient_id="PAT-B"))
        elif mutation == "closed_case":
            connection.execute(update(t.cases).where(t.cases.c.case_id == "CASE-A").values(status="closed"))
        elif mutation == "missing_hold":
            connection.execute(delete(t.holds).where(t.holds.c.draft_id == UUID(draft["draft_id"])))
        else:
            values = {
                "clinician_id": {"clinician_id": "CLIN-B"},
                "location": {"location": "Changed Room"},
                "starts_at": {"starts_at": datetime.fromisoformat(draft["slot"]["starts_at"]).astimezone(timezone.utc) + timedelta(minutes=5)},
                "ends_at": {"ends_at": datetime.fromisoformat(draft["slot"]["ends_at"]).astimezone(timezone.utc) + timedelta(minutes=5)},
                "appointment_type": {"appointment_type": "physiotherapy"},
                "disabled_slot": {"enabled": False},
            }[mutation]
            connection.execute(update(t.slots).where(t.slots.c.slot_id == slot_id).values(**values))
    assert_rejected_without_writes(clinic, draft, digest, version, code)


def test_refresh_is_not_a_tool_or_a_model_asserted_confirmation(clinic):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    before = state(clinic)
    assert "refresh_confirmation_prompt" not in FRONTDESK_TOOLS
    with pytest.raises(ValueError, match="Unknown front-desk tool"):
        clinic.service.invoke("refresh_confirmation_prompt", {
            "draft_id": draft["draft_id"], "details_hash": digest,
            "previous_readback_version": version}, context=clinic.context)
    error(clinic.service.invoke("book_appointment", {
        "case_id": clinic.context.case_id, "draft_id": draft["draft_id"],
        "confirmed": True, "readback_version": version, "details_hash": digest,
    }, context=clinic.context), "INVALID_ARGUMENT")
    assert state(clinic) == before


def test_refresh_uses_one_server_clock_snapshot_at_expiry_boundary(clinic):
    draft, digest, version = prepare_receipt(clinic)
    advance(clinic)
    expiry = datetime.fromisoformat(draft["hold_expires_at"])
    calls = []
    def clock():
        calls.append(True)
        return expiry - timedelta(microseconds=1) if len(calls) == 1 else expiry
    clinic.service._clock = clock
    assert refresh(clinic, draft, digest, version) == digest
    assert len(calls) == 1
