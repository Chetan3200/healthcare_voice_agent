"""Bounded provenance import for one canonical pre-existing synthetic appointment.

This is not a booking API and does not create or change an appointment. It only
adds the booking metadata required by the normal ownership joins. The caller
must independently verify the isolated demo target and the canonical fixture
rows. All inspection and writes occur under the shared booking guard lock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Callable, Mapping
from uuid import UUID, uuid5
from zoneinfo import ZoneInfo

from sqlalchemy import and_, insert, or_, select, text
from sqlalchemy.exc import SQLAlchemyError

from healthcare_voice_agent.booking import tables as t

UTC = timezone.utc
_NAMESPACE = UUID("710cf03d-a112-58c1-b8be-82a6ce90c85f")
APPOINTMENT_ID = "AP-1042-01"
CASE_ID = "1042"
CLINIC_ID = "DEMO-CLINIC"
STAFF_ID = "SYN-USER-STAFF"


class SeededBookingLinkError(RuntimeError):
    """The canonical receipt cannot be linked without changing unknown data."""


@dataclass(frozen=True)
class CanonicalSeededBooking:
    appointment_id: str
    case_id: str
    patient_id: str
    patient_display_name: str
    clinician_id: str
    clinician_display_name: str
    staff_id: str
    clinic_id: str
    appointment_type: str
    starts_at: datetime
    ends_at: datetime
    timezone: str
    status: str
    location: str
    record_version: int
    created_at: datetime
    updated_at: datetime
    fixture_rows: Mapping[str, Mapping] = field(repr=False, compare=False)

    @property
    def session_id(self) -> UUID:
        return uuid5(_NAMESPACE, f"{self.appointment_id}:bootstrap-session")

    @property
    def slot_id(self) -> UUID:
        return uuid5(_NAMESPACE, f"{self.appointment_id}:canonical-slot")

    @property
    def draft_id(self) -> UUID:
        return uuid5(_NAMESPACE, f"{self.appointment_id}:bootstrap-draft")

    @property
    def booking_request_id(self) -> UUID:
        return uuid5(_NAMESPACE, f"{self.appointment_id}:bootstrap-request")

    @property
    def provenance_event_id(self) -> UUID:
        return uuid5(_NAMESPACE, f"{self.appointment_id}:bootstrap-existing-receipt")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SeededBookingLinkError("Canonical fixture timestamps must be timezone-aware.")
    return parsed.astimezone(UTC)


def _one(rows, key, value, label):
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise SeededBookingLinkError(f"Canonical fixture must contain exactly one {label}.")
    return matches[0]


def canonical_seeded_booking(dataset: Mapping) -> CanonicalSeededBooking:
    """Extract the one approved fixture appointment after caller checksum validation."""
    try:
        tables = dataset["tables"]
        appointment = _one(tables["appointments"], "appointment_id", APPOINTMENT_ID, "appointment")
        case = _one(tables["cases"], "case_id", CASE_ID, "case")
        patient = _one(tables["patients"], "patient_id", case["patient_id"], "patient")
        clinician = _one(tables["clinicians"], "clinician_id", appointment["clinician_id"], "clinician")
        staff = _one(tables["app_users"], "user_id", STAFF_ID, "staff principal")
    except (KeyError, TypeError):
        raise SeededBookingLinkError("Canonical fixture data is incomplete.") from None

    expected = {
        "appointment_id": APPOINTMENT_ID,
        "case_id": CASE_ID,
        "appointment_type": "fracture_follow_up",
        "starts_at": "2026-09-24T10:30:00+05:30",
        "ends_at": "2026-09-24T05:20:00Z",
        "timezone": "Asia/Kolkata",
        "status": "scheduled",
        "clinician_id": "SYN-CLIN-02",
        "location": "Synthetic Fracture Clinic, Room 2",
        "record_version": 1,
        "created_at": "2026-09-10T12:00:00Z",
        "updated_at": "2026-09-17T09:00:00Z",
    }
    if any(appointment.get(key) != value for key, value in expected.items()):
        raise SeededBookingLinkError("AP-1042-01 no longer matches the approved canonical fixture.")
    if (case.get("patient_id") != "SYN-P001" or case.get("status") != "open"
            or patient.get("is_synthetic") is not True
            or clinician.get("is_synthetic") is not True
            or staff.get("role") != "staff" or staff.get("is_active") is not True):
        raise SeededBookingLinkError("Canonical synthetic case, patient, clinician, or staff identity changed.")

    starts_at, ends_at = _parse_timestamp(appointment["starts_at"]), _parse_timestamp(appointment["ends_at"])
    zone = ZoneInfo(appointment["timezone"])
    if (starts_at.astimezone(zone).strftime("%Y-%m-%d %H:%M") != "2026-09-24 10:30"
            or ends_at.astimezone(zone).strftime("%Y-%m-%d %H:%M") != "2026-09-24 10:50"):
        raise SeededBookingLinkError("Canonical appointment interval changed.")
    return CanonicalSeededBooking(
        appointment_id=appointment["appointment_id"], case_id=appointment["case_id"],
        patient_id=case["patient_id"], patient_display_name=patient["display_name"],
        clinician_id=appointment["clinician_id"], clinician_display_name=clinician["display_name"],
        staff_id=staff["user_id"], clinic_id=CLINIC_ID,
        appointment_type=appointment["appointment_type"], starts_at=starts_at, ends_at=ends_at,
        timezone=appointment["timezone"], status=appointment["status"], location=appointment["location"],
        record_version=appointment["record_version"], created_at=_parse_timestamp(appointment["created_at"]),
        updated_at=_parse_timestamp(appointment["updated_at"]),
        fixture_rows={"appointments": appointment, "cases": case, "patients": patient,
                      "clinicians": clinician, "app_users": staff})


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _same_time(left, right) -> bool:
    return _aware(left) == _aware(right)


def _hash(details: dict) -> str:
    return hashlib.sha256(json.dumps(details, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _details(canonical: CanonicalSeededBooking) -> dict:
    zone = ZoneInfo(canonical.timezone)
    return {
        "case_id": canonical.case_id,
        "patient_display_name": canonical.patient_display_name,
        "slot": {
            "slot_id": str(canonical.slot_id),
            "appointment_type": canonical.appointment_type,
            "starts_at": canonical.starts_at.astimezone(zone).isoformat(),
            "ends_at": canonical.ends_at.astimezone(zone).isoformat(),
            "timezone": canonical.timezone,
            "clinician": {"clinician_id": canonical.clinician_id,
                          "display_name": canonical.clinician_display_name},
            "location": canonical.location,
        },
        "provenance": {
            "kind": "bootstrap_existing_receipt",
            "source": "canonical_synthetic_fixture",
            "source_appointment_id": canonical.appointment_id,
            "created_new_appointment": False,
            "user_voice_booking": False,
            "audio_confirmation_recorded": False,
        },
    }


def _source_rows_are_present(connection, canonical: CanonicalSeededBooking):
    appointment = connection.execute(select(t.appointments).where(
        t.appointments.c.appointment_id == canonical.appointment_id)).mappings().one_or_none()
    case = connection.execute(select(t.cases).where(t.cases.c.case_id == canonical.case_id)).mappings().one_or_none()
    patient = connection.execute(select(t.patients).where(
        t.patients.c.patient_id == canonical.patient_id)).mappings().one_or_none()
    clinician = connection.execute(select(t.clinicians).where(
        t.clinicians.c.clinician_id == canonical.clinician_id)).mappings().one_or_none()
    staff = connection.execute(select(t.users).where(t.users.c.user_id == canonical.staff_id)).mappings().one_or_none()
    clinic = connection.execute(select(t.clinics).where(t.clinics.c.clinic_id == canonical.clinic_id)).mappings().one_or_none()
    if not all((appointment, case, patient, clinician, staff, clinic)):
        raise SeededBookingLinkError("Canonical source rows are missing; no provenance was added.")
    if (appointment["case_id"] != canonical.case_id
            or appointment["appointment_type"] != canonical.appointment_type
            or not _same_time(appointment["starts_at"], canonical.starts_at)
            or not _same_time(appointment["ends_at"], canonical.ends_at)
            or appointment["timezone"] != canonical.timezone
            or appointment["status"] != canonical.status
            or appointment["clinician_id"] != canonical.clinician_id
            or appointment["location"] != canonical.location
            or appointment["record_version"] != canonical.record_version
            or not _same_time(appointment["created_at"], canonical.created_at)
            or not _same_time(appointment["updated_at"], canonical.updated_at)
            or case["patient_id"] != canonical.patient_id or case["status"] != "open"
            or patient["display_name"] != canonical.patient_display_name
            or clinician["display_name"] != canonical.clinician_display_name
            or not staff["is_active"] or staff["role"] != "staff"
            or clinic["timezone"] != canonical.timezone):
        raise SeededBookingLinkError("Database source rows differ from the canonical fixture; no provenance was added.")


def _candidate_conflict(connection, canonical: CanonicalSeededBooking) -> bool:
    slot = connection.execute(select(t.slots.c.slot_id).where(or_(
        t.slots.c.slot_id == canonical.slot_id,
        and_(t.slots.c.clinic_id == canonical.clinic_id,
             t.slots.c.clinician_id == canonical.clinician_id,
             t.slots.c.appointment_type == canonical.appointment_type,
             t.slots.c.starts_at == canonical.starts_at,
             t.slots.c.ends_at == canonical.ends_at)))).first()
    session = connection.execute(select(t.sessions.c.session_id).where(
        t.sessions.c.session_id == canonical.session_id)).first()
    draft = connection.execute(select(t.drafts.c.draft_id).where(or_(
        t.drafts.c.draft_id == canonical.draft_id,
        t.drafts.c.booking_request_id == canonical.booking_request_id,
        t.drafts.c.appointment_id == canonical.appointment_id))).first()
    allocation = connection.execute(select(t.allocations.c.draft_id).where(or_(
        t.allocations.c.draft_id == canonical.draft_id,
        t.allocations.c.slot_id == canonical.slot_id,
        t.allocations.c.booking_request_id == canonical.booking_request_id))).first()
    return any((slot, session, draft, allocation))


def _verify_existing(connection, canonical: CanonicalSeededBooking, allocation) -> None:
    details = _details(canonical)
    session = connection.execute(select(t.sessions).where(
        t.sessions.c.session_id == canonical.session_id)).mappings().one_or_none()
    slot = connection.execute(select(t.slots).where(
        t.slots.c.slot_id == canonical.slot_id)).mappings().one_or_none()
    draft = connection.execute(select(t.drafts).where(
        t.drafts.c.draft_id == canonical.draft_id)).mappings().one_or_none()
    appointment = connection.execute(select(t.appointments).where(
        t.appointments.c.appointment_id == canonical.appointment_id)).mappings().one_or_none()
    if not all((session, slot, draft, appointment)):
        raise SeededBookingLinkError("Existing AP-1042-01 provenance is incomplete or conflicting.")
    allocation_expected_active = appointment["status"] in {"scheduled", "confirmed"}
    if (allocation["draft_id"] != canonical.draft_id
            or allocation["slot_id"] != canonical.slot_id
            or allocation["booking_request_id"] != canonical.booking_request_id
            or bool(allocation["active"]) != allocation_expected_active
            or session["clinic_id"] != canonical.clinic_id or session["user_id"] != canonical.staff_id
            or session["case_id"] != canonical.case_id or session["case_context_version"] != 1
            or session["conversation_version"] != 1 or session["active"] or session["current_draft_id"] is not None
            or slot["clinic_id"] != canonical.clinic_id or slot["appointment_type"] != canonical.appointment_type
            or not _same_time(slot["starts_at"], canonical.starts_at)
            or not _same_time(slot["ends_at"], canonical.ends_at)
            or slot["clinician_id"] != canonical.clinician_id or slot["location"] != canonical.location
            or slot["enabled"]
            or draft["booking_request_id"] != canonical.booking_request_id
            or draft["session_id"] != canonical.session_id or draft["case_id"] != canonical.case_id
            or draft["patient_id"] != canonical.patient_id or draft["case_context_version"] != 1
            or draft["slot_id"] != canonical.slot_id or draft["state"] != "booked"
            or draft["failure_code"] is not None or draft["details"] != details
            or draft["details_hash"] != _hash(details) or draft["readback_version"] is not None
            or draft["readback_at"] is not None or draft["confirmation_version"] != 1
            or draft["confirmation_event_id"] != canonical.provenance_event_id
            or draft["authorization_kind"] != "direct_request"
            or draft["appointment_id"] != canonical.appointment_id):
        raise SeededBookingLinkError("Existing AP-1042-01 provenance conflicts with the bounded bootstrap receipt.")
    if (appointment["case_id"] != canonical.case_id
            or appointment["appointment_type"] != canonical.appointment_type
            or not _same_time(appointment["starts_at"], canonical.starts_at)
            or not _same_time(appointment["ends_at"], canonical.ends_at)
            or appointment["timezone"] != canonical.timezone
            or appointment["clinician_id"] != canonical.clinician_id
            or appointment["location"] != canonical.location
            or not _same_time(appointment["created_at"], canonical.created_at)
            or appointment["record_version"] < canonical.record_version):
        raise SeededBookingLinkError("The linked appointment's immutable canonical identity changed.")
    offered = connection.execute(select(t.offers.c.slot_id).where(
        t.offers.c.slot_id == canonical.slot_id)).first()
    held = connection.execute(select(t.holds.c.draft_id).where(or_(
        t.holds.c.slot_id == canonical.slot_id, t.holds.c.draft_id == canonical.draft_id))).first()
    if offered or held:
        raise SeededBookingLinkError("Bootstrap provenance must remain disabled, unoffered, and unheld.")


def link_existing_seeded_booking(
    engine,
    canonical: CanonicalSeededBooking,
    *,
    validate_target: Callable | None = None,
    validate_canonical_rows: Callable | None = None,
) -> dict:
    """Atomically add or verify the one receipt under the shared guard lock."""
    if engine.dialect.name == "postgresql" and (
            validate_target is None or validate_canonical_rows is None):
        raise SeededBookingLinkError(
            "PostgreSQL linking requires the isolated-target and canonical-row validators.")
    connection = engine.connect()
    try:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        elif engine.dialect.name == "postgresql":
            connection.begin()
            connection.execute(text("SET LOCAL lock_timeout = '2s'"))
            connection.execute(text("SET LOCAL statement_timeout = '5s'"))
        else:
            raise SeededBookingLinkError("Existing-booking linking supports PostgreSQL and offline SQLite tests only.")
        if validate_target is not None:
            validate_target(connection, canonical)
        guard = connection.execute(select(t.guard.c.id).where(t.guard.c.id == 1).with_for_update()).first()
        if guard is None:
            raise SeededBookingLinkError("Booking guard row is unavailable; no provenance was added.")

        allocation = connection.execute(select(t.allocations).where(
            t.allocations.c.appointment_id == canonical.appointment_id)).mappings().one_or_none()
        if allocation is not None:
            _verify_existing(connection, canonical, allocation)
            connection.commit()
            return {"appointment_id": canonical.appointment_id, "linked": False,
                    "status": "already_linked", "receipt_kind": "bootstrap_existing_receipt"}
        if _candidate_conflict(connection, canonical):
            raise SeededBookingLinkError("Conflicting booking provenance already exists; no data was changed.")

        _source_rows_are_present(connection, canonical)
        if validate_canonical_rows is not None:
            validate_canonical_rows(connection, canonical)
        details = _details(canonical)
        connection.execute(insert(t.sessions).values(
            session_id=canonical.session_id, clinic_id=canonical.clinic_id,
            user_id=canonical.staff_id, case_id=canonical.case_id,
            case_context_version=1, conversation_version=1, active=False,
            current_draft_id=None, created_at=canonical.created_at))
        connection.execute(insert(t.slots).values(
            slot_id=canonical.slot_id, clinic_id=canonical.clinic_id,
            appointment_type=canonical.appointment_type, starts_at=canonical.starts_at,
            ends_at=canonical.ends_at, clinician_id=canonical.clinician_id,
            location=canonical.location, enabled=False))
        # Revision 005's direct_request branch is the only valid booked-draft
        # shape without a fabricated readback/audio receipt. The details JSON is
        # authoritative that this event is administrative bootstrap provenance,
        # not evidence of a user voice or direct booking request.
        connection.execute(insert(t.drafts).values(
            draft_id=canonical.draft_id, booking_request_id=canonical.booking_request_id,
            session_id=canonical.session_id, case_id=canonical.case_id,
            patient_id=canonical.patient_id, case_context_version=1,
            slot_id=canonical.slot_id, state="booked", failure_code=None,
            details=details, details_hash=_hash(details), hold_expires_at=canonical.starts_at,
            readback_version=None, readback_at=None, confirmation_version=1,
            confirmation_event_id=canonical.provenance_event_id,
            confirmed_at=canonical.created_at, authorization_kind="direct_request",
            appointment_id=canonical.appointment_id, created_at=canonical.created_at,
            updated_at=canonical.created_at))
        connection.execute(insert(t.allocations).values(
            slot_id=canonical.slot_id, draft_id=canonical.draft_id,
            booking_request_id=canonical.booking_request_id,
            appointment_id=canonical.appointment_id, created_at=canonical.created_at,
            active=True))
        inserted = connection.execute(select(t.allocations).where(
            t.allocations.c.appointment_id == canonical.appointment_id)).mappings().one()
        _verify_existing(connection, canonical, inserted)
        connection.commit()
        return {"appointment_id": canonical.appointment_id, "linked": True,
                "status": "linked", "receipt_kind": "bootstrap_existing_receipt"}
    except SeededBookingLinkError:
        connection.rollback()
        raise
    except SQLAlchemyError:
        connection.rollback()
        raise SeededBookingLinkError("Existing-booking provenance could not be linked atomically; no data was changed.") from None
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
