"""Four front-desk booking contracts, separate from the five clinical tools.

All input fields are required, including nullable restrictions. Dates/times are
clinic-local strings; serialized appointment timestamps carry the clinic's actual
UTC offset. New opaque selection/draft/request identifiers are UUIDs. These
schemas describe data, not authorization, availability or a confirmation detector.

Export: python -m healthcare_voice_agent.tools.frontdesk_contracts
The separate backend/controller must enforce ownership, holds and atomic writes.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Generic, Literal, TypeVar, Union
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, Field, StrictBool, StrictInt, StrictStr, model_validator

from healthcare_voice_agent.tools.contracts import (
    AppointmentType, Candidate, Clinician, ContractModel, RecordId,
    SuccessResponse, ToolContract, ToolMetadata, ToolWarning,
)

NonBlank = Annotated[StrictStr, Field(min_length=1, pattern=r"\S")]
LocalDate = Annotated[StrictStr, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
LocalTime = Annotated[StrictStr, Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")]
BookingErrorCode = Literal[
    "INVALID_ARGUMENT", "IDENTITY_UNRESOLVED", "CASE_CONTEXT_MISMATCH",
    "SESSION_CONTEXT_MISMATCH", "CLINIC_CONTEXT_MISMATCH", "CASE_NOT_FOUND",
    "RECORD_NOT_FOUND", "SLOT_NOT_OFFERED", "SLOT_UNAVAILABLE",
    "DRAFT_SUPERSEDED", "HOLD_EXPIRED", "CONFIRMATION_REQUIRED",
    "CONFIRMATION_STALE", "PATIENT_CONFLICT", "CASE_NOT_BOOKABLE",
    "TIMEOUT", "BACKEND_UNAVAILABLE", "INTERNAL_ERROR",
]


class FindAvailableSlotsArgs(ContractModel):
    appointment_type: AppointmentType
    start_date: LocalDate = Field(description="Clinic-local YYYY-MM-DD, inclusive.")
    end_date: LocalDate = Field(description="Inclusive; at most 31 calendar days including start_date.")
    earliest_start_time: LocalTime | None = Field(description="Inclusive HH:MM start-time restriction on EACH day; null means unrestricted.")
    latest_start_time: LocalTime | None = Field(description="Inclusive HH:MM start-time restriction on EACH day; null means unrestricted. Overnight windows are not accepted.")
    clinician_id: RecordId | None = Field(description="Exact known clinician ID, or null for any eligible clinician.")
    max_results: Annotated[StrictInt, Field(ge=1, le=100)]

    @model_validator(mode="after")
    def valid_window(self):
        start, end = date.fromisoformat(self.start_date), date.fromisoformat(self.end_date)
        if end < start:
            raise ValueError("end_date must not precede start_date")
        if (end - start).days > 30:
            raise ValueError("Search is limited to 31 inclusive calendar days")
        if (self.earliest_start_time is not None and self.latest_start_time is not None
                and self.earliest_start_time > self.latest_start_time):
            raise ValueError("latest_start_time must not precede earliest_start_time")
        return self


class PrepareBookingArgs(ContractModel):
    case_id: RecordId = Field(description="Must equal the server-managed confirmed case; never invent an ID.")
    slot_id: UUID = Field(description="Exact opaque slot ID returned by find_available_slots in this permitted conversation.")


class BookAppointmentArgs(ContractModel):
    case_id: RecordId
    draft_id: UUID = Field(description="Exact current prepared draft. Explicit confirmation is recorded by the controller, never supplied as a tool argument.")


class GetBookingStatusArgs(ContractModel):
    case_id: RecordId
    booking_request_id: UUID = Field(description="Persistent booking-attempt ID returned by preparation; NOT meta.request_id.")


def _validate_interval(starts_at, ends_at, timezone):
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("timezone must be an installed IANA timezone") from None
    if ends_at <= starts_at:
        raise ValueError("ends_at must be later than starts_at")
    for value in (starts_at, ends_at):
        if value.utcoffset() != value.astimezone(zone).utcoffset():
            raise ValueError("Appointment timestamps must carry the clinic timezone's offset at that instant")


class AvailableSlot(ContractModel):
    slot_id: UUID
    appointment_type: AppointmentType
    starts_at: AwareDatetime
    ends_at: AwareDatetime
    timezone: NonBlank = Field(description="IANA clinic timezone; the timestamps use its offset at their respective instants.")
    clinician: Clinician
    location: NonBlank

    @model_validator(mode="after")
    def valid_interval(self):
        _validate_interval(self.starts_at, self.ends_at, self.timezone)
        if not self.clinician.display_name.strip():
            raise ValueError("clinician display_name must be nonblank")
        return self


class AvailableSlotsData(ContractModel):
    slots: Annotated[list[AvailableSlot], Field(max_length=100)]
    as_of: AwareDatetime = Field(description="Server observation time; availability is not a reservation.")
    has_more: StrictBool = Field(description="Additional matching slots existed at as_of beyond the requested result limit.")

    @model_validator(mode="after")
    def ordered_unique_slots(self):
        if len({slot.slot_id for slot in self.slots}) != len(self.slots):
            raise ValueError("slots must have unique IDs")
        if any(a.starts_at > b.starts_at for a, b in zip(self.slots, self.slots[1:])):
            raise ValueError("slots must be ordered earliest first")
        if any(slot.starts_at < self.as_of for slot in self.slots):
            raise ValueError("Available slots must not start before as_of")
        if self.has_more and not self.slots:
            raise ValueError("An empty result cannot have has_more=true")
        return self


class PreparedBookingData(ContractModel):
    draft_id: UUID
    booking_request_id: UUID
    hold_expires_at: AwareDatetime = Field(description="Exclusive expiry boundary; retrying the same active selection never extends it.")
    case_id: RecordId
    patient_display_name: NonBlank
    slot: AvailableSlot = Field(description="Authoritative appointment details to read back before asking for confirmation.")

    @model_validator(mode="after")
    def hold_precedes_appointment(self):
        if self.hold_expires_at > self.slot.starts_at:
            raise ValueError("A hold cannot remain valid after the appointment starts")
        if self.draft_id == self.booking_request_id:
            raise ValueError("Draft and booking request have distinct identifiers")
        return self


class BookedAppointment(ContractModel):
    appointment_id: RecordId
    case_id: RecordId
    patient_display_name: NonBlank
    slot_id: UUID
    appointment_type: AppointmentType
    starts_at: AwareDatetime
    ends_at: AwareDatetime
    timezone: NonBlank
    clinician: Clinician
    location: NonBlank
    status: Literal["confirmed"]

    @model_validator(mode="after")
    def valid_interval(self):
        _validate_interval(self.starts_at, self.ends_at, self.timezone)
        if not self.clinician.display_name.strip():
            raise ValueError("clinician display_name must be nonblank")
        return self


class BookAppointmentData(ContractModel):
    booking_request_id: UUID
    appointment: BookedAppointment


class BookingFailure(ContractModel):
    code: Literal["DRAFT_SUPERSEDED", "SLOT_UNAVAILABLE", "PATIENT_CONFLICT", "CASE_NOT_BOOKABLE"]
    message: NonBlank = Field(description="Persisted safe failure reason; no SQL, secrets or another patient's data.")


class _BookingStatusBase(ContractModel):
    booking_request_id: UUID
    draft_id: UUID
    as_of: AwareDatetime
    hold_expires_at: AwareDatetime = Field(description="Historical hold expiry; it does not undo an already committed appointment.")


class PendingBookingStatus(_BookingStatusBase):
    booking_state: Literal["pending"]
    appointment: None
    failure: None

    @model_validator(mode="after")
    def active_hold(self):
        if self.hold_expires_at <= self.as_of:
            raise ValueError("An uncommitted hold at/past its expiry is expired, not pending")
        return self


class BookedBookingStatus(_BookingStatusBase):
    booking_state: Literal["booked"]
    appointment: BookedAppointment
    failure: None


class ExpiredBookingStatus(_BookingStatusBase):
    booking_state: Literal["expired"]
    appointment: None
    failure: None

    @model_validator(mode="after")
    def elapsed_hold(self):
        if self.hold_expires_at > self.as_of:
            raise ValueError("An uncommitted hold is not expired before its expiry")
        return self


class FailedBookingStatus(_BookingStatusBase):
    booking_state: Literal["failed"]
    appointment: None
    failure: BookingFailure


BookingStatusData = Annotated[
    Union[PendingBookingStatus, BookedBookingStatus, ExpiredBookingStatus, FailedBookingStatus],
    Field(discriminator="booking_state"),
]


class BookingToolError(ContractModel):
    code: BookingErrorCode
    message: NonBlank
    retryable: StrictBool = Field(description="Only transient failures; for an uncertain commit, recover using get_booking_status before announcing an outcome.")
    candidates: Annotated[list[Candidate], Field(max_length=0)] = Field(description="Always empty; never disclose another case or substitute a different slot.")


ToolNameT = TypeVar("ToolNameT", bound=str)


class BookingErrorResponse(ContractModel, Generic[ToolNameT]):
    contract_version: Literal["1.0.0"]
    status: Literal["error"]
    data: None
    error: BookingToolError
    warnings: Annotated[list[ToolWarning], Field(max_length=0)]
    meta: ToolMetadata[ToolNameT]


FindAvailableSlotsResponse = Annotated[Union[
    SuccessResponse[AvailableSlotsData, Literal["find_available_slots"]],
    BookingErrorResponse[Literal["find_available_slots"]],
], Field(discriminator="status")]
PrepareBookingResponse = Annotated[Union[
    SuccessResponse[PreparedBookingData, Literal["prepare_booking"]],
    BookingErrorResponse[Literal["prepare_booking"]],
], Field(discriminator="status")]
BookAppointmentResponse = Annotated[Union[
    SuccessResponse[BookAppointmentData, Literal["book_appointment"]],
    BookingErrorResponse[Literal["book_appointment"]],
], Field(discriminator="status")]
GetBookingStatusResponse = Annotated[Union[
    SuccessResponse[BookingStatusData, Literal["get_booking_status"]],
    BookingErrorResponse[Literal["get_booking_status"]],
], Field(discriminator="status")]


FRONTDESK_TOOLS: dict[str, ToolContract] = {
    "find_available_slots": ToolContract(
        "find_available_slots",
        "Find at most max_results available slots in the configured clinic timezone, earliest first. "
        "Date bounds and daily appointment START-time bounds are inclusive; null restrictions mean any. "
        "At most 31 inclusive calendar days per search. Duration, hours and clinician eligibility are backend-owned. "
        "Needs permitted clinic/conversation context, not a confirmed patient. Empty slots is success. "
        "Does not reserve a slot; availability may change before preparation. Records session offer receipts "
        "so preparation can reject invented or unoffered slot IDs; this metadata write is why read_only is false.",
        FindAvailableSlotsArgs, FindAvailableSlotsResponse, read_only=False,
    ),
    "prepare_booking": ToolContract(
        "prepare_booking",
        "Prepare the exact offered slot for the server-confirmed case and permitted conversation. Recheck "
        "availability, place a short hold (initially up to two minutes), and return authoritative readback details. "
        "This is NOT a confirmed appointment. Retrying the same active selection returns the same draft and "
        "booking_request_id without extending expiry. Changing selection supersedes the old draft and releases "
        "its hold. Read back details and ask Shall I book this? before explicit controller-recorded confirmation.",
        PrepareBookingArgs, PrepareBookingResponse, read_only=False,
    ),
    "book_appointment": ToolContract(
        "book_appointment",
        "Commit only the current held draft for the confirmed case/permitted conversation AFTER explicit user "
        "confirmation of those exact readback details. Confirmation is recorded by the trusted conversation "
        "controller, never a confirmed boolean or LLM assertion. Subsequent correction invalidates confirmation. "
        "Appointment, allocation and successful request outcome commit atomically. Repeating the same successfully "
        "booked draft returns the same appointment; do not create a new request after a lost response. "
        "Use get_booking_status for uncertain outcomes and announce success only from a committed result.",
        BookAppointmentArgs, BookAppointmentResponse, read_only=False,
    ),
    "get_booking_status": ToolContract(
        "get_booking_status",
        "Look up the persistent booking attempt for the server-confirmed case/permitted conversation. "
        "Read-only recovery after a timeout or lost booking response; never create, retry or cancel a booking. "
        "status=ok means lookup success, not booking success: only data.booking_state=booked establishes a "
        "committed appointment. pending is unresolved, expired needs new preparation, failed has a recorded "
        "terminal reason. A committed attempt stays booked even after its former hold expiry.",
        GetBookingStatusArgs, GetBookingStatusResponse, read_only=True,
    ),
}

FRONTDESK_RULES = {
    "required_fields": "All argument fields are required; nullable restrictions must be supplied as null. Unknown fields are rejected.",
    "time": "Search dates use YYYY-MM-DD and daily start-time bounds HH:MM in the server-configured clinic IANA timezone. Both bounds are inclusive; overnight windows require separate searches. At most 31 inclusive days. Appointment timestamps carry that zone's actual offset, including DST changes.",
    "identity": "Only availability search omits case_id. All other tools require the confirmed server case, session/clinic authorization and current context version. The LLM cannot set session, identity, confirmation or authorization fields. Nonexistent and cross-case child IDs must not reveal cross-case existence.",
    "ids": "IDs come from tool results or resolved context. slot_id, draft_id and booking_request_id are opaque UUIDs. booking_request_id persists across invocations; meta.request_id uniquely identifies one invocation and must not be used as an idempotency key.",
    "holds": "Preparation is not booking. Holds initially last at most two minutes and cannot outlast appointment start. At exact expiry the hold is expired. Same active selection retries do not extend expiry. A selection change supersedes the old draft and releases its hold.",
    "confirmation": "The controller records completed readback and explicit user confirmation of the exact current draft. No confirmed argument exists. A later user correction/context revision invalidates confirmation before booking. The backend must validate persisted confirmation under the booking transaction.",
    "atomicity": "A successful transaction stores the appointment, exclusive allocation and successful booking request outcome together. A retry for an already committed draft returns the original appointment, even after hold expiry.",
    "recovery": "Get status never mutates the booking or retries it. pending is not failure; booked requires appointment details; expired and failed contain no appointment. A successful lookup envelope is distinct from a successful booking.",
    "validation_boundary": "Pydantic enforces shapes and local temporal constraints only. Backend checks authorization, offered-slot membership, clinic configuration, availability, ownership, expiry, confirmation and idempotency. Output fields/validators exceed some provider tool-schema subsets; adapt provider inputs without dropping backend validation.",
}


def export_frontdesk_contracts() -> dict:
    return {
        "contract_version": "1.0.0",
        "schema_dialect": "JSON Schema 2020-12",
        "conventions": FRONTDESK_RULES,
        "tools": [tool.export() for tool in FRONTDESK_TOOLS.values()],
    }


def validate_example_bundle(example_path: str | Path) -> int:
    bundle = json.loads(Path(example_path).read_text(encoding="utf-8"))
    for example in bundle["examples"]:
        tool = FRONTDESK_TOOLS[example["tool"]]
        tool.validate_arguments_json(json.dumps(example["arguments"]))
        tool.validate_response_json(json.dumps(example["expected_response"]))
    return len(bundle["examples"])


if __name__ == "__main__":
    print(json.dumps(export_frontdesk_contracts(), indent=2))
