"""Transactional front-desk tools plus trusted, non-LLM controller hooks.

Construct with an injected SQLAlchemy Engine. Nothing connects on import. Every
mutation uses the same guard-row lock; all future calendar writers must follow
that protocol. This intentionally serializes a small single-clinic prototype,
not a high-throughput scheduling service. Alembic alone creates production DDL.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from time import perf_counter
from types import SimpleNamespace
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from pydantic import ValidationError
from sqlalchemy import and_, delete, insert, or_, select, text, update
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from healthcare_voice_agent.booking import tables as t
from healthcare_voice_agent.booking.scheduling import ScheduleRule, generate_slots
from healthcare_voice_agent.tools.frontdesk_contracts import FRONTDESK_TOOLS, FindAvailableSlotsArgs

UTC = timezone.utc
MESSAGES = {
    "INVALID_ARGUMENT": "Arguments do not match this tool's contract.",
    "IDENTITY_UNRESOLVED": "Confirm the patient case before continuing.",
    "CASE_CONTEXT_MISMATCH": "The case does not match the current confirmed context.",
    "SESSION_CONTEXT_MISMATCH": "The conversation context changed or is not permitted.",
    "CLINIC_CONTEXT_MISMATCH": "The clinic context is unavailable or does not match.",
    "CASE_NOT_FOUND": "The confirmed case is unavailable.",
    "RECORD_NOT_FOUND": "The requested booking record is unavailable in this context.",
    "SLOT_NOT_OFFERED": "Search for and select an offered slot before preparing it.",
    "SLOT_UNAVAILABLE": "That exact slot is no longer available. Search again.",
    "SLOT_TIME_MISMATCH": "The selected slot does not start at the requested time. Search the exact requested start time before trying again.",
    "DRAFT_SUPERSEDED": "That draft was replaced. Use the current selection.",
    "HOLD_EXPIRED": "The hold expired. Prepare an available slot again.",
    "CONFIRMATION_REQUIRED": "Read back this draft and obtain explicit user confirmation.",
    "CONFIRMATION_STALE": "The conversation changed after confirmation. Confirm the current draft again.",
    "PATIENT_CONFLICT": "The patient already has an overlapping appointment or active hold.",
    "CASE_NOT_BOOKABLE": "This case is not eligible for this prepared booking.",
    "TIMEOUT": "The operation timed out. For a booking attempt, check its booking_request_id before announcing an outcome.",
    "BACKEND_UNAVAILABLE": "The booking backend is unavailable. No booking outcome can be inferred.",
    "INTERNAL_ERROR": "The booking operation could not be verified. Check its request status before announcing an outcome.",
}


class BookingError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(MESSAGES[code])


@dataclass(frozen=True)
class FrontDeskContext:
    """Trusted server snapshot, never part of a tool's JSON arguments."""
    session_id: UUID
    clinic_id: str
    user_id: str
    case_id: str | None
    case_context_version: int
    conversation_version: int


def _aware(value: datetime) -> datetime:
    # SQLite's isolated test dialect loses tzinfo. Production TIMESTAMPTZ does not.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _jsonable(value):
    if isinstance(value, (datetime, UUID)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    raise TypeError("Unsupported response value")


def _hash(details: dict) -> str:
    return hashlib.sha256(json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class BookingService:
    def __init__(self, engine, *, test_clock=None, hold_seconds: int = 120):
        if type(hold_seconds) is not int or not 1 <= hold_seconds <= 120:
            raise ValueError("Hold duration must be 1 to 120 seconds")
        if engine.dialect.name not in {"postgresql", "sqlite"}:
            raise ValueError("Booking requires PostgreSQL (SQLite is offline-test only)")
        if test_clock is not None and engine.dialect.name != "sqlite":
            raise ValueError("Production expiry uses the PostgreSQL server clock")
        self.engine = engine
        self._clock = test_clock
        self.hold_seconds = hold_seconds

    @contextmanager
    def _transaction(self):
        with self.engine.connect() as connection:
            if self.engine.dialect.name == "sqlite":
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
                connection.execute(text("SET LOCAL lock_timeout = '2s'"))
                connection.execute(text("SET LOCAL statement_timeout = '5s'"))
            try:
                row = connection.execute(select(t.guard.c.id).where(t.guard.c.id == 1).with_for_update()).first()
                if row is None:
                    raise BookingError("BACKEND_UNAVAILABLE")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _now(self, connection):
        if self.engine.dialect.name == "postgresql":
            return _aware(connection.execute(select(text("clock_timestamp()"))).scalar_one())
        value = self._clock() if self._clock else datetime.now(UTC)
        if value.tzinfo is None:
            raise ValueError("Test clock must be timezone aware")
        return value.astimezone(UTC)

    @staticmethod
    def _context(row):
        return FrontDeskContext(**{key: row[key] for key in FrontDeskContext.__dataclass_fields__})

    def _check(self, connection, context, *, case_id=None):
        row = connection.execute(select(t.sessions).where(t.sessions.c.session_id == context.session_id)).mappings().first()
        if not row or not row["active"] or self._context(row) != context:
            raise BookingError("SESSION_CONTEXT_MISMATCH")
        allowed = connection.execute(select(t.users.c.user_id).where(
            t.users.c.user_id == context.user_id, t.users.c.is_active.is_(True),
            t.users.c.role.in_(["staff", "clinician"]))).first()
        if not allowed:
            raise BookingError("SESSION_CONTEXT_MISMATCH")
        if case_id is not None:
            if context.case_id is None:
                raise BookingError("IDENTITY_UNRESOLVED")
            if case_id != context.case_id:
                raise BookingError("CASE_CONTEXT_MISMATCH")
            if connection.execute(select(t.cases.c.case_id).where(t.cases.c.case_id == case_id)).first() is None:
                raise BookingError("CASE_NOT_FOUND")
        return row

    def _clinic(self, connection, clinic_id):
        row = connection.execute(select(t.clinics).where(t.clinics.c.clinic_id == clinic_id)).mappings().first()
        if row is None:
            raise BookingError("CLINIC_CONTEXT_MISMATCH")
        return row

    def _patient(self, connection, case_id):
        row = connection.execute(select(t.cases.c.patient_id, t.cases.c.status, t.patients.c.display_name)
            .join(t.patients, t.patients.c.patient_id == t.cases.c.patient_id)
            .where(t.cases.c.case_id == case_id)).mappings().first()
        if row is None:
            raise BookingError("CASE_NOT_FOUND")
        return row

    def _slot(self, connection, context, slot_id):
        row = connection.execute(select(t.slots).where(t.slots.c.slot_id == slot_id,
            t.slots.c.clinic_id == context.clinic_id)).mappings().first()
        if row is None:
            raise BookingError("RECORD_NOT_FOUND")
        return row

    def get_slot_details(self, context: FrontDeskContext, *, slot_id) -> dict:
        """Read an offered slot for consistency checks; never prepare or change it.

        Slot start/end identity is immutable through the schedule API. Preparation
        still independently rechecks availability, ownership, and offer history.
        """
        try:
            selected_id = UUID(str(slot_id))
        except (ValueError, TypeError, AttributeError):
            raise BookingError("INVALID_ARGUMENT") from None
        with self._transaction() as connection:
            self._check(connection, context, case_id=context.case_id)
            if context.case_id is None:
                raise BookingError("IDENTITY_UNRESOLVED")
            slot = self._slot(connection, context, selected_id)
            offered = connection.execute(select(t.offers.c.slot_id).where(
                t.offers.c.session_id == context.session_id,
                t.offers.c.slot_id == selected_id)).first()
            if offered is None:
                raise BookingError("SLOT_NOT_OFFERED")
            return self._slot_data(connection, slot, self._clinic(connection, context.clinic_id))

    def _slot_data(self, connection, slot, clinic):
        name = connection.execute(select(t.clinicians.c.display_name).where(
            t.clinicians.c.clinician_id == slot["clinician_id"])).scalar_one()
        zone = ZoneInfo(clinic["timezone"])
        return {"slot_id": str(slot["slot_id"]), "appointment_type": slot["appointment_type"],
            "starts_at": _aware(slot["starts_at"]).astimezone(zone).isoformat(),
            "ends_at": _aware(slot["ends_at"]).astimezone(zone).isoformat(),
            "timezone": clinic["timezone"], "clinician": {"clinician_id": slot["clinician_id"], "display_name": name},
            "location": slot["location"]}

    def _busy(self, connection, slot, now, *, patient_id=None, exclude_draft=None, exclude_appointment=None):
        """Check clinician or patient conflicts, including legacy appointments.

        Unknown legacy end times conservatively block later overlapping searches;
        no guessed duration is used. Intervals are half-open [start, end).
        """
        start, end = _aware(slot["starts_at"]), _aware(slot["ends_at"])
        if patient_id is None and connection.execute(select(t.allocations.c.slot_id).where(
                t.allocations.c.slot_id == slot["slot_id"], t.allocations.c.active.is_(True),
                t.allocations.c.appointment_id != exclude_appointment if exclude_appointment else True)).first():
            return True
        query = select(t.appointments.c.appointment_id).where(
            t.appointments.c.status.in_(["scheduled", "confirmed"]),
            t.appointments.c.starts_at < end,
            or_(t.appointments.c.ends_at.is_(None), t.appointments.c.ends_at > start))
        if exclude_appointment is not None:
            query = query.where(t.appointments.c.appointment_id != exclude_appointment)
        if patient_id is None:
            query = query.where(t.appointments.c.clinician_id == slot["clinician_id"])
        else:
            query = query.join(t.cases, t.cases.c.case_id == t.appointments.c.case_id).where(t.cases.c.patient_id == patient_id)
        if connection.execute(query.limit(1)).first():
            return True
        query = select(t.holds.c.draft_id).select_from(t.holds.join(t.slots, t.slots.c.slot_id == t.holds.c.slot_id)
            .join(t.drafts, t.drafts.c.draft_id == t.holds.c.draft_id)).where(
            t.holds.c.expires_at > now, t.slots.c.starts_at < end, t.slots.c.ends_at > start)
        if exclude_draft is not None:
            query = query.where(t.holds.c.draft_id != exclude_draft)
        if patient_id is None:
            query = query.where(t.slots.c.clinician_id == slot["clinician_id"])
        else:
            query = query.join(t.cases, t.cases.c.case_id == t.drafts.c.case_id).where(t.cases.c.patient_id == patient_id)
        return connection.execute(query.limit(1)).first() is not None

    def _owned_draft(self, connection, context, *, draft_id=None, booking_request_id=None):
        predicate = t.drafts.c.draft_id == draft_id if draft_id else t.drafts.c.booking_request_id == booking_request_id
        row = connection.execute(select(t.drafts).where(predicate,
            t.drafts.c.session_id == context.session_id, t.drafts.c.case_id == context.case_id)).mappings().first()
        if row is None:
            raise BookingError("RECORD_NOT_FOUND")
        return row

    def _terminal(self, connection, draft, now, state, code=None):
        connection.execute(delete(t.holds).where(t.holds.c.draft_id == draft["draft_id"]))
        connection.execute(update(t.drafts).where(t.drafts.c.draft_id == draft["draft_id"]).values(
            state=state, failure_code=code, updated_at=now))

    def _expire_holds(self, connection, now):
        # No scheduler is needed for correctness. Reads also compare expiry.
        connection.execute(update(t.drafts).where(t.drafts.c.state == "pending", t.drafts.c.hold_expires_at <= now)
            .values(state="expired", updated_at=now))
        connection.execute(delete(t.holds).where(t.holds.c.expires_at <= now))

    def _current_pending(self, connection, context, draft, session, now):
        if draft["state"] == "failed":
            raise BookingError(draft["failure_code"])
        if draft["state"] == "expired" or _aware(draft["hold_expires_at"]) <= now:
            if draft["state"] == "pending":
                self._terminal(connection, draft, now, "expired")
            raise BookingError("HOLD_EXPIRED")
        if draft["state"] != "pending" or session["current_draft_id"] != draft["draft_id"]:
            raise BookingError("DRAFT_SUPERSEDED")
        if draft["case_context_version"] != context.case_context_version:
            raise BookingError("CASE_CONTEXT_MISMATCH")

    def invoke(self, tool_name: str, arguments: str | dict, *, context: FrontDeskContext) -> dict:
        """Synchronous, validated tool dispatcher. No raw database exceptions escape.

        TIMEOUT may occur after COMMIT reached the server. Never record a failed
        booking solely from a connection exception; get_booking_status recovers.
        """
        if tool_name not in FRONTDESK_TOOLS:
            raise ValueError("Unknown front-desk tool")
        started, request_id = perf_counter(), uuid4()
        retrieved_at = datetime.now(UTC)
        data, error = None, None
        try:
            payload = arguments if isinstance(arguments, str) else json.dumps(arguments)
            args = FRONTDESK_TOOLS[tool_name].validate_arguments_json(payload)
        except (ValidationError, ValueError, TypeError):
            error = BookingError("INVALID_ARGUMENT")
        else:
            try:
                with self._transaction() as connection:
                    retrieved_at = self._now(connection)
                    try:
                        session = self._check(connection, context, case_id=getattr(args, "case_id", None))
                        data = getattr(self, "_" + tool_name)(connection, context, session, args, retrieved_at)
                    except BookingError as exc:
                        # Persist intentional terminal outcomes (expired,
                        # superseded, unavailable). SQL failures still roll back.
                        error = exc
            except BookingError as exc:
                error = exc
            except TimeoutError:
                error = BookingError("TIMEOUT")
            except DBAPIError as exc:
                code = getattr(exc.orig, "sqlstate", None)
                error = BookingError("TIMEOUT" if code in {"57014", "55P03"} else "BACKEND_UNAVAILABLE")
            except SQLAlchemyError:
                error = BookingError("BACKEND_UNAVAILABLE")
            except Exception:
                error = BookingError("INTERNAL_ERROR")
        result = {"contract_version": "1.0.0", "status": "error" if error else "ok",
            "data": None if error else data,
            "error": None if error is None else {"code": error.code, "message": MESSAGES[error.code],
                "retryable": error.code in {"TIMEOUT", "BACKEND_UNAVAILABLE"}, "candidates": []},
            "warnings": [], "meta": {"request_id": str(request_id), "tool_name": tool_name,
                "retrieved_at": retrieved_at.isoformat(), "duration_ms": int((perf_counter() - started) * 1000),
                "case_context_version": context.case_context_version if context.case_id is not None and tool_name != "find_available_slots" else None}}
        try:
            return FRONTDESK_TOOLS[tool_name].validate_response_json(json.dumps(result, default=_jsonable)).model_dump(mode="json")
        except (ValidationError, ValueError, TypeError):
            # The transaction may already have committed. Hide malformed internal
            # values and preserve the recovery instruction rather than infer loss.
            result.update(status="error", data=None, error={"code": "INTERNAL_ERROR", "message": MESSAGES["INTERNAL_ERROR"], "retryable": False, "candidates": []})
            return result

    async def invoke_async(self, tool_name, arguments, *, context):
        # Cancellation of the waiter cannot undo a database COMMIT. Recover by
        # booking_request_id; don't start an unrelated replacement booking.
        return await asyncio.to_thread(self.invoke, tool_name, arguments, context=context)

    def _matching_available_slots(self, connection, context, args, now, *, clinic=None, location=None):
        """Yield all slots matching the public finder's availability semantics.

        The caller decides whether to cap results or write offer receipts. Patient
        conflicts are intentionally not checked because find_available_slots does
        not require a confirmed case and does not check them either.
        """
        clinic = clinic or self._clinic(connection, context.clinic_id)
        zone = ZoneInfo(clinic["timezone"])
        start = datetime.combine(date.fromisoformat(args.start_date), time.min, zone)
        end = datetime.combine(date.fromisoformat(args.end_date), time.max, zone)
        query = select(t.slots).where(t.slots.c.clinic_id == context.clinic_id,
            t.slots.c.enabled.is_(True), t.slots.c.appointment_type == args.appointment_type,
            t.slots.c.starts_at > now, t.slots.c.starts_at >= start.astimezone(UTC), t.slots.c.starts_at <= end.astimezone(UTC))
        if args.clinician_id is not None:
            query = query.where(t.slots.c.clinician_id == args.clinician_id)
        if location is not None:
            query = query.where(t.slots.c.location == location)
        for slot in connection.execute(query.order_by(t.slots.c.starts_at, t.slots.c.slot_id)).mappings():
            local_time = _aware(slot["starts_at"]).astimezone(zone).strftime("%H:%M")
            if args.earliest_start_time is not None and local_time < args.earliest_start_time:
                continue
            if args.latest_start_time is not None and local_time > args.latest_start_time:
                continue
            if not self._busy(connection, slot, now):
                yield slot

    def _find_available_slots(self, connection, context, session, args, now):
        clinic = self._clinic(connection, context.clinic_id)
        found = []
        for slot in self._matching_available_slots(connection, context, args, now, clinic=clinic):
            found.append(self._slot_data(connection, slot, clinic))
            if len(found) > args.max_results:
                break
        selected = found[:args.max_results]
        for slot in selected:
            slot_id = UUID(slot["slot_id"])
            connection.execute(delete(t.offers).where(t.offers.c.session_id == context.session_id, t.offers.c.slot_id == slot_id))
            connection.execute(insert(t.offers).values(session_id=context.session_id, slot_id=slot_id, offered_at=now))
        return {"slots": selected, "as_of": now.isoformat(), "has_more": len(found) > args.max_results}

    def _prepare_booking(self, connection, context, session, args, now, *, reschedule_source=None):
        self._expire_holds(connection, now)
        source_id = reschedule_source["appointment_id"] if reschedule_source else None
        patient = self._patient(connection, args.case_id)
        if patient["status"] != "open":
            raise BookingError("CASE_NOT_BOOKABLE")
        slot = self._slot(connection, context, args.slot_id)
        if not connection.execute(select(t.offers.c.slot_id).where(t.offers.c.session_id == context.session_id,
                t.offers.c.slot_id == args.slot_id)).first():
            raise BookingError("SLOT_NOT_OFFERED")
        current = None
        if session["current_draft_id"]:
            current = self._owned_draft(connection, context, draft_id=session["current_draft_id"])
            intent = self._reschedule_intent(connection, current["draft_id"])
            same_purpose = (intent is None and source_id is None) or (
                intent is not None and source_id == intent["source_appointment_id"]
                and reschedule_source["record_version"] == intent["source_record_version"])
            if (same_purpose and current["state"] == "pending" and current["slot_id"] == args.slot_id
                    and current["case_context_version"] == context.case_context_version
                    and _aware(current["hold_expires_at"]) > now):
                held = connection.execute(select(t.holds.c.draft_id).where(t.holds.c.draft_id == current["draft_id"],
                    t.holds.c.slot_id == args.slot_id, t.holds.c.expires_at > now)).first()
                retry_details = {"case_id": args.case_id, "patient_display_name": patient["display_name"],
                    "slot": self._slot_data(connection, slot, self._clinic(connection, context.clinic_id))}
                code = None
                if current["patient_id"] != patient["patient_id"]:
                    code = "CASE_NOT_BOOKABLE"
                elif (not held or not slot["enabled"] or _aware(slot["starts_at"]) <= now
                        or _hash(retry_details) != current["details_hash"]
                        or self._busy(connection, slot, now, exclude_draft=current["draft_id"], exclude_appointment=source_id)):
                    code = "SLOT_UNAVAILABLE"
                elif self._busy(connection, slot, now, patient_id=patient["patient_id"], exclude_draft=current["draft_id"], exclude_appointment=source_id):
                    code = "PATIENT_CONFLICT"
                if code:
                    self._terminal(connection, current, now, "failed", code)
                    raise BookingError(code)
                return self._prepared(current)
        exclude = current["draft_id"] if current else None
        if current and current["state"] == "pending":
            # Selecting a different offered slot invalidates the old choice even
            # if the new one has just become unavailable. Never silently retain
            # an earlier confirmable booking after a correction.
            self._terminal(connection, current, now, "failed", "DRAFT_SUPERSEDED")
            connection.execute(update(t.sessions).where(t.sessions.c.session_id == context.session_id)
                .values(current_draft_id=None))
        if not slot["enabled"] or _aware(slot["starts_at"]) <= now or self._busy(connection, slot, now, exclude_draft=exclude, exclude_appointment=source_id):
            raise BookingError("SLOT_UNAVAILABLE")
        if self._busy(connection, slot, now, patient_id=patient["patient_id"], exclude_draft=exclude, exclude_appointment=source_id):
            raise BookingError("PATIENT_CONFLICT")
        draft_id, request_id = uuid4(), uuid4()
        details = {"case_id": args.case_id, "patient_display_name": patient["display_name"],
            "slot": self._slot_data(connection, slot, self._clinic(connection, context.clinic_id))}
        expiry = min(now + timedelta(seconds=self.hold_seconds), _aware(slot["starts_at"]))
        row = {"draft_id": draft_id, "booking_request_id": request_id, "session_id": context.session_id,
            "case_id": args.case_id, "patient_id": patient["patient_id"],
            "case_context_version": context.case_context_version, "slot_id": args.slot_id,
            "state": "pending", "failure_code": None, "details": details, "details_hash": _hash(details),
            "hold_expires_at": expiry, "created_at": now, "updated_at": now}
        connection.execute(insert(t.drafts).values(**row))
        connection.execute(insert(t.holds).values(slot_id=args.slot_id, draft_id=draft_id, expires_at=expiry))
        connection.execute(update(t.sessions).where(t.sessions.c.session_id == context.session_id).values(current_draft_id=draft_id))
        return self._prepared(row)

    @staticmethod
    def _prepared(draft):
        return {"draft_id": str(draft["draft_id"]), "booking_request_id": str(draft["booking_request_id"]),
            "hold_expires_at": _aware(draft["hold_expires_at"]).isoformat(), **draft["details"]}

    @staticmethod
    def _booked(draft):
        details = draft["details"]
        return {"appointment_id": draft["appointment_id"], "case_id": draft["case_id"],
            "patient_display_name": details["patient_display_name"], **details["slot"], "status": "confirmed"}

    def _book_appointment(self, connection, context, session, args, now, *, allow_reschedule=False):
        draft = self._owned_draft(connection, context, draft_id=args.draft_id)
        if self._reschedule_intent(connection, draft["draft_id"]) is not None and not allow_reschedule:
            raise BookingError("INVALID_ARGUMENT")
        if draft["state"] == "booked":
            return {"booking_request_id": str(draft["booking_request_id"]), "appointment": self._booked(draft)}
        self._current_pending(connection, context, draft, session, now)
        if draft["confirmation_version"] is None:
            raise BookingError("CONFIRMATION_REQUIRED")
        if draft["confirmation_version"] != context.conversation_version:
            raise BookingError("CONFIRMATION_STALE")
        hold = connection.execute(select(t.holds).where(t.holds.c.draft_id == draft["draft_id"],
            t.holds.c.slot_id == draft["slot_id"], t.holds.c.expires_at > now)).first()
        patient = self._patient(connection, args.case_id)
        slot = self._slot(connection, context, draft["slot_id"])
        current_details = {"case_id": args.case_id, "patient_display_name": patient["display_name"],
            "slot": self._slot_data(connection, slot, self._clinic(connection, context.clinic_id))}
        code = None
        if patient["status"] != "open" or patient["patient_id"] != draft["patient_id"]:
            code = "CASE_NOT_BOOKABLE"
        elif (not hold or not slot["enabled"] or _aware(slot["starts_at"]) <= now
                or _hash(current_details) != draft["details_hash"]
                or self._busy(connection, slot, now, exclude_draft=draft["draft_id"])):
            code = "SLOT_UNAVAILABLE"
        elif self._busy(connection, slot, now, patient_id=patient["patient_id"], exclude_draft=draft["draft_id"]):
            code = "PATIENT_CONFLICT"
        if code:
            self._terminal(connection, draft, now, "failed", code)
            raise BookingError(code)
        appointment_id = "AP-" + uuid4().hex
        connection.execute(insert(t.appointments).values(appointment_id=appointment_id, case_id=args.case_id,
            appointment_type=slot["appointment_type"], starts_at=_aware(slot["starts_at"]), ends_at=_aware(slot["ends_at"]),
            timezone=current_details["slot"]["timezone"], status="confirmed", clinician_id=slot["clinician_id"],
            location=slot["location"], record_version=1, created_at=now, updated_at=now))
        connection.execute(insert(t.allocations).values(slot_id=slot["slot_id"], draft_id=draft["draft_id"],
            booking_request_id=draft["booking_request_id"], appointment_id=appointment_id, created_at=now))
        connection.execute(delete(t.holds).where(t.holds.c.draft_id == draft["draft_id"]))
        connection.execute(update(t.drafts).where(t.drafts.c.draft_id == draft["draft_id"]).values(
            state="booked", appointment_id=appointment_id, updated_at=now))
        return {"booking_request_id": str(draft["booking_request_id"]),
            "appointment": self._booked({**draft, "appointment_id": appointment_id})}

    def _get_booking_status(self, connection, context, session, args, now):
        draft = self._owned_draft(connection, context, booking_request_id=args.booking_request_id)
        state = draft["state"]
        if state == "pending" and _aware(draft["hold_expires_at"]) <= now:
            state = "expired"  # Read-only derivation, no retry/creation/cancellation.
        return {"booking_request_id": str(draft["booking_request_id"]), "draft_id": str(draft["draft_id"]),
            "as_of": now.isoformat(), "hold_expires_at": _aware(draft["hold_expires_at"]).isoformat(),
            "booking_state": state, "appointment": self._booked(draft) if state == "booked" else None,
            "failure": {"code": draft["failure_code"], "message": MESSAGES[draft["failure_code"]]} if state == "failed" else None}

    # ---------- Trusted administrative/controller API, NOT LLM tools ----------

    @staticmethod
    def _catalog_restrictions(restrictions, context):
        """Validate finder restrictions without changing the public tool contract.

        The optional scope/location keys support trusted controller wrappers. They
        are not added to FindAvailableSlotsArgs or exposed as LLM tool arguments.
        """
        if isinstance(restrictions, FindAvailableSlotsArgs):
            raw = restrictions.model_dump(mode="json")
        elif isinstance(restrictions, dict):
            raw = dict(restrictions)
        elif hasattr(restrictions, "model_dump"):
            raw = restrictions.model_dump(mode="json")
        else:
            try:
                raw = vars(restrictions)
            except TypeError:
                raise BookingError("INVALID_ARGUMENT") from None
        allowed = set(FindAvailableSlotsArgs.model_fields) | {"case_id", "clinic_id", "location"}
        if any(key not in allowed for key in raw):
            raise BookingError("INVALID_ARGUMENT")
        if "clinic_id" in raw and raw["clinic_id"] != context.clinic_id:
            raise BookingError("CLINIC_CONTEXT_MISMATCH")
        if "case_id" in raw and raw["case_id"] != context.case_id:
            raise BookingError("CASE_CONTEXT_MISMATCH")
        location = raw.get("location")
        if location is not None and (not isinstance(location, str) or not location.strip()):
            raise BookingError("INVALID_ARGUMENT")
        try:
            args = FindAvailableSlotsArgs.model_validate(
                {key: raw[key] for key in FindAvailableSlotsArgs.model_fields})
        except (ValidationError, KeyError, TypeError, ValueError):
            raise BookingError("INVALID_ARGUMENT") from None
        scope = args.model_dump(mode="json")
        if location is not None:
            scope["location"] = location
        return args, location, scope

    def booking_catalog(self, context: FrontDeskContext, *, restrictions=None) -> dict:
        """Return this clinic's future schedule directory and optional availability.

        This trusted controller read does not create offers, expire holds, alter a
        draft, or expose patient/case data. Availability enumerates clinicians
        across every matching slot rather than applying the finder's result cap.
        """
        with self._transaction() as connection:
            self._check(connection, context)
            clinic = self._clinic(connection, context.clinic_id)
            now = self._now(connection)
            rows = connection.execute(select(
                    t.slots.c.appointment_type, t.slots.c.clinician_id,
                    t.clinicians.c.display_name)
                .join(t.clinicians, t.clinicians.c.clinician_id == t.slots.c.clinician_id)
                .where(t.slots.c.clinic_id == context.clinic_id,
                    t.slots.c.enabled.is_(True), t.slots.c.starts_at > now)
                .order_by(t.clinicians.c.display_name, t.slots.c.clinician_id,
                    t.slots.c.appointment_type)).mappings().all()
            canonical = ("initial_consultation", "fracture_follow_up", "imaging", "physiotherapy", "other")
            present_types = {row["appointment_type"] for row in rows}
            directory = {}
            for row in rows:
                item = directory.setdefault(row["clinician_id"], {
                    "clinician_id": row["clinician_id"], "display_name": row["display_name"],
                    "appointment_types": set()})
                item["appointment_types"].add(row["appointment_type"])
            clinicians = [{**item, "appointment_types": [kind for kind in canonical
                    if kind in item["appointment_types"]]}
                for item in directory.values()]
            result = {"clinic_id": context.clinic_id, "timezone": clinic["timezone"],
                "as_of": now.isoformat(),
                "appointment_types": [kind for kind in canonical if kind in present_types],
                "clinicians": clinicians, "availability_checked": restrictions is not None,
                "search_scope": None, "available_clinicians": None}
            if restrictions is None:
                return result
            args, location, scope = self._catalog_restrictions(restrictions, context)
            available_ids = {slot["clinician_id"] for slot in self._matching_available_slots(
                connection, context, args, now, clinic=clinic, location=location)}
            result["search_scope"] = scope
            result["available_clinicians"] = [
                {"clinician_id": item["clinician_id"], "display_name": item["display_name"],
                    "appointment_types": [args.appointment_type]}
                for item in clinicians if item["clinician_id"] in available_ids]
            return result

    def register_clinic(self, clinic_id: str, timezone_name: str):
        ZoneInfo(timezone_name)
        with self._transaction() as connection:
            existing = connection.execute(select(t.clinics).where(t.clinics.c.clinic_id == clinic_id)).mappings().first()
            if existing:
                if existing["timezone"] != timezone_name:
                    raise ValueError("Clinic timezone changes require a separate migration/policy")
                return
            connection.execute(insert(t.clinics).values(clinic_id=clinic_id, timezone=timezone_name))

    def publish_schedule(self, clinic_id: str, rules: tuple[ScheduleRule, ...], start_date: date, end_date: date):
        """Trusted schedule configuration supplies hours, duration and eligible clinicians."""
        ids = []
        with self._transaction() as connection:
            clinic = self._clinic(connection, clinic_id)
            for values in generate_slots(rules, start_date, end_date, clinic["timezone"]):
                if connection.execute(select(t.clinicians.c.clinician_id).where(t.clinicians.c.clinician_id == values["clinician_id"])).first() is None:
                    raise ValueError("Schedule clinician must already exist")
                predicates = [getattr(t.slots.c, key) == values[key] for key in ("clinician_id", "appointment_type", "starts_at", "ends_at")]
                existing = connection.execute(select(t.slots).where(t.slots.c.clinic_id == clinic_id, *predicates)).mappings().first()
                if existing:
                    if existing["location"] != values["location"]:
                        raise ValueError("Published slots are immutable; do not silently change location")
                    ids.append(existing["slot_id"])
                else:
                    slot_id = uuid4()
                    connection.execute(insert(t.slots).values(slot_id=slot_id, clinic_id=clinic_id, enabled=True, **values))
                    ids.append(slot_id)
        return ids

    def open_session(self, *, clinic_id: str, user_id: str) -> FrontDeskContext:
        with self._transaction() as connection:
            self._clinic(connection, clinic_id)
            if not connection.execute(select(t.users.c.user_id).where(t.users.c.user_id == user_id,
                    t.users.c.is_active.is_(True), t.users.c.role.in_(["staff", "clinician"]))).first():
                raise BookingError("SESSION_CONTEXT_MISMATCH")
            row = {"session_id": uuid4(), "clinic_id": clinic_id, "user_id": user_id, "case_id": None,
                "case_context_version": 1, "conversation_version": 1, "active": True,
                "current_draft_id": None, "created_at": self._now(connection)}
            connection.execute(insert(t.sessions).values(**row))
            return self._context(row)

    def resume_context(self, *, session_id: UUID, user_id: str) -> FrontDeskContext:
        """Called only after authenticating the staff principal outside this module."""
        with self._transaction() as connection:
            row = connection.execute(select(t.sessions).where(t.sessions.c.session_id == session_id,
                t.sessions.c.user_id == user_id)).mappings().first()
            if row is None:
                raise BookingError("SESSION_CONTEXT_MISMATCH")
            context = self._context(row)
            self._check(connection, context)
            return context

    def _supersede_current(self, connection, session, now):
        if session["current_draft_id"]:
            draft = connection.execute(select(t.drafts).where(t.drafts.c.draft_id == session["current_draft_id"])).mappings().first()
            if draft and draft["state"] == "pending":
                if _aware(draft["hold_expires_at"]) <= now:
                    self._terminal(connection, draft, now, "expired")
                else:
                    self._terminal(connection, draft, now, "failed", "DRAFT_SUPERSEDED")

    @staticmethod
    def _abandon_pending_cancellation(connection, session_id, now):
        rows = connection.execute(select(t.cancellations).where(
            t.cancellations.c.session_id == session_id,
            t.cancellations.c.state == "pending")).mappings().all()
        for row in rows:
            state = "expired" if _aware(row["expires_at"]) <= now else "abandoned"
            connection.execute(update(t.cancellations).where(
                t.cancellations.c.cancellation_request_id == row["cancellation_request_id"])
                .values(state=state, updated_at=now))

    def set_confirmed_case(self, context: FrontDeskContext, case_id: str) -> FrontDeskContext:
        """Case resolution/identity verification must have succeeded outside the LLM."""
        with self._transaction() as connection:
            session = self._check(connection, context)
            self._patient(connection, case_id)
            now = self._now(connection)
            self._supersede_current(connection, session, now)
            self._abandon_pending_cancellation(connection, context.session_id, now)
            values = {"case_id": case_id, "case_context_version": context.case_context_version + 1,
                "conversation_version": context.conversation_version + 1, "current_draft_id": None}
            connection.execute(update(t.sessions).where(t.sessions.c.session_id == context.session_id).values(**values))
            return self._context({**session, **values})

    def advance_conversation(self, context: FrontDeskContext, *, selection_changed: bool = False) -> FrontDeskContext:
        """Before EVERY new user utterance, even 'yes'. Corrections invalidate selection.

        A confirmation on an earlier conversation version never authorizes a
        later booking. Commit vs correction is serialized by the guard lock.
        A correction cannot retroactively undo an appointment already committed.
        """
        with self._transaction() as connection:
            session = self._check(connection, context)
            values = {"conversation_version": context.conversation_version + 1}
            if selection_changed:
                now = self._now(connection)
                self._supersede_current(connection, session, now)
                self._abandon_pending_cancellation(connection, context.session_id, now)
                values["current_draft_id"] = None
            connection.execute(update(t.sessions).where(t.sessions.c.session_id == context.session_id).values(**values))
            return self._context({**session, **values})

    def invalidate_selection(self, context: FrontDeskContext):
        """Invalidate a corrected selection without advancing the same turn twice.

        Advance the finalized user turn first, then invalidate its correction
        before planning/preparing a replacement. Acoustic output invalidation
        remains separate: VAD-only activity need not advance database context.
        """
        with self._transaction() as connection:
            session = self._check(connection, context)
            now = self._now(connection)
            self._supersede_current(connection, session, now)
            self._abandon_pending_cancellation(connection, context.session_id, now)
            connection.execute(update(t.sessions).where(t.sessions.c.session_id == context.session_id)
                .values(current_draft_id=None))

    def mark_readback_presented(self, context: FrontDeskContext, *, draft_id: UUID, details: dict) -> str:
        """Call only when this exact authoritative readback was actually presented.

        details = prepared data restricted to case_id/patient_display_name/slot.
        Do not call merely because an LLM generated or queued readback text.
        """
        with self._transaction() as connection:
            session = self._check(connection, context, case_id=context.case_id)
            draft = self._owned_draft(connection, context, draft_id=draft_id)
            now = self._now(connection)
            self._current_pending(connection, context, draft, session, now)
            if _hash(details) != draft["details_hash"]:
                raise BookingError("CONFIRMATION_STALE")
            connection.execute(update(t.drafts).where(t.drafts.c.draft_id == draft_id).values(
                readback_version=context.conversation_version, readback_at=now, updated_at=now,
                confirmation_version=None, confirmation_event_id=None, confirmed_at=None,
                authorization_kind="confirmation"))
            return draft["details_hash"]

    def refresh_confirmation_prompt(self, context: FrontDeskContext, *, draft_id: UUID,
                                    details_hash: str, previous_readback_version: int) -> str:
        """Carry a delivered full readback across ONE clarification-only turn.

        Trusted controller hook, never a model-facing tool. Call only after the
        complete short confirmation clarification passed the ordered output-audio
        barrier, and only if this turn did not change the selected appointment.
        A queued/generated/interrupted clarification is insufficient.

        ``readback_version`` is the effective consent-context version; the
        original ``readback_at`` still records when the FULL authoritative
        details were delivered. This does not create a new full-readback receipt,
        record consent, extend a hold, or allow skipped conversation versions.
        The NEXT finalized affirmative turn must still pass record_confirmation.
        """
        with self._transaction() as connection:
            session = self._check(connection, context, case_id=context.case_id)
            draft = self._owned_draft(connection, context, draft_id=draft_id)
            now = self._now(connection)
            self._current_pending(connection, context, draft, session, now)
            if (type(previous_readback_version) is not int
                    or draft["readback_at"] is None or draft["readback_version"] is None
                    or draft["readback_version"] != previous_readback_version
                    or context.conversation_version != previous_readback_version + 1
                    or details_hash != draft["details_hash"]
                    or any(draft[key] is not None for key in (
                        "confirmation_version", "confirmation_event_id", "confirmed_at"))):
                raise BookingError("CONFIRMATION_STALE")
            patient = self._patient(connection, context.case_id)
            if patient["status"] != "open" or patient["patient_id"] != draft["patient_id"]:
                raise BookingError("CASE_NOT_BOOKABLE")
            slot = self._slot(connection, context, draft["slot_id"])
            current_details = {"case_id": context.case_id,
                "patient_display_name": patient["display_name"],
                "slot": self._slot_data(connection, slot, self._clinic(connection, context.clinic_id))}
            if _hash(current_details) != draft["details_hash"]:
                raise BookingError("CONFIRMATION_STALE")
            source = self._validate_reschedule_source(connection, context, draft_id, now)
            source_id = source["appointment_id"] if source else None
            hold = connection.execute(select(t.holds.c.draft_id).where(
                t.holds.c.draft_id == draft_id, t.holds.c.slot_id == draft["slot_id"],
                t.holds.c.expires_at > now)).first()
            if (not hold or not slot["enabled"] or _aware(slot["starts_at"]) <= now
                    or self._busy(connection, slot, now, exclude_draft=draft_id, exclude_appointment=source_id)):
                raise BookingError("SLOT_UNAVAILABLE")
            if self._busy(connection, slot, now, patient_id=patient["patient_id"], exclude_draft=draft_id, exclude_appointment=source_id):
                raise BookingError("PATIENT_CONFLICT")
            connection.execute(update(t.drafts).where(t.drafts.c.draft_id == draft_id).values(
                readback_version=context.conversation_version, updated_at=now))
            return draft["details_hash"]

    def record_confirmation(self, context: FrontDeskContext, *, draft_id: UUID,
                            details_hash: str, confirmation_event_id: UUID):
        """Controller-only affirmative event, after advance_conversation for that utterance.

        This method deliberately is NOT in FRONTDESK_TOOLS. The trusted controller
        determines explicit consent; a tool's arguments never assert confirmation.
        """
        with self._transaction() as connection:
            session = self._check(connection, context, case_id=context.case_id)
            draft = self._owned_draft(connection, context, draft_id=draft_id)
            now = self._now(connection)
            self._current_pending(connection, context, draft, session, now)
            if (details_hash != draft["details_hash"] or draft["readback_version"] is None
                    or context.conversation_version != draft["readback_version"] + 1):
                raise BookingError("CONFIRMATION_STALE")
            reused = connection.execute(select(t.drafts.c.draft_id).where(
                t.drafts.c.confirmation_event_id == confirmation_event_id,
                t.drafts.c.draft_id != draft_id)).first()
            if reused:
                raise BookingError("CONFIRMATION_STALE")
            if draft["confirmation_event_id"] is not None:
                if draft["confirmation_event_id"] != confirmation_event_id:
                    raise BookingError("CONFIRMATION_STALE")
                return
            connection.execute(update(t.drafts).where(t.drafts.c.draft_id == draft_id).values(
                confirmation_version=context.conversation_version, confirmation_event_id=confirmation_event_id,
                confirmed_at=now, updated_at=now))

    def authorize_direct_request(self, context: FrontDeskContext, *, operation: str,
                                 request_id: UUID, details: dict, request_event_id: UUID):
        """Record one exact direct user request without creating a readback receipt."""
        if operation not in {"book", "reschedule", "cancel"} or not isinstance(details, dict):
            raise BookingError("INVALID_ARGUMENT")
        try:
            request_id = UUID(str(request_id))
            request_event_id = UUID(str(request_event_id))
        except (TypeError, ValueError, AttributeError):
            raise BookingError("INVALID_ARGUMENT") from None
        with self._transaction() as connection:
            session = self._check(connection, context, case_id=context.case_id)
            now = self._now(connection)
            if operation == "cancel":
                intent = self._owned_cancellation(connection, context, request_id)
                self._current_cancellation(connection, context, intent, now)
                source = self._cancellation_source(connection, context, intent["source_appointment_id"], now)
                if (_hash(details) != intent["details_hash"]
                        or source["record_version"] != intent["source_record_version"]
                        or _hash(source) != intent["source_hash"]):
                    raise BookingError("CONFIRMATION_STALE")
                table, record, key = t.cancellations, intent, "cancellation_request_id"
            else:
                draft = self._owned_draft(connection, context, draft_id=request_id)
                self._current_pending(connection, context, draft, session, now)
                reschedule = self._reschedule_intent(connection, request_id)
                if (operation == "book" and reschedule is not None) or (operation == "reschedule" and reschedule is None):
                    raise BookingError("INVALID_ARGUMENT")
                if _hash(details) != draft["details_hash"]:
                    raise BookingError("CONFIRMATION_STALE")
                if reschedule is not None:
                    self._validate_reschedule_source(connection, context, request_id, now)
                table, record, key = t.drafts, draft, "draft_id"
            reused_draft = connection.execute(select(t.drafts.c.draft_id).where(
                t.drafts.c.confirmation_event_id == request_event_id,
                t.drafts.c.draft_id != request_id if table is t.drafts else True)).first()
            reused_cancellation = connection.execute(select(t.cancellations.c.cancellation_request_id).where(
                t.cancellations.c.confirmation_event_id == request_event_id,
                t.cancellations.c.cancellation_request_id != request_id if table is t.cancellations else True)).first()
            if reused_draft or reused_cancellation:
                raise BookingError("CONFIRMATION_STALE")
            if record["confirmation_event_id"] is not None:
                if (record["confirmation_event_id"] != request_event_id
                        or record["confirmation_version"] != context.conversation_version):
                    raise BookingError("CONFIRMATION_STALE")
                return
            connection.execute(update(table).where(table.c[key] == request_id).values(
                confirmation_version=context.conversation_version, confirmation_event_id=request_event_id,
                confirmed_at=now, authorization_kind="direct_request", updated_at=now))

    def close_session(self, context: FrontDeskContext):
        with self._transaction() as connection:
            session = self._check(connection, context)
            now = self._now(connection)
            self._supersede_current(connection, session, now)
            self._abandon_pending_cancellation(connection, context.session_id, now)
            connection.execute(update(t.sessions).where(t.sessions.c.session_id == context.session_id).values(
                active=False, current_draft_id=None, conversation_version=context.conversation_version + 1))

    # ---------- Current booking listing and atomic changes (controller only) ----------

    @staticmethod
    def _reschedule_intent(connection, draft_id):
        return connection.execute(select(t.reschedules).where(t.reschedules.c.draft_id == draft_id)).mappings().first()

    def _source_booking(self, connection, context, appointment_id, now):
        row = connection.execute(select(t.appointments,
                t.allocations.c.slot_id.label("allocated_slot_id"),
                t.allocations.c.booking_request_id.label("original_request_id"),
                t.drafts.c.patient_id.label("booked_patient_id"))
            .join(t.allocations, t.allocations.c.appointment_id == t.appointments.c.appointment_id)
            .join(t.drafts, t.drafts.c.draft_id == t.allocations.c.draft_id)
            .join(t.sessions, t.sessions.c.session_id == t.drafts.c.session_id)
            .where(t.appointments.c.appointment_id == appointment_id,
                t.appointments.c.case_id == context.case_id,
                t.sessions.c.clinic_id == context.clinic_id,
                t.allocations.c.active.is_(True),
                t.appointments.c.status.in_(["scheduled", "confirmed"]),
                t.appointments.c.starts_at > now)).mappings().first()
        if row is None:
            raise BookingError("RECORD_NOT_FOUND")
        patient = self._patient(connection, context.case_id)
        if patient["status"] != "open" or patient["patient_id"] != row["booked_patient_id"]:
            raise BookingError("CASE_NOT_BOOKABLE")
        zone = ZoneInfo(row["timezone"])
        name = connection.execute(select(t.clinicians.c.display_name).where(
            t.clinicians.c.clinician_id == row["clinician_id"])).scalar_one()
        return {"appointment_id": row["appointment_id"], "case_id": row["case_id"],
            "patient_display_name": patient["display_name"], "slot_id": str(row["allocated_slot_id"]),
            "appointment_type": row["appointment_type"],
            "starts_at": _aware(row["starts_at"]).astimezone(zone).isoformat(),
            "ends_at": _aware(row["ends_at"]).astimezone(zone).isoformat(),
            "timezone": row["timezone"], "clinician": {"clinician_id": row["clinician_id"], "display_name": name},
            "location": row["location"], "status": row["status"], "record_version": row["record_version"],
            "booking_request_id": str(row["original_request_id"])}

    def list_bookings(self, context: FrontDeskContext) -> list[dict]:
        """Current active future appointments, not historical booking receipts.

        Scope is the confirmed case and clinic. A new authenticated session may
        list/change its case's bookings; original session ownership is not needed.
        get_booking_status continues to describe its immutable request receipt,
        which can predate a later reschedule. Use this listing for current truth.
        """
        with self._transaction() as connection:
            self._check(connection, context, case_id=context.case_id)
            if context.case_id is None:
                raise BookingError("IDENTITY_UNRESOLVED")
            now = self._now(connection)
            ids = connection.execute(select(t.appointments.c.appointment_id)
                .join(t.allocations, t.allocations.c.appointment_id == t.appointments.c.appointment_id)
                .join(t.drafts, t.drafts.c.draft_id == t.allocations.c.draft_id)
                .join(t.sessions, t.sessions.c.session_id == t.drafts.c.session_id)
                .where(t.appointments.c.case_id == context.case_id,
                    t.sessions.c.clinic_id == context.clinic_id,
                    t.allocations.c.active.is_(True),
                    t.appointments.c.status.in_(["scheduled", "confirmed"]),
                    t.appointments.c.starts_at > now)
                .order_by(t.appointments.c.starts_at, t.appointments.c.appointment_id)).scalars().all()
            return [self._source_booking(connection, context, item, now) for item in ids]

    def _cancellation_source(self, connection, context, appointment_id, now):
        row = connection.execute(select(t.appointments,
                t.allocations.c.slot_id.label("allocated_slot_id"),
                t.allocations.c.booking_request_id.label("original_request_id"),
                t.drafts.c.patient_id.label("booked_patient_id"))
            .join(t.allocations, t.allocations.c.appointment_id == t.appointments.c.appointment_id)
            .join(t.drafts, t.drafts.c.draft_id == t.allocations.c.draft_id)
            .join(t.sessions, t.sessions.c.session_id == t.drafts.c.session_id)
            .where(t.appointments.c.appointment_id == appointment_id,
                t.appointments.c.case_id == context.case_id,
                t.sessions.c.clinic_id == context.clinic_id,
                t.allocations.c.active.is_(True),
                t.appointments.c.status.in_(["scheduled", "confirmed"]),
                t.appointments.c.starts_at > now)).mappings().first()
        if row is None:
            raise BookingError("RECORD_NOT_FOUND")
        patient = self._patient(connection, context.case_id)
        if patient["patient_id"] != row["booked_patient_id"]:
            raise BookingError("CASE_CONTEXT_MISMATCH")
        zone = ZoneInfo(row["timezone"])
        name = connection.execute(select(t.clinicians.c.display_name).where(
            t.clinicians.c.clinician_id == row["clinician_id"])).scalar_one()
        return {"appointment_id": row["appointment_id"], "case_id": row["case_id"],
            "patient_display_name": patient["display_name"], "slot_id": str(row["allocated_slot_id"]),
            "appointment_type": row["appointment_type"],
            "starts_at": _aware(row["starts_at"]).astimezone(zone).isoformat(),
            "ends_at": _aware(row["ends_at"]).astimezone(zone).isoformat(),
            "timezone": row["timezone"], "clinician": {"clinician_id": row["clinician_id"], "display_name": name},
            "location": row["location"], "status": row["status"], "record_version": row["record_version"],
            "booking_request_id": str(row["original_request_id"])}

    @staticmethod
    def _owned_cancellation(connection, context, cancellation_request_id):
        row = connection.execute(select(t.cancellations).where(
            t.cancellations.c.cancellation_request_id == cancellation_request_id,
            t.cancellations.c.session_id == context.session_id,
            t.cancellations.c.case_id == context.case_id)).mappings().first()
        if row is None:
            raise BookingError("RECORD_NOT_FOUND")
        if (row["clinic_id"] != context.clinic_id
                or row["case_context_version"] != context.case_context_version):
            raise BookingError("CASE_CONTEXT_MISMATCH")
        return row

    @staticmethod
    def _cancellation_envelope(intent, state=None):
        state = state or intent["state"]
        appointment = intent["result_snapshot"] if state == "cancelled" else intent["source_snapshot"]
        return {"cancellation_request_id": str(intent["cancellation_request_id"]),
            "cancellation_state": state, "appointment": appointment,
            "expires_at": _aware(intent["expires_at"]).isoformat()}

    def _current_cancellation(self, connection, context, intent, now):
        if intent["state"] == "expired" or _aware(intent["expires_at"]) <= now:
            if intent["state"] == "pending":
                connection.execute(update(t.cancellations).where(
                    t.cancellations.c.cancellation_request_id == intent["cancellation_request_id"])
                    .values(state="expired", updated_at=now))
            raise BookingError("HOLD_EXPIRED")
        if intent["state"] != "pending":
            raise BookingError("DRAFT_SUPERSEDED")
        if intent["case_context_version"] != context.case_context_version:
            raise BookingError("CASE_CONTEXT_MISMATCH")

    def prepare_cancellation(self, context: FrontDeskContext, *, appointment_id: str,
                             expected_record_version: int) -> dict:
        """Persist an expiring cancellation intent without changing the appointment."""
        if (not isinstance(appointment_id, str) or not appointment_id
                or type(expected_record_version) is not int or expected_record_version < 1):
            raise BookingError("INVALID_ARGUMENT")
        with self._transaction() as connection:
            self._check(connection, context, case_id=context.case_id)
            if context.case_id is None:
                raise BookingError("IDENTITY_UNRESOLVED")
            now = self._now(connection)
            current = connection.execute(select(t.cancellations).where(
                t.cancellations.c.session_id == context.session_id,
                t.cancellations.c.state == "pending")).mappings().first()
            if current is not None and _aware(current["expires_at"]) <= now:
                connection.execute(update(t.cancellations).where(
                    t.cancellations.c.cancellation_request_id == current["cancellation_request_id"])
                    .values(state="expired", updated_at=now))
                current = None
            if current is not None:
                if current["source_appointment_id"] == appointment_id:
                    if (current["source_record_version"] != expected_record_version
                            or current["clinic_id"] != context.clinic_id
                            or current["case_id"] != context.case_id
                            or current["case_context_version"] != context.case_context_version):
                        raise BookingError("CONFIRMATION_STALE")
                    source = self._cancellation_source(connection, context, appointment_id, now)
                    if (_hash(source) != current["source_hash"]
                            or source["record_version"] != expected_record_version):
                        raise BookingError("CONFIRMATION_STALE")
                    return self._cancellation_envelope(current)
                connection.execute(update(t.cancellations).where(
                    t.cancellations.c.cancellation_request_id == current["cancellation_request_id"])
                    .values(state="abandoned", updated_at=now))
            source = self._cancellation_source(connection, context, appointment_id, now)
            if source["record_version"] != expected_record_version:
                raise BookingError("CONFIRMATION_STALE")
            request_id = uuid4()
            expiry = min(now + timedelta(seconds=self.hold_seconds),
                         datetime.fromisoformat(source["starts_at"]).astimezone(UTC))
            row = {"cancellation_request_id": request_id, "session_id": context.session_id,
                "case_id": context.case_id, "clinic_id": context.clinic_id,
                "case_context_version": context.case_context_version,
                "prepared_conversation_version": context.conversation_version,
                "source_appointment_id": appointment_id,
                "source_record_version": expected_record_version,
                "source_snapshot": source, "source_hash": _hash(source),
                "details_hash": _hash({"appointment": source}), "state": "pending",
                "expires_at": expiry, "created_at": now, "updated_at": now}
            connection.execute(insert(t.cancellations).values(**row))
            return self._cancellation_envelope(row)

    def mark_cancellation_presented(self, context: FrontDeskContext, *,
                                    cancellation_request_id: UUID, details: dict) -> str:
        try:
            cancellation_request_id = UUID(str(cancellation_request_id))
        except (TypeError, ValueError, AttributeError):
            raise BookingError("INVALID_ARGUMENT") from None
        with self._transaction() as connection:
            self._check(connection, context, case_id=context.case_id)
            intent = self._owned_cancellation(connection, context, cancellation_request_id)
            now = self._now(connection)
            self._current_cancellation(connection, context, intent, now)
            if not isinstance(details, dict) or _hash(details) != intent["details_hash"]:
                raise BookingError("CONFIRMATION_STALE")
            source = self._cancellation_source(connection, context, intent["source_appointment_id"], now)
            if (source["record_version"] != intent["source_record_version"]
                    or _hash(source) != intent["source_hash"]):
                raise BookingError("CONFIRMATION_STALE")
            connection.execute(update(t.cancellations).where(
                t.cancellations.c.cancellation_request_id == cancellation_request_id).values(
                readback_version=context.conversation_version, readback_at=now, updated_at=now,
                confirmation_version=None, confirmation_event_id=None, confirmed_at=None,
                authorization_kind="confirmation"))
            return intent["details_hash"]

    def record_cancellation_confirmation(self, context: FrontDeskContext, *,
                                         cancellation_request_id: UUID, details_hash: str,
                                         confirmation_event_id: UUID):
        try:
            cancellation_request_id = UUID(str(cancellation_request_id))
            confirmation_event_id = UUID(str(confirmation_event_id))
        except (TypeError, ValueError, AttributeError):
            raise BookingError("INVALID_ARGUMENT") from None
        with self._transaction() as connection:
            self._check(connection, context, case_id=context.case_id)
            intent = self._owned_cancellation(connection, context, cancellation_request_id)
            now = self._now(connection)
            self._current_cancellation(connection, context, intent, now)
            if (details_hash != intent["details_hash"] or intent["readback_version"] is None
                    or context.conversation_version != intent["readback_version"] + 1):
                raise BookingError("CONFIRMATION_STALE")
            reused = connection.execute(select(t.cancellations.c.cancellation_request_id).where(
                t.cancellations.c.session_id == context.session_id,
                t.cancellations.c.confirmation_event_id == confirmation_event_id,
                t.cancellations.c.cancellation_request_id != cancellation_request_id)).first()
            reused_booking = connection.execute(select(t.drafts.c.draft_id).where(
                t.drafts.c.session_id == context.session_id,
                t.drafts.c.confirmation_event_id == confirmation_event_id)).first()
            if reused or reused_booking:
                raise BookingError("CONFIRMATION_STALE")
            if intent["confirmation_event_id"] is not None:
                if intent["confirmation_event_id"] != confirmation_event_id:
                    raise BookingError("CONFIRMATION_STALE")
                return
            connection.execute(update(t.cancellations).where(
                t.cancellations.c.cancellation_request_id == cancellation_request_id).values(
                confirmation_version=context.conversation_version,
                confirmation_event_id=confirmation_event_id, confirmed_at=now, updated_at=now))

    @staticmethod
    def _persist_cancellation_result(connection, cancellation_request_id, result, now):
        connection.execute(update(t.cancellations).where(
            t.cancellations.c.cancellation_request_id == cancellation_request_id).values(
            state="cancelled", result_snapshot=result, cancelled_at=now, updated_at=now))

    def cancel_appointment(self, context: FrontDeskContext, *, cancellation_request_id: UUID) -> dict:
        """Commit one confirmed cancellation or recover its persisted result."""
        try:
            cancellation_request_id = UUID(str(cancellation_request_id))
        except (TypeError, ValueError, AttributeError):
            raise BookingError("INVALID_ARGUMENT") from None
        with self._transaction() as connection:
            self._check(connection, context, case_id=context.case_id)
            intent = self._owned_cancellation(connection, context, cancellation_request_id)
            if intent["state"] == "cancelled":
                return self._cancellation_envelope(intent)
            now = self._now(connection)
            self._current_cancellation(connection, context, intent, now)
            if intent["confirmation_version"] is None:
                raise BookingError("CONFIRMATION_REQUIRED")
            if intent["confirmation_version"] != context.conversation_version:
                raise BookingError("CONFIRMATION_STALE")
            source = self._cancellation_source(connection, context, intent["source_appointment_id"], now)
            if (source["record_version"] != intent["source_record_version"]
                    or _hash(source) != intent["source_hash"]):
                raise BookingError("CONFIRMATION_STALE")
            changed = connection.execute(update(t.appointments).where(
                t.appointments.c.appointment_id == source["appointment_id"],
                t.appointments.c.case_id == context.case_id,
                t.appointments.c.record_version == source["record_version"],
                t.appointments.c.status.in_(["scheduled", "confirmed"])).values(
                status="cancelled", record_version=source["record_version"] + 1,
                updated_at=now))
            released = connection.execute(update(t.allocations).where(
                t.allocations.c.appointment_id == source["appointment_id"],
                t.allocations.c.active.is_(True)).values(active=False))
            if changed.rowcount != 1 or released.rowcount != 1:
                raise BookingError("CONFIRMATION_STALE")
            result = {**source, "status": "cancelled",
                "record_version": source["record_version"] + 1}
            self._persist_cancellation_result(connection, cancellation_request_id, result, now)
            return self._cancellation_envelope({**intent, "state": "cancelled",
                "result_snapshot": result, "cancelled_at": now})

    def get_cancellation_status(self, context: FrontDeskContext, *,
                                cancellation_request_id: UUID) -> dict:
        try:
            cancellation_request_id = UUID(str(cancellation_request_id))
        except (TypeError, ValueError, AttributeError):
            raise BookingError("INVALID_ARGUMENT") from None
        with self._transaction() as connection:
            self._check(connection, context, case_id=context.case_id)
            intent = self._owned_cancellation(connection, context, cancellation_request_id)
            now = self._now(connection)
            if intent["state"] == "pending" and _aware(intent["expires_at"]) <= now:
                # Status recovery observes expiry; it never mutates or renews an intent.
                return self._cancellation_envelope(intent, state="expired")
            return self._cancellation_envelope(intent)

    def invalidate_cancellation(self, context: FrontDeskContext, *, cancellation_request_id: UUID):
        try:
            cancellation_request_id = UUID(str(cancellation_request_id))
        except (TypeError, ValueError, AttributeError):
            raise BookingError("INVALID_ARGUMENT") from None
        with self._transaction() as connection:
            self._check(connection, context, case_id=context.case_id)
            intent = self._owned_cancellation(connection, context, cancellation_request_id)
            if intent["state"] != "pending":
                return
            now = self._now(connection)
            state = "expired" if _aware(intent["expires_at"]) <= now else "abandoned"
            connection.execute(update(t.cancellations).where(
                t.cancellations.c.cancellation_request_id == cancellation_request_id)
                .values(state=state, updated_at=now))

    def _validate_reschedule_source(self, connection, context, draft_id, now):
        intent = self._reschedule_intent(connection, draft_id)
        if intent is None:
            return None
        if (intent["session_id"] != context.session_id or intent["case_id"] != context.case_id
                or intent["clinic_id"] != context.clinic_id):
            raise BookingError("CASE_CONTEXT_MISMATCH")
        source = self._source_booking(connection, context, intent["source_appointment_id"], now)
        if (source["record_version"] != intent["source_record_version"]
                or _hash(source) != intent["source_hash"]):
            raise BookingError("CONFIRMATION_STALE")
        return source

    def prepare_reschedule(self, context: FrontDeskContext, *, appointment_id: str,
                           expected_record_version: int, slot_id: UUID) -> dict:
        """Hold a replacement without cancelling the existing appointment.

        An exact source/version is persisted with the draft. Retries never renew
        the hold. Source overlap is excluded only for this reschedule operation.
        """
        if (not isinstance(appointment_id, str) or not appointment_id
                or type(expected_record_version) is not int or expected_record_version < 1):
            raise BookingError("INVALID_ARGUMENT")
        try:
            slot_id = UUID(str(slot_id))
        except (TypeError, ValueError, AttributeError):
            raise BookingError("INVALID_ARGUMENT") from None
        failure = None
        with self._transaction() as connection:
            session = self._check(connection, context, case_id=context.case_id)
            if context.case_id is None:
                raise BookingError("IDENTITY_UNRESOLVED")
            now = self._now(connection)
            source = self._source_booking(connection, context, appointment_id, now)
            if source["record_version"] != expected_record_version:
                raise BookingError("CONFIRMATION_STALE")
            slot = self._slot(connection, context, slot_id)
            if str(slot_id) == source["slot_id"] or slot["appointment_type"] != source["appointment_type"]:
                raise BookingError("INVALID_ARGUMENT")
            try:
                data = self._prepare_booking(connection, context, session,
                    SimpleNamespace(case_id=context.case_id, slot_id=slot_id), now,
                    reschedule_source=source)
                draft_id = UUID(data["draft_id"])
                intent = self._reschedule_intent(connection, draft_id)
                if intent is None:
                    connection.execute(insert(t.reschedules).values(draft_id=draft_id,
                        session_id=context.session_id, case_id=context.case_id, clinic_id=context.clinic_id,
                        source_appointment_id=appointment_id, source_record_version=expected_record_version,
                        source_snapshot=source, source_hash=_hash(source), created_at=now))
                elif intent["source_hash"] != _hash(source):
                    raise BookingError("CONFIRMATION_STALE")
            except BookingError as exc:
                # Preserve intentional supersession/expiry, never source changes.
                failure = exc
        if failure:
            raise failure
        return {**data, "reschedule_from": source}

    def reschedule_appointment(self, context: FrontDeskContext, *, draft_id: UUID) -> dict:
        """Atomically cancel-and-replace after the controller records confirmation.

        Cancel, active-allocation transfer, replacement booking and linkage share
        one guard-locked transaction. ANY error rolls everything back. No caller
        parameter asserts consent. Repeated commits return the same replacement.
        """
        try:
            draft_id = UUID(str(draft_id))
        except (TypeError, ValueError, AttributeError):
            raise BookingError("INVALID_ARGUMENT") from None
        with self._transaction() as connection:
            session = self._check(connection, context, case_id=context.case_id)
            draft = self._owned_draft(connection, context, draft_id=draft_id)
            intent = self._reschedule_intent(connection, draft_id)
            if intent is None:
                raise BookingError("INVALID_ARGUMENT")
            if (intent["session_id"] != context.session_id or intent["case_id"] != context.case_id
                    or intent["clinic_id"] != context.clinic_id):
                raise BookingError("CASE_CONTEXT_MISMATCH")
            if intent["replacement_appointment_id"] is not None:
                if draft["state"] != "booked" or draft["appointment_id"] != intent["replacement_appointment_id"]:
                    raise BookingError("INTERNAL_ERROR")
                return {"booking_request_id": str(draft["booking_request_id"]),
                    "appointment": self._booked(draft), "rescheduled_from": intent["source_appointment_id"]}
            now = self._now(connection)
            self._current_pending(connection, context, draft, session, now)
            if draft["confirmation_version"] is None:
                raise BookingError("CONFIRMATION_REQUIRED")
            if draft["confirmation_version"] != context.conversation_version:
                raise BookingError("CONFIRMATION_STALE")
            source = self._validate_reschedule_source(connection, context, draft_id, now)
            connection.execute(update(t.appointments).where(
                t.appointments.c.appointment_id == source["appointment_id"]).values(
                status="cancelled", record_version=source["record_version"] + 1, updated_at=now))
            connection.execute(update(t.allocations).where(
                t.allocations.c.appointment_id == source["appointment_id"]).values(active=False))
            result = self._book_appointment(connection, context, session,
                SimpleNamespace(case_id=context.case_id, draft_id=draft_id), now, allow_reschedule=True)
            connection.execute(update(t.reschedules).where(t.reschedules.c.draft_id == draft_id).values(
                replacement_appointment_id=result["appointment"]["appointment_id"], completed_at=now))
            return {**result, "rescheduled_from": source["appointment_id"]}
