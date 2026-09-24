"""Native tool handlers. No parsing or verification of conversational wording.

The LLM supplies structured selections. BookingService enforces domain rules;
this module only serializes calls and retains in-flight operation IDs for recovery.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime
from functools import wraps
import json
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from healthcare_voice_agent.booking.service import BookingError, MESSAGES

_UNCERTAIN = {"TIMEOUT", "BACKEND_UNAVAILABLE", "INTERNAL_ERROR"}


def _error(code):
    return {"status": "error", "error": {"code": code,
            "message": MESSAGES.get(code, "The operation could not be completed.")}}


def _tool(handler):
    """Shared lock and safe errors, not a dialogue controller."""
    @wraps(handler)
    async def call(args, flow_manager):
        session = flow_manager.state["session"]
        async with session.lock:
            if session.closed:
                return _error("SESSION_CONTEXT_MISMATCH"), None
            try:
                result = await handler(args, session)
            except BookingError as exc:
                result = _error(exc.code)
            except (KeyError, TypeError, ValueError):
                result = _error("INVALID_ARGUMENT")
            except Exception:
                result = _error("BACKEND_UNAVAILABLE")
            session.emit("frontdesk_flow_tool_result", tool_name=handler.__name__,
                         status=result["status"], error_code=result.get("error", {}).get("code"),
                         epoch=session.epoch)
            return result, None  # No conversation-node transitions.
    return call


async def _invoke(session, name, arguments):
    result = await session.db(session.service.invoke, name, arguments, context=session.context)
    if result["status"] != "ok":
        raise BookingError(result["error"]["code"])
    return result["data"]


@_tool
async def clinic_information(args, session):
    data = await session.db(session.service.booking_catalog, session.context)
    return {"status": "ok", "data": {**data, "result_kind": "clinic_directory",
                                     "slot_times_included": False}}


@_tool
async def list_appointments(args, session):
    start, end = args.get("start_date"), args.get("end_date")
    start = date.fromisoformat(start) if start else None
    end = date.fromisoformat(end) if end else None
    if start and end and end < start:
        raise BookingError("INVALID_ARGUMENT")
    rows = await session.db(session.service.list_bookings, session.context)
    zone = ZoneInfo(session.summary["timezone"])
    selected = []
    for row in rows:
        day = datetime.fromisoformat(row["starts_at"]).astimezone(zone).date()
        if (start is None or day >= start) and (end is None or day <= end):
            selected.append(row)
    session.emit("frontdesk_appointments_listed", total_count=len(rows),
                 matching_count=len(selected), date_filtered=bool(start or end))
    return {"status": "ok", "data": {"appointments": selected,
            "matching_count": len(selected), "total_active_future_count": len(rows),
            "active_future_only": True, "timezone": session.summary["timezone"],
            "filters": {"start_date": start.isoformat() if start else None,
                        "end_date": end.isoformat() if end else None}}}


def _slot_groups(slots):
    """Summarize actual results, without interpreting or validating speech."""
    groups = {}
    for slot in slots:
        day = datetime.fromisoformat(slot["starts_at"]).astimezone(ZoneInfo(slot["timezone"])).date().isoformat()
        groups.setdefault((day, slot["clinician"]["clinician_id"], slot["location"]), []).append(slot)
    result = []
    for (day, _, location), rows in groups.items():
        starts = [datetime.fromisoformat(row["starts_at"]) for row in rows]
        intervals = {(b - a).total_seconds() / 60 for a, b in zip(starts, starts[1:])}
        interval = next(iter(intervals)) if len(intervals) == 1 else None
        result.append({"date": day, "clinician": rows[0]["clinician"], "location": location,
                       "count": len(rows), "first_start": rows[0]["starts_at"],
                       "last_start": rows[-1]["starts_at"],
                       "interval_minutes": interval if interval and interval > 0 else None})
    return result


@_tool
async def search_availability(args, session):
    arguments = {key: args.get(key) for key in (
        "appointment_type", "start_date", "end_date", "earliest_start_time",
        "latest_start_time", "clinician_id")}
    arguments["max_results"] = 100
    data = await _invoke(session, "find_available_slots", arguments)
    slots = data["slots"]
    session.emit("frontdesk_availability_searched", slot_count=len(slots),
                 has_more=data["has_more"], appointment_type=arguments["appointment_type"],
                 time_filtered=bool(arguments["earliest_start_time"] or arguments["latest_start_time"]),
                 clinician_filtered=arguments["clinician_id"] is not None)
    return {"status": "ok", "data": {**data, "returned_count": len(slots),
            "matching_count": None if data["has_more"] else len(slots),
            "slot_groups": _slot_groups(slots), "search_scope": arguments}}


@_tool
async def book_slot(args, session):
    return await _action(session, "book", {key: args.get(key) for key in (
        "slot_id", "requested_starts_at")})


@_tool
async def reschedule_appointment(args, session):
    return await _action(session, "reschedule", {key: args.get(key) for key in (
        "appointment_id", "record_version", "slot_id", "requested_starts_at")})


def _batch_result(sources, completed, outcome):
    result = {"status": "ok" if len(completed) == len(sources) else "partial",
              "data": {"operation": "cancel", "cancelled_appointments": completed,
                       "cancelled_count": len(completed), "requested_count": len(sources),
                       "remaining_count": len(sources) - len(completed),
                       "remaining_appointments": sources[len(completed):]}}
    if result["status"] != "ok":
        result["stopped_reason"] = outcome
    return result


@_tool
async def cancel_appointments(args, session):
    sources = args.get("appointments")
    if (not isinstance(sources, list) or not 1 <= len(sources) <= 100
            or any(not isinstance(row, dict) for row in sources)
            or len({row.get("appointment_id") for row in sources}) != len(sources)):
        raise BookingError("INVALID_ARGUMENT")
    completed = []
    for source in sources:
        outcome = await _action(session, "cancel", source)
        if outcome["status"] != "ok":
            if session.pending and session.pending["operation"] == "cancel":
                session.pending.setdefault("batch", {"sources": deepcopy(sources),
                                                      "completed": deepcopy(completed)})
            break
        completed.append(outcome["data"]["appointment"])
    result = _batch_result(sources, completed, outcome)
    session.last_result = deepcopy(result["data"])
    return result


async def _discard(session):
    """Release an uncommitted preparation; never cancel a saved appointment."""
    await session.db(session.service.invalidate_selection, session.context)
    session.pending = None


def _finish(session, pending, data):
    result = {"status": "ok", "data": {"operation": pending["operation"], **data}}
    if pending["operation"] == "reschedule":
        result["data"]["rescheduled_from"] = pending["arguments"]["appointment_id"]
    session.results[pending["key"]] = deepcopy(result)
    session.last_result = deepcopy(result["data"])
    session.pending = None
    session.emit("frontdesk_operation_committed", operation=pending["operation"])
    return result


async def _prepare(session, operation, args):
    if operation == "book":
        return await _invoke(session, "prepare_booking", {
            "case_id": session.context.case_id, "slot_id": args.get("slot_id")})
    if operation == "reschedule":
        return await session.db(session.service.prepare_reschedule, session.context,
            appointment_id=args.get("appointment_id"), expected_record_version=args.get("record_version"),
            slot_id=args.get("slot_id"))
    return await session.db(session.service.prepare_cancellation, session.context,
        appointment_id=args.get("appointment_id"), expected_record_version=args.get("record_version"))


def _requested_start(value):
    if not isinstance(value, str) or not value:
        raise BookingError("INVALID_ARGUMENT")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise BookingError("INVALID_ARGUMENT") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BookingError("INVALID_ARGUMENT")
    return parsed.astimezone(UTC)


async def _guard_requested_start(session, arguments, requested_start):
    """Read persisted slot truth before any preparation or pending-state change."""
    try:
        slot_id = UUID(str(arguments.get("slot_id")))
    except (TypeError, ValueError, AttributeError):
        return _error("INVALID_ARGUMENT")
    try:
        slot = await session.db(session.service.get_slot_details, session.context,
                                slot_id=slot_id)
    except BookingError as exc:
        return _error(exc.code)
    except Exception:
        return _error("BACKEND_UNAVAILABLE")
    try:
        selected_start = datetime.fromisoformat(slot["starts_at"])
        if selected_start.tzinfo is None or selected_start.utcoffset() is None:
            raise ValueError("Persisted slot start is not timezone-aware")
    except (KeyError, TypeError, ValueError):
        return _error("BACKEND_UNAVAILABLE")
    if selected_start.astimezone(UTC) != requested_start:
        return _error("SLOT_TIME_MISMATCH")
    return None


async def _action(session, operation, arguments):
    """Guard, prepare, authorize and commit one structured request."""
    pending = session.pending
    normalized = deepcopy(arguments)
    requested_start = None
    validation_error = None
    if operation in {"book", "reschedule"}:
        try:
            requested_start = _requested_start(arguments.get("requested_starts_at"))
        except BookingError as exc:
            validation_error = exc
        else:
            normalized["requested_starts_at"] = requested_start.isoformat()
    key = None if validation_error else json.dumps([operation, normalized], sort_keys=True)
    if key is not None and key in session.results:
        return deepcopy(session.results[key])
    if pending and pending["uncertain"]:
        return {"status": "operation_unresolved", "next_action": "Call operation_status before another write."}
    if validation_error:
        return _error(validation_error.code)
    epoch = session.epoch
    if session.final_turn_epoch != epoch:
        return {"status": "stale_turn"}
    if operation in {"book", "reschedule"}:
        guarded = await _guard_requested_start(session, normalized, requested_start)
        if guarded is not None:
            return guarded
    arguments = normalized
    try:
        if pending and (pending["key"] != key or pending["context"] != session.context):
            await _discard(session)
            pending = None
        if pending is None:
            data = await _prepare(session, operation, arguments)
            pending = {"operation": operation, "arguments": deepcopy(arguments), "key": key,
                       "data": data, "uncertain": False, "context": session.context,
                       "request_event_id": uuid4(),
                       "request_id": data["cancellation_request_id"] if operation == "cancel" else data["booking_request_id"]}
            session.pending = pending
        else:
            data = pending["data"]  # Retry the same request; do not renew its hold.
        if epoch != session.epoch:
            await _discard(session)
            return {"status": "stale_turn"}
        request_id = UUID(data["cancellation_request_id"] if operation == "cancel" else data["draft_id"])
        details = ({"appointment": data["appointment"]} if operation == "cancel" else
                   {key: data[key] for key in ("case_id", "patient_display_name", "slot")})
        await session.db(session.service.authorize_direct_request, session.context,
            operation=operation, request_id=request_id, details=details,
            request_event_id=pending["request_event_id"])
        if epoch != session.epoch:
            await _discard(session)
            return {"status": "stale_turn"}
        pending["uncertain"] = True  # Keep this ID if the commit acknowledgement is lost.
        if operation == "cancel":
            committed = await session.db(session.service.cancel_appointment, session.context,
                                         cancellation_request_id=request_id)
        elif operation == "reschedule":
            committed = await session.db(session.service.reschedule_appointment, session.context, draft_id=request_id)
        else:
            committed = await _invoke(session, "book_appointment", {
                "case_id": session.context.case_id, "draft_id": data["draft_id"]})
        return _finish(session, pending, committed)
    except Exception as exc:
        code = exc.code if isinstance(exc, BookingError) else "BACKEND_UNAVAILABLE"
        if session.pending and session.pending["uncertain"] and code in _UNCERTAIN:
            return {"status": "operation_unresolved", "next_action": "Call operation_status; do not start a new attempt."}
        if session.pending:
            try:
                await _discard(session)
            except Exception:
                session.pending["uncertain"] = True
                return {"status": "operation_unresolved"}
        return _error(code)


@_tool
async def operation_status(args, session):
    pending = session.pending
    if pending is None:
        return {"status": "ok", "data": {"operation_state": "none", "last_result": session.last_result}}
    operation = pending["operation"]
    if operation == "cancel":
        data = await session.db(session.service.get_cancellation_status, session.context,
                               cancellation_request_id=UUID(pending["request_id"]))
        state = data["cancellation_state"]
    else:
        data = await _invoke(session, "get_booking_status", {
            "case_id": session.context.case_id, "booking_request_id": pending["request_id"]})
        state = data["booking_state"]
    if state in {"booked", "cancelled"}:
        result = _finish(session, pending, data)
    elif state in {"expired", "failed", "abandoned"}:
        session.pending = None
        result = {"status": "not_committed", "data": {"operation": operation, **data}}
    else:
        # The database has established that no commit happened. A later action
        # may retry this SAME request or replace it; status itself never commits.
        pending["uncertain"] = False
        result = {"status": "not_committed", "data": {"operation": operation, **data}}
    if pending.get("batch"):
        batch = pending["batch"]
        completed = deepcopy(batch["completed"])
        if state == "cancelled":
            completed.append(data["appointment"])
        result = _batch_result(batch["sources"], completed, result)
        session.last_result = deepcopy(result["data"])
    return result
