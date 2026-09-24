"""Front-desk write tools must not turn a requested time into a nearby slot."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

from healthcare_voice_agent.demo import flow, tools

BOOK_SLOT_ID = "00000000-0000-4000-8000-000000000101"
RESCHEDULE_SLOT_ID = "00000000-0000-4000-8000-000000000202"
DRAFT_ID = "00000000-0000-4000-8000-000000000303"
REQUEST_ID = "00000000-0000-4000-8000-000000000404"


class FakeService:
    def __init__(self, slots):
        self.slots = deepcopy(slots)
        self.calls = []

    def get_slot_details(self, context, *, slot_id):
        self.calls.append(("get_slot_details", str(slot_id)))
        return deepcopy(self.slots[str(slot_id)])

    def invoke(self, name, arguments, *, context):
        self.calls.append((name, deepcopy(arguments)))
        if name == "prepare_booking":
            return {"status": "ok", "data": self._prepared(arguments["slot_id"])}
        if name == "book_appointment":
            return {"status": "ok", "data": {"appointment": {"appointment_id": "booked-1"}}}
        raise AssertionError(f"Unexpected invoke: {name}")

    def prepare_reschedule(self, context, *, appointment_id, expected_record_version, slot_id):
        self.calls.append(("prepare_reschedule", appointment_id, expected_record_version, str(slot_id)))
        return self._prepared(str(slot_id))

    def authorize_direct_request(self, context, **kwargs):
        self.calls.append(("authorize_direct_request", kwargs["operation"]))

    def reschedule_appointment(self, context, *, draft_id):
        self.calls.append(("reschedule_appointment", str(draft_id)))
        return {"appointment": {"appointment_id": "replacement-1"}}

    def invalidate_selection(self, context):
        self.calls.append(("invalidate_selection",))

    def _prepared(self, slot_id):
        return {
            "draft_id": DRAFT_ID,
            "booking_request_id": REQUEST_ID,
            "case_id": "case-1",
            "patient_display_name": "Alex",
            "slot": deepcopy(self.slots[str(slot_id)]),
        }


def make_session(slots, *, pending=None):
    service = FakeService(slots)
    events = []

    async def db(function, *args, **kwargs):
        return function(*args, **kwargs)

    session = SimpleNamespace(
        service=service,
        context=SimpleNamespace(case_id="case-1"),
        db=db,
        lock=asyncio.Lock(),
        closed=False,
        emit=lambda name, **fields: events.append((name, fields)),
        epoch=7,
        final_turn_epoch=7,
        pending=pending,
        results={},
        last_result=None,
    )
    return session, events


def call(handler, args, session):
    result, node = asyncio.run(handler(args, SimpleNamespace(state={"session": session})))
    assert node is None
    return result


def slot(slot_id, starts_at):
    return {
        "slot_id": slot_id,
        "appointment_type": "physiotherapy",
        "starts_at": starts_at,
        "ends_at": "2026-10-02T10:30:00+05:30",
        "timezone": "Asia/Calcutta",
        "clinician": {"clinician_id": "physio-1", "display_name": "Dr Test"},
        "location": "Room 1",
    }


def test_native_write_schemas_require_timezone_aware_requested_start():
    schemas = {schema.name: schema for schema in flow.global_functions()}
    for name in ("book_slot", "reschedule_appointment"):
        schema = schemas[name]
        assert "requested_starts_at" in schema.required
        assert schema.properties["requested_starts_at"]["format"] == "date-time"
        assert schema.cancel_on_interruption is False


def test_book_rejects_physio_slot_thirty_minutes_before_requested_time_without_writes():
    session, _ = make_session({BOOK_SLOT_ID: slot(BOOK_SLOT_ID, "2026-10-02T10:00:00+05:30")})

    result = call(tools.book_slot, {
        "slot_id": BOOK_SLOT_ID,
        "requested_starts_at": "2026-10-02T10:30:00+05:30",
    }, session)

    assert result["status"] == "error"
    assert result["error"]["code"] == "SLOT_TIME_MISMATCH"
    assert session.service.calls == [("get_slot_details", BOOK_SLOT_ID)]
    assert session.pending is None and session.results == {}


def test_reschedule_rejects_three_pm_slot_for_three_thirty_request_without_writes():
    session, _ = make_session({RESCHEDULE_SLOT_ID: slot(
        RESCHEDULE_SLOT_ID, "2026-10-02T15:00:00+05:30")})

    result = call(tools.reschedule_appointment, {
        "appointment_id": "appointment-1",
        "record_version": 2,
        "slot_id": RESCHEDULE_SLOT_ID,
        "requested_starts_at": "2026-10-02T15:30:00+05:30",
    }, session)

    assert result["status"] == "error"
    assert result["error"]["code"] == "SLOT_TIME_MISMATCH"
    assert session.service.calls == [("get_slot_details", RESCHEDULE_SLOT_ID)]
    assert session.pending is None and session.results == {}


def test_matching_requested_starts_allow_book_and_atomic_reschedule():
    book_session, _ = make_session({BOOK_SLOT_ID: slot(
        BOOK_SLOT_ID, "2026-10-02T10:30:00+05:30")})
    book_result = call(tools.book_slot, {
        "slot_id": BOOK_SLOT_ID,
        "requested_starts_at": "2026-10-02T10:30:00+05:30",
    }, book_session)
    assert book_result["status"] == "ok"
    assert [entry[0] for entry in book_session.service.calls] == [
        "get_slot_details", "prepare_booking", "authorize_direct_request", "book_appointment"]

    reschedule_session, _ = make_session({RESCHEDULE_SLOT_ID: slot(
        RESCHEDULE_SLOT_ID, "2026-10-02T15:30:00+05:30")})
    reschedule_result = call(tools.reschedule_appointment, {
        "appointment_id": "appointment-1",
        "record_version": 2,
        "slot_id": RESCHEDULE_SLOT_ID,
        "requested_starts_at": "2026-10-02T15:30:00+05:30",
    }, reschedule_session)
    assert reschedule_result["status"] == "ok"
    assert [entry[0] for entry in reschedule_session.service.calls] == [
        "get_slot_details", "prepare_reschedule", "authorize_direct_request", "reschedule_appointment"]


def test_equivalent_timezone_instant_is_accepted_and_idempotent():
    session, _ = make_session({BOOK_SLOT_ID: slot(
        BOOK_SLOT_ID, "2026-10-02T10:30:00+05:30")})
    local = call(tools.book_slot, {
        "slot_id": BOOK_SLOT_ID,
        "requested_starts_at": "2026-10-02T10:30:00+05:30",
    }, session)
    utc_retry = call(tools.book_slot, {
        "slot_id": BOOK_SLOT_ID,
        "requested_starts_at": "2026-10-02T05:00:00Z",
    }, session)

    assert local == utc_retry
    assert local["status"] == "ok"
    assert [entry[0] for entry in session.service.calls].count("get_slot_details") == 1
    assert [entry[0] for entry in session.service.calls].count("book_appointment") == 1


def test_missing_naive_or_bad_requested_timestamp_is_rejected_before_any_lookup_or_write():
    invalid_values = (None, "", "2026-10-02T10:30:00", "not-a-timestamp", 123)
    for value in invalid_values:
        session, _ = make_session({BOOK_SLOT_ID: slot(
            BOOK_SLOT_ID, "2026-10-02T10:30:00+05:30")})
        args = {"slot_id": BOOK_SLOT_ID}
        if value is not None:
            args["requested_starts_at"] = value

        result = call(tools.book_slot, args, session)

        assert result["status"] == "error"
        assert result["error"]["code"] == "INVALID_ARGUMENT"
        assert session.service.calls == []
        assert session.pending is None and session.results == {}


def test_mismatch_does_not_discard_or_replace_existing_pending_selection():
    existing = {
        "operation": "book",
        "arguments": {"slot_id": "old-slot"},
        "key": "old-key",
        "data": {},
        "uncertain": False,
        "context": SimpleNamespace(case_id="case-1"),
    }
    session, _ = make_session({BOOK_SLOT_ID: slot(
        BOOK_SLOT_ID, "2026-10-02T10:00:00+05:30")}, pending=existing)

    result = call(tools.book_slot, {
        "slot_id": BOOK_SLOT_ID,
        "requested_starts_at": "2026-10-02T10:30:00+05:30",
    }, session)

    assert result["error"]["code"] == "SLOT_TIME_MISMATCH"
    assert session.pending is existing
    assert session.service.calls == [("get_slot_details", BOOK_SLOT_ID)]


def test_uncertain_pending_operation_is_not_overridden_even_by_bad_new_timestamp():
    uncertain = {
        "operation": "reschedule",
        "request_id": REQUEST_ID,
        "uncertain": True,
    }
    session, _ = make_session({BOOK_SLOT_ID: slot(
        BOOK_SLOT_ID, "2026-10-02T10:30:00+05:30")}, pending=uncertain)

    result = call(tools.book_slot, {"slot_id": BOOK_SLOT_ID}, session)

    assert result["status"] == "operation_unresolved"
    assert session.pending is uncertain
    assert session.service.calls == []
