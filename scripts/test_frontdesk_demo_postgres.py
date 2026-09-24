"""Opt-in PostgreSQL verification in the fixed, isolated synthetic demo only.

Requires --run. Retains uniquely named QA patients/cases, two slots, sessions and
one confirmed QA appointment for audit. Never selects root .env, changes case
1042, deletes data, creates DDL, contacts models or makes paid calls. A simulated
lost response means a successful return value is discarded, not a network fault.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import time
import json
import time as clock
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from healthcare_voice_agent.booking.scheduling import ScheduleRule
from healthcare_voice_agent.booking.service import BookingError, BookingService
from healthcare_voice_agent.demo.database import (
    DEMO_CASE_ID, DEMO_STAFF_ID, DEMO_TIMEZONE, build_demo_engine, verify_demo_database,
)


class VerificationFailure(RuntimeError):
    """Only fixed test labels, never raw database/provider errors."""


def require(condition, label):
    if not condition:
        raise VerificationFailure(label)


def good(result, label):
    require(result.get("status") == "ok", label)
    return result["data"]


def rejected(result, code, label):
    require(result.get("status") == "error" and result["error"]["code"] == code, label)


@contextmanager
def bounded_transaction(engine):
    with engine.begin() as connection:
        connection.execute(text("SET LOCAL statement_timeout='5s'"))
        connection.execute(text("SET LOCAL lock_timeout='2s'"))
        connection.execute(text("SELECT id FROM clinic.booking_guard WHERE id=1 FOR UPDATE")).one()
        yield connection


def reject_dml(connection, sql, arguments, *, constraint, sqlstate="23503", deferred=False):
    """Attempt only QA DML in a savepoint; rollback whether it succeeds or fails."""
    transaction = connection.begin_nested()
    observed = None
    try:
        connection.execute(text(sql), arguments)
        if deferred:
            # Constraint names below are fixed literals, never caller/model input.
            connection.execute(text('SET CONSTRAINTS clinic."' + constraint + '" IMMEDIATE'))
    except IntegrityError as exc:
        observed = (getattr(exc.orig, "sqlstate", None), getattr(exc.orig.diag, "constraint_name", None))
    finally:
        transaction.rollback()
    require(observed == (sqlstate, constraint), "constraint_" + constraint)


def fixture_case_snapshot(connection):
    row = connection.execute(text("SELECT to_jsonb(c)::text FROM clinic.cases c WHERE case_id=:case"),
                             {"case": DEMO_CASE_ID}).scalar_one()
    appointments = connection.execute(text("SELECT to_jsonb(a)::text FROM clinic.appointments a "
        "WHERE case_id=:case ORDER BY appointment_id"), {"case": DEMO_CASE_ID}).scalars().all()
    return row, appointments


def run_verification(engine=None):
    owned = engine is None
    engine = engine or build_demo_engine()
    sessions = {}
    service = None
    checks = []
    try:
        # This validates the fixed URL before connecting, then validates the
        # instance marker, schema and synthetic fixture. No QA query precedes it.
        readiness = verify_demo_database(engine)
        checks.append("fixed_demo_target_and_instance_marker")
        suffix = uuid4().hex[:16]
        clinic_id = "DEMO-QA-" + suffix
        case_ids = ["QA-CASE-" + suffix + "-A", "QA-CASE-" + suffix + "-B"]
        patient_ids = ["QA-PAT-" + suffix + "-A", "QA-PAT-" + suffix + "-B"]
        with bounded_transaction(engine) as connection:
            before = fixture_case_snapshot(connection)
            # Keep QA dates beyond the normal 15-day demo schedule and beyond
            # earlier retained QA bookings for the existing clinician.
            day = connection.execute(text("SELECT GREATEST("
                "(clock_timestamp() AT TIME ZONE :zone)::date + 20, "
                "COALESCE((SELECT (max(starts_at) AT TIME ZONE :zone)::date + 1 "
                "FROM clinic.booking_slots WHERE clinic_id LIKE 'DEMO-QA-%'), "
                "(clock_timestamp() AT TIME ZONE :zone)::date + 20))"),
                {"zone": DEMO_TIMEZONE}).scalar_one()
            for index, (case_id, patient_id) in enumerate(zip(case_ids, patient_ids)):
                connection.execute(text("INSERT INTO clinic.patients(patient_id,display_name,is_synthetic) "
                    "VALUES (:patient,:name,true)"),
                    {"patient": patient_id, "name": "Synthetic QA Patient " + str(index + 1)})
                connection.execute(text("INSERT INTO clinic.cases(case_id,patient_id,description,status,opened_at) "
                    "VALUES (:case,:patient,'Isolated PostgreSQL verification only','open',clock_timestamp())"),
                    {"case": case_id, "patient": patient_id})
        service = BookingService(engine)
        service.register_clinic(clinic_id, DEMO_TIMEZONE)
        ids = service.publish_schedule(clinic_id, (ScheduleRule("fracture_follow_up", "SYN-CLIN-01",
            tuple(range(7)), time(9), time(10), 30, "Synthetic QA only"),), day, day)
        require(len(ids) == 2, "two_qa_slots")

        def session(case_id):
            context = service.open_session(clinic_id=clinic_id, user_id=DEMO_STAFF_ID)
            context = service.set_confirmed_case(context, case_id)
            sessions[context.session_id] = context
            return context

        def invoke(name, args, context, provider=service):
            return provider.invoke(name, args, context=context)

        def search(context):
            return good(invoke("find_available_slots", {
                "appointment_type": "fracture_follow_up", "start_date": day.isoformat(),
                "end_date": day.isoformat(), "earliest_start_time": None,
                "latest_start_time": None, "clinician_id": "SYN-CLIN-01", "max_results": 2,
            }, context), "search")

        def prepare(context, slot, provider=service):
            return invoke("prepare_booking", {"case_id": context.case_id, "slot_id": str(slot)}, context, provider)

        def book(context, draft):
            return invoke("book_appointment", {"case_id": context.case_id, "draft_id": draft["draft_id"]}, context)

        def status(context, draft):
            return invoke("get_booking_status", {"case_id": context.case_id,
                "booking_request_id": draft["booking_request_id"]}, context)

        def readback(context, draft):
            return service.mark_readback_presented(context, draft_id=UUID(draft["draft_id"]),
                details={key: draft[key] for key in ("case_id", "patient_display_name", "slot")})

        def advance(context, **kwargs):
            current = service.advance_conversation(context, **kwargs)
            sessions[current.session_id] = current
            return current

        contenders = [session(case) for case in case_ids]
        for context in contenders:
            require(len(search(context)["slots"]) == 2, "two_available_qa_slots")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(prepare, context, ids[0]) for context in contenders]
            outcomes = [future.result(timeout=15) for future in futures]
        successful = [i for i, result in enumerate(outcomes) if result["status"] == "ok"]
        require(len(successful) == 1, "concurrent_hold_single_winner")
        winner_index = successful[0]
        rejected(outcomes[1-winner_index], "SLOT_UNAVAILABLE", "concurrent_hold_loser")
        owner = contenders[winner_index]
        other = contenders[1-winner_index]
        draft = outcomes[winner_index]["data"]
        checks.append("concurrent_competing_holds_single_winner")
        rejected(book(owner, draft), "CONFIRMATION_REQUIRED", "book_without_confirmation")
        rejected(invoke("book_appointment", {"case_id": other.case_id, "draft_id": draft["draft_id"]}, owner),
                 "CASE_CONTEXT_MISMATCH", "wrong_case")
        foreign_session = session(owner.case_id)
        rejected(status(foreign_session, draft), "RECORD_NOT_FOUND", "foreign_session")
        checks.extend(["confirmation_required", "wrong_case_rejected", "foreign_session_rejected"])
        digest = readback(owner, draft)
        owner = advance(owner)
        service.record_confirmation(owner, draft_id=UUID(draft["draft_id"]),
            details_hash=digest, confirmation_event_id=uuid4())
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(book, owner, draft) for _ in range(2)]
            booked = [good(future.result(timeout=15), "concurrent_book") for future in futures]
        require(booked[0] == booked[1], "duplicate_booking_same_result")
        appointment_id = booked[0]["appointment"]["appointment_id"]
        # Discard the booking response, recovering only from its stable request ID.
        del booked
        recovered = good(status(owner, draft), "status_after_discarded_response")
        require(recovered["booking_state"] == "booked" and
                recovered["appointment"]["appointment_id"] == appointment_id, "recovered_persisted_booking")
        checks.extend(["concurrent_duplicate_booking_idempotent", "discarded_response_status_recovery"])

        pending = good(prepare(other, ids[1]), "second_slot_prepare")
        readback(other, pending)
        with bounded_transaction(engine) as connection:
            count = connection.execute(text("SELECT count(*) FROM clinic.appointments "
                "WHERE case_id IN (:a,:b)"), {"a": case_ids[0], "b": case_ids[1]}).scalar_one()
            require(count == 1, "one_qa_appointment")
            allocations = connection.execute(text("SELECT count(*) FROM clinic.booking_allocations "
                "WHERE booking_request_id=:request"), {"request": UUID(draft["booking_request_id"])}).scalar_one()
            require(allocations == 1, "one_allocation")
            expected_deferred = {"booking_session_current_draft_fk", "booking_hold_matches_draft_fk",
                "booking_allocation_matches_draft_fk", "booking_draft_committed_allocation_fk"}
            actual = set(connection.execute(text("SELECT conname FROM pg_constraint WHERE "
                "connamespace='clinic'::regnamespace AND condeferrable AND condeferred")).scalars())
            require(expected_deferred <= actual, "native_deferred_constraints")
            args = {"draft": UUID(draft["draft_id"]), "pending": UUID(pending["draft_id"]),
                    "pending_request": UUID(pending["booking_request_id"]), "slot": ids[1],
                    "foreign_session": foreign_session.session_id, "other_case": other.case_id,
                    "event": uuid4()}
            reject_dml(connection, "UPDATE clinic.booking_sessions SET current_draft_id=:draft "
                "WHERE session_id=:foreign_session", args,
                constraint="booking_session_current_draft_fk", deferred=True)
            reject_dml(connection, "UPDATE clinic.booking_holds SET expires_at=expires_at+interval '1 second' "
                "WHERE draft_id=:pending", args, constraint="booking_hold_matches_draft_fk", deferred=True)
            reject_dml(connection, "UPDATE clinic.booking_allocations SET booking_request_id=:pending_request "
                "WHERE draft_id=:draft", args, constraint="booking_allocation_matches_draft_fk", deferred=True)
            reject_dml(connection, "UPDATE clinic.booking_drafts SET slot_id=:slot WHERE draft_id=:draft", args,
                constraint="booking_draft_committed_allocation_fk", deferred=True)
            reject_dml(connection, "UPDATE clinic.booking_drafts SET case_id=:other_case WHERE draft_id=:draft", args,
                constraint="booking_draft_appointment_case_fk")
            reject_dml(connection, "INSERT INTO clinic.booking_allocations SELECT * FROM clinic.booking_allocations "
                "WHERE draft_id=:draft", args, constraint="booking_allocations_pkey", sqlstate="23505")
            reject_dml(connection, "UPDATE clinic.booking_drafts SET confirmation_version=readback_version, "
                "confirmation_event_id=:event,confirmed_at=clock_timestamp() WHERE draft_id=:pending", args,
                constraint="booking_draft_confirmation_shape", sqlstate="23514")
        checks.extend(["exactly_one_persisted_appointment_and_allocation", "four_native_initially_deferred_fks",
                       "seven_invalid_dml_constraint_rejections"])
        other = advance(other, selection_changed=True)
        rejected(book(other, pending), "DRAFT_SUPERSEDED", "correction_rejects_old_booking")
        corrected = good(status(other, pending), "corrected_status")
        require(corrected["booking_state"] == "failed" and corrected["failure"]["code"] == "DRAFT_SUPERSEDED",
                "correction_persisted")
        checks.append("correction_after_readback_invalidates_selection")
        short_service = BookingService(engine, hold_seconds=1)
        expiring = good(prepare(other, ids[1], short_service), "short_hold_prepare")
        clock.sleep(1.1)
        require(good(status(other, expiring), "expired_status")["booking_state"] == "expired", "server_clock_expiry")
        rejected(book(other, expiring), "HOLD_EXPIRED", "expired_hold_cannot_book")
        checks.append("postgres_server_clock_expiry")
        with bounded_transaction(engine) as connection:
            require(fixture_case_snapshot(connection) == before, "user_demo_case_unchanged")
            from scripts import seed
            dataset = seed.load_dataset(seed.DEFAULT_FIXTURE_DIR)
            cursor = connection.connection.driver_connection.cursor()
            try:
                fixture_rows = seed._verify_fixture_rows(cursor, dataset)
            finally:
                cursor.close()
        checks.extend(["user_demo_case_1042_unchanged", "all_original_fixture_rows_unchanged"])
        return {"status": "passed", "database": readiness["database"], "port": readiness["port"],
                "revision": readiness["revision"], "checks_passed": len(checks), "checks": checks,
                "constraint_rejections": 7, "verified_original_fixture_rows": fixture_rows,
                "qa_clinic_id": clinic_id, "qa_case_ids": case_ids, "qa_slots_created": 2,
                "qa_appointments_created": 1, "appointment_id": appointment_id,
                "booking_request_id": draft["booking_request_id"],
                "qa_records_retained_for_audit": True, "paid_or_model_calls": 0}
    finally:
        if service is not None:
            for context in sessions.values():
                try:
                    service.close_session(context)
                except BookingError:
                    pass
        if owned:
            engine.dispose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Create retained synthetic QA records and run PostgreSQL checks")
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"status": "not_run", "instruction": "Pass --run to test only the isolated synthetic demo."}))
        return 0
    try:
        result = run_verification()
    except Exception as exc:
        result = {"status": "failed", "exception_type": type(exc).__name__}
        if isinstance(exc, VerificationFailure):
            result["check"] = str(exc)
        print(json.dumps(result))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
