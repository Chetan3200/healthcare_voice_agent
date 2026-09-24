# Front-desk appointment booking tools

These are **four new contracts and a transactional backend**, separate from the
existing five clinical tools. They book an appointment for an already-resolved
case. They do not create patients/cases, provide clinical advice or implement the
clinical agent. The four backend contracts remain unchanged. The voice agent exposes seven native tools from one Pipecat Flows node, with
no Python conversation parsing. The wrappers in `demo/tools.py` these use the backend methods
for listings, booking, atomic rescheduling and cancellation. Authorization/receipt
recording methods are not model-facing tools. The current voice demo uses direct
request authorization (revision 005), not the older two-turn readback protocol
documented below for explicit-confirmation callers. See the
[front-desk flow guide](frontdesk-demo.md) for the current implementation.

## Files and implementation status

| File | Purpose |
|---|---|
| `src/healthcare_voice_agent/tools/frontdesk_contracts.py` | Strict Pydantic arguments/responses and `FRONTDESK_TOOLS` registry |
| [`contracts/frontdesk-tool-contracts.json`](contracts/frontdesk-tool-contracts.json) | Generated, provider-independent JSON Schemas |
| [`contracts/frontdesk-tool-examples.json`](contracts/frontdesk-tool-examples.json) | Synthetic successful/error/status examples |
| `src/healthcare_voice_agent/booking/service.py` | Four tool handlers and separate trusted controller/admin methods |
| `src/healthcare_voice_agent/booking/scheduling.py` | Backend-owned schedule rules and slot generation |
| `migrations/versions/002_frontdesk_booking.py` | Persisted booking tables, after revision `001`, unchanged |
| `migrations/versions/003_frontdesk_rescheduling.py` | Active allocations and atomic replacement linkage, not automatically applied |
| `migrations/versions/004_frontdesk_cancellation.py` | Expiring cancellation intents and persisted outcomes, not automatically applied |
| `migrations/versions/005_direct_request_authorization.py` | Explicit direct-request authorization mode, no fabricated readback; not automatically applied |

No new dependencies are needed. The application injects an existing SQLAlchemy
engine; the booking module does not read `.env` or connect on import. Alembic is
the production schema mechanism. `booking/tables.py` supplies query mappings,
not an alternative production `create_all()` setup.

**Implemented separately:** the opt-in synthetic demo includes voice tool
registration, exact-target direct-request authorization without a second confirmation,
interruption/stale-result safeguards, and booking-status recovery. Its isolated
PostgreSQL target previously passed migration-002 live checks without selecting
the existing clinic database; newer rescheduling and cancellation implementations
have only been verified offline in this work. The normal conversation demo and [read-only clinician assistant](clinician-demo.md)
remain separate. Production authentication and patient identity resolution are
not implemented. Offline tests and live database checks do not
establish real OpenAI/browser-audio behavior; see [demo scope](frontdesk-demo.md).

## Shared schema conventions

- Every argument field is required, including nullable fields. In Python use
  `None`; in JSON use `null`. Unexpected fields are rejected.
- IDs are opaque. Use only IDs from tool results or trusted resolved context.
- `case_id`, `clinician_id`, and `appointment_id` are nonempty strings of at most
  64 characters. `slot_id`, `draft_id`, and `booking_request_id` are UUID strings
  in JSON and UUID objects in directly constructed Pydantic models.
- Appointment types are exactly `initial_consultation`, `fracture_follow_up`,
  `imaging`, `physiotherapy`, or `other`.
- Dates/times in searches use the configured clinic timezone, never the caller's
  laptop timezone. Returned appointment timestamps are timezone-aware and carry
  that IANA zone's actual offset at each instant.
- `booking_request_id` identifies a **persistent booking attempt**. It survives
  invocation retries. `meta.request_id` is a **new UUID for each tool invocation**;
  it is not the booking idempotency key. `draft_id` identifies the prepared details
  and is distinct from both.

### Common response envelope

Every response has all six fields below. This is Python shape notation; `data`
contains one of the concrete payloads in the following sections.

```python
{
    "contract_version": "1.0.0",
    "status": "ok",                  # "ok" or "error"
    "data": data,                    # tool-specific payload, or None on error
    "error": None,                   # structured error below, or None on success
    "warnings": [],
    "meta": {
        "request_id": "00000000-0000-4000-8000-000000000901",
        "tool_name": "find_available_slots",
        "retrieved_at": "2026-09-30T10:00:00+00:00",
        "duration_ms": 8,             # nonnegative integer
        "case_context_version": None # positive integer for resolved case tools
    },
}
```

The current backend emits no warnings. The shared successful-envelope warning
schema allows `HISTORICAL_RECORD` and `INCOMPLETE_RECORD`, with `message` and
`evidence_ids`; error responses require an empty warning list. Availability
search reports `case_context_version=None` because it does not depend on a case.

An error replaces `data` with `None` and uses:

```python
{
    "code": "HOLD_EXPIRED",
    "message": "The hold expired. Prepare an available slot again.",
    "retryable": False,
    "candidates": [],
}
```

Errors never return another patient's details, SQL exceptions, secret values, or
a silently substituted slot. A successful lookup envelope does **not** by itself
mean an appointment was booked.

## 1. `find_available_slots`

### Arguments

| Field | Type | Meaning/validation |
|---|---|---|
| `appointment_type` | Appointment-type enum | Exact type requested |
| `start_date` | `str` | Valid `YYYY-MM-DD`, inclusive |
| `end_date` | `str` | Valid `YYYY-MM-DD`, inclusive; not before start; at most 31 calendar days including both endpoints |
| `earliest_start_time` | `str \| None` | `HH:MM`, inclusive start-time filter on each day; `None` removes the lower bound |
| `latest_start_time` | `str \| None` | `HH:MM`, inclusive start-time filter on each day; `None` removes the upper bound |
| `clinician_id` | `str \| None` | Exact known clinician ID, or any eligible clinician |
| `max_results` | `int` | 1 through 100; booleans and numeric strings are rejected |

```python
arguments = {
    "appointment_type": "fracture_follow_up",
    "start_date": "2026-10-01",
    "end_date": "2026-10-03",
    "earliest_start_time": "09:00",
    "latest_start_time": "16:00",
    "clinician_id": None,
    "max_results": 3,
}
```

The time window cannot cross midnight; use separate searches instead. It filters
**start time**, not the entire duration: a latest start of `11:00` can include an
`11:00–11:30` appointment. Duration, opening hours, clinician eligibility, location,
and timezone come from published backend schedules, not the LLM.

### Success data

Every `AvailableSlot` has these required fields. Clinician and location are
non-null in this front-desk implementation.

```python
slot = {
    "slot_id": "00000000-0000-4000-8000-000000000101",
    "appointment_type": "fracture_follow_up",
    "starts_at": "2026-10-01T09:00:00+05:30",
    "ends_at": "2026-10-01T09:30:00+05:30",
    "timezone": "Asia/Kolkata",
    "clinician": {
        "clinician_id": "CLIN-SYN-01",
        "display_name": "Dr Synthetic Asha",
    },
    "location": "Synthetic Clinic, Room 1",
}

search_data = {
    "slots": [slot],
    "as_of": "2026-09-30T10:00:00+00:00",
    "has_more": False,
}
```

- `slots` is sorted earliest first and contains at most `max_results` items.
- `has_more=True` means additional matching available slots existed beyond that
  limit at `as_of`. It is not a reservation or a pagination cursor.
- An empty result is `status="ok"`, `slots=[]`, `has_more=False`.
- Search needs a permitted clinic/conversation context, but no confirmed case.
- **No hold is created.** Search records session-scoped offer receipts, so
  preparation can reject a guessed or unoffered slot ID. Because this is a
  persisted metadata write, its registry flag is conservatively
  `read_only=False`, even though availability itself is not changed.

## 2. `prepare_booking`

### Arguments

| Field | Type | Meaning |
|---|---|---|
| `case_id` | `str` | Must equal the server-confirmed case |
| `slot_id` | UUID | Exact slot returned by search in this permitted conversation |

```python
arguments = {
    "case_id": "1042",
    "slot_id": "00000000-0000-4000-8000-000000000101",
}
```

### Success data

```python
prepared_data = {
    "draft_id": "00000000-0000-4000-8000-000000000201",
    "booking_request_id": "00000000-0000-4000-8000-000000000301",
    "hold_expires_at": "2026-09-30T10:02:00+00:00",
    "case_id": "1042",
    "patient_display_name": "Synthetic Patient A",
    "slot": slot,  # the complete AvailableSlot object above
}
```

The backend rechecks the exact slot, case eligibility, clinician conflicts,
patient conflicts, and offer membership. It persists a draft and an exclusive
short-lived hold. The default is 120 seconds, shortened if the appointment would
start sooner. At `now == hold_expires_at`, the hold is already expired.

Retrying the same still-valid active selection returns its existing IDs and
original expiry, without refreshing the two-minute clock. It rechecks validity;
it does not treat an obsolete hold as usable. A new selection supersedes the
old draft and releases its hold. The controller should invalidate a selection
immediately on a correction, even if the proposed replacement later proves
unavailable.

**This is not a confirmed appointment.** Read back the returned patient name,
appointment type, clinician, date/time with timezone, and location. Ask “Shall I
book this?” rather than announcing a booking.

## 3. `book_appointment`

### Arguments

| Field | Type | Meaning |
|---|---|---|
| `case_id` | `str` | Must equal the confirmed case |
| `draft_id` | UUID | Exact prepared draft being confirmed |

```python
arguments = {
    "case_id": "1042",
    "draft_id": "00000000-0000-4000-8000-000000000201",
}
```

There is deliberately **no `confirmed` boolean**, confirmation text, session ID,
case-context version, or authorization field in the tool arguments. Supplying
one is an argument-validation error.

The backend checks that the uncommitted draft belongs to the permitted session
and case, is current, has a valid hold, and has a controller-recorded confirmation
for its exact authoritative readback on the current conversation version. It
rechecks the underlying patient/slot details and conflicts before inserting.

### Success data

```python
appointment = {
    "appointment_id": "AP-SYNTHETIC-001",
    "case_id": "1042",
    "patient_display_name": "Synthetic Patient A",
    **slot,                          # flattened slot fields, not a nested "slot"
    "status": "confirmed",
}

booking_data = {
    "booking_request_id": "00000000-0000-4000-8000-000000000301",
    "appointment": appointment,
}
```

The appointment, exclusive allocation, and successful draft/request outcome are
saved in one transaction. Repeating a successfully booked draft returns the
original appointment, including after the old hold expiry. It does not create a
second appointment or require confirmation again for that completed outcome.

The returned `confirmed` appointment is the **committed booking snapshot**.
A persistent `booked` request means this attempt committed, not that the
appointment remains active after a later change. Use the controller-only
`list_bookings(context)` for current active future bookings, not old request
receipts. Rescheduling preserves immutable historical receipts.

## 4. `get_booking_status`

### Arguments

| Field | Type | Meaning |
|---|---|---|
| `case_id` | `str` | Must equal the confirmed case |
| `booking_request_id` | UUID | Persistent attempt ID returned by preparation, not `meta.request_id` |

```python
arguments = {
    "case_id": "1042",
    "booking_request_id": "00000000-0000-4000-8000-000000000301",
}
```

### Success data

All variants include `booking_request_id`, `draft_id`, `as_of`,
`hold_expires_at`, `booking_state`, `appointment`, and `failure`.

```python
status_data = {
    "booking_request_id": "00000000-0000-4000-8000-000000000301",
    "draft_id": "00000000-0000-4000-8000-000000000201",
    "as_of": "2026-09-30T10:01:00+00:00",
    "hold_expires_at": "2026-09-30T10:02:00+00:00",
    "booking_state": "booked",
    "appointment": appointment,
    "failure": None,
}
```

| `booking_state` | `appointment` | `failure` | Meaning/action |
|---|---|---|---|
| `pending` | `None` | `None` | No terminal outcome; do not announce success or failure |
| `booked` | Complete `BookedAppointment` | `None` | Commit is established; use the existing appointment |
| `expired` | `None` | `None` | Uncommitted hold expired; select/prepare again |
| `failed` | `None` | `{"code": ..., "message": ...}` | Persisted definitive failure; follow its reason |

A persisted failure reason is one of `DRAFT_SUPERSEDED`, `SLOT_UNAVAILABLE`,
`PATIENT_CONFLICT`, or `CASE_NOT_BOOKABLE`. A missing/foreign request instead
returns an error envelope with `RECORD_NOT_FOUND`; it is not a successful
`failed`-state lookup.

This tool is read-only. It can derive `expired` from the persisted expiry and
server time without updating the row. It never retries a commit, creates a new
attempt, or cancels an appointment. A committed request stays `booked` after its
former hold expiry.

## Structured errors and recovery

| Code | Meaning / next step |
|---|---|
| `INVALID_ARGUMENT` | Fix malformed, missing, extra, or incompatible fields |
| `IDENTITY_UNRESOLVED` | Complete trusted case/identity resolution first |
| `CASE_CONTEXT_MISMATCH` | Discard stale case work; obtain the current confirmed context |
| `SESSION_CONTEXT_MISMATCH` | Session is stale, inactive, or not permitted; resolve server context |
| `CLINIC_CONTEXT_MISMATCH` | Clinic configuration/context is unavailable or mismatched |
| `CASE_NOT_FOUND` | Confirmed case is unavailable; do not invent a replacement |
| `RECORD_NOT_FOUND` | Request/draft/slot is unavailable in this context; no cross-case existence is disclosed |
| `SLOT_NOT_OFFERED` | Search and select a returned slot in this conversation |
| `SLOT_UNAVAILABLE` | Re-search; never silently substitute another slot |
| `DRAFT_SUPERSEDED` | Use the current selection and new authoritative readback |
| `HOLD_EXPIRED` | Prepare an available slot again; old confirmation cannot revive it |
| `CONFIRMATION_REQUIRED` | Present the exact draft and obtain explicit consent |
| `CONFIRMATION_STALE` | A later conversation change invalidated consent; present/confirm again |
| `PATIENT_CONFLICT` | An overlapping appointment or hold exists; ask for a different time without disclosing other case details |
| `CASE_NOT_BOOKABLE` | Case is not eligible for this prepared booking |
| `TIMEOUT` | Outcome may be unknown; use the prepared `booking_request_id` to recover |
| `BACKEND_UNAVAILABLE` | Backend could not establish an outcome; recover when available |
| `INTERNAL_ERROR` | Result could not be verified; do not infer that a commit failed |

Only `TIMEOUT` and `BACKEND_UNAVAILABLE` are marked `retryable=True` by this
backend. That flag does not mean “start a new booking.” After an uncertain
`book_appointment`, query status using the same request ID. Even an error after
the database committed must not be translated into a false “booking failed.”
An eventual retry of the same draft is idempotent; a new draft is a new attempt.

## Trusted controller integration

`FrontDeskContext` is a server-only snapshot containing `session_id`, `clinic_id`,
`user_id`, `case_id`, `case_context_version`, and `conversation_version`. The LLM
never supplies these fields. Only the four registry entries are LLM tools.

The current authorization model reuses the existing active `staff`/`clinician`
application-user records. It is not an authentication system, per-clinic access
control system, or an anonymous patient-facing login. Authenticate the principal
and perform identity/case resolution outside these tools before trusting IDs.

### Python lifecycle example

This is an integration outline, not a consent detector. `engine`,
`authenticated_staff_id`, and `resolved_case_id` below must come from trusted
application code. `engine` must be a PostgreSQL SQLAlchemy engine built from
private configuration, ideally with `hide_parameters=True` and a bounded connect
timeout. Do not put credentials in source or tool arguments.

```python
from uuid import UUID, uuid4
from healthcare_voice_agent.booking.service import BookingService

service = BookingService(engine)  # injected; no .env loading here
context = service.open_session(
    clinic_id="SYNTHETIC-CLINIC", user_id=authenticated_staff_id,
)
# Call only after external identity/case resolution has succeeded.
context = service.set_confirmed_case(context, resolved_case_id)

# At the start of EVERY new user utterance, advance the server context.
# This includes a later affirmative utterance. Do not wait for an old tool
# to finish before invalidating that tool's captured context.
context = service.advance_conversation(context)
search = service.invoke("find_available_slots", {
    "appointment_type": "fracture_follow_up",
    "start_date": "2026-10-01", "end_date": "2026-10-03",
    "earliest_start_time": None, "latest_start_time": None,
    "clinician_id": None, "max_results": 3,
}, context=context)
# Handle errors and an empty slots list before proceeding.
assert search["status"] == "ok" and search["data"]["slots"]

# A separate user turn selects one of the offered slots.
context = service.advance_conversation(context)
selected_slot_id = search["data"]["slots"][0]["slot_id"]  # example only: user's selection
prepared = service.invoke("prepare_booking", {
    "case_id": context.case_id, "slot_id": selected_slot_id,
}, context=context)
assert prepared["status"] == "ok"
draft = prepared["data"]

# Present the authoritative details and ask "Shall I book this?".
# ONLY on actual completed delivery, not merely generation/queueing:
details = {key: draft[key] for key in ("case_id", "patient_display_name", "slot")}
details_hash = service.mark_readback_presented(
    context, draft_id=UUID(draft["draft_id"]), details=details,
)

# On the next user utterance's start, invalidate previous conversation consent.
context = service.advance_conversation(context)
# Wait for the complete final utterance. Enter this block ONLY when the trusted
# controller has established explicit consent to this exact draft, without a
# correction. Do not classify consent using a substring check for "yes".
confirmation_event_id = uuid4()  # stable controller event ID; retain on retries
service.record_confirmation(
    context,
    draft_id=UUID(draft["draft_id"]),
    details_hash=details_hash,
    confirmation_event_id=confirmation_event_id,
)
result = service.invoke("book_appointment", {
    "case_id": context.case_id, "draft_id": draft["draft_id"],
}, context=context)

# Recovery when result is uncertain, using the SAME persisted attempt:
status = service.invoke("get_booking_status", {
    "case_id": context.case_id,
    "booking_request_id": draft["booking_request_id"],
}, context=context)
if status["status"] == "ok" and status["data"]["booking_state"] == "booked":
    existing_appointment = status["data"]["appointment"]
```

The `assert` statements illustrate branching only; a real controller must handle
all errors and ask the user rather than crash. The example does not implement
speech delivery, user selection, or consent classification.

Important controller rules:

1. Invalidate stale output at acoustic speech start, but advance database context
   once per nonempty finalized user utterance, not for VAD-only noise. Use the
   complete final utterance for consent, never an ASR partial or LLM claim.
   Capture the resulting context for all work on that turn.
2. Discard stale tool results and queued speech before presentation. A successful
   tool response from a prior context is not permission to read it in a new case.
3. Confirmation is accepted only on the conversation version immediately after
   the effective readback receipt. A fully delivered, short confirmation repair
   may use the controller-only `refresh_confirmation_prompt` hook to carry an
   existing delivered selection summary across exactly one clarification turn. It rechecks the
   unchanged appointment, active hold and ownership, preserves `readback_at`, and
   updates only `readback_version`/`updated_at`. It never authorizes a booking.
   Missing initial summary proof, a version gap, changed details or an interrupted
   repair cannot use this shortcut; present the selected summary again. The natural
   demo speaks type/clinician/day/start in a short permission question while the
   complete authoritative payload, including unspoken fixed-case metadata, stays
   bound to the receipt.
4. If a selection change is already known at turn start (for example, a UI
   selection), use `advance_conversation(context, selection_changed=True)`.
   For voice, advance on the finalized utterance; if it reveals a correction,
   call `invalidate_selection(context)` before planning the replacement to
   release/supersede the old draft without advancing that utterance twice.
5. For a changed confirmed case, call `set_confirmed_case` only after re-resolving
   identity; it increments context versions and invalidates a pending selection.
6. A correction that wins the transaction lock prevents a stale commit. A
   correction arriving after a completed commit cannot retroactively undo that
   appointment. A subsequent move or cancellation must use its own controller-owned
   preparation, delivered readback and fresh explicit permission.
7. Keep the stable session UUID and booking-request ID in trusted application
   state for recovery. After re-authenticating the same principal,
   `resume_context(session_id=..., user_id=...)` reloads an active persisted
   context after restart. A new unrelated session cannot read the old request.
8. `close_session(context)` explicitly invalidates the session and releases its
   pending hold. Do not close it merely to recover a response lost during a
   transient reconnect. Closed-session recovery would require a separately
   designed authorization flow.

For async callers, `await service.invoke_async(name, arguments, context=context)`
runs the synchronous database work off-loop. Cancelling that awaiter does not
prove the database transaction was cancelled or uncommitted; recover by request
ID. Controller/admin methods can raise `BookingError`; tool `invoke` converts
failures into the validated envelope.

## Backend schedule configuration

A trusted administrator publishes inventory before availability searches. The
LLM cannot set appointment duration, opening hours, or clinician eligibility.
The staff/clinician/case records referenced here must already exist.

```python
from datetime import date, time
from healthcare_voice_agent.booking.scheduling import ScheduleRule

service.register_clinic("SYNTHETIC-CLINIC", "Asia/Kolkata")
rules = (
    ScheduleRule(
        appointment_type="fracture_follow_up",
        clinician_id="CLIN-SYN-01",  # existing, eligible synthetic clinician
        weekdays=(0, 1, 2, 3, 4),    # Monday=0 through Sunday=6
        opens_at=time(9, 0),
        closes_at=time(17, 0),
        duration_minutes=30,
        location="Synthetic Clinic, Room 1",
    ),
)
slot_ids = service.publish_schedule(
    "SYNTHETIC-CLINIC", rules, date(2026, 10, 1), date(2026, 10, 31),
)
```

Run this administrative setup before `open_session` for that clinic. The snippet
is not a default schedule and is not executed automatically. Supply actual
approved configuration rather than having the agent invent it.

Rules use same-day, minute-precision local opening hours, durations of 5–240
minutes, and explicit clinician/type associations. A publication covers at most
366 inclusive days. Repeating the same published slot identity reuses its ID;
it does not silently relocate the slot. Clinic timezone changes are rejected by
the registration helper and need a separate administrative policy.

DST-ambiguous/nonexistent local endpoints are skipped rather than guessed. A
slot whose actual elapsed duration differs from its configured duration across
a transition is also skipped. Search uses the resulting materialized inventory.

Intervals are half-open `[start, end)`, so adjacent appointments do not overlap.
The backend checks clinician conflicts and, after case resolution, patient
conflicts across that patient's cases. Existing `scheduled`/`confirmed`
appointments participate. A legacy appointment with an unknown end time is
handled conservatively rather than assigning a guessed duration.

## Persistence, concurrency, and rollout

Revision `002` adds:

| Table | Responsibility |
|---|---|
| `booking_guard` | Singleton transaction lock for booking/controller/schedule writers |
| `booking_clinics` | Clinic timezone configuration |
| `booking_sessions` | Trusted user/clinic/case context, versions and current draft |
| `booking_slots` | Published appointment inventory |
| `booking_offers` | Session-scoped search receipts, not reservations |
| `booking_drafts` | Persistent attempt, authoritative readback snapshot, confirmation and outcome |
| `booking_holds` | Exclusive active slot hold and expiry |
| `booking_allocations` | Committed slot/request/appointment linkage |

PostgreSQL's clock is authoritative for hold validity. Expiry checks do not
require a timer job to run at exactly two minutes. Cleanup during preparation
can persist expiry and remove expired hold rows; status lookup derives expiry
without writing. The persisted patient binding/readback snapshot is rechecked
before an uncommitted booking, not silently retargeted.

All booking, controller, and schedule operations lock guard row `1`. This is an
intentionally serialized prototype, not a high-throughput scheduling design.
**Every future appointment/calendar writer must follow the same lock protocol**
for arbitrary clinician/patient overlap guarantees. Unique keys and deferred
composite foreign keys additionally protect exact slot exclusivity and atomic
request/draft/allocation/appointment linkage. Direct SQL or external calendar
writers that ignore the protocol are not made safe merely by these tools.

The migration extends `001`; it does not rewrite that already-applied revision,
seed patient data, or publish schedules. **It has not been applied as part of this
implementation.** When ready, the following is an explicit database-changing
administrative step, not a startup side effect:

```bash
uv run --locked alembic upgrade head
```

Keep existing `.env` and database credentials unchanged. Existing synthetic
patients, cases, clinicians and staff must be seeded separately through the
existing fixture workflow if absent. Then register the clinic and publish its
approved schedule. Do not use `metadata.create_all()` against the live database
and do not downgrade an applied revision to edit it.

### Verification boundary

Offline tests cover schema validation, synthetic examples, transactional booking
logic, retries, expiry, context/confirmation guards, conflicts and recovery.
SQLite tests use isolated files and `BEGIN IMMEDIATE`; they are not proof of
PostgreSQL lock behavior or deferred-constraint execution. Offline Alembic SQL
rendering is not a migration application. Before enabling real booking, verify
`002` and concurrent transactions in a disposable PostgreSQL environment, then
complete authenticated controller/voice integration and synthetic end-to-end
readback, correction, interruption and recovery tests.

### Implementation verification checkpoint

- **718 tests passed** in the final full offline run, including **131 new
  booking/schema/scheduling tests**. An existing voice-worker startup test timed
  out on an earlier run, then passed in isolation and in the full rerun; no voice
  runtime code was changed for that timeout.
- Lockfile validation, Python compilation, Alembic `001:002` offline SQL rendering,
  and the original fixture dry-run (288 normal rows) passed.
- Fingerprints confirmed private `.env`, dependencies, Compose, fixture bytes,
  migration `001` and the existing local speech implementations were unchanged.
- No migration was applied, fixture rows inserted, schedules published, live
  PostgreSQL queries executed or voice-agent booking calls enabled by this work.
