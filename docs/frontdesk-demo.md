# Synthetic front-desk voice agent

This is an English-only appointment demo for a fixed fictional patient and clinic.
It uses OpenAI STT → a tool-calling OpenAI LLM → OpenAI TTS through Pipecat.
Pipecat Flows (included in the pinned Pipecat 1.11.0) keeps one front-desk node
and its tools available throughout the conversation.
There is no separate action planner, speech-writing model, or prose validator.

For caller lines and presenter cues covering both agents, see
[voice demo conversations](voice-demo-conversations.md).

## Run

Use the project Python 3.11 environment and committed `uv.lock`.

```bash
# Read-only configuration/database readiness check; no model calls:
./scripts/voice_frontdesk_demo.sh --check

# Explicit database preparation, only if the isolated database is not ready:
.venv/bin/python scripts/setup_frontdesk_demo.py --prepare

# Stop any old demo process before restarting:
LIVE_API_ENABLED=true ./scripts/voice_frontdesk_demo.sh
```

Open http://127.0.0.1:7862/client/ and click Connect. Provider use is paid.
Use synthetic speech only. Direct-request actions require migration **005**.
If the database is not already at revision 005, run the explicit preparation
command above with the server stopped. It preserves existing bookings. The
single-node simplification requires no additional migration.

The isolated database is `frontdesk_synthetic_demo` on port **55432**, with case
**1042** at **DEMO-CLINIC** in **Asia/Kolkata**. The original clinic database on
port 5433 is separate. Preparation preserves additional demo bookings.

## Read the code in this order

Paths are relative to `src/healthcare_voice_agent/`:

1. **`demo/flow.py`**: one front-desk node, one prompt and seven native tools.
2. **`demo/tools.py`**: thin handlers and the shared prepare/authorize/commit operation.
3. **`booking/service.py`**: existing database transactions and domain safeguards.
4. **`demo/runtime.py`**: fixed identity, turn boundary and in-flight operation state.
5. **`voice/frontdesk_demo.py`**: native FlowManager and finalized-turn tracking.

Startup remains `scripts/voice_frontdesk_demo.sh` → `__main__.py` →
`voice/server.py` → `voice/session.py` → `voice/pipeline.py`.
Provider construction remains in `voice/providers.py`.

## One agent, no conversation phases

The single **frontdesk** node keeps these tools available throughout:

| Tool | Arguments / purpose |
|---|---|
| `clinic_information` | Services and clinicians, not date-specific availability |
| `search_availability` | Type, date range, optional time/clinician filters |
| `list_appointments` | Optional date range; null dates list all active future bookings |
| `book_slot` | Exact `slot_id` plus timezone-aware `requested_starts_at` |
| `cancel_appointments` | Explicit list of `appointment_id` and `record_version` pairs |
| `reschedule_appointment` | Original ID/version, replacement `slot_id`, and `requested_starts_at` |
| `operation_status` | Recover the stored in-flight request without starting a write |

The LLM interprets dates, times, ordinals, references, intent and ambiguity.
It asks for genuinely missing information and calls tools when the request is
clear. Python does not independently parse the caller's words or verify linguistic
consent. There are no collect/choose/done transitions, selection-route tools,
phrase validators, local offer-history copies or speech-delivery receipts.
Ordinary model text is not rewritten. For the current speech experiment, a small
`FullReplyTTSBuffer` waits for the complete LLM reply, then sends one native
pre-aggregated text frame to TTS instead of separate sentences. Interrupted
unfinished replies are discarded; empty/tool-only responses produce no speech.

The greeting uses the patient's first name and lists booking, viewing,
rescheduling and cancellation. Replies should be short, without repeating
established details or asking for a second confirmation.

The search requests up to 100 slots and includes counts, `has_more`, the actual
search scope and cadence-checked groups for concise summaries. An empty follow-up
search does not delete earlier backend offers. The model can select a previously
returned ID; the backend checks its current availability when preparing a booking.

For cancellation, the model supplies exactly the requested IDs from a listing.
Python never interprets “all,” “these,” spoken times or ordinal selections.
A batch executes individual transactions and reports completed/remaining counts,
stopping on an error, interruption or uncertain outcome. A reschedule replaces
its original atomically; the model must not cancel the original separately.

## What Python still checks

Only structured arguments and transaction state: fixed case/clinic ownership,
real slot IDs, requested-start consistency, availability, conflicts, source record versions, expiry,
idempotency and stale in-flight turns. The domain checks remain in BookingService,
not duplicated as dialogue rules in the voice layer. Appointment listings apply
the model's explicit date arguments rather than overriding them from caller text.

Explicit requested times use an exact-start availability search (equal earliest/latest
bounds). Both write tools require `requested_starts_at`; the handler compares its
instant with the persisted, authorized, offered slot's **start**, not its end,
before replacing pending state or preparing a write. Missing/naive timestamps
are rejected; a different instant returns `SLOT_TIME_MISMATCH`. Equivalent offsets
normalize to the same instant for same-turn idempotency. An uncertain operation
still requires status recovery. Both fields come from the model, so this checks
consistency, not whether the original speech was understood correctly. Garbled
time selections require clarification; clear requests need no second confirmation.

The trusted backend `authorize_direct_request` records
`authorization_kind=direct_request` without a fabricated readback or extra yes.
The existing migration 005 and older confirmation APIs remain unchanged. This
simplification adds no migration or dependency.

The session stores only an in-flight operation and same-turn results needed for
retries. A lost commit acknowledgement preserves its request ID and blocks a new
write until `operation_status` establishes the outcome. Status itself does not
commit. Retrying an uncommitted request in the same turn reuses its existing hold.

The audio path is STT → user context → LLM → full-reply buffer → TTS → output,
with one input processor for finalized-turn tracking. The front-desk launcher
uses `marin` with `gpt-4o-mini-tts`; other profiles are unchanged. Audio still
streams in 500 ms PCM chunks, but speech begins only after the reply finishes
generating. This may improve within-reply intonation at the cost of later first
audio; audible quality and latency still need a live comparison. Real VAD and final-only
transcription remain enabled; interim fragments do not start ghost turns.
English-only STT and existing provider configuration are unchanged.

## Existing fixture appointment visibility

Listings still enforce the original case/clinic allocation joins. The canonical
`AP-1042-01` fixture originally had no front-desk provenance and was invisible to
that listing. The bounded, idempotent data-only command is:

```bash
.venv/bin/python scripts/setup_frontdesk_demo.py --link-existing
```

It verifies the marked isolated database and exact canonical synthetic source,
then links an inactive bootstrap session, a disabled/non-offerable slot, a
provenance draft and an allocation. It does not create or modify an appointment,
start a container, migrate, publish a schedule, or access the clinician database.
Metadata explicitly identifies an imported existing receipt, not a voice booking
or audio confirmation. Conflicting linkage fails closed. Run this after initial
`--prepare` for a fresh demo; it is separate from read-only `--check`. Do not
remove ownership joins or treat a restricted empty listing as proof that no other
clinical appointments exist. The canonical seeder still rejects changed original
fixture rows; it must not reset appointments after legitimate cancellations.

## Verification boundary

A bounded check using the real service and isolated SQLite fixture passed:

- Search 16 slots, perform an empty evening search, then book the earlier 9 a.m.
  slot without any speech receipt; repeating the tool does not duplicate it.
- Atomically reschedule using an explicit source ID/version and replacement slot.
- Reject a stale source version; cancel exactly two explicit appointments and
  reuse the results on duplicate calls.
- Recover a committed booking after a simulated lost acknowledgement without
  creating another booking; reject a write from an interrupted old turn.

Modified modules parse and the native tool schemas and voice adapter import.
These checks do not verify model decisions, speech recognition or browser audio.
No full suite, paid provider call, PostgreSQL write, migration or restart was run.
The full-reply speech experiment also passed focused offline checks using native
Pipecat with mocked speech HTTP: three sentences become one `marin` request,
spoken-text context remains intact, and interrupted/empty replies are not spoken.

## Manual checkpoint

Restart the existing demo, then try one conversation: request imaging slots,
ask about an unavailable time window, choose an earlier returned slot, list the
booking, reschedule it and cancel it. Check that there is no extra confirmation,
no repeated full summary and no missing tool after completing an action.
