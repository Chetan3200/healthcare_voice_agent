# Healthcare Voice Agent

A Python project for a synthetic fracture-clinic assistant. All patients,
clinical records and predictions are fictional test data. The clinician profile
can separately retrieve external public web guidance through Exa.

## Current implementation status

- Present: Alembic database schema, deterministic synthetic fixtures, safe seed
  loader, Pydantic tool contracts, validated provider configuration, three provider
  factories, a minimal browser voice runtime, per-session traces, four separate
  front-desk tool contracts and a transactional booking backend, an isolated
  synthetic front-desk voice profile with trusted direct-request authorization, a
  read-only clinician profile with Exa guidance retrieval, and offline tests.
- Not implemented: production authentication/identity resolution, guaranteed
  audible-step resumption and semantic backchannel handling. Exa is external
  unversioned guidance, not clinic-approved policy. Setup does not generate embeddings.
- The runtime uses Pipecat's bundled UI and a simple connectivity-demo prompt.
  Its microphone, live provider access, and audible reconnect checkpoint still
  require a manual live test; passing offline tests does not establish that.
- The separate synthetic PostgreSQL demo has passed live booking/constraint checks.
  Offline tests do not establish real model access, browser playback or voice quality.

## Isolated appointment-booking demo

The separate front-desk profile uses OpenAI STT, LLM and TTS, a fictional fixed
patient, and its own PostgreSQL database on port `55432`. It does not copy or
modify the existing clinic database. See [setup, safeguards and testing](docs/frontdesk-demo.md).

```bash
./scripts/voice_frontdesk_demo.sh --check     # no model calls
LIVE_API_ENABLED=true ./scripts/voice_frontdesk_demo.sh
```

Open `http://127.0.0.1:7862/client/`. Clicking Connect starts paid OpenAI services.
The front desk uses **one Pipecat Flows node with seven native tools**, all
available throughout the conversation. The LLM interprets requests and writes
ordinary replies directly. There are no conversation phases, Python phrase/time
parsers, extra confirmation turns or audio-delivery gates for slot selection.
Python keeps structured transaction checks, duplicate-write protection and
uncertain-write recovery; the database enforces ownership and availability.
Booking and rescheduling also require `requested_starts_at`, checked against the
persisted slot start before preparation. This is a consistency guard, not a
second speech parser.

**Start reading:** `demo/flow.py` → `demo/tools.py` → `booking/service.py`
under `src/healthcare_voice_agent/`. See [the short flow guide](docs/frontdesk-demo.md)
for the complete code map, safeguards and manual checks.

The database must be at revision **005** for direct-request authorization.
This adds a migration but no dependency. If necessary, explicitly prepare the isolated database
with `.venv/bin/python scripts/setup_frontdesk_demo.py --prepare` while the server
is stopped. To link the canonical existing follow-up into the front-desk listing,
run `.venv/bin/python scripts/setup_frontdesk_demo.py --link-existing` after preparation.
This separate, idempotent data-only action preserves the appointment itself and
does not migrate, start containers, or publish slots. Existing additional demo
bookings are preserved; the canonical seeder refuses changed original fixture rows.
Syntax/import checks are not
live-model or browser-audio verification; the new flow needs a manual live pass.

## Read-only clinician demo

A separate **single-node assistant with five clinical tools plus `open_case`**
uses the existing synthetic clinical database. It starts with no selected patient;
say “Open case 1042” or search by name and clarify the case. It returns study metadata, stored model results, reviewed
reports, appointments and source-bounded Exa excerpts. It cannot book or modify
records. It combines Smart Turn with Pipecat's native LLM incomplete-turn filtering
for hesitant speech; front-desk turn handling is unchanged. Keep existing
database/OpenAI settings; add `EXA_API_KEY` privately.

```bash
./scripts/voice_clinician.sh --check
LIVE_API_ENABLED=true ./scripts/voice_clinician.sh
```

Open `http://127.0.0.1:7863/client/`. See [setup, verification and the five difficult
voice scenarios](docs/clinician-demo.md). No installs, migrations or paid calls
were performed to implement this profile. Exa live behavior is unverified; key
presence alone is not provider validation. Clinical reads and case selection were
checked against PostgreSQL. See [50 varied test queries and expected outcomes](docs/clinician-test-queries.md).
Failed case selections now clear old context without replaying the failed request;
study metadata exposes pending/failed current model status separately from completed
historical IDs. Both profiles record hashes of loaded role/tool definitions for
version verification. Restart servers manually before live retesting these changes.
For a guided walkthrough of both agents, use [the demo conversations](docs/voice-demo-conversations.md):
caller lines, expected tool use, selected edge cases, and delay/interruption cues.

## Prerequisites

- uv; this migration was checked with uv 0.9.9.
- Python 3.11.16, selected by `.python-version`. uv can install it if needed.
- Docker Desktop or Docker Engine with Compose v2 for database checks.

Run every command below from the repository root. Do not activate the old
standalone database environment. Dependencies are managed by `pyproject.toml`
and `uv.lock`, not by a second requirements file.

## Install and run offline checks

For the maintainer's first setup only, run `uv lock` to create `uv.lock`.
Commit that lockfile. Subsequent checkouts use:

```bash
uv sync --locked
uv run --locked python -m healthcare_voice_agent
uv run --locked pytest
uv run --locked python scripts/seed.py --dry-run
uv run --locked alembic heads
```

The application entrypoint validates provider settings, saves a non-secret
`runs/<unique-id>/config.json`, and exits. Without `--serve`, it never constructs
provider services or calls an API, even if live mode is enabled.
The offline suite retains the 24 seed tests and 4 project-layout/contract checks,
and adds configuration, redaction, provider wiring, timeout, and cleanup tests.
Fixture validation expects 288 normal rows and 92 scenario count checks.

## Provider configuration (no API calls)

```bash
uv run --locked python -m healthcare_voice_agent --check-config
```

Defaults: OpenAI streaming transcription (`gpt-live-transcribe`, English input
hint `languages: ["en"]`), a text/tool LLM (`gpt-4.1-mini-2025-04-14`), and TTS
(`gpt-4o-mini-tts`, voice `coral`). These are configurable baseline candidates,
not verified model access or measured performance. Your private `.env` is not
rewritten; add settings from `.env.example` only when needed.

Live services default to disabled. The factories require explicit opt-in and a
private API key for any selected OpenAI stage. An entirely self-hosted stack
needs no OpenAI key. No test budget setting is required.

Install the optional Pipecat/OpenAI dependencies in your normal Terminal:

```bash
uv sync --locked --extra voice --no-cache --link-mode copy
uv run --locked --extra voice pytest
```

An actual-service constructor/cleanup test is skipped without the voice extra.
With it installed, the test uses a dummy key and blocks network connections; it
does not start a voice session or validate live APIs. See
[provider configuration and limitations](docs/providers.md).

## Local browser voice loop (no clinic tools)

First install the updated `voice` extra in your normal Terminal using the command
above. Add your API key privately and set `LIVE_API_ENABLED=true` in the existing
`.env`. Do not paste a key into chat or replace existing database settings.

After those prerequisites are satisfied:

```bash
uv run --locked --extra voice python -m healthcare_voice_agent --serve --env-file .env
```

Open **http://127.0.0.1:7860/client/**, allow microphone access, and click Connect.
Speak first. Connecting starts provider services and can incur API usage,
including streaming silence. Disconnect when finished. Automatic idle and
session-duration cutoffs are disabled by default; stalled-request and cleanup
timeouts remain enabled.

A new context, provider set, and `runs/<run-id>/sessions/<id>/events.jsonl` are
created for every connection. See [startup, logs, and the manual checkpoint](docs/voice-loop.md).
No database connection is needed for this stage.

## Optional Nemotron / Breeze / HybridDiffusion stack

Opt-in adapters now support Nemotron 3.5 streaming ASR, Breeze TTS 2 streaming
audio, and HybridDiffusion-2B streamed chat. They connect to separately prepared
model servers; CUDA packages never enter the app's existing environment.

See **[GPU setup and startup](docs/gpu-models.md)** for pinned runtime/model
revisions, explicit installation/download steps, GPU launchers, and SSH tunnels.
After preparing and starting those servers:

```bash
./scripts/voice_gpu.sh --check-config
LIVE_API_ENABLED=true ./scripts/voice_gpu.sh
```

The launcher does not install/download anything, change `.env`, or replace the
OpenAI/Whisper/Kokoro defaults. Offline tests verify protocol and lifecycle
handling, not GPU inference, model quality, memory fit, or native latency.

## Front-desk booking tools

Four booking contracts are implemented separately from the five clinician tools:
`find_available_slots`, `prepare_booking`, `book_appointment`, and
`get_booking_status`.

- [Detailed schemas, response dictionaries, and controller integration](docs/frontdesk-booking.md)
- [Generated JSON Schemas](docs/contracts/frontdesk-tool-contracts.json)
- [13 synthetic success/error/recovery examples](docs/contracts/frontdesk-tool-examples.json)

The backend stores slot offers, short holds, drafts, confirmation versions and
idempotent outcomes. Revision **002** added eight booking tables; new revision
**003** adds allocation activity and reschedule linkage without rewriting applied
migrations. Controller-only `list_bookings`, `prepare_reschedule` and
`reschedule_appointment` support current appointment queries and atomic changes.
Revision **004** adds expiring cancellation intents and durable outcome receipts.
Native flow tools wrap cancellation preparation, commit and recovery. The model
cannot call the receipt/consent-recording backend hooks directly or assert a
confirmation flag. Revision **005** adds an explicit direct-request authorization
mode: the session binds the real caller request to the exact target and commits
without fabricating a readback or requiring another yes.
The isolated front-desk voice profile wires these operations; the normal voice
profile remains separate. Production authentication and patient identity
resolution are still outside this fixed-case synthetic demo.

## Start and verify the database

```bash
uv run --locked python scripts/setup_env.py
docker compose up -d --wait
uv run --locked alembic upgrade head
uv run --locked python scripts/check_db.py
```

`setup_env.py` preserves an existing `.env` byte-for-byte. On a fresh checkout it
creates a private file with a generated database password. `.env.example` is a
reference, not a file with a usable password. Never commit `.env` or API keys.
Keep existing credentials when reusing a PostgreSQL volume; changing `.env` does
not change the password stored inside PostgreSQL. If needed, set `DB_PORT=5433`
in the private `.env` before starting Docker.

Docker starts PostgreSQL only. Alembic owns schema changes. The existing Compose
project name and named volume are preserved. Do not run `docker compose down -v`
or downgrade the applied schema as part of setup.

After explicitly applying the current migration head, `check_db.py` should
report pgvector, Alembic revision **005**, and **24 clinic tables** (14 baseline,
eight original booking tables, reschedule linkage, and cancellation receipts). Existing databases are
not upgraded automatically. Patient count depends on fixture loading; the checker
inserts no rows. The unchanged revision-001 fixture format is accepted on database
revisions 001 through 005; it does not seed booking schedules.

## Synthetic fixtures

```bash
# No database connection or writes:
uv run --locked python scripts/seed.py --dry-run

# Read-only verification; requires the fixtures to have been loaded:
uv run --locked python scripts/seed.py --verify-db

# Explicitly load missing rows when needed, then verify:
uv run --locked python scripts/seed.py --apply
uv run --locked python scripts/seed.py --verify-db
```

`--apply` inserts missing rows, skips identical rows, and aborts on conflicts.
It never overwrites or deletes existing rows. Fixture files and their manifest
are versioned together; do not edit hashes just to make a failed check pass.

## Code map

- `src/healthcare_voice_agent/`: installable application package.
- `src/healthcare_voice_agent/config.py`: validated settings and public run records.
- `src/healthcare_voice_agent/demo/flow.py`: single front-desk node, prompt and native tools.
- `src/healthcare_voice_agent/demo/tools.py`: native LLM tool handlers.
- `src/healthcare_voice_agent/demo/runtime.py`: isolated session and direct-action state.
- `src/healthcare_voice_agent/voice/providers.py`: `build_stt`, `build_llm`, `build_tts`.
- `src/healthcare_voice_agent/voice/pipeline.py`: provider-neutral pipeline assembly.
- `src/healthcare_voice_agent/voice/session.py`: per-connection lifecycle and cleanup.
- `src/healthcare_voice_agent/voice/server.py`: bundled runner adapter, loopback only.
- `src/healthcare_voice_agent/voice/{tracing,observers,rtvi}.py`: logs and safe UI errors.
- `src/healthcare_voice_agent/agent/prompts.md`: the non-clinical demo prompt.
- `src/healthcare_voice_agent/tools/contracts.py`: typed clinician contracts.
- `src/healthcare_voice_agent/clinician/`: executable read-only clinical tools and Exa retrieval.
- `scripts/`: environment setup, database checking, and deterministic seeding.
- `migrations/`: Alembic revisions. Leave applied revision 001 unchanged.
- `fixtures/`: synthetic clinical/document data and coverage scenarios.
- `tests/`: offline tests, including real framework assembly and mocked session lifecycles.
- `evals/`: future frozen conversation scenarios and audio fixtures.
- `results/`: selected, sanitized evaluation outputs for submission.
- `runs/`: ignored local runtime output, created when needed.

## Submission notes

Start reviewers at `docs/frontdesk-demo.md`, then `demo/flow.py` and `demo/tools.py`.
Include source, `pyproject.toml`, `uv.lock`, migrations, synthetic fixtures and docs.
Do not include `.env`, `.venv`, `.demo`, local model caches, or unsanitized `runs/`.
The new flow has had local syntax/import checks, not a completed live voice evaluation.

## More detail

- [Local Kokoro + Whisper turbo paired experiment](docs/local-speech.md)
- [Browser voice loop, logs, and manual checkpoint](docs/voice-loop.md)
- [Provider configuration, factories, and timeout semantics](docs/providers.md)
- [Database design](docs/database.md)
- [Seeding, fixture cases, and limitations](docs/seeding.md)
- [JSON tool contracts and examples](docs/contracts/)

The fixture coverage scenarios are not a completed voice-evaluation suite.
Live tests require model access. Record measured results separately from
configuration or constructor checks. OpenAI credentials are not needed for the
offline checks.
