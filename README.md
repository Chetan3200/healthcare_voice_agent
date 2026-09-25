# Healthcare Voice Agent

Synthetic fracture-clinic voice agents built with Python, PostgreSQL and Pipecat.
All patient records, predictions and appointments are fictional. This is a local
demo, not production authentication or a system for real patient data.

- **Front desk:** book, reschedule and cancel appointments in an isolated demo database.
- **Clinician:** select a case and read clinical records; optional external guidance through Exa.
- **Conversation:** a basic voice-connectivity mode without clinical tools or a database.

## Demo recordings

[Watch the demo recordings on Google Drive](https://drive.google.com/drive/folders/1v54tiLTUsC-Ek2aI4ZPXAyyYkpKNDAOH?usp=sharing).

- **Front-desk agent:** `frontdesk-agent.mp4`
- **Clinician agent:** `clinician-agent.mp4`
- **Background-task handling:** `background-task-test.mp4`

## Setup

Use Python **3.11.16** (see `.python-version`), `uv`, and Docker with Compose v2
for the databases. Run commands from the repository root.

```bash
uv sync --locked --extra voice
.venv/bin/python scripts/setup_env.py
.venv/bin/python -m healthcare_voice_agent --check-config
```

`setup_env.py` preserves an existing `.env`; on a fresh checkout it creates a
private file with a generated database password. Add provider keys privately:
`OPENAI_API_KEY` for any OpenAI stage and optional `EXA_API_KEY` for guidance search.
Use `.env.example` as a reference, never overwrite existing credentials with it.

The configuration check starts no services or provider requests. Live access
requires `LIVE_API_ENABLED=true`. Connecting the browser to an OpenAI profile can
incur charges; guidance searches use Exa. Keep the application and model servers
on loopback, and use SSH tunnels for remote GPU servers.

## Clinical database

Required for the clinician, not for the isolated front desk or conversation mode.
These are explicit setup actions, not automatic application-startup behavior:

```bash
docker compose up -d --wait
.venv/bin/alembic upgrade head
.venv/bin/python scripts/check_db.py
.venv/bin/python scripts/seed.py --dry-run
.venv/bin/python scripts/seed.py --apply
.venv/bin/python scripts/seed.py --verify-db
```

- Alembic owns the schema. Add new revisions; never rewrite applied migrations.
- Keep credentials when reusing a PostgreSQL volume. Editing `.env` does not
  change the password stored in PostgreSQL. Set `DB_PORT=5433` before startup if
  the default port is occupied.
- Do not use `docker compose down -v` or downgrade the schema during normal setup.
- Seeding inserts missing rows transactionally, skips identical rows, and aborts
  on conflicts without overwriting or deleting existing rows. If the connection
  fails near commit, run `--verify-db` before retrying.
- Fixtures and their manifest are versioned together. Do not change hashes to
  hide validation failures. Optional `.venv/bin/python scripts/seed.py --check-invalid-db`
  checks invalid rows inside rolled-back transactions; it is not required for startup.

## Run the agents

The launchers use the existing environment. They do not install dependencies,
apply migrations, seed databases or download models. Stop an old process before
reusing its port; restart the server to apply changed settings.

### Front desk

Uses OpenAI STT, LLM and TTS. Its synthetic database is separate from the clinical
one: `frontdesk_synthetic_demo` on `127.0.0.1:55432`, fixed case `1042`, clinic
`DEMO-CLINIC`, timezone `Asia/Kolkata`.

```bash
# Explicit preparation when missing or out of date, with the agent stopped:
.venv/bin/python scripts/setup_frontdesk_demo.py --prepare
./scripts/voice_frontdesk_demo.sh --check
LIVE_API_ENABLED=true ./scripts/voice_frontdesk_demo.sh
```

Open **http://127.0.0.1:7862/client/** and Connect. Preparation starts only the
isolated demo container, migrates, seeds and publishes slots; it does not modify
the clinical database. `--check` is read-only and makes no model calls.

If the canonical fixture appointment should appear in front-desk listings, run
`.venv/bin/python scripts/setup_frontdesk_demo.py --link-existing` after preparation.
This only links provenance; it does not change the appointment or publish slots.

The agent uses native Pipecat tools. Python enforces ownership, exact slot details,
duplicate protection and uncertain-write recovery. Clear booking/change requests
use direct-request authorization without requiring an extra confirmation turn.

### Clinician

Uses the clinical database above. The OpenAI launcher defaults to synthetic user
`SYN-USER-CLIN`; no patient is preselected.

```bash
./scripts/voice_clinician.sh --check
LIVE_API_ENABLED=true ./scripts/voice_clinician.sh
```

Open **http://127.0.0.1:7863/client/** and start with “Open case 1042.” Clinical
reads require successful case selection; ambiguous names require clarification.
This profile cannot book, reschedule or cancel appointments. Its readiness check
verifies database access, not OpenAI or Exa keys.

`EXA_API_KEY` is optional: without it, database reads still work and guidance
search reports a configuration error. Exa results are external guidance, not
clinic-approved policy. Native Pipecat tasks handle background reads and
cancellation; Smart Turn determines turn completion without an LLM completion filter.

Front-desk and clinician launchers default to a 120-second idle timeout, with
session limits of 600 and 3600 seconds respectively. Override
`VOICE_IDLE_TIMEOUT_SECONDS` / `VOICE_MAX_SESSION_SECONDS` explicitly; `none`
disables a limit. `CLINICIAN_DEMO_WEB_SEARCH_DELAY_SECONDS=15` optionally delays
only the first valid guidance search to exercise background delivery, not to
measure provider latency.

### Nemotron + Breeze with a selectable LLM

Prepare the servers and tunnels using the **[GPU setup guide](docs/gpu-models.md)**.
The following clinician example uses OpenAI as the LLM while keeping speech on the
GPU servers:

```bash
AGENT_MODE=clinician \
GPU_LLM_PROVIDER=openai \
STT_LANGUAGE=en TTS_LANGUAGE=en \
VOICE_PORT=7864 \
VOICE_IDLE_TIMEOUT_SECONDS=120 VOICE_MAX_SESSION_SECONDS=3600 \
LIVE_API_ENABLED=true ./scripts/voice_gpu.sh
```

Open **http://127.0.0.1:7864/client/**. Set `GPU_LLM_PROVIDER=qwen` or
`hybrid_diffusion` only after the corresponding LLM server is ready; an unset
selector still defaults to HybridDiffusion. The current Qwen pin is
`Qwen/Qwen3.8-27B-FP8`.

The launcher defaults to Nemotron on `8080`, Breeze on `7861`, and the self-hosted
LLM on `30000`. Match `STT_BASE_URL`, `TTS_BASE_URL` and, for a self-hosted LLM,
`LLM_BASE_URL` to your servers/tunnels. On the current Vast setup, Nemotron uses **18080** because
Jupyter occupies 8080; add
`STT_BASE_URL=ws://127.0.0.1:18080/v1/audio/transcriptions/realtime` to the command.
An entirely self-hosted selection needs no OpenAI key; the OpenAI LLM selection does.

### Conversation-only mode

Uses the configured providers, defaults to OpenAI, and requires no database:

```bash
AGENT_MODE=conversation LIVE_API_ENABLED=true \
  .venv/bin/python -m healthcare_voice_agent --serve
```

Open **http://127.0.0.1:7860/client/** with the default `VOICE_PORT`, Connect and
speak first. Unlike the two agent launchers, default conversation settings have
no idle or session-duration cutoff; disconnect when finished.

## Checks and logs

```bash
.venv/bin/python -m pytest
```

Offline/mock tests are separate from live provider and database checks.
Some legacy completion-filter and deferred-tool tests still need updating;
historical pass counts are not a current full-suite result. Optional
`.venv/bin/python scripts/test_frontdesk_demo_postgres.py --run` performs live,
isolated PostgreSQL checks and creates retained synthetic QA records.

Before trusting a provider change, verify tool choice/arguments with frozen
synthetic inputs, then check browser speech, correction, cancellation, barge-in
and reconnect. Offline tests do not establish model access, audible playback,
latency or safe handling of real patient data.

Each run writes `runs/<run-id>/config.json`; connections have separate
`sessions/<id>/events.jsonl` files. Traces include transcripts and generated/TTS
text, but no raw audio. Credential redaction is **not** general PHI redaction.
Keep logs private, use synthetic inputs, and distinguish server timing from
what was actually heard in the browser. Exact interrupted-audio resumption is
not guaranteed.

## Local speech compatibility

Whisper/Kokoro provider code and the optional `local-voice` extra remain; their
old launcher and benchmarks have been retired. They are not used by the front-desk
or clinician profiles. Apple Silicon local experiments require an explicit
`uv sync --locked --extra voice --extra local-voice` and correctly configured
local providers; `--prepare-local-models` explicitly prepares their pinned assets.

For native loading failures, use a normal Terminal and reinstall the named failing
package with the locked extras, for example:

```bash
uv sync --locked --extra voice --extra local-voice --no-cache --link-mode copy \
  --reinstall-package PACKAGE_NAME
```

Replace `PACKAGE_NAME` with the failing dependency. Do not disable Gatekeeper,
re-sign third-party libraries or clear quarantine flags. Reinstalling packages
does not fix missing Metal GPU access; use a normal local graphical Terminal.

## Code and contract references

- `src/healthcare_voice_agent/demo/` and its sibling `booking/`: front-desk flow and backend.
- `src/healthcare_voice_agent/clinician/` and its sibling `tools/`: clinical reads and contracts.
- `src/healthcare_voice_agent/voice/`: providers, pipeline, browser sessions and tracing.
- `src/healthcare_voice_agent/agent/prompts.md`: runtime prompt for conversation-only mode.
- `scripts/`: explicit setup, database checks and launchers; `migrations/`: schema revisions.
- `fixtures/`: synthetic seed data and validation scenarios; `evals/`: evaluation inputs.
- [Contract schemas and examples](docs/contracts/): retained JSON references and test inputs.
- [GPU setup](docs/gpu-models.md): separate model environments, downloads and serving.

Keep `.env`, private keys, `.venv`, `.demo`, model weights, logs and generated
results out of Git. Clinical identity/access handling here is a synthetic demo,
not a production authorization system.
