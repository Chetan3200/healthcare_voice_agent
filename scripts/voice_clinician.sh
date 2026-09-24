#!/bin/sh
# Synthetic read-only clinician profile. No installs, migrations, seeding or model downloads.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export AGENT_MODE=clinician
export CLINICIAN_USER_ID="${CLINICIAN_USER_ID:-SYN-USER-CLIN}"
export STT_PROVIDER=openai
export STT_MODEL=gpt-live-transcribe
export STT_BASE_URL=wss://api.openai.com/v1/realtime
export STT_LANGUAGE=en
export STT_FINALIZE_TRANSCRIPTS=true
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4.1-mini-2025-04-14
export LLM_BASE_URL=https://api.openai.com/v1
export TTS_PROVIDER=openai
export TTS_MODEL=gpt-4o-mini-tts
export TTS_BASE_URL=https://api.openai.com/v1
export TTS_VOICE=marin
export TTS_LANGUAGE=en
export TTS_AUDIO_CHUNK_MS="${TTS_AUDIO_CHUNK_MS:-500}"
export TTS_INSTRUCTIONS='Speak clearly and naturally in English with a warm, professional tone. Read only the supplied text faithfully. Do not translate, add commentary or read instructions aloud.'
export VOICE_PORT="${VOICE_PORT:-7863}"
export VOICE_IDLE_TIMEOUT_SECONDS="${VOICE_IDLE_TIMEOUT_SECONDS:-120}"
export VOICE_MAX_SESSION_SECONDS="${VOICE_MAX_SESSION_SECONDS:-3600}"
if [ ! -x .venv/bin/python ]; then
  echo 'Project environment is missing. Install the locked voice extra explicitly first.' >&2
  exit 1
fi
if [ "${1:-}" = --check ]; then
  exec .venv/bin/python - <<'PY'
from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.clinician.runtime import clinical_records
config = load_config()
try:
    case_count = clinical_records(config.clinician).verify_access()
except Exception:
    raise SystemExit('Clinical database/case check failed. Check the local synthetic database and configured clinician/case. No changes were made.') from None
print('Clinical database: read-only access verified')
print('Accessible synthetic cases:', case_count)
print('No case is preselected. Say: Open case 1042.')
print('Exa key:', 'configured (not live-tested)' if config.clinician.exa_api_key else 'MISSING: set EXA_API_KEY privately in .env')
print('No provider requests or database writes performed.')
PY
fi
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x /opt/homebrew/bin/uv ]; then UV_BIN=/opt/homebrew/bin/uv; fi
if [ -z "$UV_BIN" ]; then echo 'uv is required; no dependencies were installed.' >&2; exit 1; fi
if [ "$#" -eq 0 ]; then set -- --serve; fi
exec "$UV_BIN" run --no-sync --no-python-downloads --locked --extra voice python -m healthcare_voice_agent "$@"
