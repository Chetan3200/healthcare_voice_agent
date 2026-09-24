#!/bin/sh
# Paired local speech experiment. Never installs dependencies, edits .env or enables paid APIs.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
# Standard HTTP downloads avoid requiring the optional native HF Xet loader.
export HF_HUB_DISABLE_XET=1
export STT_PROVIDER=whisper
export STT_MODEL=mlx-community/whisper-large-v3-turbo
export STT_BASE_URL=
export STT_LANGUAGE="${STT_LANGUAGE:-auto}"
export STT_FINALIZE_TRANSCRIPTS=true
export TTS_PROVIDER=kokoro
export TTS_MODEL=kokoro-v1.0
export TTS_BASE_URL=
export TTS_INSTRUCTIONS=off
export TTS_LANGUAGE="${TTS_LANGUAGE:-auto}"
export TTS_VOICE="${TTS_VOICE:-af_heart}"
export TTS_HINDI_VOICE="${TTS_HINDI_VOICE:-hf_alpha}"
if [ "$#" -eq 0 ]; then set -- --serve; fi
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x /opt/homebrew/bin/uv ]; then UV_BIN=/opt/homebrew/bin/uv; fi
if [ -z "$UV_BIN" ]; then echo "Install uv first." >&2; exit 1; fi
# Dependency installation is an explicit user action, never a launcher side effect.
if [ ! -x .venv/bin/python ]; then
  echo "Project environment is missing. Please run uv sync --locked --extra voice --extra local-voice yourself first." >&2
  exit 1
fi
exec "$UV_BIN" run --no-sync --no-python-downloads --locked --extra voice --extra local-voice python -m healthcare_voice_agent "$@"
