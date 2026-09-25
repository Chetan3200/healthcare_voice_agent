#!/bin/sh
# Self-hosted speech; HybridDiffusion by default, Qwen or OpenAI by explicit selection.
# No installs, downloads or .env changes.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export STT_PROVIDER=nemotron
export STT_MODEL=nemotron-3.5-asr-streaming-0.6b
export STT_BASE_URL="${STT_BASE_URL:-ws://127.0.0.1:8080/v1/audio/transcriptions/realtime}"
export STT_LANGUAGE="${STT_LANGUAGE:-auto}"
export STT_FINALIZE_TRANSCRIPTS=true
case "${GPU_LLM_PROVIDER:-hybrid_diffusion}" in
  hybrid_diffusion)
    export LLM_PROVIDER=hybrid_diffusion
    export LLM_MODEL=yuchen-zhu-zyc/HybridDiffusion-2B
    export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:30000/v1}"
    ;;
  qwen)
    export LLM_PROVIDER=qwen
    export LLM_MODEL=Qwen/Qwen3.8-27B-FP8
    export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:30000/v1}"
    ;;
  openai)
    export LLM_PROVIDER=openai
    export LLM_MODEL=gpt-4.1-mini-2025-04-14
    # Never reuse an inherited self-hosted endpoint for the OpenAI client/key.
    export LLM_BASE_URL=https://api.openai.com/v1
    ;;
  *) echo 'GPU_LLM_PROVIDER must be hybrid_diffusion, qwen or openai.' >&2; exit 1 ;;
esac
export TTS_PROVIDER=breeze
export TTS_MODEL=BreezeBlue/Breeze-TTS-2
export TTS_BASE_URL="${TTS_BASE_URL:-http://127.0.0.1:7861}"
export TTS_VOICE=S0
export TTS_LANGUAGE="${TTS_LANGUAGE:-auto}"
export TTS_SPEED=1.0
export TTS_INSTRUCTIONS="${TTS_INSTRUCTIONS:-A warm, calm conversational voice.}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [ "$#" -eq 0 ]; then set -- --serve; fi
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x /opt/homebrew/bin/uv ]; then UV_BIN=/opt/homebrew/bin/uv; fi
if [ -z "$UV_BIN" ]; then echo "Install uv first." >&2; exit 1; fi
if [ ! -x .venv/bin/python ]; then
  echo "Project environment missing. Run uv sync --locked --extra voice explicitly first." >&2
  exit 1
fi
# LIVE_API_ENABLED is deliberately NOT set here. Existing opt-in still applies.
exec "$UV_BIN" run --no-sync --no-python-downloads --locked --extra voice python -m healthcare_voice_agent "$@"
