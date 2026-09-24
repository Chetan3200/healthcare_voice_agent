#!/bin/sh
# Fixed synthetic comparison only: no installs, downloads, .env reads or APIs.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/benchmark_speech_optimizations.sh NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
if [ ! -x .venv/bin/python ]; then
  echo "The existing project environment is required; this script never installs it." >&2
  exit 2
fi
.venv/bin/python scripts/benchmark_local_speech.py --output "$1"
.venv/bin/python scripts/benchmark_whisper_reuse.py --audio-dir "$1/audio" --output "$1/whisper"
printf '\nFixed comparison complete. TTS: %s/summary.json\nWhisper: %s/whisper/summary.json\n' "$1" "$1"
