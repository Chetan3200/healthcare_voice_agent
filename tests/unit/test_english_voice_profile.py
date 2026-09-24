"""Voice-profile checks; only isolated shell parameter expansion, no provider access."""

from pathlib import Path
import subprocess

import pytest

from healthcare_voice_agent.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_frontdesk_launcher_forces_english_without_changing_demo_buffering():
    script = (ROOT / "scripts/voice_frontdesk_demo.sh").read_text(encoding="utf-8")

    assert "export STT_LANGUAGE=en" in script
    assert "export TTS_LANGUAGE=en" in script
    assert 'TTS_AUDIO_CHUNK_MS:-500' in script
    assert "VOICE_IDLE_TIMEOUT_SECONDS:-120" in script
    assert "VOICE_MAX_SESSION_SECONDS:-600" in script
    assert all(term not in script.lower() for term in ("hindi", "hinglish", "code-switch"))
    assert "Do not translate or transliterate." in script


@pytest.mark.parametrize("override, expected", [
    (None, 3600), ("", 3600), ("1800", 1800), ("none", None),
])
def test_clinician_launcher_one_hour_default_and_explicit_overrides(override, expected):
    script = (ROOT / "scripts/voice_clinician.sh").read_text(encoding="utf-8")
    limit_line, = [line for line in script.splitlines()
                   if line.startswith("export VOICE_MAX_SESSION_SECONDS=")]
    # Execute only the actual limit assignment, never the launcher or providers.
    env = {} if override is None else {"VOICE_MAX_SESSION_SECONDS": override}
    result = subprocess.run(
        ["/bin/sh", "-c", limit_line + '\nprintf "%s" "$VOICE_MAX_SESSION_SECONDS"'],
        env=env, capture_output=True, text=True, check=True, timeout=5,
    )
    config = load_config(None, environ={"VOICE_MAX_SESSION_SECONDS": result.stdout})
    assert config.voice.max_session_seconds == expected
    assert "VOICE_IDLE_TIMEOUT_SECONDS:-120" in script
