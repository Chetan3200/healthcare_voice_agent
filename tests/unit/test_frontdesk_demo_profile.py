"""Profile isolation, startup and resource-lifetime checks, without network."""
import asyncio
from pathlib import Path

import pytest

from healthcare_voice_agent.config import ConfigurationError, load_config, write_run_config
from healthcare_voice_agent.demo import runtime


def test_baseline_remains_the_default():
    assert load_config(None, environ={}).agent_mode == "conversation"


@pytest.mark.parametrize("environment", [
    {"AGENT_MODE": "unknown"},
    {"AGENT_MODE": "frontdesk_demo", "STT_PROVIDER": "whisper"},
    {"AGENT_MODE": "frontdesk_demo", "TTS_PROVIDER": "kokoro"},
    {"AGENT_MODE": "frontdesk_demo", "LLM_PROVIDER": "hybrid_diffusion"},
    {"AGENT_MODE": "frontdesk_demo", "STT_FINALIZE_TRANSCRIPTS": "false"},
])
def test_profile_rejects_unsupported_confirmation_pipelines(environment):
    with pytest.raises(ConfigurationError):
        load_config(None, environ=environment)


def test_demo_record_is_explicit_but_secret_free(tmp_path):
    config = load_config(None, environ={"AGENT_MODE": "frontdesk_demo", "OPENAI_API_KEY": "private-key-marker"})
    assert not config.live_api_enabled
    contents = write_run_config(config, tmp_path / "run").read_text()
    assert '"frontdesk_tools_enabled": true' in contents
    assert '"clinic_tools_enabled": false' in contents
    assert "private-key-marker" not in contents


@pytest.mark.parametrize("twice", [False, True])
def test_cancelled_runtime_creation_is_drained_and_closed(monkeypatch, twice):
    import threading
    started, release = threading.Event(), threading.Event()
    closed = []
    class Resource:
        async def close(self):
            closed.append(True)
    def slow_open(trace):
        started.set()
        release.wait(2)
        return Resource()
    monkeypatch.setattr(runtime, "_open_demo", slow_open)
    async def exercise():
        task = asyncio.create_task(runtime.open_demo_runtime(None))
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        if twice:
            await asyncio.sleep(0)
            task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(exercise())
    assert closed == [True]


def test_launcher_is_explicit_without_install_migration_or_paid_opt_in():
    script = (Path(__file__).resolve().parents[2] / "scripts/voice_frontdesk_demo.sh").read_text()
    assert "AGENT_MODE=frontdesk_demo" in script
    for stage in ("STT", "LLM", "TTS"):
        assert f"{stage}_PROVIDER=openai" in script
    assert "--no-sync --no-python-downloads --locked" in script
    assert "setup_frontdesk_demo.py --check" in script
    assert "LIVE_API_ENABLED=true" not in script
    assert "--prepare" not in script
    assert "uv sync" not in script


def test_frontdesk_restores_audio_reservoir_without_changing_global_experiment():
    script = (Path(__file__).resolve().parents[2] / "scripts/voice_frontdesk_demo.sh").read_text()
    assert 'export TTS_AUDIO_CHUNK_MS="${TTS_AUDIO_CHUNK_MS:-500}"' in script
    assert "VOICE_SMART_TURN_STOP_SECONDS" not in script
    assert load_config(None, environ={}).tts.audio_chunk_ms == 100
    assert load_config(None, environ={"TTS_AUDIO_CHUNK_MS": "500"}).tts.audio_chunk_ms == 500
