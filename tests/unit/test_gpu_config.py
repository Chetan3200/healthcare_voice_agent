"""Opt-in self-hosted configuration; no network or native model imports."""

import json

import pytest

from healthcare_voice_agent.config import ConfigurationError, load_config, write_run_config


GPU = {"STT_PROVIDER": "nemotron", "LLM_PROVIDER": "hybrid_diffusion", "TTS_PROVIDER": "breeze"}


def test_self_hosted_defaults_leave_baseline_unchanged():
    config = load_config(None, environ=GPU)
    assert config.stt.model == "nemotron-3.5-asr-streaming-0.6b"
    assert config.llm.model == "yuchen-zhu-zyc/HybridDiffusion-2B"
    assert config.tts.model == "BreezeBlue/Breeze-TTS-2"
    assert config.tts.voice == "S0"
    assert config.tts.instructions
    baseline = load_config(None, environ={})
    assert baseline.stt.provider == baseline.llm.provider == baseline.tts.provider == "openai"


def test_self_hosted_still_needs_live_opt_in_but_not_openai_key():
    with pytest.raises(ConfigurationError, match="disabled"):
        load_config(None, environ=GPU).require_live_ready()
    load_config(None, environ={**GPU, "LIVE_API_ENABLED": "true"}).require_live_ready()


@pytest.mark.parametrize("stage", ["STT", "LLM", "TTS"])
def test_mixed_stack_still_requires_openai_key(stage):
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        load_config(None, environ={**GPU, f"{stage}_PROVIDER": "openai", "LIVE_API_ENABLED": "true"}).require_live_ready()


@pytest.mark.parametrize("setting,value", [
    ("STT_MODEL", "gpt-live-transcribe"),
    ("LLM_MODEL", "gpt-4.1-mini-2025-04-14"),
    ("TTS_MODEL", "gpt-4o-mini-tts"),
    ("STT_BASE_URL", "wss://api.openai.com/v1/realtime"),
    ("LLM_BASE_URL", "https://api.openai.com/v1"),
    ("TTS_BASE_URL", "https://api.openai.com/v1"),
    ("STT_FINALIZE_TRANSCRIPTS", "false"),
    ("TTS_INSTRUCTIONS", "off"),
    ("TTS_VOICE", "coral"), ("TTS_SPEED", "1.2"), ("TTS_LANGUAGE", "hi"),
    ("TTS_BREEZE_CFG_SCALE", "nan"), ("TTS_BREEZE_CFG_SCALE", "11"),
    ("LLM_BASE_URL", "http://gpu.example/v1"),
    ("STT_BASE_URL", "ws://gpu.example/v1/realtime"),
])
def test_incompatible_or_unsafe_settings_fail_closed(setting, value):
    with pytest.raises(ConfigurationError):
        load_config(None, environ={**GPU, setting: value})


def test_stage_keys_never_enter_repr_or_run_records(tmp_path):
    keys = {f"{stage}_API_KEY": f"private-{stage}-marker" for stage in ("STT", "LLM", "TTS")}
    config = load_config(None, environ={**GPU, **keys})
    record = write_run_config(config, tmp_path).read_text()
    assert set(config.redaction_secrets()) == set(keys.values())
    for value in keys.values():
        assert value not in repr(config)
        assert value not in record
        assert value not in config.model_dump_json()
    assert json.loads(record)["configuration"]["openai_key_configured"] is False


def test_breeze_chinese_does_not_expand_kokoro_languages():
    assert load_config(None, environ={**GPU, "TTS_LANGUAGE": "zh"}).tts.language == "zh"
    with pytest.raises(ConfigurationError):
        load_config(None, environ={"TTS_PROVIDER": "kokoro", "TTS_LANGUAGE": "zh"})
