"""Configuration and run-record tests. No real .env, network, or API access."""

import json

import pytest

from healthcare_voice_agent.config import (
    DEFAULT_TTS_INSTRUCTIONS, ConfigurationError, load_config, write_run_config,
)


def test_defaults_need_no_key_and_disable_live_services():
    config = load_config(None, environ={})
    assert config.stt.model == "gpt-live-transcribe"
    assert config.stt.language == "en"
    assert config.llm.model == "gpt-4.1-mini-2025-04-14"
    assert config.tts.voice == "coral"
    assert config.tts.language == "en"
    assert config.tts.instructions == DEFAULT_TTS_INSTRUCTIONS
    assert all(term not in config.tts.instructions.lower() for term in ("hindi", "hinglish", "code-switch"))
    assert config.tts.audio_chunk_ms == 100
    assert config.voice.smart_turn_stop_seconds == 1.5
    assert config.openai_api_key is None
    assert config.live_api_enabled is False
    assert "api_budget_usd" not in config.public_dict()
    assert config.voice.idle_timeout_seconds is None
    assert config.voice.max_session_seconds is None


def test_clinician_demo_web_search_delay_is_bounded_and_clinician_only():
    assert load_config(None, environ={}).clinician.demo_web_search_delay_seconds == 0.0
    clinician = load_config(None, environ={
        "AGENT_MODE": "clinician", "CLINICIAN_DEMO_WEB_SEARCH_DELAY_SECONDS": "2.5",
    })
    assert clinician.clinician.demo_web_search_delay_seconds == 2.5
    frontdesk = load_config(None, environ={
        "AGENT_MODE": "frontdesk_demo", "CLINICIAN_DEMO_WEB_SEARCH_DELAY_SECONDS": "2.5",
    })
    assert frontdesk.clinician.demo_web_search_delay_seconds == 0.0
    for invalid in ("-0.1", "15.1", "nan", "inf"):
        with pytest.raises(ConfigurationError):
            load_config(None, environ={
                "AGENT_MODE": "clinician", "CLINICIAN_DEMO_WEB_SEARCH_DELAY_SECONDS": invalid,
            })


def test_environment_overrides_dotenv_without_mutating_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-file\nTTS_VOICE=nova\n", encoding="utf-8")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    config = load_config(env_file, environ={"LLM_MODEL": "from-environment"})
    assert config.llm.model == "from-environment"
    assert config.tts.voice == "nova"
    import os
    assert "LLM_MODEL" not in os.environ


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto "])
def test_local_auto_language_remains_an_explicit_opt_in(value):
    config = load_config(None, environ={
        "STT_PROVIDER": "whisper", "STT_BASE_URL": "", "STT_LANGUAGE": value,
    })
    assert config.stt.language is None


@pytest.mark.parametrize("value", ["auto", "hi"])
def test_live_transcribe_requires_explicit_english(value):
    with pytest.raises(ConfigurationError, match="STT_LANGUAGE=en"):
        load_config(None, environ={"STT_LANGUAGE": value})


def test_other_stt_models_can_configure_a_single_language():
    config = load_config(None, environ={"STT_MODEL": "gpt-transcribe", "STT_LANGUAGE": "hi"})
    assert config.stt.language == "hi"


@pytest.mark.parametrize("setting", ["STT_PROVIDER", "LLM_PROVIDER", "TTS_PROVIDER"])
def test_unsupported_provider_never_falls_back(setting):
    with pytest.raises(ConfigurationError, match="no fallback"):
        load_config(None, environ={setting: "not-implemented"})


@pytest.mark.parametrize("setting", ["STT_MODEL", "LLM_MODEL", "TTS_VOICE"])
def test_empty_required_settings_are_rejected(setting):
    with pytest.raises(ConfigurationError):
        load_config(None, environ={setting: ""})


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_http_timeouts_are_rejected(value):
    with pytest.raises(ConfigurationError):
        load_config(None, environ={"LLM_TIMEOUT_SECONDS": value})


@pytest.mark.parametrize("endpoint", [
    "https://private-marker@api.openai.com/v1",
    "https://api.openai.com/v1?key=private-marker",
    "https://api.openai.com/v1#private-marker",
    "http://untrusted.example/v1",
    "wss://api.openai.com/v1",
])
def test_invalid_http_endpoints_do_not_leak_values(endpoint):
    with pytest.raises(ConfigurationError) as error:
        load_config(None, environ={"LLM_BASE_URL": endpoint})
    assert "private-marker" not in str(error.value)


def test_local_http_endpoint_can_be_configured():
    config = load_config(None, environ={"LLM_BASE_URL": "http://127.0.0.1:8000/v1/"})
    assert config.llm.base_url == "http://127.0.0.1:8000/v1"


@pytest.mark.parametrize("setting", ["STT_CONNECT_TIMEOUT_SECONDS", "STT_LANGUAGES", "LLM_MODLE"])
def test_unknown_provider_knobs_are_not_silently_ignored(setting):
    with pytest.raises(ConfigurationError, match="Unknown provider setting"):
        load_config(None, environ={setting: "1"})


@pytest.mark.parametrize("environment, message", [
    ({}, "disabled"),
    ({"LIVE_API_ENABLED": "true"}, "OPENAI_API_KEY"),
])
def test_live_prerequisites(environment, message):
    with pytest.raises(ConfigurationError, match=message):
        load_config(None, environ=environment).require_live_ready()


def test_config_and_record_exclude_api_key_and_database_settings(tmp_path):
    config = load_config(None, environ={
        "OPENAI_API_KEY": "private-key-marker", "POSTGRES_PASSWORD": "private-db-marker",
    })
    assert "private-key-marker" not in repr(config)
    assert "private-key-marker" not in config.model_dump_json()
    snapshot = write_run_config(config, tmp_path / "run")
    text = snapshot.read_text(encoding="utf-8")
    assert "private-key-marker" not in text
    assert "private-db-marker" not in text
    record = json.loads(text)
    assert record["configuration"]["openai_key_configured"] is True
    assert record["configuration"]["stt"]["language"] == "en"
    assert record["configuration"]["tts"]["language"] == "en"
    assert "pipecat-ai" in record["package_versions"]


def test_run_record_cannot_be_overwritten(tmp_path):
    config = load_config(None, environ={})
    write_run_config(config, tmp_path)
    with pytest.raises(FileExistsError):
        write_run_config(config, tmp_path)


def test_configuration_cli_is_offline_and_records_a_run(tmp_path, monkeypatch, capsys):
    from healthcare_voice_agent.__main__ import main
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LIVE_API_ENABLED=false\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    assert main(["--env-file", str(env_file), "--run-dir", str(run_dir), "--check-config"]) == 0
    assert (run_dir / "config.json").exists()
    assert "No provider services or API calls were started" in capsys.readouterr().out


def test_default_env_path_is_explicitly_current_working_directory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLM_MODEL=cwd-selected-model\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_config(environ={}).llm.model == "cwd-selected-model"


def test_live_opt_in_and_key_are_sufficient_without_a_budget():
    config = load_config(None, environ={
        "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "dummy-offline-key",
    })
    config.require_live_ready()


@pytest.mark.parametrize("value", ["", "none", "off", "disabled", "NONE"])
def test_explicitly_disabling_session_and_idle_cutoffs(value):
    config = load_config(None, environ={
        "VOICE_IDLE_TIMEOUT_SECONDS": value, "VOICE_MAX_SESSION_SECONDS": value,
    })
    assert config.voice.idle_timeout_seconds is None
    assert config.voice.max_session_seconds is None


def test_tts_style_override_is_loaded_and_recorded_without_changing_models(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('TTS_INSTRUCTIONS="Speak calmly."\n', encoding="utf-8")
    config = load_config(env_file, environ={"TTS_INSTRUCTIONS": "  Speak measured English clearly.  "})
    assert config.tts.instructions == "Speak measured English clearly."
    assert config.tts.model == "gpt-4o-mini-tts"
    assert config.tts.voice == "coral"
    record = json.loads(write_run_config(config, tmp_path / "run").read_text(encoding="utf-8"))
    assert record["configuration"]["tts"]["instructions"] == config.tts.instructions


@pytest.mark.parametrize("value", ["", " ", "none", "OFF", "disabled", " DiSaBlEd "])
def test_tts_instructions_can_be_disabled_for_the_baseline(value):
    config = load_config(None, environ={"TTS_INSTRUCTIONS": value})
    assert config.tts.instructions is None
    assert config.public_dict()["tts"]["instructions"] is None


@pytest.mark.parametrize("model", ["tts-1", "tts-1-hd"])
def test_legacy_tts_models_require_instructions_to_be_disabled(model):
    with pytest.raises(ConfigurationError, match="TTS_INSTRUCTIONS=off"):
        load_config(None, environ={"TTS_MODEL": model})
    config = load_config(None, environ={"TTS_MODEL": model, "TTS_INSTRUCTIONS": "off"})
    assert config.tts.model == model
    assert config.tts.instructions is None


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False)])
def test_transcript_finalization_switch_is_recorded(value, expected, tmp_path):
    config = load_config(None, environ={"STT_FINALIZE_TRANSCRIPTS": value})
    assert config.stt.finalize_transcripts is expected
    record = json.loads(write_run_config(config, tmp_path).read_text())
    assert record["configuration"]["stt"]["finalize_transcripts"] is expected
    assert ("ordered committed items" in record["voice_policy"]["stt_finalization"]) is expected


def test_transcript_finalization_defaults_on_and_rejects_invalid_values():
    assert load_config(None, environ={}).stt.finalize_transcripts is True
    with pytest.raises(ConfigurationError):
        load_config(None, environ={"STT_FINALIZE_TRANSCRIPTS": "not-a-boolean"})


@pytest.mark.parametrize("chunk_ms,stop_seconds", [(100, 1.5), (500, 3.0), (200, 2.0)])
def test_latency_settings_are_loaded_and_recorded(chunk_ms, stop_seconds, tmp_path):
    config = load_config(None, environ={
        "TTS_AUDIO_CHUNK_MS": str(chunk_ms),
        "VOICE_SMART_TURN_STOP_SECONDS": str(stop_seconds),
    })
    assert config.tts.audio_chunk_ms == chunk_ms
    assert config.voice.smart_turn_stop_seconds == stop_seconds
    record = json.loads(write_run_config(config, tmp_path).read_text())
    assert record["configuration"]["tts"]["audio_chunk_ms"] == chunk_ms
    assert record["configuration"]["voice"]["smart_turn_stop_seconds"] == stop_seconds
    assert config.stt.finalize_transcripts is True
    assert config.voice.idle_timeout_seconds is None
    assert config.voice.max_session_seconds is None


@pytest.mark.parametrize("value", ["0", "-1", "19", "1001", "100.5", "nan", "inf", "invalid"])
def test_invalid_audio_chunk_sizes_are_rejected(value):
    with pytest.raises(ConfigurationError):
        load_config(None, environ={"TTS_AUDIO_CHUNK_MS": value})


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_smart_turn_silence_fallbacks_are_rejected(value):
    with pytest.raises(ConfigurationError):
        load_config(None, environ={"VOICE_SMART_TURN_STOP_SECONDS": value})


def test_latency_process_overrides_take_precedence_over_file(tmp_path):
    env_file = tmp_path / "latency.env"
    env_file.write_text("TTS_AUDIO_CHUNK_MS=500\nVOICE_SMART_TURN_STOP_SECONDS=3\n")
    config = load_config(env_file, environ={
        "TTS_AUDIO_CHUNK_MS": "100", "VOICE_SMART_TURN_STOP_SECONDS": "1.5",
    })
    assert config.tts.audio_chunk_ms == 100
    assert config.voice.smart_turn_stop_seconds == 1.5


def test_http_keepalive_defaults_are_sixty_seconds():
    config = load_config(None, environ={})
    assert config.llm.keepalive_expiry_seconds == 60.0
    assert config.tts.keepalive_expiry_seconds == 60.0
    assert config.llm.connect_timeout_seconds == config.tts.connect_timeout_seconds == 5.0


@pytest.mark.parametrize("llm_expiry,tts_expiry", [(60, 60), (5, 5), (30, 45)])
def test_keepalive_can_be_overridden_independently_and_is_recorded(llm_expiry, tts_expiry, tmp_path):
    config = load_config(None, environ={
        "LLM_KEEPALIVE_EXPIRY_SECONDS": str(llm_expiry),
        "TTS_KEEPALIVE_EXPIRY_SECONDS": str(tts_expiry),
    })
    record = json.loads(write_run_config(config, tmp_path).read_text())
    assert record["configuration"]["llm"]["keepalive_expiry_seconds"] == llm_expiry
    assert record["configuration"]["tts"]["keepalive_expiry_seconds"] == tts_expiry
    assert config.tts.audio_chunk_ms == 100
    assert config.voice.smart_turn_stop_seconds == 1.5
    assert config.stt.finalize_transcripts is True


@pytest.mark.parametrize("setting", ["LLM_KEEPALIVE_EXPIRY_SECONDS", "TTS_KEEPALIVE_EXPIRY_SECONDS"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_keepalive_expiry_is_rejected(setting, value):
    with pytest.raises(ConfigurationError):
        load_config(None, environ={setting: value})


def test_keepalive_process_overrides_file_settings(tmp_path):
    env_file = tmp_path / "keepalive.env"
    env_file.write_text("LLM_KEEPALIVE_EXPIRY_SECONDS=5\nTTS_KEEPALIVE_EXPIRY_SECONDS=5\n")
    config = load_config(env_file, environ={
        "LLM_KEEPALIVE_EXPIRY_SECONDS": "60", "TTS_KEEPALIVE_EXPIRY_SECONDS": "60",
    })
    assert config.llm.keepalive_expiry_seconds == config.tts.keepalive_expiry_seconds == 60.0


@pytest.mark.parametrize("mode,retries", [
    ("clinician", 1), ("frontdesk_demo", 0), ("conversation", 0),
])
def test_run_record_reports_stage_specific_retry_policy(tmp_path, mode, retries):
    config = load_config(None, environ={"AGENT_MODE": mode})
    record = json.loads(write_run_config(config, tmp_path / "run").read_text())
    policy = record["provider_policy"]
    assert config.llm_http_max_retries == retries
    assert policy["llm_http_max_retries"] == retries
    assert policy["tts_http_max_retries"] == 0
    assert policy["stt_auto_reconnect"] is False
