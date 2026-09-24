"""Local speech configuration/wiring tests with no native MLX or model downloads.

Factory tests construct actual Pipecat adapters, but every model boundary is an
explicit fake. No microphone, provider API, network or latency measurement.
"""

import asyncio
import hashlib
import io
import json
import socket
import sys
import threading
from types import SimpleNamespace

import pytest

from healthcare_voice_agent.config import (
    DEFAULT_TTS_INSTRUCTIONS, ConfigurationError, load_config, write_run_config,
)
from healthcare_voice_agent.voice import local_assets, providers


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Local provider wiring tests must remain offline")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def local_config(**overrides):
    return load_config(None, environ={
        "STT_PROVIDER": "whisper", "TTS_PROVIDER": "kokoro",
        "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "offline-placeholder",
        **overrides,
    })


def unexpected(*args, **kwargs):
    raise AssertionError("This provider must not be imported, loaded or invoked")


def test_openai_defaults_and_local_profile_do_not_change_llm():
    original = load_config(None, environ={})
    assert original.stt.provider == original.llm.provider == original.tts.provider == "openai"
    assert original.stt.model == "gpt-live-transcribe"
    assert original.tts.model == "gpt-4o-mini-tts"
    assert original.tts.voice == "coral"
    assert original.tts.instructions == DEFAULT_TTS_INSTRUCTIONS
    local = local_config()
    assert local.llm == original.llm
    assert local.voice == original.voice
    assert local.stt.model == local_assets.WHISPER_MODEL
    assert local.stt.language == "en"
    assert local.stt.base_url is None
    assert local.stt.finalize_transcripts is True
    assert local.tts.model == "kokoro-v1.0"
    assert local.tts.voice == "af_heart"
    assert local.tts.hindi_voice == "hf_alpha"
    assert local.tts.language == "en"
    assert local.tts.instructions is None
    assert local.tts.base_url is None


@pytest.mark.parametrize("language,expected", [("auto", None), ("hi", "hi"), ("en", "en")])
def test_local_whisper_alias_language_and_empty_endpoint(language, expected):
    config = local_config(STT_MODEL="large-v3-turbo", STT_LANGUAGE=language,
                          STT_BASE_URL="", TTS_BASE_URL="", TTS_INSTRUCTIONS="off")
    assert config.stt.model == local_assets.WHISPER_MODEL
    assert config.stt.language == expected
    assert config.stt.base_url is config.tts.base_url is config.tts.instructions is None


@pytest.mark.parametrize("overrides,match", [
    ({"STT_MODEL": "gpt-live-transcribe"}, "STT_MODEL"),
    ({"STT_MODEL": "whisper-tiny"}, "STT_MODEL"),
    ({"TTS_MODEL": "gpt-4o-mini-tts"}, "TTS_MODEL"),
    ({"TTS_VOICE": "coral"}, "TTS_VOICE"),
    ({"TTS_VOICE": "hf_alpha"}, "TTS_VOICE"),
    ({"TTS_HINDI_VOICE": "af_heart"}, "TTS_HINDI_VOICE"),
    ({"TTS_INSTRUCTIONS": "Speak softly"}, "TTS_INSTRUCTIONS=off"),
    ({"STT_BASE_URL": "wss://api.openai.com/v1/realtime"}, "STT_BASE_URL"),
    ({"TTS_BASE_URL": "https://api.openai.com/v1"}, "TTS_BASE_URL"),
    ({"STT_FINALIZE_TRANSCRIPTS": "false"}, "STT_FINALIZE_TRANSCRIPTS=true"),
    ({"STT_LANGUAGE": "fr"}, "STT_LANGUAGE"),
    ({"TTS_LANGUAGE": "fr"}, "tts.language"),
    ({"TTS_SPEED": "0.1"}, "tts.speed"),
    ({"TTS_SPEED": "nan"}, "tts.speed"),
    ({"STT_MAX_PENDING_SEGMENTS": "0"}, "stt.max_pending_segments"),
    ({"STT_MAX_SEGMENT_SECONDS": "0"}, "stt.max_segment_seconds"),
    ({"STT_INFERENCE_TIMEOUT_SECONDS": "0"}, "stt.inference_timeout_seconds"),
    ({"TTS_INFERENCE_TIMEOUT_SECONDS": "inf"}, "tts.inference_timeout_seconds"),
])
def test_invalid_local_settings_fail_instead_of_falling_back(overrides, match):
    with pytest.raises(ConfigurationError, match=match):
        local_config(**overrides)


@pytest.mark.parametrize("key,value", [
    ("STT_PROVIDER", "kokoro"), ("TTS_PROVIDER", "whisper"),
    ("LLM_PROVIDER", "whisper"), ("LLM_PROVIDER", "kokoro"),
    ("STT_PROVIDER", "other"), ("TTS_PROVIDER", "other"),
])
def test_provider_is_validated_per_stage(key, value):
    with pytest.raises(ConfigurationError, match="no fallback"):
        load_config(None, environ={key: value})


def test_paired_environment_overrides_existing_dotenv_without_editing_it(tmp_path):
    env = tmp_path / ".env"
    source = ("STT_PROVIDER=openai\nSTT_MODEL=gpt-live-transcribe\n"
              "STT_BASE_URL=wss://api.openai.com/v1/realtime\n"
              "TTS_PROVIDER=openai\nTTS_MODEL=gpt-4o-mini-tts\nTTS_VOICE=coral\n"
              "TTS_BASE_URL=https://api.openai.com/v1\nTTS_INSTRUCTIONS=Speak calmly\n"
              "LLM_MODEL=chosen-openai-model\nOPENAI_API_KEY=private-marker\n")
    env.write_text(source)
    config = load_config(env, environ={
        "STT_PROVIDER": "whisper", "STT_MODEL": "large-v3-turbo", "STT_BASE_URL": "",
        "TTS_PROVIDER": "kokoro", "TTS_MODEL": "kokoro-v1.0", "TTS_BASE_URL": "",
        "TTS_VOICE": "af_heart", "TTS_INSTRUCTIONS": "off",
    })
    assert config.llm.provider == "openai"
    assert config.llm.model == "chosen-openai-model"
    assert config.tts.instructions is None
    assert config.stt.base_url is config.tts.base_url is None
    assert env.read_text() == source
    assert "private-marker" not in config.model_dump_json()


def test_run_record_includes_pins_runtime_policy_and_no_secrets(tmp_path):
    config = local_config(OPENAI_API_KEY="private-marker")
    record = json.loads(write_run_config(config, tmp_path).read_text())
    assert record["local_models"] == local_assets.local_asset_record("whisper", "kokoro")
    assert record["local_models"]["stt"]["revision"] == local_assets.WHISPER_REVISION
    assert len(record["local_models"]["stt"]["revision"]) == 40
    assert record["local_models"]["tts"]["files"] == local_assets.KOKORO_FILES
    for name, spec in record["local_models"]["tts"]["files"].items():
        assert name in {"kokoro-v1.0.onnx", "voices-v1.0.bin"}
        assert spec["size"] > 0
        # Parent may finish pinning fetched release digests while these tests run.
        assert spec["sha256"] is None or len(spec["sha256"]) == 64
    assert "FIFO local segments" in record["voice_policy"]["stt_finalization"]
    assert record["voice_policy"]["raw_audio_recording"] is False
    assert record["voice_policy"]["clinic_tools_enabled"] is False
    assert "mlx-whisper" in record["package_versions"]
    assert "kokoro-onnx" in record["package_versions"]
    assert "private-marker" not in json.dumps(record)
    assert local_assets.local_asset_record("openai", "openai") == {}


@pytest.mark.parametrize("factory", [providers.build_stt, providers.build_tts])
@pytest.mark.parametrize("overrides,match", [
    ({"LIVE_API_ENABLED": "false"}, "disabled"),
    ({"OPENAI_API_KEY": ""}, "OPENAI_API_KEY"),
])
def test_local_factory_gate_precedes_every_runtime_import(factory, overrides, match, monkeypatch):
    monkeypatch.setattr(providers, "_load_openai_components", unexpected)
    monkeypatch.setattr(providers, "_get_whisper_runtime", unexpected)
    monkeypatch.setattr(providers, "_get_kokoro_runtime", unexpected)
    with pytest.raises(ConfigurationError, match=match):
        factory(local_config(**overrides))


def test_factories_construct_real_local_adapters_without_cloud_speech(monkeypatch):
    pytest.importorskip("pipecat")
    from healthcare_voice_agent.voice.local_stt import LocalWhisperSTTService
    from healthcare_voice_agent.voice.local_tts import LocalKokoroTTSService
    monkeypatch.setattr(providers, "_load_openai_components", unexpected)
    monkeypatch.setattr(providers, "_get_whisper_runtime", lambda: (object(), "offline-path"))
    monkeypatch.setattr(providers, "_get_kokoro_runtime", lambda **kwargs: SimpleNamespace(voices={"af_heart", "hf_alpha"}))

    async def exercise():
        config = local_config(STT_LANGUAGE="hi", STT_MAX_SEGMENT_SECONDS="42",
                              STT_MAX_PENDING_SEGMENTS="3", STT_INFERENCE_TIMEOUT_SECONDS="7",
                              TTS_AUDIO_CHUNK_MS="20", TTS_SPEED="1.2")
        stt = providers.build_stt(config)
        trace = SimpleNamespace(emit=lambda *args, **kwargs: None)
        tts = providers.build_tts(config, trace=trace)
        try:
            assert isinstance(stt, LocalWhisperSTTService)
            assert isinstance(tts, LocalKokoroTTSService)
            assert stt._settings.model == local_assets.WHISPER_MODEL
            assert stt._settings.language == "hi"
            assert stt._max_segment_bytes == 42 * 16000 * 2
            assert stt._max_pending_segments == 3
            assert stt._inference_timeout_seconds == 7
            assert tts._settings.model == "kokoro-v1.0"
            assert tts._settings.voice == "af_heart"
            assert tts._trace is trace
            assert tts._stop_frame_timeout_s == 3.0
        finally:
            await stt.cleanup()
            await tts.cleanup()
    asyncio.run(exercise())


def test_unknown_stock_voice_is_rejected_before_synthesis(monkeypatch):
    pytest.importorskip("pipecat")
    monkeypatch.setattr(providers, "_get_kokoro_runtime", lambda **kwargs: SimpleNamespace(voices={"af_heart", "hf_alpha"}))
    with pytest.raises(providers.ProviderDependencyError, match="voice"):
        providers.build_tts(local_config(TTS_VOICE="af_not_in_release"))


def test_openai_llm_is_still_constructed_in_paired_profile(monkeypatch):
    from tests.unit.test_providers import FakeLimits, FakeClient, FakeLLM
    components = providers._OpenAIComponents(
        stt=unexpected, llm=FakeLLM, tts=unexpected,
        async_client=FakeClient, http_client=FakeClient,
        timeout=lambda seconds, connect: SimpleNamespace(default=seconds, connect=connect),
        connection_limits=FakeLimits(),
    )
    monkeypatch.setattr(providers, "_load_openai_components", lambda: components)
    service = providers.build_llm(local_config(), system_instruction="Test instruction")
    try:
        assert service.kwargs["settings"].model == "gpt-4.1-mini-2025-04-14"
        assert service.kwargs["settings"].system_instruction == "Test instruction"
        assert service._client.max_retries == 0
    finally:
        asyncio.run(service.cleanup())


def test_whisper_callback_preserves_audio_and_transcribes_without_translation(monkeypatch):
    audio = object()
    calls = []
    def transcribe(value, **kwargs):
        calls.append((value, kwargs))
        return {"text": "आज appointment है", "language": "hi"}
    monkeypatch.setattr(providers, "_get_whisper_runtime", lambda: (SimpleNamespace(transcribe=transcribe), "pinned-local-path"))
    result = providers._whisper_transcribe(audio, language="hi")
    assert result["text"] == "आज appointment है"
    assert calls == [(audio, {
        "path_or_hf_repo": "pinned-local-path", "language": "hi", "task": "transcribe",
        "temperature": 0.0, "condition_on_previous_text": False, "verbose": None,
    })]


def test_kokoro_callback_preserves_exact_text_and_settings(monkeypatch):
    calls = []
    def create(text, **kwargs):
        calls.append((text, kwargs))
        return "fake-waveform", 24000
    monkeypatch.setattr(providers, "_get_kokoro_runtime", lambda **kwargs: SimpleNamespace(create=create))
    text = "  नमस्ते,  appointment 7:30 पर है. "
    assert providers._kokoro_synthesize(text, voice="hf_alpha", lang="hi", speed=1.1) == ("fake-waveform", 24000)
    assert calls == [(text, {"voice": "hf_alpha", "lang": "hi", "speed": 1.1})]


@pytest.mark.parametrize("kind", ["whisper", "kokoro"])
def test_cancelled_waiter_does_not_release_shared_native_inference_lock(kind, monkeypatch):
    first_entered, release_first, second_lock_attempt = (threading.Event() for _ in range(3))
    entered = []
    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.attempts = 0
            self.counter_lock = threading.Lock()
        def __enter__(self):
            with self.counter_lock:
                self.attempts += 1
                if self.attempts == 2:
                    second_lock_attempt.set()
            self.lock.acquire()
        def __exit__(self, *args):
            self.lock.release()
    def inference(value, **kwargs):
        entered.append(value)
        if value == "first":
            first_entered.set()
            if not release_first.wait(3):
                raise AssertionError("Test failed to release first native worker")
        return {"text": value} if kind == "whisper" else (value, 24000)
    monkeypatch.setattr(providers, f"_{kind}_lock", ObservedLock())
    if kind == "whisper":
        monkeypatch.setattr(providers, "_get_whisper_runtime", lambda: (SimpleNamespace(transcribe=inference), "offline"))
        callback = providers._whisper_transcribe
    else:
        monkeypatch.setattr(providers, "_get_kokoro_runtime", lambda **kwargs: SimpleNamespace(create=inference))
        callback = lambda value: providers._kokoro_synthesize(value, voice="af_heart", lang="en-us", speed=1)

    async def exercise():
        first = asyncio.create_task(asyncio.to_thread(callback, "first"))
        second = None
        try:
            assert await asyncio.to_thread(first_entered.wait, 2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(asyncio.to_thread(callback, "second"))
            assert await asyncio.to_thread(second_lock_attempt.wait, 2)
            assert entered == ["first"]
            release_first.set()
            await asyncio.wait_for(second, 2)
            assert entered == ["first", "second"]
        finally:
            release_first.set()
            if second is not None:
                await asyncio.gather(second, return_exceptions=True)
    asyncio.run(exercise())


def test_whisper_runtime_is_pinned_cache_only_and_loaded_once(tmp_path, monkeypatch):
    for name in ("config.json", "weights.safetensors"):
        (tmp_path / name).write_bytes(b"offline-fixture")
    downloads, loaded = [], []
    module = object()
    def download(**kwargs):
        downloads.append(kwargs)
        assert kwargs["local_files_only"] is True
        assert kwargs["token"] is False
        assert kwargs["revision"] == local_assets.WHISPER_REVISION
        return str(tmp_path)
    holder = SimpleNamespace(get_model=lambda local, dtype: loaded.append((local, dtype)))
    monkeypatch.setattr(providers, "_whisper_runtime", None)
    monkeypatch.setattr(providers, "_load_whisper_components", lambda: (SimpleNamespace(float16="fp16"), module, download, holder))
    monkeypatch.setattr(providers, "model_cache", lambda: tmp_path / "cache")
    with providers._whisper_lock:
        assert providers._get_whisper_runtime() == (module, str(tmp_path))
        assert providers._get_whisper_runtime() == (module, str(tmp_path))
    assert len(downloads) == 1
    assert loaded == [(str(tmp_path), "fp16")]


def test_missing_whisper_cache_fails_safely_without_fallback(monkeypatch):
    def failed(**kwargs):
        assert kwargs["local_files_only"] is True
        raise OSError("private-marker")
    monkeypatch.setattr(providers, "_whisper_runtime", None)
    monkeypatch.setattr(providers, "_load_whisper_components", lambda: (None, None, failed, None))
    with pytest.raises(providers.ProviderDependencyError) as error:
        providers._get_whisper_runtime()
    assert "prepare-local-models" in str(error.value)
    assert "private-marker" not in str(error.value)


def test_kokoro_integrity_detects_same_size_corruption(tmp_path):
    file = tmp_path / "asset"
    file.write_bytes(b"good")
    spec = {"size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    assert providers._valid_kokoro_file(file, spec)
    file.write_bytes(b"evil")
    assert not providers._valid_kokoro_file(file, spec)
    file.unlink()
    assert not providers._valid_kokoro_file(file, spec)


def test_explicit_preparation_downloads_pinned_assets_atomically_without_native_models(tmp_path, monkeypatch):
    downloaded = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **kwargs: downloaded.append(kwargs)))
    monkeypatch.setattr(providers, "_load_whisper_components", unexpected)
    monkeypatch.setattr(providers, "_load_kokoro_components", unexpected)
    monkeypatch.setattr(providers, "_load_openai_components", unexpected)
    monkeypatch.setattr(providers, "model_cache", lambda: tmp_path)
    payload = b"public-model-fixture"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(providers, "KOKORO_FILES", {"fixture.bin": {"size": len(payload), "sha256": digest}})
    requests = []
    def urlopen(url, timeout):
        requests.append((url, timeout))
        return io.BytesIO(payload)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    result = providers.prepare_local_models(local_config(LIVE_API_ENABLED="false", OPENAI_API_KEY=""))
    assert result["kokoro"] == {"fixture.bin": digest}
    assert (tmp_path / "kokoro/fixture.bin").read_bytes() == payload
    assert not list(tmp_path.rglob("*.partial"))
    assert downloaded[0]["revision"] == local_assets.WHISPER_REVISION
    assert downloaded[0]["token"] is False
    assert set(downloaded[0]["allow_patterns"]) == {"config.json", "weights.safetensors"}
    providers.prepare_local_models(local_config(LIVE_API_ENABLED="false", OPENAI_API_KEY=""))
    assert len(requests) == 1  # Valid Kokoro files are reused.


def test_prepare_cli_needs_no_key_and_does_not_construct_providers(tmp_path, monkeypatch, capsys):
    from healthcare_voice_agent.__main__ import main
    env = tmp_path / ".env"
    env.write_text("STT_PROVIDER=whisper\nTTS_PROVIDER=kokoro\nLIVE_API_ENABLED=false\n")
    for key in list(__import__("os").environ):
        if key.startswith(("STT_", "TTS_", "LLM_", "VOICE_")) or key in {"OPENAI_API_KEY", "LIVE_API_ENABLED"}:
            monkeypatch.delenv(key, raising=False)
    seen = []
    monkeypatch.setattr(providers, "prepare_local_models", lambda config: seen.append(config) or {"whisper": {"revision": local_assets.WHISPER_REVISION}})
    for name in ("build_stt", "build_llm", "build_tts", "warm_local_models"):
        monkeypatch.setattr(providers, name, unexpected)
    assert main(["--prepare-local-models", "--env-file", str(env), "--run-dir", str(tmp_path / "run")]) == 0
    assert seen[0].live_api_enabled is False
    assert seen[0].openai_api_key is None
    assert "No OpenAI API calls were made" in capsys.readouterr().out


def test_non_apple_whisper_reports_explicit_no_fallback(monkeypatch):
    monkeypatch.setattr(providers.platform, "system", lambda: "Linux")
    with pytest.raises(providers.ProviderDependencyError, match="no CPU fallback"):
        providers._load_whisper_components()


def test_kokoro_runtime_pins_cpu_threads_and_reuses_loaded_engine(tmp_path, monkeypatch):
    created = []
    engine = object()
    class Options:
        pass
    def session(model, **kwargs):
        created.append((model, kwargs))
        return "fake-session"
    def from_session(instance, voices):
        assert instance == "fake-session"
        assert voices.endswith("voices-v1.0.bin")
        return engine
    monkeypatch.setattr(providers, "_kokoro_runtime", None)
    monkeypatch.setattr(providers, "model_cache", lambda: tmp_path)
    monkeypatch.setattr(providers, "_valid_kokoro_file", lambda *args: True)
    monkeypatch.setattr(providers, "_load_kokoro_components", lambda: (
        SimpleNamespace(SessionOptions=Options, InferenceSession=session),
        SimpleNamespace(from_session=from_session),
    ))
    with providers._kokoro_lock:
        assert providers._get_kokoro_runtime() is engine
        assert providers._get_kokoro_runtime() is engine
    assert len(created) == 1
    assert created[0][1]["providers"] == ["CPUExecutionProvider"]
    assert created[0][1]["sess_options"].intra_op_num_threads == 2
    assert created[0][1]["sess_options"].inter_op_num_threads == 1


def test_kokoro_load_failure_is_sanitized(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("private-marker")
    monkeypatch.setattr(providers, "_kokoro_runtime", None)
    monkeypatch.setattr(providers, "_valid_kokoro_file", lambda *args: True)
    monkeypatch.setattr(providers, "_load_kokoro_components", lambda: (
        SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=failed), None,
    ))
    with pytest.raises(providers.ProviderDependencyError) as error:
        providers._get_kokoro_runtime()
    assert "private-marker" not in str(error.value)
    assert "Kokoro" in str(error.value)
    assert providers._kokoro_runtime is None


def test_failed_download_does_not_install_partial_file(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setattr(providers, "model_cache", lambda: tmp_path)
    payload = b"wrong"
    monkeypatch.setattr(providers, "KOKORO_FILES", {"fixture.bin": {
        "size": len(payload), "sha256": hashlib.sha256(b"right").hexdigest(),
    }})
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    config = load_config(None, environ={"TTS_PROVIDER": "kokoro"})
    with pytest.raises(providers.ProviderDependencyError, match="integrity"):
        providers.prepare_local_models(config)
    assert not (tmp_path / "kokoro/fixture.bin").exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_local_check_config_remains_offline_without_live_opt_in(tmp_path, monkeypatch, capsys):
    from healthcare_voice_agent.__main__ import main
    for key in list(__import__("os").environ):
        if key.startswith(("STT_", "TTS_", "LLM_", "VOICE_")) or key in {"OPENAI_API_KEY", "LIVE_API_ENABLED"}:
            monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("STT_PROVIDER=whisper\nTTS_PROVIDER=kokoro\n")
    for name in ("prepare_local_models", "build_stt", "build_llm", "build_tts", "warm_local_models"):
        monkeypatch.setattr(providers, name, unexpected)
    assert main(["--check-config", "--env-file", str(env), "--run-dir", str(tmp_path / "run")]) == 0
    output = capsys.readouterr().out
    assert '"provider": "whisper"' in output
    assert '"provider": "kokoro"' in output
    assert "No provider services or API calls were started" in output
