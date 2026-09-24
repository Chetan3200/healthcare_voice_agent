"""Offline selection, rollback and process-lifetime settings contracts."""

from types import SimpleNamespace

import pytest

from healthcare_voice_agent.config import ConfigurationError, load_config, write_run_config
from healthcare_voice_agent.voice import providers


def test_speech_optimization_environment_settings_and_rollback(tmp_path):
    config = load_config(None, environ={
        "STT_PROVIDER": "whisper", "TTS_PROVIDER": "kokoro",
        "STT_REUSE_ENCODER": "false", "TTS_CPU_THREADS": "4",
        "TTS_FIRST_CHUNK_CHARS": "0",
    })
    assert config.stt.reuse_encoder is False
    assert config.tts.cpu_threads == 4
    assert config.tts.first_chunk_chars == 0
    text = write_run_config(config, tmp_path).read_text()
    assert '"reuse_encoder": false' in text
    assert '"cpu_threads": 4' in text
    assert '"first_chunk_chars": 0' in text


@pytest.mark.parametrize("key,value", [
    ("TTS_CPU_THREADS", "0"), ("TTS_CPU_THREADS", "9"),
    ("TTS_CPU_THREADS", "1.5"), ("TTS_FIRST_CHUNK_CHARS", "1"),
    ("TTS_FIRST_CHUNK_CHARS", "31"), ("TTS_FIRST_CHUNK_CHARS", "257"),
    ("STT_REUSE_ENCODER", "perhaps"),
])
def test_invalid_optimization_settings_fail_closed(key, value):
    with pytest.raises(ConfigurationError):
        load_config(None, environ={key: value})


@pytest.mark.parametrize("fallback", [False, True])
def test_whisper_optimized_provider_passes_language_and_original_fallback(monkeypatch, fallback):
    from healthcare_voice_agent.voice import whisper_reuse
    audio = object()
    native_calls, helper_calls = [], []
    model, dependencies = object(), object()
    def stock(value, **kwargs):
        native_calls.append((value, kwargs))
        return {"text": "synthetic stock", "language": "hi"}
    def optimized(value, **kwargs):
        helper_calls.append((value, kwargs))
        if fallback:
            return kwargs["stock_transcribe"](value)
        return {"text": "synthetic optimized", "language": "hi"}
    monkeypatch.setattr(providers, "_get_whisper_runtime", lambda: (SimpleNamespace(transcribe=stock), "cached-model"))
    monkeypatch.setattr(providers, "_whisper_reuse_dependencies", lambda path: (model, dependencies))
    monkeypatch.setattr(whisper_reuse, "transcribe_with_encoder_reuse", optimized)
    result = providers._whisper_transcribe(audio, language=None, reuse_encoder=True)
    assert result["local_decode_path"] == ("stock_fallback" if fallback else "single_encode")
    assert helper_calls[0][0] is audio
    assert helper_calls[0][1]["language"] is None
    assert helper_calls[0][1]["model"] is model
    assert helper_calls[0][1]["dependencies"] is dependencies
    assert len(native_calls) == int(fallback)
    if fallback:
        assert native_calls == [(audio, {
            "path_or_hf_repo": "cached-model", "language": None, "task": "transcribe",
            "temperature": 0.0, "condition_on_previous_text": False, "verbose": None,
        })]


def test_kokoro_thread_setting_is_fixed_for_engine_lifetime(monkeypatch):
    created = []
    engine = object()
    monkeypatch.setattr(providers, "_kokoro_runtime", None)
    monkeypatch.setattr(providers, "_kokoro_runtime_threads", None)
    monkeypatch.setattr(providers, "_create_kokoro_runtime", lambda n: created.append(n) or engine)
    with providers._kokoro_lock:
        assert providers._get_kokoro_runtime(cpu_threads=4) is engine
        assert providers._get_kokoro_runtime() is engine
        assert providers._get_kokoro_runtime(cpu_threads=4) is engine
        with pytest.raises(providers.ProviderDependencyError, match="restart"):
            providers._get_kokoro_runtime(cpu_threads=2)
    assert created == [4]


def test_expired_kokoro_piece_does_not_start_native_work():
    import asyncio
    import time
    from healthcare_voice_agent.voice.local_tts import LocalKokoroTTSService
    calls = []
    async def scenario():
        service = LocalKokoroTTSService(synthesize=lambda *a, **k: calls.append(a))
        try:
            with pytest.raises(TimeoutError):
                await service._await_synthesis("synthetic", "context", 0, "af_heart", "en-us", time.monotonic() - 1)
            assert calls == []
        finally:
            await service.cleanup()
    asyncio.run(scenario())
