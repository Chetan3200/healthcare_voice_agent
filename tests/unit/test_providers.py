"""Factory wiring tests using explicit test doubles, not real provider services."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from healthcare_voice_agent.config import ConfigurationError, load_config
from healthcare_voice_agent.voice import providers


@dataclass
class FakeLimits:
    max_connections: int = 1000
    max_keepalive_connections: int = 100
    keepalive_expiry: float = 5.0


class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_retries = kwargs.get("max_retries", 2)
        self.close_count = 0

    async def close(self):
        self.close_count += 1


class FakeService:
    Settings = SimpleNamespace

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cleanup_count = 0

    async def cleanup(self):
        self.cleanup_count += 1


class FakeLLM(FakeService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = self.create_client(api_key=kwargs["api_key"], base_url=kwargs["base_url"])


class FakeTTS(FakeService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = FakeClient(http_client=kwargs["http_client"])
        self.sample_rate = kwargs["sample_rate"]


@pytest.fixture
def components(monkeypatch):
    value = providers._OpenAIComponents(
        stt=FakeService, llm=FakeLLM, tts=FakeTTS,
        async_client=FakeClient, http_client=FakeClient,
        timeout=lambda seconds, connect: SimpleNamespace(default=seconds, connect=connect),
        connection_limits=FakeLimits(),
    )
    monkeypatch.setattr(providers, "_load_openai_components", lambda: value)
    return value


@pytest.fixture
def config():
    return load_config(None, environ={
        "OPENAI_API_KEY": "offline-placeholder", "LIVE_API_ENABLED": "true",
        "LLM_MODEL": "chosen-llm-model", "LLM_TIMEOUT_SECONDS": "17",
        "LLM_CONNECT_TIMEOUT_SECONDS": "3", "LLM_MAX_OUTPUT_TOKENS": "123",
        "TTS_VOICE": "nova", "TTS_TIMEOUT_SECONDS": "11", "TTS_CONNECT_TIMEOUT_SECONDS": "2",
        "STT_WS_CLOSE_TIMEOUT_SECONDS": "1.5",
    })


@pytest.mark.parametrize("factory", [providers.build_stt, providers.build_llm, providers.build_tts])
def test_live_gate_runs_before_any_provider_import(factory, monkeypatch):
    def unexpected_import():
        raise AssertionError("Live gate must run before importing provider services")
    monkeypatch.setattr(providers, "_load_openai_components", unexpected_import)
    with pytest.raises(ConfigurationError, match="disabled"):
        factory(load_config(None, environ={}))


def test_stt_uses_explicit_english_local_turn_control_and_close_timeout(config, components):
    service = providers.build_stt(config)
    assert service.kwargs["settings"].model == config.stt.model
    assert service.kwargs["settings"].language == "en"
    assert service.kwargs["base_url"] == config.stt.base_url
    assert service.kwargs["turn_detection"] is False
    assert service.kwargs["finalize_transcripts"] is True
    assert service.kwargs["reconnect_on_error"] is False
    assert service.kwargs["ws_close_timeout"] == 1.5


def test_llm_applies_configured_timeouts_to_the_actual_sdk_client(config, components):
    service = providers.build_llm(config, system_instruction="A test agent instruction")
    assert service.kwargs["settings"].model == "chosen-llm-model"
    assert service.kwargs["settings"].system_instruction == "A test agent instruction"
    assert service.kwargs["settings"].max_completion_tokens == 123
    assert service._client.kwargs["base_url"] == config.llm.base_url
    assert service._client.kwargs["timeout"].default == 17
    assert service._client.kwargs["timeout"].connect == 3
    assert service._client.max_retries == 0
    assert service.kwargs["retry_on_timeout"] is False
    http = service._client.kwargs["http_client"]
    assert http.kwargs["limits"] == FakeLimits(keepalive_expiry=60.0)
    assert components.connection_limits.keepalive_expiry == 5.0


@pytest.mark.parametrize(("agent_mode", "expected_llm_retries"), [
    ("conversation", 0),
    ("frontdesk_demo", 0),
    ("clinician", 1),
])
def test_llm_native_retry_policy_is_profile_specific_and_keeps_pipecat_retry_disabled(
    agent_mode, expected_llm_retries, components,
):
    config = load_config(None, environ={
        "OPENAI_API_KEY": "offline-placeholder", "LIVE_API_ENABLED": "true",
        "AGENT_MODE": agent_mode,
    })

    llm = providers.build_llm(config)
    tts = providers.build_tts(config)

    assert config.llm_http_max_retries == expected_llm_retries
    assert llm._client.max_retries == expected_llm_retries
    assert llm.kwargs["retry_on_timeout"] is False
    assert tts._client.max_retries == 0


def test_tts_uses_configured_voice_endpoint_and_http_client(config, components):
    service = providers.build_tts(config)
    assert service.kwargs["settings"].voice == "nova"
    assert service.kwargs["settings"].model == config.tts.model
    assert service.kwargs["settings"].instructions == config.tts.instructions
    assert service.kwargs["base_url"] == config.tts.base_url
    timeout = service.kwargs["http_client"].kwargs["timeout"]
    assert timeout.default == 11
    assert timeout.connect == 2
    assert service.kwargs["sample_rate"] == 24000
    assert service.chunk_size == 4800
    assert service.kwargs["http_client"].kwargs["limits"] == FakeLimits(keepalive_expiry=60.0)
    assert components.connection_limits.keepalive_expiry == 5.0
    assert service._client.max_retries == 0


@pytest.mark.parametrize("factory", [providers.build_llm, providers.build_tts])
def test_http_services_close_their_owned_clients_once(factory, config, components):
    service = factory(config)
    async def exercise():
        await service.cleanup()
        await service.cleanup()
    asyncio.run(exercise())
    assert service._client.close_count == 1


def test_cleanup_closes_client_even_if_framework_cleanup_fails(config, components, monkeypatch):
    async def failing_cleanup(self):
        raise RuntimeError("test cleanup failure")
    monkeypatch.setattr(FakeLLM, "cleanup", failing_cleanup)
    service = providers.build_llm(config)
    with pytest.raises(RuntimeError, match="test cleanup failure"):
        asyncio.run(service.cleanup())
    assert service._client.close_count == 1


def test_provider_specific_imports_stay_in_the_factory_module():
    import ast
    from pathlib import Path
    package = Path(__file__).resolve().parents[2] / "src/healthcare_voice_agent"
    for source in package.rglob("*.py"):
        if source == package / "voice/providers.py":
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(
                name == "openai" or name.startswith(("openai.", "pipecat.services.openai"))
                for name in names
            ), f"Provider-specific import outside factory module: {source}"


@pytest.mark.parametrize("value,expected", [
    ("Speak calmly without changing the text.", "Speak calmly without changing the text."),
    ("off", None),
])
def test_tts_factory_forwards_custom_or_disabled_instructions(value, expected, components):
    config = load_config(None, environ={
        "OPENAI_API_KEY": "offline-placeholder", "LIVE_API_ENABLED": "true",
        "TTS_INSTRUCTIONS": value,
    })
    service = providers.build_tts(config)
    try:
        assert service.kwargs["settings"].instructions == expected
        assert service.kwargs["settings"].model == "gpt-4o-mini-tts"
        assert service.kwargs["settings"].voice == "coral"
    finally:
        asyncio.run(service.cleanup())
