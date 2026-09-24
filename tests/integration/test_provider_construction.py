"""Optional real-Pipecat constructor smoke test. No sockets or inference allowed.

Install the voice extra in Terminal to enable this test. Constructor smoke tests
are not live provider, transcription, speech-quality, or pipeline validation.
"""

import asyncio
import importlib.util
import socket

import pytest

from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice.providers import build_llm, build_stt, build_tts

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pipecat") is None,
    reason="Install the voice extra in Terminal to test actual Pipecat constructors",
)


def test_real_provider_construction_and_cleanup_without_network(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Provider construction must not open a network connection")
    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    config = load_config(None, environ={
        "OPENAI_API_KEY": "offline-placeholder", "LIVE_API_ENABLED": "true",
    })

    async def exercise():
        services = []
        try:
            for factory in (build_stt, build_llm, build_tts):
                services.append(factory(config))
            stt, llm, tts = services
            from pipecat.transcriptions.language import Language
            assert stt._settings.language is Language.EN
            assert stt._ws_close_timeout == config.stt.ws_close_timeout_seconds
            assert llm._client.timeout.read == config.llm.timeout_seconds
            assert llm._client.timeout.connect == config.llm.connect_timeout_seconds
            assert tts._client.timeout.read == config.tts.timeout_seconds
            assert tts._settings.instructions == config.tts.instructions
            assert tts._settings.speed is None  # Keep the provider's default speed.
            assert llm._client.max_retries == tts._client.max_retries == 0
        finally:
            for service in reversed(services):
                await service.cleanup()

    asyncio.run(exercise())
