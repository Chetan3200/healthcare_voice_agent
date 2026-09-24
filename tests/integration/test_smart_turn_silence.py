"""Test the real analyzer's silence fallback, not learned turn accuracy.

Synthetic PCM and explicit speech flags exercise its timer without invoking ONNX
inference, recording audio, starting providers, or waiting on wall-clock time.
"""

import asyncio
import socket

import pytest

pytest.importorskip("pipecat")

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

from healthcare_voice_agent.config import load_config

# Ten milliseconds at 16 kHz, mono PCM16. VAD decisions are supplied explicitly.
PCM_10_MS = b"\x00\x00" * 160


@pytest.fixture(autouse=True)
def block_network_and_audio_logging(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No network in a silence-fallback test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv("PIPECAT_SMART_TURN_LOG_DATA", raising=False)


@pytest.mark.parametrize("override,expected_chunks", [(None, 150), ("3", 300)])
def test_fallback_uses_configured_continuous_silence(override, expected_chunks):
    environment = {} if override is None else {"VOICE_SMART_TURN_STOP_SECONDS": override}
    config = load_config(None, environ=environment)
    analyzer = LocalSmartTurnAnalyzerV3(
        sample_rate=16000,
        params=SmartTurnParams(stop_secs=config.voice.smart_turn_stop_seconds),
    )
    try:
        # The turn strategy normally does this during pipeline setup.
        analyzer.set_sample_rate(16000)
        # Silence before any speech must not create a turn.
        for _ in range(expected_chunks + 1):
            assert analyzer.append_audio(PCM_10_MS, False) == EndOfTurnState.INCOMPLETE
        assert analyzer.append_audio(PCM_10_MS, True) == EndOfTurnState.INCOMPLETE
        for _ in range(expected_chunks - 1):
            assert analyzer.append_audio(PCM_10_MS, False) == EndOfTurnState.INCOMPLETE
        assert analyzer.append_audio(PCM_10_MS, False) == EndOfTurnState.COMPLETE
        assert analyzer.speech_triggered is False
    finally:
        asyncio.run(analyzer.cleanup())


def test_resumed_speech_restarts_silence_fallback():
    config = load_config(None, environ={})
    analyzer = LocalSmartTurnAnalyzerV3(
        sample_rate=16000,
        params=SmartTurnParams(stop_secs=config.voice.smart_turn_stop_seconds),
    )
    try:
        analyzer.set_sample_rate(16000)
        analyzer.append_audio(PCM_10_MS, True)
        for _ in range(140):
            assert analyzer.append_audio(PCM_10_MS, False) == EndOfTurnState.INCOMPLETE
        analyzer.append_audio(PCM_10_MS, True)
        for _ in range(149):
            assert analyzer.append_audio(PCM_10_MS, False) == EndOfTurnState.INCOMPLETE
        assert analyzer.append_audio(PCM_10_MS, False) == EndOfTurnState.COMPLETE
    finally:
        asyncio.run(analyzer.cleanup())
