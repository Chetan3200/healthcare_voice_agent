"""Exercise the pinned TTS adapter's outgoing body with a mocked response.

No inference, network, microphone, audio recording, or quality evaluation occurs.
Only pipeline sample-rate setup is bypassed; run_tts and request construction are
real. The owned SDK client is retained and closed after each test.
"""

import asyncio
import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import TTSAudioRawFrame

from healthcare_voice_agent.config import DEFAULT_TTS_INSTRUCTIONS, load_config
from healthcare_voice_agent.voice.providers import build_tts


@pytest.mark.parametrize("override,expected", [
    (None, DEFAULT_TTS_INSTRUCTIONS),
    ("Speak gently and preserve the supplied language.", "Speak gently and preserve the supplied language."),
    ("off", None),
])
def test_actual_tts_request_preserves_text_and_forwards_style_on_every_request(override, expected, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("This TTS parameter test must never contact a provider")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    environment = {"LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "offline-placeholder"}
    if override is not None:
        environment["TTS_INSTRUCTIONS"] = override
    config = load_config(None, environ=environment)
    calls = []
    chunk_sizes = []
    sample_chunks = [b"\x01\x02", b"\x03\x04"]

    class FakeResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def iter_bytes(self, chunk_size):
            chunk_sizes.append(chunk_size)
            for chunk in sample_chunks:
                yield chunk

    def capture_request(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    async def exercise():
        service = build_tts(config)
        try:
            # FrameProcessorSetup normally supplies this. No pipeline is started.
            service._sample_rate = 24000
            monkeypatch.setattr(
                service._client.audio.speech.with_streaming_response,
                "create", capture_request,
            )
            cases_path = Path(__file__).resolve().parents[2] / "evals/scenarios/tts-naturalness-v1.json"
            suite = json.loads(cases_path.read_text(encoding="utf-8"))
            assert suite["schema_version"] == 1
            assert suite["suite_id"] == "tts-naturalness-v1"
            cases = suite["cases"]
            assert len(cases) == 4
            assert len({case["id"] for case in cases}) == len(cases)
            for case in cases:
                text = case["text"]
                assert text.strip()
                context_id = case["id"]
                frames = [frame async for frame in service.run_tts(text, context_id)]
                assert len(frames) == 2
                assert all(isinstance(frame, TTSAudioRawFrame) for frame in frames)
                assert [frame.audio for frame in frames] == sample_chunks
                assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in frames)
                assert all(frame.context_id == context_id for frame in frames)
                expected_body = {
                    "input": text, "model": "gpt-4o-mini-tts", "voice": "coral",
                    "response_format": "pcm",
                }
                if expected is not None:
                    expected_body["instructions"] = expected
                assert calls[-1] == expected_body  # No speed override or text rewrite.
            assert len(calls) == len(cases)
            assert chunk_sizes == [4800] * len(cases)
        finally:
            await service.cleanup()
        assert service._client.is_closed()

    asyncio.run(exercise())
