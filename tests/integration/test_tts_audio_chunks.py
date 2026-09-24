"""Exercise configured TTS PCM chunks with a fully mocked streaming response.

No provider request, inference, microphone, audio playback, or quality/latency
claim occurs. The real Pipecat TTS subclass and its ``run_tts`` implementation
are used; only its SDK streaming response is replaced.
"""

import asyncio
import socket

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.openai.tts import OpenAITTSService

from healthcare_voice_agent.config import DEFAULT_TTS_INSTRUCTIONS, load_config
from healthcare_voice_agent.voice.providers import build_tts


def _block_network(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("This offline TTS chunk test must never contact a provider")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)


@pytest.mark.parametrize(("audio_chunk_ms", "expected_chunk_bytes"), [(100, 4800), (500, 24000)])
def test_actual_tts_uses_configured_pcm_chunk_size_and_streams_before_completion(
    audio_chunk_ms, expected_chunk_bytes, monkeypatch,
):
    _block_network(monkeypatch)
    config = load_config(None, environ={
        "LIVE_API_ENABLED": "true",
        "OPENAI_API_KEY": "offline-placeholder",
        "TTS_AUDIO_CHUNK_MS": str(audio_chunk_ms),
    })
    requested = []
    chunk_sizes = []
    first_pcm = bytes(index % 251 for index in range(expected_chunk_bytes))
    final_pcm = b"\x00\xff\x10\xef\x22\xdd"

    class FakeResponse:
        status_code = 200

        def __init__(self):
            self.finished = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def iter_bytes(self, chunk_size):
            chunk_sizes.append(chunk_size)
            yield first_pcm
            yield final_pcm
            self.finished = True

    response = FakeResponse()

    def capture_request(**kwargs):
        requested.append(kwargs)
        return response

    async def exercise():
        service = build_tts(config)
        try:
            assert isinstance(service, OpenAITTSService)
            # FrameProcessorSetup normally supplies this. No pipeline is started.
            service._sample_rate = 24000
            assert service.chunk_size == expected_chunk_bytes
            monkeypatch.setattr(
                service._client.audio.speech.with_streaming_response, "create", capture_request,
            )

            stream = service.run_tts("Please rest your ankle.", "chunk-context")
            first_frame = await anext(stream)
            # The first frame is available while the fake provider still has audio to stream.
            assert response.finished is False
            remaining_frames = [frame async for frame in stream]
            frames = [first_frame, *remaining_frames]

            assert all(isinstance(frame, TTSAudioRawFrame) for frame in frames)
            assert [frame.audio for frame in frames] == [first_pcm, final_pcm]
            assert [len(frame.audio) for frame in frames] == [expected_chunk_bytes, len(final_pcm)]
            assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in frames)
            assert all(frame.context_id == "chunk-context" for frame in frames)
            assert chunk_sizes == [expected_chunk_bytes]
            assert requested == [{
                "input": "Please rest your ankle.",
                "model": "gpt-4o-mini-tts",
                "voice": "coral",
                "instructions": DEFAULT_TTS_INSTRUCTIONS,
                "response_format": "pcm",
            }]
        finally:
            await service.cleanup()
        assert service._client.is_closed()

    asyncio.run(exercise())
