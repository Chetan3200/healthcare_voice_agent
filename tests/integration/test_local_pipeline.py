"""Both local adapters in the real graph, with model and LLM boundaries faked."""

import asyncio
import socket
from types import SimpleNamespace

import pytest

pytest.importorskip("pipecat")

from pipecat.processors.frame_processor import FrameProcessor
from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice import pipeline as wiring
from healthcare_voice_agent.voice.local_stt import LocalWhisperSTTService
from healthcare_voice_agent.voice.local_tts import LocalKokoroTTSService
from healthcare_voice_agent.voice.tracing import SessionTrace


class PassThrough(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def test_real_paired_local_graph_and_session_isolation(monkeypatch, tmp_path):
    def blocked(*args, **kwargs):
        raise AssertionError("Paired graph test is offline")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(wiring.providers, "_get_whisper_runtime", lambda: (object(), "offline"))
    monkeypatch.setattr(wiring.providers, "_get_kokoro_runtime", lambda **kwargs: SimpleNamespace(
        voices={"af_heart": object(), "hf_alpha": object()},
    ))
    monkeypatch.setattr(wiring.providers, "build_llm", lambda *args, **kwargs: PassThrough())
    config = load_config(None, environ={
        "STT_PROVIDER": "whisper", "TTS_PROVIDER": "kokoro",
        "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "offline-placeholder",
    })

    async def run():
        bundles, traces = [], []
        try:
            for index in range(2):
                trace = SessionTrace(tmp_path / f"{index}.jsonl", session_id=str(index))
                traces.append(trace)
                incoming, outgoing = PassThrough(), PassThrough()
                transport = SimpleNamespace(input=lambda: incoming, output=lambda: outgoing)
                bundle = await wiring.build_pipeline(config, transport, trace, prompt="Synthetic test")
                bundles.append(bundle)
                chain = bundle.pipeline.processors[1:-1]
                assert isinstance(chain[2], LocalWhisperSTTService)
                assert isinstance(chain[5], LocalKokoroTTSService)
                assert chain[5].chunk_size == 4800
                assert not bundle.audio_gate.ready
            assert bundles[0].context is not bundles[1].context
            left = bundles[0].pipeline.processors[1:-1]
            right = bundles[1].pipeline.processors[1:-1]
            assert left[2]._segment_queue is not right[2]._segment_queue
            assert left[2]._audio_buffer is not right[2]._audio_buffer
            assert left[5] is not right[5]
        finally:
            for bundle in bundles:
                await bundle.cleanup_unstarted()
            for trace in traces:
                trace.close()
    asyncio.run(run())
