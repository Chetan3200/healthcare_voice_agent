"""Bounded full-reply TTS checks. Native Pipecat, mocked speech HTTP; no API calls."""
import asyncio
from pathlib import Path

from pipecat.frames.frames import (
    AggregatedTextFrame, InterruptionFrame, LLMFullResponseEndFrame,
    LLMFullResponseStartFrame, LLMTextFrame, TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import SleepFrame, run_test

from healthcare_voice_agent.config import load_config
from healthcare_voice_agent.voice.providers import build_tts
from healthcare_voice_agent.voice.reply_tts import FullReplyTTSBuffer


def test_complete_reply_is_one_marin_request(monkeypatch):
    async def check():
        config = load_config(None, environ={
            "LIVE_API_ENABLED": "true", "OPENAI_API_KEY": "offline-placeholder",
            "TTS_VOICE": "marin", "TTS_MODEL": "gpt-4o-mini-tts",
        })
        tts = build_tts(config)
        requests = []

        class Audio:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def iter_bytes(self, chunk_size):
                yield bytes(480)  # Synthetic silence, never played.

        def speech(**kwargs):
            requests.append(kwargs)
            return Audio()

        monkeypatch.setattr(tts._client.audio.speech.with_streaming_response, "create", speech)
        text = "There are 16 slots. They run from 9 a.m. to 4:30 p.m. Which time works?"
        down, _ = await run_test(Pipeline([FullReplyTTSBuffer(), tts]), start_timeout=5, frames_to_send=[
            LLMFullResponseStartFrame(), LLMTextFrame(text[:20]),
            LLMTextFrame(text[20:44]), LLMTextFrame(text[44:]), LLMFullResponseEndFrame(),
        ])
        assert len(requests) == 1
        assert requests[0]["input"] == text
        assert requests[0]["voice"] == "marin"
        assert [frame.text for frame in down if isinstance(frame, TTSTextFrame)] == [text]
    asyncio.run(check())
    script = (Path(__file__).resolve().parents[2] / "scripts/voice_frontdesk_demo.sh").read_text()
    assert "export TTS_VOICE=marin" in script


def test_interrupted_and_empty_replies_are_not_spoken():
    async def check():
        down, _ = await run_test(FullReplyTTSBuffer(), frames_to_send=[
            LLMFullResponseStartFrame(), LLMTextFrame("Discard this unfinished reply."),
            SleepFrame(sleep=0.03), InterruptionFrame(), SleepFrame(sleep=0.03),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(), LLMFullResponseEndFrame(),  # Tool-only turn.
            LLMFullResponseStartFrame(), LLMTextFrame("The replacement reply."),
            LLMFullResponseEndFrame(),
        ])
        assert [frame.text for frame in down if isinstance(frame, AggregatedTextFrame)] == [
            "The replacement reply."]
    asyncio.run(check())
