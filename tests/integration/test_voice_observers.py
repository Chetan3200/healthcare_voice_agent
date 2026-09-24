"""Offline observer wiring checks. No transport, provider, or inference calls."""

import asyncio
import json

import pytest

pytest.importorskip("pipecat", reason="Install the voice extra in Terminal to test Pipecat observers")


def test_observers_trace_text_boundaries_metrics_and_deduplicate(tmp_path):
    from pipecat.frames.frames import (
        InputAudioRawFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        MetricsFrame,
        TTSTextFrame,
    )
    from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    from healthcare_voice_agent.voice.observers import make_observers
    from healthcare_voice_agent.voice.tracing import SessionTrace

    class Source:
        name = "test-source"

    async def exercise():
        trace_path = tmp_path / "voice.jsonl"
        trace = SessionTrace(trace_path, session_id="observer-test")
        observer = make_observers(trace)[0]
        source = Source()
        audio = InputAudioRawFrame(audio=b"raw", sample_rate=16000, num_channels=1)
        audio_data = FramePushed(source=source, destination=source, frame=audio, direction=FrameDirection.DOWNSTREAM, timestamp=0)
        await observer.on_push_frame(audio_data)
        assert audio.id not in observer._seen

        frames = [
            LLMFullResponseStartFrame(),
            LLMTextFrame(text="Hello"),
            TTSTextFrame(text="Hello", aggregated_by="sentence", context_id="turn-7"),
            MetricsFrame(data=[
                TTFBMetricsData(processor="llm", value=0.1),
                ProcessingMetricsData(processor="llm", value=0.1),
            ]),
        ]
        for frame in frames:
            data = FramePushed(source=source, destination=source, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0)
            await observer.on_push_frame(data)
            await observer.on_push_frame(data)
        trace.close()
        return [json.loads(line) for line in trace_path.read_text().splitlines()]

    events = asyncio.run(exercise())
    assert [item["event"] for item in events] == [
        "llm_response_start", "assistant_generated_text", "tts_output_text", "metrics"
    ]
    assert events[2]["data"]["context_id"] == "turn-7"
    assert [metric["metric_type"] for metric in events[-1]["data"]["metrics"]] == [
        "TTFBMetricsData", "ProcessingMetricsData"
    ]


def test_safe_rtvi_hides_provider_error_but_preserves_error_shape():
    from pipecat.frames.frames import ErrorFrame

    from healthcare_voice_agent.voice.rtvi import SafeRTVIProcessor

    async def exercise():
        processor = SafeRTVIProcessor()
        captured = []

        async def capture(message):
            captured.append(message)

        processor.push_transport_message = capture
        await processor._send_error_frame(ErrorFrame(error="provider failed: top-secret", fatal=True))
        return captured[0].model_dump(mode="json")

    message = asyncio.run(exercise())
    assert message["type"] == "error"
    assert message["data"]["fatal"] is True
    assert message["data"]["error"] == "The voice service encountered an error. Please try again."
    assert "top-secret" not in json.dumps(message)


def test_stt_finalization_metadata_is_traced_without_raw_provider_result(tmp_path):
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection
    from healthcare_voice_agent.voice.observers import make_observers
    from healthcare_voice_agent.voice.tracing import SessionTrace

    async def exercise():
        trace_path = tmp_path / "finalization.jsonl"
        trace = SessionTrace(trace_path, session_id="finalization-test")
        observer = make_observers(trace)[0]
        source = type("Source", (), {"name": "test-stt"})()
        frame = TranscriptionFrame("complete sentence", "user", "now", finalized=True,
                                   result={"raw": "must-not-be-serialized"})
        frame.metadata.update({
            "stt_speech_generation": 2, "stt_commit_sequences": [1, 2],
            "stt_item_ids": ["one", "two"], "stt_completion_received_at": ["t1", "t2"],
            "unrelated": "must-not-be-serialized",
        })
        await observer.on_push_frame(FramePushed(source=source, destination=source,
            frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0))
        trace.close()
        text = trace_path.read_text()
        assert "must-not-be-serialized" not in text
        event = json.loads(text)
        assert event["event"] == "stt_final"
        assert event["data"]["finalized"] is True
        assert event["data"]["speech_generation"] == 2
        assert event["data"]["item_ids"] == ["one", "two"]
    asyncio.run(exercise())
