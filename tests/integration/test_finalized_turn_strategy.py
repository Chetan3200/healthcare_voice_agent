"""Actual Pipecat strategy/controller/aggregation; no providers or audio."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import pytest
pytest.importorskip("pipecat")
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import InterimTranscriptionFrame, STTMetadataFrame, TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from healthcare_voice_agent.voice.turns import SpeechGenerationTurnStopStrategy

class Analyzer(BaseTurnAnalyzer):
    def __init__(self, verdict=EndOfTurnState.COMPLETE):
        super().__init__()
        self.verdict = verdict
    @property
    def speech_triggered(self): return False
    @property
    def params(self): return None
    def append_audio(self, buffer, is_speech): return EndOfTurnState.INCOMPLETE
    async def analyze_end_of_turn(self): return self.verdict, None
    def update_vad_start_secs(self, start_secs): pass
    def clear(self): pass

def transcript(text, generation=1, *, finalized=True, item_ids=None):
    frame = TranscriptionFrame(text, "synthetic-user", "now", finalized=finalized)
    if generation is not None: frame.metadata["stt_speech_generation"] = generation
    if item_ids is not None: frame.metadata["stt_item_ids"] = item_ids
    return frame

@asynccontextmanager
async def strategy_only(verdict=EndOfTurnState.COMPLETE, *, enabled=True):
    manager = TaskManager()
    s = SpeechGenerationTurnStopStrategy(turn_analyzer=Analyzer(verdict), finalize_transcripts=enabled)
    await s.setup(SimpleNamespace(task_manager=manager, audio_in_sample_rate=16000))
    stopped = []
    async def on_stop(*args): stopped.append(s._text)
    s.add_event_handler("on_user_turn_stopped", on_stop)
    try:
        await s.process_frame(STTMetadataFrame(service_name="offline-stt", ttfs_p99_latency=60.0))
        yield s, manager, stopped
    finally:
        await s.cleanup()
        assert not manager.current_tasks()

async def expire(s, manager):
    await manager.cancel_task(s._timeout_task)
    await s._timeout_handler(0)  # Real timeout handler, no long wall-clock wait.

async def stopped_speech(s):
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))

def test_current_final_stops_before_long_p99_deadline():
    async def exercise():
        async with strategy_only() as (s, manager, stopped):
            await stopped_speech(s)
            assert s._timeout_task is not None
            await s.process_frame(transcript("complete sentence"))
            assert stopped == ["complete sentence"]
            assert s._timeout_task is None
            assert s.wait_for_transcript is True
    asyncio.run(exercise())

def test_nonfinal_waits_and_native_fallback_is_retained():
    async def exercise():
        async with strategy_only() as (s, manager, stopped):
            await stopped_speech(s)
            await s.process_frame(transcript("legacy text", finalized=False))
            assert stopped == []
            await expire(s, manager)
            assert stopped == ["legacy text"]
    asyncio.run(exercise())

def test_final_cannot_override_incomplete_smart_turn():
    async def exercise():
        async with strategy_only(EndOfTurnState.INCOMPLETE) as (s, manager, stopped):
            await stopped_speech(s)
            await s.process_frame(transcript("I was going to"))
            await expire(s, manager)
            assert stopped == []
    asyncio.run(exercise())

def test_resume_clears_older_text_and_rejects_late_final_even_after_timeout():
    async def exercise():
        async with strategy_only(EndOfTurnState.INCOMPLETE) as (s, manager, stopped):
            await stopped_speech(s)
            await s.process_frame(transcript("earlier segment"))
            await s.process_frame(VADUserStartedSpeakingFrame())
            assert s._text == ""
            s._turn_analyzer.verdict = EndOfTurnState.COMPLETE
            await s.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
            await s.process_frame(transcript("late older text", generation=1))
            assert s._text == "" and not s._transcript_finalized
            await expire(s, manager)
            assert stopped == []
            await s.process_frame(transcript("current ending", generation=2))
            assert stopped == ["current ending"]
    asyncio.run(exercise())

def test_new_interim_item_rejects_old_queued_final_before_vad_resumes():
    async def exercise():
        async with strategy_only() as (s, manager, stopped):
            await stopped_speech(s)
            frame = InterimTranscriptionFrame("more", "user", "now")
            frame.metadata["stt_item_id"] = "newer"
            await s.process_frame(frame)
            await s.process_frame(transcript("old text", item_ids=["old"]))
            assert s._text == "" and stopped == []
            await expire(s, manager)
            assert stopped == []
            await s.process_frame(transcript("new text", item_ids=["newer"]))
            assert stopped == ["new text"]
    asyncio.run(exercise())

def test_disabled_fix_preserves_native_untagged_path():
    async def exercise():
        async with strategy_only(enabled=False) as (s, manager, stopped):
            await stopped_speech(s)
            await s.process_frame(transcript("baseline", generation=None, finalized=False))
            assert stopped == [] and s._speech_generation == 0
            await expire(s, manager)
            assert stopped == ["baseline"]
    asyncio.run(exercise())

@asynccontextmanager
async def aggregator(verdict=EndOfTurnState.COMPLETE):
    manager = TaskManager()
    analyzer = Analyzer(verdict)
    strategy = SpeechGenerationTurnStopStrategy(turn_analyzer=analyzer)
    context = LLMContext()
    user = LLMUserAggregator(context, params=LLMUserAggregatorParams(user_turn_strategies=UserTurnStrategies(stop=[strategy])))
    await user.setup(FrameProcessorSetup(clock=SystemClock(), task_manager=manager, pipeline_worker=SimpleNamespace(app_resources={})))
    committed = []
    ready = asyncio.Event()
    async def discard(*args, **kwargs): pass
    async def added(agg, message):
        committed.append(message.content)
        ready.set()
    user.push_frame = discard
    user.broadcast_frame = discard
    user.broadcast_interruption = discard
    user.add_event_handler("on_user_turn_message_added", added)
    try:
        await user.process_frame(STTMetadataFrame(service_name="offline-stt", ttfs_p99_latency=60.0), FrameDirection.DOWNSTREAM)
        yield SimpleNamespace(user=user, strategy=strategy, analyzer=analyzer, context=context, committed=committed, ready=ready)
    finally:
        await user.cleanup()
        assert not manager.current_tasks()

async def feed(b, frame):
    await b.user.process_frame(frame, FrameDirection.DOWNSTREAM)

def test_actual_aggregator_and_controller_commit_ordered_pieces_once():
    async def exercise():
        async with aggregator() as b:
            await feed(b, VADUserStartedSpeakingFrame())
            await feed(b, VADUserStoppedSpeakingFrame(stop_secs=0.2))
            await feed(b, transcript("Use the right side,", finalized=False))
            assert b.committed == []
            await feed(b, transcript("not the left."))
            await asyncio.wait_for(b.ready.wait(), timeout=1)
            assert b.committed == ["Use the right side, not the left."]
            assert b.context.get_messages() == [{"role": "user", "content": b.committed[0]}]
            await b.strategy.trigger_user_turn_stopped()
            assert len(b.committed) == 1
    asyncio.run(exercise())

def test_actual_aggregator_keeps_old_text_but_waits_for_resumed_ending():
    async def exercise():
        async with aggregator(EndOfTurnState.INCOMPLETE) as b:
            await feed(b, VADUserStartedSpeakingFrame())
            await feed(b, VADUserStoppedSpeakingFrame(stop_secs=0.2))
            await feed(b, VADUserStartedSpeakingFrame())
            b.analyzer.verdict = EndOfTurnState.COMPLETE
            await feed(b, VADUserStoppedSpeakingFrame(stop_secs=0.2))
            await feed(b, transcript("The earlier part", generation=1))
            assert b.committed == []
            await feed(b, transcript("and the corrected ending.", generation=2))
            await asyncio.wait_for(b.ready.wait(), timeout=1)
            assert b.committed == ["The earlier part and the corrected ending."]
            assert b.context.get_messages()[0]["content"] == b.committed[0]
    asyncio.run(exercise())
