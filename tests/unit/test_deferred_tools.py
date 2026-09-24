"""Offline unit checks for deferred native-tool delivery coordination."""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from healthcare_voice_agent.voice.deferred_tools import (
    PROCESSING_NOTICE,
    DeferredToolDelivery,
    _Call,
    _DeliveryCheck,
)


class Context:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)


class Trace:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


class GateHarness:
    def __init__(self, *, notice_seconds=0.01, trace=None):
        self.context = Context()
        self.gate = DeferredToolDelivery(SimpleNamespace(trace=trace), notice_seconds=notice_seconds)
        self.gate.bind(SimpleNamespace(), object(), object(), self.context)
        self.queued = []
        self.pushed = []
        self.tasks = []

        async def queue(frame, direction=FrameDirection.DOWNSTREAM, callback=None):
            self.queued.append((frame, direction))

        async def push(frame, direction=FrameDirection.DOWNSTREAM):
            self.pushed.append((frame, direction))

        def create(coroutine, name=None, context=None):
            task = asyncio.create_task(coroutine, name=name, context=context)
            self.tasks.append(task)
            return task

        self.gate.queue_frame = queue
        self.gate.push_frame = push
        self.gate.create_task = create

    def call(self, call_id, group=None, *, epoch=None, generation=None):
        call = _Call(group or call_id,
                     self.gate.epoch if epoch is None else epoch,
                     self.gate.generation if generation is None else generation)
        self.gate._calls[call_id] = call
        return call

    async def settle(self):
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def deliver(self, frame):
        await self.gate.process_frame(frame, FrameDirection.DOWNSTREAM)

    def finish_reply(self, text="An answer.", context_id="reply"):
        self.gate.observe(self.gate.llm, LLMTextFrame(text))
        self.gate.observe(self.gate.tts, TTSStartedFrame(context_id))
        self.gate.observe(self.gate.llm, LLMFullResponseEndFrame())
        self.gate.observe(self.gate.output, TTSStoppedFrame(context_id))


def test_notice_is_exactly_once_at_threshold_and_fast_completion_suppresses_it():
    async def check():
        slow = GateHarness(notice_seconds=0.001)
        slow_call = slow.call("slow")
        slow_call.timer = asyncio.create_task(slow.gate._notice_after("slow", slow_call))
        await asyncio.sleep(0.01)
        notices = [frame for frame, _ in slow.queued if isinstance(frame, _DeliveryCheck)]
        assert [(notice.epoch, notice.notice_id) for notice in notices] == [(0, "slow")]
        await slow.deliver(notices[0])
        assert [frame.text for frame, _ in slow.pushed if isinstance(frame, TTSSpeakFrame)] == [PROCESSING_NOTICE]
        await slow.deliver(notices[0])
        assert len([frame for frame, _ in slow.pushed if isinstance(frame, TTSSpeakFrame)]) == 1

        fast = GateHarness(notice_seconds=0.01)
        fast_call = fast.call("fast")
        fast_call.timer = asyncio.create_task(fast.gate._notice_after("fast", fast_call))
        fast.gate.finish("fast")
        await asyncio.sleep(0.02)
        assert fast.queued == []

    asyncio.run(check())


def test_new_user_turn_suppresses_already_queued_notice():
    async def check():
        harness = GateHarness()
        harness.call("old")
        queued = _DeliveryCheck(harness.gate.epoch, notice_id="old")
        harness.gate.user_started()
        await harness.deliver(queued)
        assert not [frame for frame, _ in harness.pushed if isinstance(frame, TTSSpeakFrame)]
        assert harness.gate._notice_epochs == set()

    asyncio.run(check())


@pytest.mark.parametrize("blocked", ["user", "llm", "tts", "audio"])
def test_pending_or_busy_states_block_ready_result_delivery(blocked):
    async def check():
        harness = GateHarness()
        call = harness.call("result")
        harness.gate.finish("result")
        harness.gate.result_in_context("result", call)
        if blocked == "user":
            harness.gate.user_started()
        elif blocked == "llm":
            harness.gate.observe(harness.gate.llm, LLMFullResponseStartFrame())
        elif blocked == "tts":
            harness.gate.observe(harness.gate.tts, TTSStartedFrame("tts-1"))
        else:
            harness.gate.observe(harness.gate.tts, TTSStartedFrame("tts-1"))
            harness.gate.observe(harness.gate.output, BotStartedSpeakingFrame())
        await harness.deliver(_DeliveryCheck(harness.gate.epoch))
        assert harness.context.messages == []
        assert "result" in harness.gate._ready

    asyncio.run(check())


def test_only_ordered_transport_tts_stopped_releases_speech_gate_and_wakes_result():
    async def check():
        harness = GateHarness()
        call = harness.call("result")
        harness.gate.finish("result")
        harness.gate.observe(harness.gate.tts, TTSStartedFrame("audio-1"))
        harness.gate.result_in_context("result", call)
        harness.gate.observe(harness.gate.llm, LLMFullResponseEndFrame())
        harness.gate.observe(harness.gate.output, BotStoppedSpeakingFrame())
        await harness.settle()
        assert not [frame for frame, _ in harness.queued if isinstance(frame, _DeliveryCheck)]
        harness.gate.observe(harness.gate.output, TTSStoppedFrame("wrong-context"))
        await harness.settle()
        assert not [frame for frame, _ in harness.queued if isinstance(frame, _DeliveryCheck)]
        harness.gate.observe(harness.gate.output, TTSStoppedFrame("audio-1"))
        await harness.settle()
        checks = [frame for frame, _ in harness.queued if isinstance(frame, _DeliveryCheck)]
        assert len(checks) == 1 and checks[0].notice_id is None
        await harness.deliver(checks[0])
        assert len(harness.context.messages) == 1
        assert "result" in harness.context.messages[0]["content"]

    asyncio.run(check())


def test_foreground_answer_cannot_claim_or_finish_a_ready_search_result():
    async def check():
        trace = Trace()
        h = GateHarness(trace=trace)
        call = h.call("search")
        h.gate.finish("search")
        h.gate.result_in_context("search", call)
        await h.settle()
        old_check = next(f for f, _ in h.queued if isinstance(f, _DeliveryCheck))
        h.gate.user_started()
        h.gate.user_stopped()
        foreground = LLMContextFrame(h.context)
        await h.deliver(foreground)
        assert h.gate._ready == {"search": call}
        assert not h.gate._presenting
        await h.deliver(old_check)
        h.finish_reply("The imaging was a left wrist X-ray.")
        await h.settle()
        assert h.gate._ready == {"search": call}
        assert not any(e == "background_result_response_finished" for e, _ in trace.events)
        check = next(f for f, _ in h.queued if isinstance(f, _DeliveryCheck)
                     and f.epoch == h.gate.epoch)
        await h.deliver(check)
        assert not h.gate._ready
        assert h.gate._presenting == {"search": call}
        h.finish_reply("Coming back to the public cast guidance.")
        assert [(e, d["tool_call_ids"]) for e, d in trace.events
                if e == "background_result_response_finished"] == [
                    ("background_result_response_finished", ["search"])]
        assert not h.context.messages  # No stale delivery instruction on future turns.

    asyncio.run(check())


def test_barge_in_retains_presenting_result_without_cancelling_tool_task():
    async def check():
        harness = GateHarness()
        result = _Call("batch", 0, 0)
        harness.gate._presenting["result"] = result
        call = harness.call("still-running", "batch")
        await harness.gate.process_frame(
            __import__("pipecat.frames.frames", fromlist=["InterruptionFrame"]).InterruptionFrame(),
            FrameDirection.DOWNSTREAM,
        )
        assert harness.gate._ready == {"result": result}
        assert harness.gate._presenting == {}
        assert harness.gate._calls["still-running"] is call
        assert call.timer is None

    asyncio.run(check())


def test_batch_siblings_wait_for_group_but_different_group_older_epoch_does_not_block():
    async def check():
        siblings = GateHarness()
        ready = siblings.call("one", "native-batch")
        siblings.call("two", "native-batch")
        siblings.gate.finish("one")
        siblings.gate.result_in_context("one", ready)
        await siblings.settle()
        assert not [frame for frame, _ in siblings.queued if isinstance(frame, _DeliveryCheck)]

        epochs = GateHarness()
        old = epochs.call("old", "old-batch", epoch=-1)
        old.timer = asyncio.create_task(asyncio.sleep(10))
        ready = epochs.call("new", "new-batch")
        epochs.gate.finish("new")
        epochs.gate.result_in_context("new", ready)
        await epochs.settle()
        checks = [frame for frame, _ in epochs.queued if isinstance(frame, _DeliveryCheck)]
        assert len(checks) == 1
        old.timer.cancel()
        await asyncio.gather(old.timer, return_exceptions=True)

    asyncio.run(check())


def test_invalidate_and_close_reject_late_callbacks_cancel_timers_and_trace_is_optional():
    async def check():
        harness = GateHarness(trace=None)
        call = harness.call("late")
        call.timer = asyncio.create_task(asyncio.sleep(10))
        harness.gate.invalidate()
        harness.gate.result_in_context("late", call)
        await harness.deliver(_DeliveryCheck(0))
        assert harness.gate._ready == {}
        await asyncio.gather(call.timer, return_exceptions=True)
        assert call.timer.cancelled()

        closed = GateHarness(trace=Trace())
        later = closed.call("later")
        closed.gate.stop()
        closed.gate.result_in_context("later", later)
        await closed.deliver(_DeliveryCheck(closed.gate.epoch))
        assert closed.gate.closed is True
        assert closed.gate._ready == {}
        assert closed.context.messages == []

        cleaned = GateHarness()
        pending = cleaned.call("pending")
        pending.timer = asyncio.create_task(asyncio.sleep(10))
        await cleaned.gate.cleanup()
        assert cleaned.gate.closed is True
        assert pending.timer.cancelled()

    asyncio.run(check())


def test_interruption_invalidates_queued_wake_without_overwriting_completed_user_turn():
    from pipecat.frames.frames import InterruptionFrame
    async def check():
        h = GateHarness()
        call = h.call('result')
        h.gate.finish('result')
        h.gate.result_in_context('result', call)
        await h.settle()
        old = next(f for f, _ in h.queued if isinstance(f, _DeliveryCheck))
        h.gate.user_started()
        h.gate.user_stopped()
        await h.deliver(InterruptionFrame())
        assert not h.gate.user_pending
        await h.deliver(old)
        assert h.context.messages == []
        h.gate.check_ready()
        await h.settle()
        assert any(isinstance(f, _DeliveryCheck) and f.epoch == h.gate.epoch for f, _ in h.queued)
    asyncio.run(check())


def test_failed_delivery_keeps_result_without_automatic_llm_retry():
    from healthcare_voice_agent.voice.reply_tts import LLMErrorFallbackFrame
    async def check():
        h = GateHarness()
        h.gate.user_pending = True
        result = _Call('batch', 0, 0)
        h.gate._presenting['result'] = result
        h.gate.observe(h.gate.llm, LLMFullResponseStartFrame())
        h.gate.observe(h.gate.llm, LLMErrorFallbackFrame())
        assert h.gate._ready == {'result': result}
        assert not h.gate.user_pending
        h.gate.observe(h.gate.tts, TTSStartedFrame('fallback'))
        h.gate.observe(h.gate.llm, LLMFullResponseEndFrame())
        h.gate.observe(h.gate.output, TTSStoppedFrame('fallback'))
        await h.settle()
        assert not h.queued
        h.gate.user_started()
        h.gate.user_stopped()
        await h.deliver(LLMContextFrame(h.context))
        assert h.gate._ready == {'result': result}
        assert not h.gate._presenting
        h.finish_reply()
        await h.settle()
        await h.deliver(next(f for f, _ in h.queued if isinstance(f, _DeliveryCheck)))
        assert h.gate._presenting == {'result': result}
    asyncio.run(check())


def test_old_reply_error_does_not_clear_new_user_or_pause_new_turn():
    from healthcare_voice_agent.voice.reply_tts import LLMErrorFallbackFrame
    h = GateHarness()
    h.gate.observe(h.gate.llm, LLMFullResponseStartFrame())
    h.gate.user_started()
    h.gate.observe(h.gate.llm, LLMErrorFallbackFrame())
    assert h.gate.user_pending and not h.gate._wait_for_user


def test_case_reset_does_not_falsely_declare_active_audio_idle():
    h = GateHarness()
    h.gate.observe(h.gate.tts, TTSStartedFrame('speech'))
    h.gate.invalidate()
    assert h.gate.speech_busy and h.gate._audio_context == 'speech'


def test_native_reprompt_or_upstream_continuation_does_not_consume_unclaimed_results():
    async def check():
        h = GateHarness()
        call = h.call('result')
        h.gate.finish('result')
        h.gate.user_pending = True
        h.gate.result_in_context('result', call)
        h.gate.begin_inference()  # Native continuation bypassing the input processor.
        h.gate.observe(h.gate.llm, LLMFullResponseEndFrame())
        assert h.gate._ready == {'result': call}
        assert not h.gate._presenting and not h.queued
        h.gate.user_stopped()
        h.gate.begin_inference()
        h.finish_reply("Answer only the new question.")
        await h.settle()
        assert h.gate._ready == {'result': call}
        assert not h.gate._presenting
        assert any(isinstance(f, _DeliveryCheck) for f, _ in h.queued)
    asyncio.run(check())


def test_current_question_results_are_claimed_before_ready_older_results():
    async def check():
        h = GateHarness()
        search = h.call('search')
        h.gate.user_started()
        h.gate.user_stopped()
        study = h.call('study')
        for key, call in [('search', search), ('study', study)]:
            h.gate.finish(key)
            h.gate.result_in_context(key, call)
        await h.deliver(_DeliveryCheck(h.gate.epoch))
        assert set(h.gate._presenting) == {'study'}
        assert set(h.gate._ready) == {'search'}
        assert 'study' in h.context.messages[-1]['content']
        assert 'search' not in h.context.messages[-1]['content']
        h.finish_reply('Imaging only.')
        await h.deliver(_DeliveryCheck(h.gate.epoch))
        assert set(h.gate._presenting) == {'search'}
        assert not h.gate._ready
        assert 'search' in h.context.messages[-1]['content']
        assert 'study' not in h.context.messages[-1]['content']
    asyncio.run(check())


def test_tool_only_delivery_retains_parent_through_dependent_lookup():
    async def check():
        trace = Trace()
        h = GateHarness(trace=trace)
        parent = h.call('parent')
        h.gate.finish('parent')
        h.gate.result_in_context('parent', parent)
        await h.deliver(_DeliveryCheck(h.gate.epoch))
        h.gate.observe(h.gate.llm, FunctionCallsStartedFrame(
            function_calls=[SimpleNamespace(tool_call_id='dependent')]))
        h.gate.observe(h.gate.llm, LLMFullResponseEndFrame())
        assert set(h.gate._presenting) == {'parent'}
        assert not any(e == 'background_result_response_finished' for e, _ in trace.events)
        child = h.call('dependent')
        h.gate.finish('dependent')
        h.gate.result_in_context('dependent', child)
        await h.deliver(_DeliveryCheck(h.gate.epoch))
        assert set(h.gate._presenting) == {'parent', 'dependent'}
        assert all(key in h.context.messages[-1]['content'] for key in ('parent', 'dependent'))
        assert len(h.context.messages) == 1
        h.finish_reply('The complete requested finding.')
        assert not h.gate._presenting and not h.context.messages
        assert [(e, d['count']) for e, d in trace.events
                if e == 'background_result_response_finished'] == [
                    ('background_result_response_finished', 2)]
    asyncio.run(check())


def test_empty_delivery_is_not_completion_and_does_not_spin():
    async def check():
        trace = Trace()
        h = GateHarness(trace=trace)
        call = h.call('result')
        h.gate.finish('result')
        h.gate.result_in_context('result', call)
        await h.settle()
        h.queued.clear()
        await h.deliver(_DeliveryCheck(h.gate.epoch))
        h.gate.observe(h.gate.llm, LLMFullResponseEndFrame())
        await h.settle()
        assert set(h.gate._presenting) == {'result'}
        assert not h.queued
        assert not any(e == 'background_result_response_finished' for e, _ in trace.events)
        h.gate.user_started()
        assert set(h.gate._ready) == {'result'}
        assert not h.context.messages
    asyncio.run(check())


def test_native_started_frame_fences_gap_before_foreground_handler_begins():
    async def check():
        h = GateHarness()
        search = h.call('search', epoch=-1)
        h.gate.finish('search')
        h.gate.result_in_context('search', search)
        await h.deliver(LLMContextFrame(h.context))
        h.gate.observe(h.gate.llm, FunctionCallsStartedFrame(
            function_calls=[SimpleNamespace(tool_call_id='study')]))
        h.gate.observe(h.gate.llm, LLMFullResponseEndFrame())
        await h.deliver(_DeliveryCheck(h.gate.epoch))
        assert set(h.gate._ready) == {'search'}
        assert not h.gate._presenting and not h.context.messages
        study = h.call('study')
        h.gate.finish('study')
        h.gate.result_in_context('study', study)
        await h.deliver(_DeliveryCheck(h.gate.epoch))
        assert set(h.gate._presenting) == {'study'}
        assert set(h.gate._ready) == {'search'}
    asyncio.run(check())
