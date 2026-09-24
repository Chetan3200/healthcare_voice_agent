"""Offline unit tests for explicit audio transcript commit ordering."""

from itertools import permutations

import pytest

from healthcare_voice_agent.voice.transcript_commits import (
    TranscriptCommits,
    TranscriptOrderError,
)


def _acknowledged_commit(tracker, item_id="item-1", text="hello", received_at="t-1"):
    sequence = tracker.reserve_commit()
    tracker.mark_sent(sequence)
    tracker.acknowledge(item_id, None)
    assert tracker.complete(item_id, text, received_at) is True
    return sequence


def test_single_complete_commit_releases_exact_batch():
    tracker = TranscriptCommits()

    assert _acknowledged_commit(tracker) == 1
    batch = tracker.take_ready()

    assert batch is not None
    assert batch.text == "hello"
    assert batch.generation == 0
    assert batch.sequences == (1,)
    assert batch.item_ids == ("item-1",)
    assert batch.received_at == ("t-1",)
    assert tracker.pending_count == 0


@pytest.mark.parametrize("completion_order", tuple(permutations(("item-1", "item-2", "item-3"))))
def test_all_completion_orders_release_one_ordered_combined_batch(completion_order):
    tracker = TranscriptCommits()
    sequences = [tracker.reserve_commit() for _ in range(3)]
    for sequence in sequences:
        tracker.mark_sent(sequence)
    tracker.acknowledge("item-1", None)
    tracker.acknowledge("item-2", "item-1")
    tracker.acknowledge("item-3", "item-2")

    values = {
        "item-1": ("first", "time-1"),
        "item-2": ("second", "time-2"),
        "item-3": ("third", "time-3"),
    }
    for item_id in completion_order:
        text, received_at = values[item_id]
        assert tracker.complete(item_id, text, received_at) is True

    batch = tracker.take_ready()
    assert batch is not None
    assert batch.text == "first second third"
    assert batch.sequences == (1, 2, 3)
    assert batch.item_ids == ("item-1", "item-2", "item-3")
    assert batch.received_at == ("time-1", "time-2", "time-3")
    assert tracker.take_ready() is None


def test_completion_before_acknowledgement_is_attached_and_released():
    tracker = TranscriptCommits()
    sequence = tracker.reserve_commit()
    tracker.mark_sent(sequence)

    assert tracker.complete("item-1", "early result", "time-1") is True
    assert tracker.take_ready() is None
    tracker.acknowledge("item-1", None)

    batch = tracker.take_ready()
    assert batch is not None
    assert batch.text == "early result"
    assert batch.item_ids == ("item-1",)


def test_ready_text_is_withheld_while_speaking_or_any_commit_is_unresolved():
    tracker = TranscriptCommits()
    _acknowledged_commit(tracker)
    tracker.speech_started()
    assert tracker.take_ready() is None

    sequence = tracker.reserve_commit()
    assert tracker.take_ready() is None  # reserved but unsent
    tracker.mark_sent(sequence)
    assert tracker.take_ready() is None  # sent but unacknowledged
    tracker.acknowledge("item-2", "item-1")
    assert tracker.take_ready() is None  # acknowledged but incomplete
    tracker.complete("item-2", "second", "time-2")

    batch = tracker.take_ready()
    assert batch is not None
    assert batch.text == "hello second"
    assert batch.generation == 1


def test_speech_resume_invalidates_readiness_until_its_own_commit_completes():
    tracker = TranscriptCommits()
    _acknowledged_commit(tracker, "item-1", "first", "time-1")

    tracker.speech_started()
    assert tracker.take_ready() is None
    sequence = tracker.reserve_commit()
    tracker.mark_sent(sequence)
    tracker.acknowledge("item-2", "item-1")
    assert tracker.take_ready() is None
    tracker.complete("item-2", "second", "time-2")

    batch = tracker.take_ready()
    assert batch is not None
    assert batch.text == "first second"
    assert batch.generation == 1


def test_duplicate_ack_completion_and_post_release_delta_complete_do_not_repeat_text():
    tracker = TranscriptCommits()
    sequence = tracker.reserve_commit()
    tracker.mark_sent(sequence)
    tracker.acknowledge("item-1", None)
    tracker.acknowledge("item-1", None)
    assert tracker.complete("item-1", "once", "time-1") is True
    assert tracker.complete("item-1", "replacement", "time-2") is False

    batch = tracker.take_ready()
    assert batch is not None
    assert batch.text == "once"
    assert tracker.complete("item-1", "late duplicate delta", "time-3") is False
    assert tracker.take_ready() is None


def test_empty_completion_resolves_without_inventing_text():
    tracker = TranscriptCommits()
    _acknowledged_commit(tracker, text="", received_at="time-1")

    batch = tracker.take_ready()
    assert batch is not None
    assert batch.text == ""
    assert batch.received_at == ("time-1",)


def test_wrong_previous_item_id_and_unmatched_early_completion_raise_order_errors():
    tracker = TranscriptCommits()
    first = tracker.reserve_commit()
    second = tracker.reserve_commit()
    tracker.mark_sent(first)
    tracker.mark_sent(second)
    tracker.acknowledge("item-1", None)

    with pytest.raises(TranscriptOrderError):
        tracker.acknowledge("item-2", "not-item-1")

    # Completion-before-ACK is permitted while a sent commit still awaits an ACK,
    # but provisional text must not release before that ACK validates the item ID.
    assert tracker.complete("unknown", "provisional", "time-x") is True
    assert tracker.take_ready() is None
    with pytest.raises(TranscriptOrderError):
        tracker.acknowledge("item-2", "item-1")


def test_pending_and_retired_caches_are_bounded():
    tracker = TranscriptCommits(max_pending=2, recent_limit=2)
    tracker.reserve_commit()
    tracker.reserve_commit()
    with pytest.raises(TranscriptOrderError):
        tracker.reserve_commit()

    tracker = TranscriptCommits(recent_limit=2)
    previous = None
    for number in range(1, 4):
        item_id = f"item-{number}"
        sequence = tracker.reserve_commit()
        tracker.mark_sent(sequence)
        tracker.acknowledge(item_id, previous)
        tracker.complete(item_id, f"text-{number}", f"time-{number}")
        assert tracker.take_ready() is not None
        previous = item_id

    assert tracker._recent == {"item-2", "item-3"}
    assert tracker.is_complete("item-1") is False
    assert tracker.complete("item-3", "duplicate", "time-4") is False
    with pytest.raises(TranscriptOrderError):
        tracker.complete("item-1", "evicted orphan", "time-4")


def test_reserving_before_await_blocks_already_ready_text():
    tracker = TranscriptCommits()
    _acknowledged_commit(tracker, "item-1", "first", "time-1")

    # VAD reserves synchronously before awaiting the outbound send operation.
    sequence = tracker.reserve_commit()
    assert tracker.take_ready() is None
    tracker.mark_sent(sequence)
    tracker.acknowledge("item-2", "item-1")
    tracker.complete("item-2", "second", "time-2")

    batch = tracker.take_ready()
    assert batch is not None
    assert batch.text == "first second"


def test_close_discards_buffered_state_and_prevents_future_release():
    tracker = TranscriptCommits()
    _acknowledged_commit(tracker)
    tracker.close()

    assert tracker.pending_count == 0
    assert tracker.take_ready() is None
    assert tracker.is_complete("item-1") is False
    with pytest.raises(TranscriptOrderError):
        tracker.reserve_commit()


def test_is_current_is_invalidated_by_new_speech_or_new_reservation():
    tracker = TranscriptCommits()
    _acknowledged_commit(tracker)
    batch = tracker.take_ready()
    assert batch is not None
    assert tracker.is_current(batch) is True

    tracker.speech_started()
    assert tracker.is_current(batch) is False
    sequence = tracker.reserve_commit()
    assert tracker.is_current(batch) is False
    tracker.mark_sent(sequence)
    tracker.acknowledge("item-2", "item-1")
    tracker.complete("item-2", "second", "time-2")
    assert tracker.take_ready() is not None


def test_multiple_early_completions_are_attached_by_later_ordered_acks():
    tracker = TranscriptCommits()
    for _ in range(3):
        tracker.speech_started()
        tracker.mark_sent(tracker.reserve_commit())
    tracker.complete("three", "third", "t3")
    tracker.complete("one", "first", "t1")
    tracker.complete("two", "second", "t2")
    assert tracker.take_ready() is None
    tracker.acknowledge("one", None)
    tracker.acknowledge("two", "one")
    assert tracker.take_ready() is None
    tracker.acknowledge("three", "two")
    batch = tracker.take_ready()
    assert batch.text == "first second third"
    assert batch.item_ids == ("one", "two", "three")


def test_duplicate_ack_is_idempotent_even_with_conflicting_duplicate_metadata():
    tracker = TranscriptCommits()
    tracker.speech_started()
    tracker.mark_sent(tracker.reserve_commit())
    tracker.acknowledge("one", None)
    tracker.acknowledge("one", "conflicting-duplicate")
    tracker.complete("one", "once", "t1")
    assert tracker.take_ready().text == "once"
    tracker.acknowledge("one", None)
    assert tracker.take_ready() is None
