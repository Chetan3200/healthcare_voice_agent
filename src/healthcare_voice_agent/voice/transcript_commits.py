"""Bounded, provider-independent bookkeeping for explicitly committed audio.

No sockets, audio, timers, or framework imports. Completed text is withheld
until all currently reserved commits are complete and the speaker is quiet.
"""

from collections import deque
from dataclasses import dataclass


class TranscriptOrderError(ValueError):
    """The adapter cannot safely establish transcript order/completeness."""


@dataclass(frozen=True)
class Completion:
    text: str
    received_at: str


@dataclass
class _Commit:
    sequence: int
    generation: int
    sent: bool = False
    item_id: str | None = None
    completion: Completion | None = None


@dataclass(frozen=True)
class CompletedBatch:
    text: str
    generation: int
    sequences: tuple[int, ...]
    item_ids: tuple[str, ...]
    received_at: tuple[str, ...]


class TranscriptCommits:
    def __init__(self, *, max_pending: int = 64, recent_limit: int = 512):
        if max_pending < 1 or recent_limit < 1:
            raise ValueError("Transcript tracking limits must be positive")
        self.active = True
        self.speaking = False
        self.generation = 0
        self._sequence = 0
        self._pending: list[_Commit] = []
        self._items: dict[str, _Commit] = {}
        self._early: dict[str, Completion] = {}
        self._previous_item: str | None = None
        self._recent: set[str] = set()
        self._recent_order: deque[str] = deque()
        self._max_pending = max_pending
        self._recent_limit = recent_limit

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def speech_started(self):
        self.generation += 1
        self.speaking = True

    def reserve_commit(self) -> int:
        """Reserve before any await in VAD-stop handling, not after sending."""
        self.speaking = False
        if not self.active or len(self._pending) >= self._max_pending:
            raise TranscriptOrderError("Audio commit tracking is unavailable or full")
        self._sequence += 1
        self._pending.append(_Commit(self._sequence, self.generation))
        return self._sequence

    def mark_sent(self, sequence: int):
        commit = next((c for c in self._pending if not c.sent), None)
        if commit is None or commit.sequence != sequence:
            raise TranscriptOrderError("Audio commits were sent out of order")
        commit.sent = True

    def acknowledge(self, item_id: str, previous_item_id: str | None):
        if not isinstance(item_id, str) or not item_id:
            raise TranscriptOrderError("Audio commit has no valid item ID")
        if item_id in self._recent or item_id in self._items:
            return  # Duplicate acknowledgement, not a second audio segment.
        commit = next((c for c in self._pending if c.sent and c.item_id is None), None)
        if commit is None or previous_item_id != self._previous_item:
            raise TranscriptOrderError("Audio commit acknowledgement order is ambiguous")
        commit.item_id = item_id
        commit.completion = self._early.pop(item_id, None)
        self._items[item_id] = commit
        self._previous_item = item_id
        if len(self._early) > sum(c.sent and c.item_id is None for c in self._pending):
            raise TranscriptOrderError("A transcript cannot be matched to a sent commit")

    def complete(self, item_id: str, text: str, received_at: str) -> bool:
        """Accept a terminal result once, including completion-before-ACK."""
        if not isinstance(item_id, str) or not item_id or not isinstance(text, str):
            raise TranscriptOrderError("Transcription completion is malformed")
        if self.is_complete(item_id):
            return False
        completion = Completion(text, received_at)
        commit = self._items.get(item_id)
        if commit is not None:
            commit.completion = completion
        else:
            unacknowledged = sum(c.sent and c.item_id is None for c in self._pending)
            if len(self._early) >= unacknowledged:
                raise TranscriptOrderError("Transcription has no matching audio commit")
            self._early[item_id] = completion
        return True

    def is_complete(self, item_id: str) -> bool:
        return (
            item_id in self._recent or item_id in self._early
            or (item_id in self._items and self._items[item_id].completion is not None)
        )

    def item_generation(self, item_id: str) -> int | None:
        commit = self._items.get(item_id)
        return commit.generation if commit is not None else None

    def take_ready(self) -> CompletedBatch | None:
        if (
            not self.active or self.speaking or not self._pending or self._early
            or self._pending[-1].generation != self.generation
            or any(c.item_id is None or c.completion is None for c in self._pending)
        ):
            return None
        commits = self._pending
        batch = CompletedBatch(
            text=" ".join(c.completion.text for c in commits if c.completion.text.strip()),
            generation=self.generation,
            sequences=tuple(c.sequence for c in commits),
            item_ids=tuple(c.item_id for c in commits),
            received_at=tuple(c.completion.received_at for c in commits),
        )
        self._pending = []
        for commit in commits:
            self._items.pop(commit.item_id)
            self._recent.add(commit.item_id)
            self._recent_order.append(commit.item_id)
        while len(self._recent_order) > self._recent_limit:
            self._recent.discard(self._recent_order.popleft())
        return batch

    def is_current(self, batch: CompletedBatch) -> bool:
        return (
            self.active and not self.speaking and not self._pending
            and batch.generation == self.generation
        )

    def close(self):
        self.active = False
        self.speaking = False
        self._pending.clear()
        self._items.clear()
        self._early.clear()
        self._recent.clear()
        self._recent_order.clear()
