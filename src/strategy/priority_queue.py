"""
Signal Priority Queue — rank and batch-execute multiple signals.

Collects signals from all pairs in a tick, scores them, then yields
them in descending priority order.  Supports:

  * Max-queue-depth cap (drop lowest-scoring signals)
  * Per-pair concurrency limit (don't execute 2 signals for same pair)
  * Score-decay: re-scores stale signals before yielding
  * Deduplication: rejects signals with same signal_id
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from strategy.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class PriorityQueueConfig:
    """Tuning knobs for the signal priority queue."""

    max_depth: int = 50  # max signals held at once
    max_per_pair: int = 1  # max concurrent signals per pair
    score_decay: bool = True  # re-apply decay before yielding
    min_score: float = 55.0  # drop signals below this after decay


class _ScoredEntry:
    """
    Wrapper so signals can live in a min-heap ordered by *negative* score
    (heapq is a min-heap; we want highest score first).
    """

    __slots__ = ("neg_score", "insert_order", "signal")

    def __init__(self, signal: Signal, insert_order: int):
        self.neg_score = -signal.score
        self.insert_order = insert_order
        self.signal = signal

    def __lt__(self, other: "_ScoredEntry") -> bool:
        # Lower neg_score = higher actual score = higher priority
        if self.neg_score != other.neg_score:
            return self.neg_score < other.neg_score
        return self.insert_order < other.insert_order


class SignalPriorityQueue:
    """
    Priority queue that collects scored signals and yields them
    in best-first order.

    Usage::

        pq = SignalPriorityQueue()

        # Collect phase (inside tick)
        for pair in pairs:
            signal = generator.generate(pair, size)
            if signal:
                signal.score = scorer.score(signal, skews)
                pq.push(signal)

        # Execute phase — yields highest-score first
        for signal in pq.drain():
            ctx = await executor.execute(signal)
    """

    def __init__(
        self,
        config: Optional[PriorityQueueConfig] = None,
        decay_fn=None,
    ):
        self.config = config or PriorityQueueConfig()
        self._heap: list[_ScoredEntry] = []
        self._counter = 0  # tie-breaker for equal scores
        self._seen_ids: set[str] = set()
        self._decay_fn = decay_fn  # Optional: scorer.apply_decay

        # Stats
        self._total_pushed = 0
        self._total_dropped = 0
        self._total_yielded = 0

    # ── public API ─────────────────────────────────────────────

    def push(self, signal: Signal) -> bool:
        """
        Add a signal to the queue.

        Returns True if accepted, False if rejected (duplicate / overflow).
        """
        if signal.signal_id in self._seen_ids:
            logger.debug("PQ: duplicate signal %s", signal.signal_id)
            return False

        entry = _ScoredEntry(signal, self._counter)
        self._counter += 1
        self._total_pushed += 1

        heapq.heappush(self._heap, entry)
        self._seen_ids.add(signal.signal_id)

        # Evict lowest-priority if over capacity
        while len(self._heap) > self.config.max_depth:
            evicted = self._evict_lowest()
            if evicted:
                self._seen_ids.discard(evicted.signal_id)
                self._total_dropped += 1
                logger.debug(
                    "PQ: evicted %s (score=%.1f)", evicted.signal_id, evicted.score
                )

        return True

    def drain(self) -> Iterator[Signal]:
        """
        Yield signals in descending score order.

        Applies score-decay and per-pair limits.  The queue is empty
        after draining.
        """
        pair_counts: dict[str, int] = {}

        while self._heap:
            entry = heapq.heappop(self._heap)
            signal = entry.signal

            # Per-pair concurrency limit
            pair = signal.pair
            if pair_counts.get(pair, 0) >= self.config.max_per_pair:
                logger.debug("PQ: pair limit reached for %s", pair)
                continue

            # Expired?
            if time.time() >= signal.expiry:
                logger.debug("PQ: signal %s expired", signal.signal_id)
                continue

            # Score decay
            if self.config.score_decay and self._decay_fn:
                signal.score = self._decay_fn(signal)

            # Min-score gate after decay
            if signal.score < self.config.min_score:
                logger.debug(
                    "PQ: signal %s below min after decay (%.1f)",
                    signal.signal_id,
                    signal.score,
                )
                continue

            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            self._total_yielded += 1
            yield signal

        # Clear bookkeeping
        self._seen_ids.clear()

    def peek(self) -> Optional[Signal]:
        """Return the highest-priority signal without removing it."""
        if not self._heap:
            return None
        return self._heap[0].signal

    def clear(self) -> None:
        """Drop all queued signals."""
        self._heap.clear()
        self._seen_ids.clear()

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def is_empty(self) -> bool:
        return len(self._heap) == 0

    @property
    def stats(self) -> dict:
        return {
            "queued": len(self._heap),
            "total_pushed": self._total_pushed,
            "total_dropped": self._total_dropped,
            "total_yielded": self._total_yielded,
        }

    # ── internals ──────────────────────────────────────────────

    def _evict_lowest(self) -> Optional[Signal]:
        """Remove and return the lowest-priority signal."""
        if not self._heap:
            return None
        # heapq is a min-heap by neg_score, so the *largest* neg_score
        # (= lowest actual score) is what we want to remove.
        # We need to find it — heapq doesn't support pop-max efficiently,
        # so we rebuild after removing the worst.
        worst_idx = 0
        for i, entry in enumerate(self._heap):
            if entry.neg_score > self._heap[worst_idx].neg_score:
                worst_idx = i
        worst = self._heap[worst_idx]
        self._heap[worst_idx] = self._heap[-1]
        self._heap.pop()
        if self._heap:
            heapq.heapify(self._heap)
        return worst.signal
