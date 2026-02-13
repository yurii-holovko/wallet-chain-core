"""Tests for strategy.priority_queue — SignalPriorityQueue."""

import time

from strategy.priority_queue import PriorityQueueConfig, SignalPriorityQueue
from strategy.signal import Direction, Signal

# ── helpers ────────────────────────────────────────────────────────


def _sig(pair="ETH/USDT", score=70.0, ttl=30.0, **kw) -> Signal:
    """Create a minimal Signal for queue testing."""
    return Signal.create(
        pair=pair,
        direction=Direction.BUY_CEX_SELL_DEX,
        cex_price=2000.0,
        dex_price=2010.0,
        spread_bps=50.0,
        size=0.1,
        expected_gross_pnl=1.0,
        expected_fees=0.2,
        expected_net_pnl=0.8,
        score=score,
        expiry=time.time() + ttl,
        inventory_ok=True,
        within_limits=True,
        **kw,
    )


# ── basic push / drain ────────────────────────────────────────────


class TestBasicOperations:
    def test_push_and_drain_single(self):
        pq = SignalPriorityQueue()
        sig = _sig(score=80)
        assert pq.push(sig) is True
        assert pq.size == 1

        drained = list(pq.drain())
        assert len(drained) == 1
        assert drained[0].signal_id == sig.signal_id
        assert pq.is_empty

    def test_drain_yields_highest_score_first(self):
        pq = SignalPriorityQueue(PriorityQueueConfig(min_score=0, max_per_pair=10))
        s1 = _sig(score=60)
        s2 = _sig(score=90)
        s3 = _sig(score=75)
        pq.push(s1)
        pq.push(s2)
        pq.push(s3)
        scores = [s.score for s in pq.drain()]
        assert scores == [90, 75, 60]

    def test_empty_drain(self):
        pq = SignalPriorityQueue()
        assert list(pq.drain()) == []


class TestDeduplication:
    def test_duplicate_signal_rejected(self):
        pq = SignalPriorityQueue()
        sig = _sig(score=80)
        assert pq.push(sig) is True
        assert pq.push(sig) is False  # same signal_id
        assert pq.size == 1

    def test_different_signals_accepted(self):
        pq = SignalPriorityQueue()
        s1 = _sig(score=80)
        s2 = _sig(score=70)
        assert pq.push(s1) is True
        assert pq.push(s2) is True
        assert pq.size == 2


class TestMaxDepth:
    def test_evicts_lowest_when_over_capacity(self):
        cfg = PriorityQueueConfig(max_depth=2, min_score=0, max_per_pair=10)
        pq = SignalPriorityQueue(config=cfg)
        s1 = _sig(score=80)
        s2 = _sig(score=90)
        s3 = _sig(score=60)  # this should be evicted
        pq.push(s1)
        pq.push(s2)
        pq.push(s3)
        assert pq.size == 2
        drained = list(pq.drain())
        ids = {s.signal_id for s in drained}
        assert s1.signal_id in ids
        assert s2.signal_id in ids
        assert s3.signal_id not in ids

    def test_stats_track_drops(self):
        cfg = PriorityQueueConfig(max_depth=1, min_score=0)
        pq = SignalPriorityQueue(config=cfg)
        pq.push(_sig(score=90))
        pq.push(_sig(score=50))
        assert pq.stats["total_dropped"] == 1


class TestPerPairLimit:
    def test_only_one_signal_per_pair(self):
        cfg = PriorityQueueConfig(max_per_pair=1, min_score=0)
        pq = SignalPriorityQueue(config=cfg)
        s1 = _sig(pair="ETH/USDT", score=90)
        s2 = _sig(pair="ETH/USDT", score=80)
        s3 = _sig(pair="BTC/USDT", score=70)
        pq.push(s1)
        pq.push(s2)
        pq.push(s3)
        drained = list(pq.drain())
        pairs = [s.pair for s in drained]
        assert pairs.count("ETH/USDT") == 1
        assert pairs.count("BTC/USDT") == 1

    def test_max_per_pair_two(self):
        cfg = PriorityQueueConfig(max_per_pair=2, min_score=0)
        pq = SignalPriorityQueue(config=cfg)
        pq.push(_sig(pair="ETH/USDT", score=90))
        pq.push(_sig(pair="ETH/USDT", score=80))
        pq.push(_sig(pair="ETH/USDT", score=70))
        drained = list(pq.drain())
        assert len(drained) == 2
        assert drained[0].score == 90
        assert drained[1].score == 80


class TestExpiry:
    def test_expired_signal_skipped(self):
        pq = SignalPriorityQueue(PriorityQueueConfig(min_score=0))
        sig = _sig(score=90, ttl=-1)  # already expired
        pq.push(sig)
        drained = list(pq.drain())
        assert len(drained) == 0


class TestScoreDecay:
    def test_decay_fn_applied_during_drain(self):
        def decay(signal):
            return signal.score * 0.5

        cfg = PriorityQueueConfig(score_decay=True, min_score=0)
        pq = SignalPriorityQueue(config=cfg, decay_fn=decay)
        sig = _sig(score=80)
        pq.push(sig)
        drained = list(pq.drain())
        assert len(drained) == 1
        assert drained[0].score == 40.0

    def test_decay_below_min_score_dropped(self):
        def decay(signal):
            return signal.score * 0.1  # drops to 8.0

        cfg = PriorityQueueConfig(score_decay=True, min_score=50)
        pq = SignalPriorityQueue(config=cfg, decay_fn=decay)
        sig = _sig(score=80)
        pq.push(sig)
        drained = list(pq.drain())
        assert len(drained) == 0

    def test_no_decay_when_disabled(self):
        call_count = {"n": 0}

        def decay(signal):
            call_count["n"] += 1
            return signal.score

        cfg = PriorityQueueConfig(score_decay=False, min_score=0)
        pq = SignalPriorityQueue(config=cfg, decay_fn=decay)
        pq.push(_sig(score=80))
        list(pq.drain())
        assert call_count["n"] == 0


class TestPeekAndClear:
    def test_peek_returns_highest(self):
        pq = SignalPriorityQueue()
        s1 = _sig(score=60)
        s2 = _sig(score=90)
        pq.push(s1)
        pq.push(s2)
        top = pq.peek()
        assert top.score == 90
        assert pq.size == 2  # not removed

    def test_peek_empty(self):
        pq = SignalPriorityQueue()
        assert pq.peek() is None

    def test_clear(self):
        pq = SignalPriorityQueue()
        pq.push(_sig(score=80))
        pq.push(_sig(score=70))
        pq.clear()
        assert pq.is_empty
        assert pq.size == 0


class TestStats:
    def test_stats_accumulate(self):
        cfg = PriorityQueueConfig(min_score=0, max_per_pair=10)
        pq = SignalPriorityQueue(config=cfg)
        pq.push(_sig(score=80))
        pq.push(_sig(score=70))
        assert pq.stats["total_pushed"] == 2
        list(pq.drain())
        assert pq.stats["total_yielded"] == 2
        assert pq.stats["queued"] == 0
