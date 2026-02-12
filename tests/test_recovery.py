"""Tests for executor.recovery — CircuitBreaker, ReplayProtection,
FailureClassifier, RecoveryManager."""

import time

from executor.recovery import (
    CircuitBreaker,
    CircuitBreakerConfig,
    FailureCategory,
    FailureClassifier,
    RecoveryConfig,
    RecoveryManager,
    ReplayConfig,
    ReplayProtection,
)
from strategy.signal import Direction, Signal

# ── helpers ────────────────────────────────────────────────────────


def _make_signal(**overrides):
    defaults = dict(
        pair="ETH/USDT",
        direction=Direction.BUY_CEX_SELL_DEX,
        cex_price=2000.0,
        dex_price=2010.0,
        spread_bps=50.0,
        size=1.0,
        expected_gross_pnl=10.0,
        expected_fees=3.0,
        expected_net_pnl=7.0,
        score=80.0,
        expiry=time.time() + 30,
        inventory_ok=True,
        within_limits=True,
    )
    defaults.update(overrides)
    return Signal.create(**defaults)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Failure Classifier                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestFailureClassifier:
    def test_timeout_is_transient(self):
        assert (
            FailureClassifier.classify("connection timeout")
            == FailureCategory.TRANSIENT
        )

    def test_rate_limit_429(self):
        assert (
            FailureClassifier.classify("HTTP 429 too many requests")
            == FailureCategory.RATE_LIMIT
        )

    def test_insufficient_funds_permanent(self):
        assert (
            FailureClassifier.classify("insufficient balance")
            == FailureCategory.PERMANENT
        )

    def test_revert_permanent(self):
        assert (
            FailureClassifier.classify("execution reverted")
            == FailureCategory.PERMANENT
        )

    def test_nonce_too_low_permanent(self):
        assert FailureClassifier.classify("nonce too low") == FailureCategory.PERMANENT

    def test_dns_network(self):
        assert (
            FailureClassifier.classify("DNS resolution failed")
            == FailureCategory.NETWORK
        )

    def test_econnrefused_network(self):
        assert (
            FailureClassifier.classify("ECONNREFUSED 127.0.0.1")
            == FailureCategory.NETWORK
        )

    def test_none_is_unknown(self):
        assert FailureClassifier.classify(None) == FailureCategory.UNKNOWN

    def test_empty_string_unknown(self):
        assert FailureClassifier.classify("") == FailureCategory.UNKNOWN

    def test_gibberish_unknown(self):
        assert FailureClassifier.classify("xyzzy foobar") == FailureCategory.UNKNOWN

    def test_retriable_categories(self):
        assert FailureClassifier.is_retriable(FailureCategory.TRANSIENT)
        assert FailureClassifier.is_retriable(FailureCategory.RATE_LIMIT)
        assert FailureClassifier.is_retriable(FailureCategory.NETWORK)

    def test_non_retriable_categories(self):
        assert not FailureClassifier.is_retriable(FailureCategory.PERMANENT)
        assert not FailureClassifier.is_retriable(FailureCategory.UNKNOWN)

    def test_case_insensitive(self):
        assert FailureClassifier.classify("TIMEOUT") == FailureCategory.TRANSIENT
        assert (
            FailureClassifier.classify("Rate Limit exceeded")
            == FailureCategory.RATE_LIMIT
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Circuit Breaker — Single Breaker                               ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestCircuitBreakerConfig:
    def test_defaults(self):
        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 3
        assert cfg.window_seconds == 300.0
        assert cfg.cooldown_seconds == 600.0
        assert cfg.max_drawdown_usd == 50.0
        assert cfg.half_open_after_pct == 0.8
        assert cfg.success_decay == 1
        assert cfg.per_pair is True


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert not cb.is_open()
        assert cb.allows_trade()

    def test_trips_after_threshold(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open()
        cb.record_failure()
        assert cb.is_open()

    def test_resets_after_cooldown(self):
        cfg = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1)
        cb = CircuitBreaker(cfg)
        cb.record_failure()
        assert cb.is_open()
        time.sleep(0.15)
        assert not cb.is_open()

    def test_time_until_reset_zero_when_closed(self):
        cb = CircuitBreaker()
        assert cb.time_until_reset() == 0.0

    def test_time_until_reset_positive_when_open(self):
        cfg = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=100)
        cb = CircuitBreaker(cfg)
        cb.record_failure()
        assert cb.time_until_reset() > 0

    def test_manual_trip(self):
        cb = CircuitBreaker()
        cb.trip()
        assert cb.is_open()

    # ── PnL-based trip ────────────────────────────────────────

    def test_pnl_drawdown_trips(self):
        cfg = CircuitBreakerConfig(
            failure_threshold=100,  # won't trip by count
            max_drawdown_usd=10.0,
        )
        cb = CircuitBreaker(cfg)
        # One failure with -$11 PnL should trip
        cb.record_failure(pnl=-11.0)
        assert cb.is_open()

    def test_pnl_drawdown_accumulates(self):
        cfg = CircuitBreakerConfig(
            failure_threshold=100,
            max_drawdown_usd=10.0,
        )
        cb = CircuitBreaker(cfg)
        cb.record_failure(pnl=-4.0)
        assert not cb.is_open()
        cb.record_failure(pnl=-4.0)
        assert not cb.is_open()
        cb.record_failure(pnl=-4.0)  # cumulative = -12
        assert cb.is_open()

    # ── success decay ─────────────────────────────────────────

    def test_success_decays_failures(self):
        cfg = CircuitBreakerConfig(failure_threshold=3, success_decay=2)
        cb = CircuitBreaker(cfg)
        cb.record_failure()
        cb.record_failure()
        # 2 failures, then 1 success decays 2 → 0 failures
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        # Should still be at 2, not tripped
        assert not cb.is_open()

    # ── per-pair isolation ────────────────────────────────────

    def test_per_pair_isolation(self):
        # Global threshold high so only pair breaker trips
        cfg = CircuitBreakerConfig(failure_threshold=2, per_pair=True)
        cb = CircuitBreaker(cfg)
        # Only record to pair breaker, not global
        cb._pair_breaker("ETH/USDT").record_failure()
        cb._pair_breaker("ETH/USDT").record_failure()
        # ETH/USDT pair breaker is open
        assert cb.is_open("ETH/USDT")
        # BTC/USDT pair breaker is still closed
        assert not cb._pair_breaker("BTC/USDT").is_open()
        # Global was not touched → still closed
        assert not cb._global.is_open()

    def test_per_pair_independent_reset(self):
        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            cooldown_seconds=0.1,
            per_pair=True,
        )
        cb = CircuitBreaker(cfg)
        cb.record_failure("ETH/USDT")
        assert cb.is_open("ETH/USDT")
        time.sleep(0.15)
        assert not cb.is_open("ETH/USDT")

    def test_per_pair_trip_does_not_affect_other_pairs(self):
        cfg = CircuitBreakerConfig(failure_threshold=100, per_pair=True)
        cb = CircuitBreaker(cfg)
        # Trip only the pair breaker, not global
        cb._pair_breaker("ETH/USDT")._trip()
        assert cb._pair_breaker("ETH/USDT").is_open()
        assert not cb._pair_breaker("BTC/USDT").is_open()
        assert not cb._global.is_open()

    # ── half-open probe ───────────────────────────────────────

    def test_half_open_transition(self):
        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            cooldown_seconds=0.2,
            half_open_after_pct=0.5,  # half-open at 50% = 0.1s
        )
        cb = CircuitBreaker(cfg)
        cb.record_failure()
        assert cb.is_open()

        # Wait for half-open
        time.sleep(0.12)
        assert cb.allows_trade()  # half-open allows probe

    def test_half_open_probe_success_closes(self):
        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            cooldown_seconds=0.5,
            half_open_after_pct=0.1,  # enter half-open quickly
        )
        cb = CircuitBreaker(cfg)
        cb.record_failure()
        time.sleep(0.08)  # enter half-open
        assert cb.allows_trade()

        cb.record_success()
        assert not cb.is_open()
        assert cb.allows_trade()

    # ── permanent errors count double ─────────────────────────

    def test_permanent_error_counts_double(self):
        cfg = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(cfg)
        # One permanent error = 2 counts
        cb.record_failure(category=FailureCategory.PERMANENT)
        assert not cb.is_open()
        # One more transient = 1 count → total 3 → trip
        cb.record_failure(category=FailureCategory.TRANSIENT)
        assert cb.is_open()

    # ── snapshot ──────────────────────────────────────────────

    def test_snapshot_global(self):
        cb = CircuitBreaker()
        snap = cb.snapshot()
        assert "global" in snap
        assert snap["global"]["state"] == "closed"
        assert snap["global"]["failures"] == 0

    def test_snapshot_pair(self):
        cfg = CircuitBreakerConfig(failure_threshold=1, per_pair=True)
        cb = CircuitBreaker(cfg)
        cb.record_failure("ETH/USDT")
        snap = cb.snapshot("ETH/USDT")
        assert "pair" in snap
        assert snap["pair"]["state"] == "open"


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Replay Protection                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestReplayProtection:
    def test_first_signal_not_duplicate(self):
        rp = ReplayProtection()
        sig = _make_signal()
        assert not rp.is_duplicate(sig)

    def test_same_signal_is_duplicate(self):
        rp = ReplayProtection()
        sig = _make_signal()
        rp.mark_executed(sig)
        assert rp.is_duplicate(sig)

    def test_different_signals_not_duplicate(self):
        rp = ReplayProtection()
        sig1 = _make_signal()
        sig2 = _make_signal()
        rp.mark_executed(sig1)
        assert not rp.is_duplicate(sig2)

    def test_expired_entries_cleaned(self):
        # Disable nonce check so only dedup TTL matters
        rp = ReplayProtection(
            ReplayConfig(ttl_seconds=0.1, max_age_seconds=999, nonce_check=False)
        )
        sig = _make_signal()
        rp.mark_executed(sig)
        assert rp.is_duplicate(sig)
        time.sleep(0.15)
        assert not rp.is_duplicate(sig)

    # ── max-age check ─────────────────────────────────────────

    def test_stale_signal_rejected(self):
        rp = ReplayProtection(ReplayConfig(max_age_seconds=1.0))
        sig = _make_signal()
        sig.timestamp = time.time() - 5  # 5 seconds old
        allowed, reason = rp.check(sig)
        assert not allowed
        assert "stale" in reason

    def test_fresh_signal_accepted(self):
        rp = ReplayProtection(ReplayConfig(max_age_seconds=30.0))
        sig = _make_signal()
        allowed, reason = rp.check(sig)
        assert allowed
        assert reason == "ok"

    # ── nonce monotonic check ─────────────────────────────────

    def test_nonce_rejects_older_signal(self):
        rp = ReplayProtection(ReplayConfig(nonce_check=True, max_age_seconds=999))
        sig1 = _make_signal()
        rp.mark_executed(sig1)

        # Create a signal with an older timestamp for the same pair
        sig2 = _make_signal()
        sig2.timestamp = sig1.timestamp - 1
        allowed, reason = rp.check(sig2)
        assert not allowed
        assert "nonce" in reason

    def test_nonce_accepts_newer_signal(self):
        rp = ReplayProtection(ReplayConfig(nonce_check=True, max_age_seconds=999))
        sig1 = _make_signal()
        rp.mark_executed(sig1)

        # New signal has a newer timestamp (created after)
        time.sleep(0.01)
        sig2 = _make_signal()
        allowed, reason = rp.check(sig2)
        assert allowed

    def test_nonce_check_disabled(self):
        rp = ReplayProtection(ReplayConfig(nonce_check=False, max_age_seconds=999))
        sig1 = _make_signal()
        rp.mark_executed(sig1)

        sig2 = _make_signal()
        sig2.timestamp = sig1.timestamp - 1
        allowed, reason = rp.check(sig2)
        assert allowed  # nonce check disabled → allowed

    # ── LRU eviction ──────────────────────────────────────────

    def test_lru_eviction(self):
        rp = ReplayProtection(
            ReplayConfig(max_entries=3, ttl_seconds=999, max_age_seconds=999)
        )
        signals = [_make_signal() for _ in range(5)]
        for sig in signals:
            rp.mark_executed(sig)

        # Only last 3 should be tracked
        assert len(rp._executed) == 3
        # First two should have been evicted
        assert signals[0].signal_id not in rp._executed
        assert signals[1].signal_id not in rp._executed
        assert signals[4].signal_id in rp._executed

    # ── audit log ─────────────────────────────────────────────

    def test_audit_log_records_checks(self):
        rp = ReplayProtection(ReplayConfig(max_age_seconds=999))
        sig = _make_signal()
        rp.check(sig)
        rp.mark_executed(sig)
        rp.check(sig)  # duplicate

        log = rp.audit_log
        assert len(log) == 2
        assert log[0]["accepted"] is True
        assert log[1]["accepted"] is False

    def test_audit_log_capped(self):
        rp = ReplayProtection(ReplayConfig(audit_log_size=5, max_age_seconds=999))
        for _ in range(10):
            sig = _make_signal()
            rp.check(sig)

        assert len(rp._audit) <= 5

    # ── stats ─────────────────────────────────────────────────

    def test_stats(self):
        rp = ReplayProtection(ReplayConfig(max_age_seconds=999))
        sig1 = _make_signal()
        sig2 = _make_signal()
        rp.check(sig1)  # accepted
        rp.mark_executed(sig1)
        rp.check(sig1)  # rejected (duplicate)
        rp.check(sig2)  # accepted

        stats = rp.stats
        assert stats["tracked_ids"] == 1
        assert stats["audit_accepted"] == 2
        assert stats["audit_rejected"] == 1


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Recovery Manager                                               ║
# ╚══════════════════════════════════════════════════════════════════╝


class TestRecoveryManager:
    def test_pre_flight_passes_clean_signal(self):
        rm = RecoveryManager()
        sig = _make_signal()
        allowed, reason = rm.pre_flight(sig)
        assert allowed
        assert reason == "ok"

    def test_pre_flight_blocks_when_cb_open(self):
        rm = RecoveryManager()
        rm.circuit_breaker.trip()
        sig = _make_signal()
        allowed, reason = rm.pre_flight(sig)
        assert not allowed
        assert "circuit breaker" in reason

    def test_pre_flight_blocks_duplicate(self):
        rm = RecoveryManager()
        sig = _make_signal()
        rm.record_outcome(sig, True)
        # Same signal again
        allowed, reason = rm.pre_flight(sig)
        assert not allowed
        assert "duplicate" in reason

    def test_pre_flight_blocks_stale_signal(self):
        cfg = RecoveryConfig(
            replay=ReplayConfig(max_age_seconds=1.0),
        )
        rm = RecoveryManager(cfg)
        sig = _make_signal()
        sig.timestamp = time.time() - 5
        allowed, reason = rm.pre_flight(sig)
        assert not allowed
        assert "stale" in reason

    # ── record_outcome ────────────────────────────────────────

    def test_record_success_updates_cb(self):
        rm = RecoveryManager()
        sig = _make_signal()
        rm.record_outcome(sig, True, pnl=5.0)
        snap = rm.snapshot()
        assert snap["circuit_breaker"]["global"]["cumulative_pnl"] == 5.0

    def test_record_failure_updates_cb(self):
        rm = RecoveryManager()
        sig = _make_signal()
        rm.record_outcome(sig, False, "timeout error", pnl=-2.0)
        snap = rm.snapshot()
        assert snap["circuit_breaker"]["global"]["failures"] >= 1
        assert snap["circuit_breaker"]["global"]["cumulative_pnl"] == -2.0

    def test_record_failure_classifies_error(self):
        rm = RecoveryManager()
        sig = _make_signal()
        rm.record_outcome(sig, False, "insufficient balance", pnl=-5.0)
        # Permanent error counts double → 2 failures from 1 call
        snap = rm.snapshot()
        assert snap["circuit_breaker"]["global"]["failures"] >= 2

    def test_record_outcome_marks_executed(self):
        rm = RecoveryManager()
        sig = _make_signal()
        rm.record_outcome(sig, True)
        # Signal should now be a duplicate
        allowed, reason = rm.pre_flight(sig)
        assert not allowed

    # ── snapshot ──────────────────────────────────────────────

    def test_snapshot_structure(self):
        rm = RecoveryManager()
        snap = rm.snapshot()
        assert "circuit_breaker" in snap
        assert "replay" in snap
        assert "recent_outcomes" in snap

    def test_snapshot_per_pair(self):
        cfg = RecoveryConfig(
            circuit_breaker=CircuitBreakerConfig(per_pair=True),
        )
        rm = RecoveryManager(cfg)
        sig = _make_signal(pair="ETH/USDT")
        rm.record_outcome(sig, False, "timeout", pnl=-1.0)

        snap = rm.snapshot("ETH/USDT")
        assert "pair" in snap["circuit_breaker"]

    def test_outcomes_log_capped(self):
        rm = RecoveryManager()
        for _ in range(30):
            sig = _make_signal()
            rm.record_outcome(sig, True, pnl=0.1)

        snap = rm.snapshot()
        assert len(snap["recent_outcomes"]) <= 20

    # ── integration: CB trips after classified failures ───────

    def test_cb_trips_after_enough_failures(self):
        cfg = RecoveryConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=3),
        )
        rm = RecoveryManager(cfg)
        for i in range(3):
            sig = _make_signal()
            rm.record_outcome(sig, False, "timeout", pnl=-1.0)

        sig = _make_signal()
        allowed, reason = rm.pre_flight(sig)
        assert not allowed
        assert "circuit breaker" in reason

    def test_cb_trips_on_drawdown(self):
        cfg = RecoveryConfig(
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=100,
                max_drawdown_usd=5.0,
            ),
        )
        rm = RecoveryManager(cfg)
        sig = _make_signal()
        rm.record_outcome(sig, False, "error", pnl=-6.0)

        sig2 = _make_signal()
        allowed, reason = rm.pre_flight(sig2)
        assert not allowed

    # ── half-open probe through RecoveryManager ───────────────

    def test_half_open_probe_allowed(self):
        cfg = RecoveryConfig(
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=1,
                cooldown_seconds=0.2,
                half_open_after_pct=0.1,
            ),
        )
        rm = RecoveryManager(cfg)
        sig = _make_signal()
        rm.record_outcome(sig, False, "timeout", pnl=-1.0)

        # CB is now open
        sig2 = _make_signal()
        allowed, _ = rm.pre_flight(sig2)
        assert not allowed

        # Wait for half-open
        time.sleep(0.05)
        sig3 = _make_signal()
        allowed, reason = rm.pre_flight(sig3)
        assert allowed  # probe allowed
