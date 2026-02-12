"""Tests for executor.engine — Executor state machine, retry, unwind, metrics."""

import asyncio
import time

import pytest

from executor.engine import (
    _VALID_TRANSITIONS,
    ExecutionContext,
    ExecutionMetrics,
    Executor,
    ExecutorConfig,
    ExecutorState,
    InvalidTransition,
    StateEvent,
)
from strategy.signal import Direction, Signal

# ── helpers ───────────────────────────────────────────────────────


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
        expiry=time.time() + 10,
        inventory_ok=True,
        within_limits=True,
    )
    defaults.update(overrides)
    return Signal.create(**defaults)


def _make_executor(**config_overrides) -> Executor:
    cfg = ExecutorConfig(simulation_mode=True, **config_overrides)
    return Executor(None, None, None, cfg)


# ══════════════════════════════════════════════════════════════════
#  State transition guards
# ══════════════════════════════════════════════════════════════════


class TestTransitionGuards:
    """Verify that only valid state edges are accepted."""

    def test_valid_transitions_complete(self):
        """Every ExecutorState has an entry in the transition map."""
        for state in ExecutorState:
            assert state in _VALID_TRANSITIONS

    def test_terminal_states_have_no_exits(self):
        assert _VALID_TRANSITIONS[ExecutorState.DONE] == set()
        assert _VALID_TRANSITIONS[ExecutorState.FAILED] == set()

    def test_idle_to_validating_allowed(self):
        ctx = ExecutionContext(signal=_make_signal())
        ctx.transition(ExecutorState.VALIDATING, "test")
        assert ctx.state == ExecutorState.VALIDATING

    def test_idle_to_failed_allowed(self):
        ctx = ExecutionContext(signal=_make_signal())
        ctx.transition(ExecutorState.FAILED, "circuit breaker")
        assert ctx.state == ExecutorState.FAILED

    def test_idle_to_done_rejected(self):
        ctx = ExecutionContext(signal=_make_signal())
        with pytest.raises(InvalidTransition):
            ctx.transition(ExecutorState.DONE, "skip everything")

    def test_done_to_anything_rejected(self):
        ctx = ExecutionContext(signal=_make_signal())
        ctx.transition(ExecutorState.VALIDATING)
        ctx.transition(ExecutorState.LEG1_PENDING)
        ctx.transition(ExecutorState.LEG1_FILLED)
        ctx.transition(ExecutorState.LEG2_PENDING)
        ctx.transition(ExecutorState.LEG2_FILLED)
        ctx.transition(ExecutorState.DONE)
        with pytest.raises(InvalidTransition):
            ctx.transition(ExecutorState.IDLE)

    def test_failed_to_anything_rejected(self):
        ctx = ExecutionContext(signal=_make_signal())
        ctx.transition(ExecutorState.FAILED)
        with pytest.raises(InvalidTransition):
            ctx.transition(ExecutorState.IDLE)

    def test_leg1_confirming_can_retry(self):
        """LEG1_CONFIRMING → LEG1_PENDING is allowed (retry)."""
        ctx = ExecutionContext(signal=_make_signal())
        ctx.transition(ExecutorState.VALIDATING)
        ctx.transition(ExecutorState.LEG1_PENDING)
        ctx.transition(ExecutorState.LEG1_CONFIRMING)
        ctx.transition(ExecutorState.LEG1_PENDING, "retry")
        assert ctx.state == ExecutorState.LEG1_PENDING

    def test_leg2_confirming_can_retry(self):
        """LEG2_CONFIRMING → LEG2_PENDING is allowed (retry)."""
        ctx = ExecutionContext(signal=_make_signal())
        ctx.transition(ExecutorState.VALIDATING)
        ctx.transition(ExecutorState.LEG1_PENDING)
        ctx.transition(ExecutorState.LEG1_FILLED)
        ctx.transition(ExecutorState.LEG2_PENDING)
        ctx.transition(ExecutorState.LEG2_CONFIRMING)
        ctx.transition(ExecutorState.LEG2_PENDING, "retry")
        assert ctx.state == ExecutorState.LEG2_PENDING


# ══════════════════════════════════════════════════════════════════
#  Event log / audit trail
# ══════════════════════════════════════════════════════════════════


class TestEventLog:
    def test_events_recorded_on_transition(self):
        ctx = ExecutionContext(signal=_make_signal())
        ctx.transition(ExecutorState.VALIDATING, "check")
        ctx.transition(ExecutorState.FAILED, "invalid")

        assert len(ctx.events) == 2
        assert ctx.events[0].from_state == ExecutorState.IDLE
        assert ctx.events[0].to_state == ExecutorState.VALIDATING
        assert ctx.events[0].detail == "check"
        assert ctx.events[1].from_state == ExecutorState.VALIDATING
        assert ctx.events[1].to_state == ExecutorState.FAILED

    def test_state_event_to_dict(self):
        ev = StateEvent(
            from_state=ExecutorState.IDLE,
            to_state=ExecutorState.VALIDATING,
            detail="test",
        )
        d = ev.to_dict()
        assert d["from"] == "IDLE"
        assert d["to"] == "VALIDATING"
        assert d["detail"] == "test"
        assert "ts" in d

    def test_context_summary_includes_events(self):
        ctx = ExecutionContext(signal=_make_signal())
        ctx.transition(ExecutorState.FAILED, "boom")
        ctx.error = "boom"
        ctx.finished_at = time.time()
        s = ctx.summary()
        assert len(s["events"]) == 1
        assert s["state"] == "FAILED"
        assert s["error"] == "boom"


# ══════════════════════════════════════════════════════════════════
#  ExecutionContext
# ══════════════════════════════════════════════════════════════════


class TestExecutionContext:
    def test_default_state(self):
        sig = _make_signal()
        ctx = ExecutionContext(signal=sig)
        assert ctx.state == ExecutorState.IDLE
        assert ctx.error is None
        assert ctx.actual_net_pnl is None
        assert ctx.events == []

    def test_duration_ms_none_before_finish(self):
        ctx = ExecutionContext(signal=_make_signal())
        assert ctx.duration_ms is None

    def test_duration_ms_calculated(self):
        ctx = ExecutionContext(signal=_make_signal())
        ctx.finished_at = ctx.started_at + 1.5
        assert abs(ctx.duration_ms - 1500.0) < 1.0

    def test_summary_keys(self):
        ctx = ExecutionContext(signal=_make_signal())
        s = ctx.summary()
        expected_keys = {
            "signal_id",
            "pair",
            "direction",
            "state",
            "leg1_venue",
            "leg1_fill_price",
            "leg1_fill_size",
            "leg2_venue",
            "leg2_fill_price",
            "leg2_fill_size",
            "leg2_tx_hash",
            "actual_net_pnl",
            "error",
            "duration_ms",
            "metrics",
            "events",
        }
        assert set(s.keys()) == expected_keys


# ══════════════════════════════════════════════════════════════════
#  ExecutionMetrics
# ══════════════════════════════════════════════════════════════════


class TestExecutionMetrics:
    def test_defaults(self):
        m = ExecutionMetrics()
        assert m.leg1_retries == 0
        assert m.leg2_retries == 0
        assert m.unwind_attempted is False

    def test_to_dict_filters_none(self):
        m = ExecutionMetrics(leg1_latency_ms=42.0)
        d = m.to_dict()
        assert d["leg1_latency_ms"] == 42.0
        assert "leg2_latency_ms" not in d  # None → filtered

    def test_to_dict_includes_all_set(self):
        m = ExecutionMetrics(
            leg1_latency_ms=10.0,
            leg2_latency_ms=20.0,
            total_latency_ms=30.0,
            leg1_slippage_bps=1.5,
            leg2_slippage_bps=2.0,
            leg1_fill_ratio=0.95,
            leg2_fill_ratio=1.0,
            leg1_retries=1,
            leg2_retries=0,
            unwind_attempted=True,
            unwind_success=True,
        )
        d = m.to_dict()
        assert len(d) == 11


# ══════════════════════════════════════════════════════════════════
#  ExecutorConfig
# ══════════════════════════════════════════════════════════════════


class TestExecutorConfig:
    def test_defaults(self):
        cfg = ExecutorConfig()
        assert cfg.leg1_timeout == 5.0
        assert cfg.leg2_timeout == 60.0
        assert cfg.simulation_mode is True
        assert cfg.max_leg1_retries == 2
        assert cfg.max_leg2_retries == 1
        assert cfg.retry_base_delay == 0.5

    def test_custom_config(self):
        cfg = ExecutorConfig(
            simulation_mode=False,
            leg1_timeout=2.0,
            max_leg1_retries=5,
        )
        assert cfg.simulation_mode is False
        assert cfg.leg1_timeout == 2.0
        assert cfg.max_leg1_retries == 5


# ══════════════════════════════════════════════════════════════════
#  Executor — simulation mode happy paths
# ══════════════════════════════════════════════════════════════════


class TestExecutorSimulation:
    @pytest.mark.asyncio
    async def test_execute_success_dex_first(self):
        """Default (use_flashbots=True) → DEX first, then CEX."""
        executor = _make_executor()
        sig = _make_signal()
        ctx = await executor.execute(sig)

        assert ctx.state == ExecutorState.DONE
        assert ctx.actual_net_pnl is not None
        assert ctx.actual_pnl is not None
        assert ctx.actual_fees is not None
        assert ctx.finished_at is not None
        assert ctx.leg1_venue == "dex"
        assert ctx.leg2_venue == "cex"

    @pytest.mark.asyncio
    async def test_execute_success_cex_first(self):
        executor = _make_executor(use_flashbots=False)
        sig = _make_signal()
        ctx = await executor.execute(sig)

        assert ctx.state == ExecutorState.DONE
        assert ctx.leg1_venue == "cex"
        assert ctx.leg2_venue == "dex"

    @pytest.mark.asyncio
    async def test_event_log_populated(self):
        executor = _make_executor()
        sig = _make_signal()
        ctx = await executor.execute(sig)

        assert len(ctx.events) >= 5  # IDLE→VALID→L1P→L1F→L2P→L2F→DONE
        states_visited = [e.to_state for e in ctx.events]
        assert ExecutorState.VALIDATING in states_visited
        assert ExecutorState.LEG1_PENDING in states_visited
        assert ExecutorState.LEG1_FILLED in states_visited
        assert ExecutorState.LEG2_PENDING in states_visited
        assert ExecutorState.DONE in states_visited

    @pytest.mark.asyncio
    async def test_metrics_populated(self):
        executor = _make_executor()
        sig = _make_signal()
        ctx = await executor.execute(sig)

        assert ctx.metrics.leg1_latency_ms is not None
        assert ctx.metrics.leg2_latency_ms is not None
        assert ctx.metrics.total_latency_ms is not None
        assert ctx.metrics.leg1_slippage_bps is not None
        assert ctx.metrics.leg2_slippage_bps is not None

    @pytest.mark.asyncio
    async def test_pnl_calculated(self):
        executor = _make_executor()
        sig = _make_signal(cex_price=2000.0, dex_price=2020.0)
        ctx = await executor.execute(sig)

        assert ctx.state == ExecutorState.DONE
        assert isinstance(ctx.actual_net_pnl, float)
        assert ctx.actual_pnl is not None
        assert ctx.actual_fees is not None

    @pytest.mark.asyncio
    async def test_pnl_uses_breakeven_from_meta(self):
        """When signal.meta has breakeven_bps, PnL uses it."""
        executor = _make_executor()
        sig = _make_signal(
            cex_price=2000.0,
            dex_price=2020.0,
            meta={"breakeven_bps": 20.0},
        )
        ctx = await executor.execute(sig)
        assert ctx.actual_fees is not None
        # 20 bps on ~2020 * 1.0 ≈ $0.40
        assert ctx.actual_fees > 0


# ══════════════════════════════════════════════════════════════════
#  Pre-flight gates
# ══════════════════════════════════════════════════════════════════


class TestPreFlightGates:
    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks(self):
        executor = _make_executor()
        executor.circuit_breaker.trip()
        ctx = await executor.execute(_make_signal())

        assert ctx.state == ExecutorState.FAILED
        assert "Circuit breaker" in ctx.error
        assert len(ctx.events) == 1  # IDLE → FAILED
        assert ctx.events[0].to_state == ExecutorState.FAILED

    @pytest.mark.asyncio
    async def test_duplicate_signal_rejected(self):
        executor = _make_executor()
        sig = _make_signal()
        ctx1 = await executor.execute(sig)
        assert ctx1.state == ExecutorState.DONE

        ctx2 = await executor.execute(sig)
        assert ctx2.state == ExecutorState.FAILED
        assert "Duplicate" in ctx2.error

    @pytest.mark.asyncio
    async def test_invalid_signal_rejected(self):
        executor = _make_executor()
        sig = _make_signal(expiry=time.time() - 1)
        ctx = await executor.execute(sig)

        assert ctx.state == ExecutorState.FAILED
        assert "invalid" in ctx.error.lower()
        # Should have IDLE→VALIDATING→FAILED
        assert len(ctx.events) == 2


# ══════════════════════════════════════════════════════════════════
#  Retry logic
# ══════════════════════════════════════════════════════════════════


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_leg1_retry_on_failure(self):
        """Leg 1 fails once then succeeds on retry."""
        executor = _make_executor(
            use_flashbots=False,
            max_leg1_retries=2,
            retry_base_delay=0.01,
        )
        call_count = 0
        original = executor._execute_cex_leg

        async def flaky_cex(signal, size=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": False, "error": "transient"}
            return await original(signal, size)

        executor._execute_cex_leg = flaky_cex
        ctx = await executor.execute(_make_signal())

        assert ctx.state == ExecutorState.DONE
        assert call_count == 2
        assert ctx.metrics.leg1_retries == 1

    @pytest.mark.asyncio
    async def test_leg2_retry_on_failure(self):
        """Leg 2 fails once then succeeds on retry."""
        executor = _make_executor(
            use_flashbots=False,
            max_leg2_retries=2,
            retry_base_delay=0.01,
        )
        call_count = 0
        original = executor._execute_dex_leg

        async def flaky_dex(signal, size):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": False, "error": "transient"}
            return await original(signal, size)

        executor._execute_dex_leg = flaky_dex
        ctx = await executor.execute(_make_signal())

        assert ctx.state == ExecutorState.DONE
        assert call_count == 2
        assert ctx.metrics.leg2_retries == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_fails(self):
        """When all retries are exhausted, the execution fails."""
        executor = _make_executor(
            use_flashbots=False,
            max_leg1_retries=1,
            retry_base_delay=0.01,
        )

        async def always_fail(signal, size=None):
            return {"success": False, "error": "permanent"}

        executor._execute_cex_leg = always_fail
        ctx = await executor.execute(_make_signal())

        assert ctx.state == ExecutorState.FAILED
        assert "permanent" in ctx.error

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self):
        """Timeout on leg1 triggers retry."""
        executor = _make_executor(
            use_flashbots=False,
            leg1_timeout=0.3,
            max_leg1_retries=1,
            retry_base_delay=0.01,
        )
        call_count = 0

        async def slow_then_fast(signal, size=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(10)  # will timeout
            # Fast success on retry (no sleep)
            return {
                "success": True,
                "price": signal.cex_price * 1.0001,
                "filled": size or signal.size,
                "order_id": "sim_retry",
            }

        executor._execute_cex_leg = slow_then_fast
        ctx = await executor.execute(_make_signal())

        assert ctx.state == ExecutorState.DONE
        assert call_count == 2


# ══════════════════════════════════════════════════════════════════
#  Unwind logic
# ══════════════════════════════════════════════════════════════════


class TestUnwind:
    @pytest.mark.asyncio
    async def test_unwind_triggered_on_leg2_failure_cex_first(self):
        """When leg2 (DEX) fails after leg1 (CEX) filled, unwind occurs."""
        executor = _make_executor(
            use_flashbots=False,
            max_leg2_retries=0,
            retry_base_delay=0.01,
        )

        async def failing_dex(signal, size):
            return {"success": False, "error": "DEX reverted"}

        executor._execute_dex_leg = failing_dex
        ctx = await executor.execute(_make_signal())

        assert ctx.state == ExecutorState.FAILED
        assert ctx.metrics.unwind_attempted is True
        assert ctx.metrics.unwind_success is True  # sim unwind always succeeds
        assert "unwound" in ctx.error.lower()

    @pytest.mark.asyncio
    async def test_unwind_triggered_on_leg2_failure_dex_first(self):
        """When leg2 (CEX) fails after leg1 (DEX) filled, unwind occurs."""
        executor = _make_executor(
            use_flashbots=True,
            max_leg1_retries=0,
            retry_base_delay=0.01,
        )

        async def failing_cex(signal, size=None):
            return {"success": False, "error": "CEX rejected"}

        executor._execute_cex_leg = failing_cex
        ctx = await executor.execute(_make_signal())

        assert ctx.state == ExecutorState.FAILED
        assert ctx.metrics.unwind_attempted is True
        assert "unwound" in ctx.error.lower()

    @pytest.mark.asyncio
    async def test_unwind_events_logged(self):
        """Unwind transitions are captured in the event log."""
        executor = _make_executor(
            use_flashbots=False,
            max_leg2_retries=0,
            retry_base_delay=0.01,
        )

        async def failing_dex(signal, size):
            return {"success": False, "error": "boom"}

        executor._execute_dex_leg = failing_dex
        ctx = await executor.execute(_make_signal())

        states = [e.to_state for e in ctx.events]
        assert ExecutorState.UNWINDING in states
        assert ExecutorState.FAILED in states


# ══════════════════════════════════════════════════════════════════
#  Aggregate stats
# ══════════════════════════════════════════════════════════════════


class TestExecutorStats:
    @pytest.mark.asyncio
    async def test_stats_after_success(self):
        executor = _make_executor()
        await executor.execute(_make_signal())
        s = executor.stats

        assert s["total"] == 1
        assert s["successful"] == 1
        assert s["failed"] == 0
        assert s["win_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_stats_after_failure(self):
        executor = _make_executor()
        executor.circuit_breaker.trip()
        await executor.execute(_make_signal())
        s = executor.stats

        assert s["total"] == 1
        assert s["successful"] == 0
        assert s["failed"] == 1
        assert s["win_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_accumulate(self):
        executor = _make_executor()
        sig = _make_signal()
        await executor.execute(sig)
        await executor.execute(sig)  # same signal → duplicate
        s = executor.stats

        assert s["total"] == 2
        assert s["successful"] == 1  # first succeeds
        assert s["failed"] == 1  # second is duplicate

    @pytest.mark.asyncio
    async def test_total_pnl_accumulates(self):
        executor = _make_executor()
        ctx = await executor.execute(_make_signal())
        pnl1 = ctx.actual_net_pnl or 0
        assert executor.stats["total_pnl"] == round(pnl1, 4)


# ══════════════════════════════════════════════════════════════════
#  Slippage calculation
# ══════════════════════════════════════════════════════════════════


class TestSlippage:
    def test_zero_expected(self):
        assert Executor._calc_slippage_bps(0, 100) == 0.0

    def test_no_slippage(self):
        assert Executor._calc_slippage_bps(100, 100) == 0.0

    def test_positive_slippage(self):
        # 1% worse = 100 bps
        bps = Executor._calc_slippage_bps(100.0, 101.0)
        assert abs(bps - 100.0) < 0.1

    def test_negative_slippage(self):
        # Improvement also measured as absolute
        bps = Executor._calc_slippage_bps(100.0, 99.0)
        assert abs(bps - 100.0) < 0.1


# ══════════════════════════════════════════════════════════════════
#  PnL calculation
# ══════════════════════════════════════════════════════════════════


class TestPnLCalculation:
    def test_buy_cex_sell_dex_positive(self):
        executor = _make_executor()
        sig = _make_signal(
            direction=Direction.BUY_CEX_SELL_DEX,
            cex_price=2000.0,
            dex_price=2050.0,
        )
        ctx = ExecutionContext(signal=sig)
        ctx.leg1_fill_price = 2000.0
        ctx.leg2_fill_price = 2050.0
        ctx.leg1_fill_size = 1.0
        executor._compute_pnl(ctx)

        assert ctx.actual_pnl == 50.0  # gross
        assert ctx.actual_fees > 0
        assert ctx.actual_net_pnl < 50.0  # net < gross

    def test_buy_dex_sell_cex_positive(self):
        executor = _make_executor()
        sig = _make_signal(
            direction=Direction.BUY_DEX_SELL_CEX,
            cex_price=2050.0,
            dex_price=2000.0,
        )
        ctx = ExecutionContext(signal=sig)
        ctx.leg1_fill_price = 2050.0
        ctx.leg2_fill_price = 2000.0
        ctx.leg1_fill_size = 1.0
        executor._compute_pnl(ctx)

        assert ctx.actual_pnl == 50.0
        assert ctx.actual_net_pnl < 50.0

    def test_breakeven_bps_from_meta(self):
        executor = _make_executor()
        sig = _make_signal(
            direction=Direction.BUY_CEX_SELL_DEX,
            meta={"breakeven_bps": 10.0},
        )
        ctx = ExecutionContext(signal=sig)
        ctx.leg1_fill_price = 2000.0
        ctx.leg2_fill_price = 2010.0
        ctx.leg1_fill_size = 1.0
        executor._compute_pnl(ctx)

        # 10 bps on ~$2010 ≈ $2.01 fees
        assert 1.5 < ctx.actual_fees < 3.0


# ══════════════════════════════════════════════════════════════════
#  Integration: full lifecycle
# ══════════════════════════════════════════════════════════════════


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_lifecycle_cex_first(self):
        """CEX-first: verify every state is visited in order."""
        executor = _make_executor(use_flashbots=False)
        ctx = await executor.execute(_make_signal())

        expected_order = [
            ExecutorState.VALIDATING,
            ExecutorState.LEG1_PENDING,
            ExecutorState.LEG1_FILLED,
            ExecutorState.LEG2_PENDING,
            ExecutorState.LEG2_FILLED,
            ExecutorState.DONE,
        ]
        actual_states = [e.to_state for e in ctx.events]
        assert actual_states == expected_order

    @pytest.mark.asyncio
    async def test_full_lifecycle_dex_first(self):
        """DEX-first: verify every state is visited in order."""
        executor = _make_executor(use_flashbots=True)
        ctx = await executor.execute(_make_signal())

        expected_order = [
            ExecutorState.VALIDATING,
            ExecutorState.LEG1_PENDING,
            ExecutorState.LEG1_FILLED,
            ExecutorState.LEG2_PENDING,
            ExecutorState.LEG2_FILLED,
            ExecutorState.DONE,
        ]
        actual_states = [e.to_state for e in ctx.events]
        assert actual_states == expected_order

    @pytest.mark.asyncio
    async def test_failure_lifecycle_with_unwind(self):
        """Leg2 failure: IDLE→VALID→L1P→L1F→L2P→UNWIND→FAILED."""
        executor = _make_executor(
            use_flashbots=False,
            max_leg2_retries=0,
        )

        async def fail_dex(signal, size):
            return {"success": False, "error": "revert"}

        executor._execute_dex_leg = fail_dex
        ctx = await executor.execute(_make_signal())

        expected_order = [
            ExecutorState.VALIDATING,
            ExecutorState.LEG1_PENDING,
            ExecutorState.LEG1_FILLED,
            ExecutorState.LEG2_PENDING,
            ExecutorState.UNWINDING,
            ExecutorState.FAILED,
        ]
        actual_states = [e.to_state for e in ctx.events]
        assert actual_states == expected_order

    @pytest.mark.asyncio
    async def test_summary_serializable(self):
        """ctx.summary() produces a JSON-serializable dict."""
        import json

        executor = _make_executor()
        ctx = await executor.execute(_make_signal())
        s = ctx.summary()
        # Should not raise
        json.dumps(s)
