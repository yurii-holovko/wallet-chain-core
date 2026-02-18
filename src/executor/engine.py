"""
Executor State Machine — the *muscles* of the arbitrage system.

Lifecycle of an execution::

    IDLE ─► VALIDATING ─► LEG1_PENDING ─► LEG1_CONFIRMING ─► LEG1_FILLED
                                                                   │
                                                                   ▼
              DONE ◄── LEG2_FILLED ◄── LEG2_CONFIRMING ◄── LEG2_PENDING
                                         │
                                         ▼
                                     UNWINDING ─► FAILED

Every state transition is guarded — only allowed edges succeed.
Each transition is logged in ``ExecutionContext.events`` for full
post-mortem analysis.

Retry logic with exponential back-off is built into each leg.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Optional

from eth_abi import encode as abi_encode
from eth_utils import keccak

from chain import ChainClient, TransactionBuilder
from core.base_types import Address, TokenAmount, TransactionRequest
from core.wallet_manager import WalletManager
from executor.recovery import RecoveryConfig, RecoveryManager
from strategy.signal import Direction, Signal

logger = logging.getLogger(__name__)

# ── States & Transitions ─────────────────────────────────────────


class ExecutorState(Enum):
    IDLE = auto()
    VALIDATING = auto()
    LEG1_PENDING = auto()
    LEG1_CONFIRMING = auto()
    LEG1_FILLED = auto()
    LEG2_PENDING = auto()
    LEG2_CONFIRMING = auto()
    LEG2_FILLED = auto()
    DONE = auto()
    FAILED = auto()
    UNWINDING = auto()


# Allowed edges — any transition not listed here will raise.
_VALID_TRANSITIONS: dict[ExecutorState, set[ExecutorState]] = {
    ExecutorState.IDLE: {ExecutorState.VALIDATING, ExecutorState.FAILED},
    ExecutorState.VALIDATING: {ExecutorState.LEG1_PENDING, ExecutorState.FAILED},
    ExecutorState.LEG1_PENDING: {
        ExecutorState.LEG1_CONFIRMING,
        ExecutorState.LEG1_FILLED,
        ExecutorState.FAILED,
    },
    ExecutorState.LEG1_CONFIRMING: {
        ExecutorState.LEG1_FILLED,
        ExecutorState.LEG1_PENDING,  # retry
        ExecutorState.FAILED,
    },
    ExecutorState.LEG1_FILLED: {ExecutorState.LEG2_PENDING},
    ExecutorState.LEG2_PENDING: {
        ExecutorState.LEG2_CONFIRMING,
        ExecutorState.LEG2_FILLED,
        ExecutorState.UNWINDING,
        ExecutorState.FAILED,
    },
    ExecutorState.LEG2_CONFIRMING: {
        ExecutorState.LEG2_FILLED,
        ExecutorState.LEG2_PENDING,  # retry
        ExecutorState.UNWINDING,
    },
    ExecutorState.LEG2_FILLED: {ExecutorState.DONE},
    ExecutorState.UNWINDING: {ExecutorState.FAILED},
    ExecutorState.DONE: set(),
    ExecutorState.FAILED: set(),
}


class InvalidTransition(Exception):
    """Raised when a state transition is not allowed."""


# ── Event Log Entry ──────────────────────────────────────────────


@dataclass
class StateEvent:
    """One row in the execution audit trail."""

    from_state: ExecutorState
    to_state: ExecutorState
    timestamp: float = field(default_factory=time.time)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "from": self.from_state.name,
            "to": self.to_state.name,
            "ts": self.timestamp,
            "detail": self.detail,
        }


# ── Execution Metrics ────────────────────────────────────────────


@dataclass
class ExecutionMetrics:
    """Latency and quality measurements for one execution."""

    leg1_latency_ms: Optional[float] = None
    leg2_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    leg1_slippage_bps: Optional[float] = None
    leg2_slippage_bps: Optional[float] = None
    leg1_fill_ratio: Optional[float] = None
    leg2_fill_ratio: Optional[float] = None
    leg1_retries: int = 0
    leg2_retries: int = 0
    unwind_attempted: bool = False
    unwind_success: Optional[bool] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ── Execution Context ────────────────────────────────────────────


@dataclass
class ExecutionContext:
    """Full context for a single execution attempt."""

    signal: Signal
    state: ExecutorState = ExecutorState.IDLE

    # Leg 1
    leg1_venue: str = ""
    leg1_order_id: Optional[str] = None
    leg1_fill_price: Optional[float] = None
    leg1_fill_size: Optional[float] = None
    leg1_started_at: Optional[float] = None

    # Leg 2
    leg2_venue: str = ""
    leg2_tx_hash: Optional[str] = None
    leg2_fill_price: Optional[float] = None
    leg2_fill_size: Optional[float] = None
    leg2_started_at: Optional[float] = None

    # Timing
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    # Results
    actual_pnl: Optional[float] = None
    actual_fees: Optional[float] = None
    actual_net_pnl: Optional[float] = None
    error: Optional[str] = None

    # Audit trail
    events: list[StateEvent] = field(default_factory=list)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)

    # ── transition helper ─────────────────────────────────────

    def transition(self, new_state: ExecutorState, detail: str = "") -> None:
        """
        Move to *new_state* if the edge is allowed; raise otherwise.

        Every transition is appended to ``self.events``.
        """
        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTransition(f"{self.state.name} → {new_state.name} not allowed")
        event = StateEvent(
            from_state=self.state,
            to_state=new_state,
            detail=detail,
        )
        self.events.append(event)
        self.state = new_state

    @property
    def duration_ms(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at) * 1_000

    def summary(self) -> dict:
        """Compact dict for logging / persistence."""
        return {
            "signal_id": self.signal.signal_id,
            "pair": self.signal.pair,
            "direction": self.signal.direction.value,
            "state": self.state.name,
            "leg1_venue": self.leg1_venue,
            "leg1_fill_price": self.leg1_fill_price,
            "leg1_fill_size": self.leg1_fill_size,
            "leg2_venue": self.leg2_venue,
            "leg2_fill_price": self.leg2_fill_price,
            "leg2_fill_size": self.leg2_fill_size,
            "leg2_tx_hash": self.leg2_tx_hash,
            "actual_net_pnl": self.actual_net_pnl,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metrics": self.metrics.to_dict(),
            "events": [e.to_dict() for e in self.events],
        }


# ── Executor Config ──────────────────────────────────────────────


@dataclass
class ExecutorConfig:
    # Timeouts
    leg1_timeout: float = 5.0
    leg2_timeout: float = 60.0

    # Fill quality
    min_fill_ratio: float = 0.8

    # Retry
    max_leg1_retries: int = 2
    max_leg2_retries: int = 1
    retry_base_delay: float = 0.5  # seconds, doubles each attempt

    # Leg ordering
    use_flashbots: bool = True

    # Simulation
    simulation_mode: bool = True

    # DEX on-chain parameters
    dex_chain_id: int = 11155111
    dex_deadline_seconds: int = 120
    dex_slippage_bps: int = 100
    dex_gas_priority: str = "medium"
    dex_rpc_url: Optional[str] = None
    dex_router_address: Optional[str] = None
    dex_weth_address: Optional[str] = None
    dex_quote_token_address: Optional[str] = None
    dex_private_key: Optional[str] = None


# ── Executor ─────────────────────────────────────────────────────


class Executor:
    """
    Execute arbitrage trades across CEX and DEX.

    The state machine enforces strict transition guards so that
    every execution follows a deterministic path.  Retry and unwind
    logic is baked in.
    """

    def __init__(
        self,
        exchange_client,
        pricing_module,
        inventory_tracker,
        config: Optional[ExecutorConfig] = None,
        recovery_config: Optional[RecoveryConfig] = None,
    ):
        self.exchange = exchange_client
        self.pricing = pricing_module
        self.inventory = inventory_tracker
        self.config = config or ExecutorConfig()

        self.recovery = RecoveryManager(recovery_config)
        # Convenience aliases so existing code keeps working
        self.circuit_breaker = self.recovery.circuit_breaker
        self.replay_protection = self.recovery.replay
        self._dex_client: Optional[ChainClient] = None
        self._dex_wallet: Optional[WalletManager] = None

        # Aggregate metrics
        self._total_executions: int = 0
        self._successful: int = 0
        self._failed: int = 0
        self._total_pnl: float = 0.0

    # ── public API ────────────────────────────────────────────

    async def execute(self, signal: Signal) -> ExecutionContext:
        """
        Run the full execution lifecycle for *signal*.

        Returns an ``ExecutionContext`` with final state, metrics,
        and a full event log regardless of outcome.
        """
        ctx = ExecutionContext(signal=signal)
        self._total_executions += 1

        # ── pre-flight gates (delegated to RecoveryManager) ──
        allowed, reason = self.recovery.pre_flight(signal)
        if not allowed:
            ctx.transition(ExecutorState.FAILED, reason)
            ctx.error = reason
            ctx.finished_at = time.time()
            self._failed += 1
            return ctx

        ctx.transition(ExecutorState.VALIDATING, "Pre-flight checks")

        if not signal.is_valid():
            ctx.transition(ExecutorState.FAILED, "Signal invalid")
            ctx.error = "Signal invalid"
            ctx.finished_at = time.time()
            self._failed += 1
            return ctx

        # ── execute legs ──────────────────────────────────────
        if self.config.use_flashbots:
            ctx = await self._execute_dex_first(ctx)
        else:
            ctx = await self._execute_cex_first(ctx)

        # ── record result ─────────────────────────────────────
        ctx.finished_at = time.time()
        ctx.metrics.total_latency_ms = ctx.duration_ms
        net_pnl = ctx.actual_net_pnl or 0.0

        if ctx.state == ExecutorState.DONE:
            self.recovery.record_outcome(signal, True, pnl=net_pnl)
            self._successful += 1
            self._total_pnl += net_pnl
        else:
            self.recovery.record_outcome(signal, False, ctx.error, pnl=net_pnl)
            self._failed += 1

        logger.info(
            "Execution %s  state=%s  pnl=%s  duration=%.0fms",
            signal.signal_id,
            ctx.state.name,
            ctx.actual_net_pnl,
            ctx.duration_ms or 0,
        )
        return ctx

    @property
    def stats(self) -> dict:
        """Aggregate executor statistics."""
        return {
            "total": self._total_executions,
            "successful": self._successful,
            "failed": self._failed,
            "win_rate": (
                self._successful / self._total_executions
                if self._total_executions
                else 0.0
            ),
            "total_pnl": round(self._total_pnl, 4),
            "circuit_breaker_open": self.circuit_breaker.is_open(),
            "recovery": self.recovery.snapshot(),
        }

    # ── CEX-first flow ────────────────────────────────────────

    async def _execute_cex_first(self, ctx: ExecutionContext) -> ExecutionContext:
        """CEX leg first (default for non-Flashbots)."""
        signal = ctx.signal

        # ── Leg 1: CEX ────────────────────────────────────────
        ctx.transition(ExecutorState.LEG1_PENDING, "Starting CEX leg")
        ctx.leg1_venue = "cex"
        ctx.leg1_started_at = time.time()

        leg1 = await self._execute_leg_with_retry(
            coro_factory=lambda: self._execute_cex_leg(signal),
            timeout=self.config.leg1_timeout,
            max_retries=self.config.max_leg1_retries,
            ctx=ctx,
            leg_name="leg1",
        )

        if leg1 is None or not leg1["success"]:
            ctx.transition(
                ExecutorState.FAILED,
                leg1.get("error", "CEX rejected") if leg1 else "CEX timeout",
            )
            ctx.error = leg1.get("error", "CEX timeout") if leg1 else "CEX timeout"
            return ctx

        fill_ratio = leg1["filled"] / signal.size
        ctx.metrics.leg1_fill_ratio = fill_ratio
        if fill_ratio < self.config.min_fill_ratio:
            ctx.transition(
                ExecutorState.FAILED,
                f"Partial fill {fill_ratio:.1%} < {self.config.min_fill_ratio:.0%}",
            )
            ctx.error = "Partial fill below threshold"
            return ctx

        ctx.leg1_fill_price = leg1["price"]
        ctx.leg1_fill_size = leg1["filled"]
        ctx.leg1_order_id = leg1.get("order_id")
        ctx.metrics.leg1_latency_ms = (time.time() - ctx.leg1_started_at) * 1_000
        ctx.metrics.leg1_slippage_bps = self._calc_slippage_bps(
            expected=signal.cex_price, actual=leg1["price"]
        )
        ctx.transition(ExecutorState.LEG1_FILLED, "CEX leg filled")

        # ── Leg 2: DEX ────────────────────────────────────────
        ctx.transition(ExecutorState.LEG2_PENDING, "Starting DEX leg")
        ctx.leg2_venue = "dex"
        ctx.leg2_started_at = time.time()

        leg2 = await self._execute_leg_with_retry(
            coro_factory=lambda: self._execute_dex_leg(signal, ctx.leg1_fill_size),
            timeout=self.config.leg2_timeout,
            max_retries=self.config.max_leg2_retries,
            ctx=ctx,
            leg_name="leg2",
        )

        if leg2 is None or not leg2["success"]:
            error_detail = leg2.get("error", "DEX failed") if leg2 else "DEX timeout"
            ctx.transition(ExecutorState.UNWINDING, f"DEX failed: {error_detail}")
            ctx.metrics.unwind_attempted = True
            await self._unwind(ctx)
            ctx.transition(ExecutorState.FAILED, "Unwound after DEX failure")
            ctx.error = f"DEX failed - unwound: {error_detail}"
            return ctx

        ctx.leg2_fill_price = leg2["price"]
        ctx.leg2_fill_size = leg2["filled"]
        ctx.leg2_tx_hash = leg2.get("tx_hash")
        ctx.metrics.leg2_latency_ms = (time.time() - ctx.leg2_started_at) * 1_000
        ctx.metrics.leg2_fill_ratio = leg2["filled"] / ctx.leg1_fill_size
        ctx.metrics.leg2_slippage_bps = self._calc_slippage_bps(
            expected=signal.dex_price, actual=leg2["price"]
        )
        ctx.transition(ExecutorState.LEG2_FILLED, "DEX leg filled")

        self._compute_pnl(ctx)
        ctx.transition(ExecutorState.DONE, "Execution complete")
        return ctx

    # ── DEX-first flow ────────────────────────────────────────

    async def _execute_dex_first(self, ctx: ExecutionContext) -> ExecutionContext:
        """DEX leg first (when using Flashbots — failed tx = no cost)."""
        signal = ctx.signal

        # ── Leg 1: DEX ────────────────────────────────────────
        ctx.transition(ExecutorState.LEG1_PENDING, "Starting DEX leg")
        ctx.leg1_venue = "dex"
        ctx.leg1_started_at = time.time()

        leg1 = await self._execute_leg_with_retry(
            coro_factory=lambda: self._execute_dex_leg(signal, signal.size),
            timeout=self.config.leg2_timeout,
            max_retries=self.config.max_leg2_retries,
            ctx=ctx,
            leg_name="leg1",
        )

        if leg1 is None or not leg1["success"]:
            error_detail = leg1.get("error", "DEX failed") if leg1 else "DEX timeout"
            ctx.transition(
                ExecutorState.FAILED,
                f"DEX failed (no cost via Flashbots): {error_detail}",
            )
            ctx.error = f"DEX failed (no cost via Flashbots): {error_detail}"
            return ctx

        ctx.leg1_fill_price = leg1["price"]
        ctx.leg1_fill_size = leg1["filled"]
        ctx.leg1_order_id = leg1.get("order_id")
        ctx.metrics.leg1_latency_ms = (time.time() - ctx.leg1_started_at) * 1_000
        ctx.metrics.leg1_slippage_bps = self._calc_slippage_bps(
            expected=signal.dex_price, actual=leg1["price"]
        )
        ctx.transition(ExecutorState.LEG1_FILLED, "DEX leg filled")

        # ── Leg 2: CEX ────────────────────────────────────────
        ctx.transition(ExecutorState.LEG2_PENDING, "Starting CEX leg")
        ctx.leg2_venue = "cex"
        ctx.leg2_started_at = time.time()

        leg2 = await self._execute_leg_with_retry(
            coro_factory=lambda: self._execute_cex_leg(signal, ctx.leg1_fill_size),
            timeout=self.config.leg1_timeout,
            max_retries=self.config.max_leg1_retries,
            ctx=ctx,
            leg_name="leg2",
        )

        if leg2 is None or not leg2["success"]:
            error_detail = leg2.get("error", "CEX failed") if leg2 else "CEX timeout"
            ctx.transition(ExecutorState.UNWINDING, f"CEX failed: {error_detail}")
            ctx.metrics.unwind_attempted = True
            await self._unwind(ctx)
            ctx.transition(ExecutorState.FAILED, "Unwound after CEX failure")
            ctx.error = f"CEX failed after DEX - unwound: {error_detail}"
            return ctx

        ctx.leg2_fill_price = leg2["price"]
        ctx.leg2_fill_size = leg2["filled"]
        ctx.leg2_tx_hash = leg2.get("tx_hash")
        ctx.metrics.leg2_latency_ms = (time.time() - ctx.leg2_started_at) * 1_000
        ctx.metrics.leg2_fill_ratio = leg2["filled"] / ctx.leg1_fill_size
        ctx.metrics.leg2_slippage_bps = self._calc_slippage_bps(
            expected=signal.cex_price, actual=leg2["price"]
        )
        ctx.transition(ExecutorState.LEG2_FILLED, "CEX leg filled")

        self._compute_pnl(ctx)
        ctx.transition(ExecutorState.DONE, "Execution complete")
        return ctx

    # ── retry wrapper ─────────────────────────────────────────

    async def _execute_leg_with_retry(
        self,
        coro_factory,
        timeout: float,
        max_retries: int,
        ctx: ExecutionContext,
        leg_name: str,
    ) -> Optional[dict]:
        """
        Execute a leg coroutine with retry + exponential back-off.

        Returns the leg result dict, or ``None`` on total failure.
        """
        retries_attr = f"{leg_name}_retries"
        last_result: Optional[dict] = None

        for attempt in range(1 + max_retries):
            try:
                last_result = await asyncio.wait_for(coro_factory(), timeout=timeout)
                if last_result and last_result.get("success"):
                    return last_result
            except asyncio.TimeoutError:
                last_result = {"success": False, "error": "timeout"}
            except Exception as exc:
                last_result = {"success": False, "error": str(exc)}

            # Record retry metric
            if attempt < max_retries:
                setattr(ctx.metrics, retries_attr, attempt + 1)
                delay = self.config.retry_base_delay * (2**attempt)
                logger.warning(
                    "%s attempt %d failed, retrying in %.1fs: %s",
                    leg_name,
                    attempt + 1,
                    delay,
                    last_result.get("error") if last_result else "unknown",
                )
                await asyncio.sleep(delay)

        return last_result

    # ── leg executors ─────────────────────────────────────────

    async def _execute_cex_leg(self, signal: Signal, size: float = None) -> dict:
        actual_size = size or signal.size
        if self.config.simulation_mode:
            await asyncio.sleep(0.1)
            return {
                "success": True,
                "price": signal.cex_price * 1.0001,
                "filled": actual_size,
                "order_id": f"sim_{signal.signal_id}_cex",
            }
        # Real execution via exchange client
        side = "buy" if signal.direction == Direction.BUY_CEX_SELL_DEX else "sell"
        result = self.exchange.create_limit_ioc_order(
            symbol=signal.pair,
            side=side,
            amount=actual_size,
            price=signal.cex_price * 1.001,
        )
        return {
            "success": result["status"] == "filled",
            "price": float(result["avg_fill_price"]),
            "filled": float(result["amount_filled"]),
            "order_id": result.get("order_id"),
            "error": result["status"],
        }

    async def _execute_dex_leg(self, signal: Signal, size: float) -> dict:
        if self.config.simulation_mode:
            await asyncio.sleep(0.5)
            return {
                "success": True,
                "price": signal.dex_price * 0.9998,
                "filled": size,
                "tx_hash": f"0xsim_{signal.signal_id}_dex",
            }
        try:
            return await asyncio.to_thread(self._execute_real_dex_leg, signal, size)
        except Exception as exc:
            return {"success": False, "error": f"DEX execution error: {exc}"}

    # ── unwind ────────────────────────────────────────────────

    async def _unwind(self, ctx: ExecutionContext) -> None:
        """
        Reverse the filled leg-1 to flatten a stuck position.

        In simulation mode this is a no-op.  In production the unwind
        places a market sell (or buy) on the venue that was already
        filled.
        """
        if self.config.simulation_mode:
            await asyncio.sleep(0.1)
            ctx.metrics.unwind_success = True
            logger.info("Simulated unwind for %s", ctx.signal.signal_id)
            return

        signal = ctx.signal
        try:
            if ctx.leg1_venue == "cex" and ctx.leg1_fill_size:
                # Reverse the CEX order
                reverse_side = (
                    "sell" if signal.direction == Direction.BUY_CEX_SELL_DEX else "buy"
                )
                result = self.exchange.create_limit_ioc_order(
                    symbol=signal.pair,
                    side=reverse_side,
                    amount=ctx.leg1_fill_size,
                    price=signal.cex_price
                    * (0.999 if reverse_side == "sell" else 1.001),
                )
                ctx.metrics.unwind_success = result.get("status") == "filled"
            elif ctx.leg1_venue == "dex" and ctx.leg1_fill_size:
                # Reverse the DEX swap
                reverse_signal = Signal.create(
                    pair=signal.pair,
                    direction=(
                        Direction.BUY_DEX_SELL_CEX
                        if signal.direction == Direction.BUY_CEX_SELL_DEX
                        else Direction.BUY_CEX_SELL_DEX
                    ),
                    cex_price=signal.cex_price,
                    dex_price=signal.dex_price,
                    spread_bps=0,
                    size=ctx.leg1_fill_size,
                    expected_gross_pnl=0,
                    expected_fees=0,
                    expected_net_pnl=0,
                    score=100,
                    expiry=time.time() + 120,
                    inventory_ok=True,
                    within_limits=True,
                )
                result = await asyncio.to_thread(
                    self._execute_real_dex_leg,
                    reverse_signal,
                    ctx.leg1_fill_size,
                )
                ctx.metrics.unwind_success = result.get("success", False)
            else:
                ctx.metrics.unwind_success = False
                logger.error("Cannot unwind: no leg1 fill data")
        except Exception as exc:
            ctx.metrics.unwind_success = False
            logger.error("Unwind failed: %s", exc)

    # ── PnL ───────────────────────────────────────────────────

    def _compute_pnl(self, ctx: ExecutionContext) -> None:
        """
        Calculate actual PnL from fill prices.

        Uses the signal's fee structure (if available in meta) or a
        flat 40 bps estimate.
        """
        signal = ctx.signal
        p1 = ctx.leg1_fill_price or 0.0
        p2 = ctx.leg2_fill_price or 0.0
        size = ctx.leg1_fill_size or 0.0

        if signal.direction == Direction.BUY_CEX_SELL_DEX:
            gross = (p2 - p1) * size
        else:
            gross = (p1 - p2) * size

        # Use actual fee data when available
        breakeven_bps = signal.meta.get("breakeven_bps", 40.0)
        trade_value = size * max(p1, p2)
        fees = (breakeven_bps / 10_000) * trade_value

        ctx.actual_pnl = round(gross, 6)
        ctx.actual_fees = round(fees, 6)
        ctx.actual_net_pnl = round(gross - fees, 6)

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _calc_slippage_bps(expected: float, actual: float) -> float:
        """Slippage in basis points (positive = worse than expected)."""
        if expected == 0:
            return 0.0
        return round(abs(actual - expected) / expected * 10_000, 2)

    # ── real DEX execution (unchanged from previous impl) ─────

    def _execute_real_dex_leg(self, signal: Signal, size: float) -> dict:
        self._ensure_dex_ready()
        assert self._dex_client is not None
        assert self._dex_wallet is not None
        wallet_addr = Address.from_string(self._dex_wallet.address)

        router = Address.from_string(
            self._resolve_dex_config("DEX_ROUTER_ADDRESS", "dex_router_address")
        )
        weth = self._resolve_dex_config("DEX_WETH_ADDRESS", "dex_weth_address")
        quote = self._resolve_dex_config(
            "DEX_QUOTE_TOKEN_ADDRESS", "dex_quote_token_address"
        )
        deadline = int(time.time()) + self.config.dex_deadline_seconds

        if signal.direction == Direction.BUY_CEX_SELL_DEX:
            eth_in_wei = self._to_wei(size)
            min_quote_out = int(
                self._to_token_units(size * signal.dex_price, 6)
                * (10_000 - self.config.dex_slippage_bps)
                / 10_000
            )
            calldata = self._encode_call(
                "swapExactETHForTokens(uint256,address[],address,uint256)",
                ["uint256", "address[]", "address", "uint256"],
                [min_quote_out, [weth, quote], wallet_addr.checksum, deadline],
            )
            receipt = (
                TransactionBuilder(self._dex_client, self._dex_wallet)
                .to(router)
                .value(TokenAmount(raw=eth_in_wei, decimals=18, symbol="ETH"))
                .data(calldata)
                .chain_id(self.config.dex_chain_id)
                .with_gas_estimate()
                .with_gas_price(self.config.dex_gas_priority)
                .send_and_wait(timeout=int(self.config.leg2_timeout))
            )
            executed_price = signal.dex_price * (
                (10_000 - self.config.dex_slippage_bps) / 10_000
            )
            return {
                "success": receipt.status,
                "price": executed_price,
                "filled": size,
                "tx_hash": receipt.tx_hash,
            }

        # Buy exact ETH on DEX: quote token -> ETH
        eth_out_wei = self._to_wei(size)
        max_quote_in = int(
            self._to_token_units(size * signal.dex_price, 6)
            * (10_000 + self.config.dex_slippage_bps)
            / 10_000
        )
        quote_token = Address.from_string(quote)
        self._ensure_allowance(
            token=quote_token,
            owner=wallet_addr,
            spender=router,
            min_amount=max_quote_in,
        )
        calldata = self._encode_call(
            "swapTokensForExactETH(uint256,uint256,address[],address,uint256)",
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [eth_out_wei, max_quote_in, [quote, weth], wallet_addr.checksum, deadline],
        )
        receipt = (
            TransactionBuilder(self._dex_client, self._dex_wallet)
            .to(router)
            .value(TokenAmount(raw=0, decimals=18, symbol="ETH"))
            .data(calldata)
            .chain_id(self.config.dex_chain_id)
            .with_gas_estimate()
            .with_gas_price(self.config.dex_gas_priority)
            .send_and_wait(timeout=int(self.config.leg2_timeout))
        )
        executed_price = signal.dex_price * (
            (10_000 + self.config.dex_slippage_bps) / 10_000
        )
        return {
            "success": receipt.status,
            "price": executed_price,
            "filled": size,
            "tx_hash": receipt.tx_hash,
        }

    def _ensure_dex_ready(self) -> None:
        if self._dex_client is not None and self._dex_wallet is not None:
            return
        rpc_url = self._resolve_dex_config("SEPOLIA_RPC_URL", "dex_rpc_url")
        private_key = self._resolve_dex_config("PRIVATE_KEY", "dex_private_key")
        self._dex_client = ChainClient([rpc_url])
        self._dex_wallet = WalletManager(private_key)

    def _resolve_dex_config(self, env_key: str, config_attr: str) -> str:
        value = getattr(self.config, config_attr) or os.getenv(env_key)
        if not value:
            raise ValueError(f"Missing DEX config: {config_attr} / {env_key}")
        return value

    def _ensure_allowance(
        self,
        token: Address,
        owner: Address,
        spender: Address,
        min_amount: int,
    ) -> None:
        assert self._dex_client is not None
        assert self._dex_wallet is not None
        allowance_data = self._encode_call(
            "allowance(address,address)",
            ["address", "address"],
            [owner.checksum, spender.checksum],
        )
        allowance_call = TransactionRequest(
            to=token,
            value=TokenAmount(raw=0, decimals=18, symbol="ETH"),
            data=allowance_data,
            chain_id=self.config.dex_chain_id,
        )
        allowance_raw = self._dex_client.call(allowance_call)
        current_allowance = int.from_bytes(allowance_raw, "big") if allowance_raw else 0
        if current_allowance >= min_amount:
            return

        max_uint256 = 2**256 - 1
        approve_data = self._encode_call(
            "approve(address,uint256)",
            ["address", "uint256"],
            [spender.checksum, max_uint256],
        )
        receipt = (
            TransactionBuilder(self._dex_client, self._dex_wallet)
            .to(token)
            .value(TokenAmount(raw=0, decimals=18, symbol="ETH"))
            .data(approve_data)
            .chain_id(self.config.dex_chain_id)
            .with_gas_estimate()
            .with_gas_price(self.config.dex_gas_priority)
            .send_and_wait(timeout=int(self.config.leg2_timeout))
        )
        if not receipt.status:
            raise RuntimeError("Token approve transaction failed")

    @staticmethod
    def _encode_call(signature: str, arg_types: list[str], args: list[Any]) -> bytes:
        selector = keccak(text=signature)[:4]
        return selector + abi_encode(arg_types, args)

    @staticmethod
    def _to_wei(value_eth: float) -> int:
        return int(Decimal(str(value_eth)) * Decimal(10**18))

    @staticmethod
    def _to_token_units(amount: float, decimals: int) -> int:
        return int(Decimal(str(amount)) * Decimal(10**decimals))
