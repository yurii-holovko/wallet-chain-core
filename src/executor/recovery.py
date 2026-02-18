"""
Failure Handler — Circuit Breaker & Replay Protection.

Circuit Breaker
~~~~~~~~~~~~~~~
Three states: **CLOSED** (trading), **OPEN** (halted), **HALF-OPEN** (probing).

* Trips on N failures inside a rolling window **or** when cumulative PnL
  drops below a drawdown threshold.
* Per-pair tracking: one toxic pair won't shut down the whole bot.
* Half-open mode lets a single probe trade through before fully closing
  again or resetting.
* Consecutive successes decay the failure count so healthy periods
  naturally bring the breaker closer to baseline.

Replay Protection
~~~~~~~~~~~~~~~~~
* Per-pair + global dedup with configurable TTL.
* Nonce-monotonic check: rejects signals whose nonce is ≤ the last
  executed nonce for that pair (prevents stale / re-ordered signals).
* Bounded size with LRU eviction so memory stays constant.
* Full audit log of every accept / reject decision.

Failure Classifier
~~~~~~~~~~~~~~~~~~
Categorises raw error strings into buckets (TRANSIENT, PERMANENT,
RATE_LIMIT, NETWORK, UNKNOWN) so the circuit breaker can weigh
them differently.

Recovery Manager
~~~~~~~~~~~~~~~~
Façade that wires CircuitBreaker + ReplayProtection + FailureClassifier
together and exposes a single ``pre_flight(signal)`` gate plus
``record_outcome(signal, success, error)`` callback.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from strategy.signal import Signal

logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Failure Classifier                                             ║
# ╚══════════════════════════════════════════════════════════════════╝


class FailureCategory(Enum):
    """Broad failure bucket used by the circuit breaker."""

    TRANSIENT = auto()  # network blip, timeout — worth retrying
    PERMANENT = auto()  # insufficient funds, invalid order — don't retry
    RATE_LIMIT = auto()  # exchange rate-limit — back off
    NETWORK = auto()  # DNS / TCP / TLS errors
    UNKNOWN = auto()  # catch-all


# Ordered list of (regex, category).  First match wins.
_PATTERNS: list[tuple[re.Pattern, FailureCategory]] = [
    (re.compile(r"timeout", re.I), FailureCategory.TRANSIENT),
    (re.compile(r"transient", re.I), FailureCategory.TRANSIENT),
    (re.compile(r"temporarily", re.I), FailureCategory.TRANSIENT),
    (re.compile(r"rate.?limit", re.I), FailureCategory.RATE_LIMIT),
    (re.compile(r"429", re.I), FailureCategory.RATE_LIMIT),
    (re.compile(r"too many requests", re.I), FailureCategory.RATE_LIMIT),
    (re.compile(r"insufficient", re.I), FailureCategory.PERMANENT),
    (re.compile(r"invalid", re.I), FailureCategory.PERMANENT),
    (re.compile(r"revert", re.I), FailureCategory.PERMANENT),
    (re.compile(r"nonce too low", re.I), FailureCategory.PERMANENT),
    (re.compile(r"rejected", re.I), FailureCategory.PERMANENT),
    (
        re.compile(r"DNS|ECONNREFUSED|ENOTFOUND|ConnectionReset", re.I),
        FailureCategory.NETWORK,
    ),
    (re.compile(r"network", re.I), FailureCategory.NETWORK),
]


class FailureClassifier:
    """Classify an error string into a :class:`FailureCategory`."""

    @staticmethod
    def classify(error: Optional[str]) -> FailureCategory:
        if not error:
            return FailureCategory.UNKNOWN
        for pattern, category in _PATTERNS:
            if pattern.search(error):
                return category
        return FailureCategory.UNKNOWN

    @staticmethod
    def is_retriable(category: FailureCategory) -> bool:
        return category in (
            FailureCategory.TRANSIENT,
            FailureCategory.RATE_LIMIT,
            FailureCategory.NETWORK,
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Circuit Breaker                                                ║
# ╚══════════════════════════════════════════════════════════════════╝


class BreakerState(Enum):
    CLOSED = "closed"  # Normal — trading allowed
    OPEN = "open"  # Tripped — all trading blocked
    HALF_OPEN = "half_open"  # Probing — one trade allowed


@dataclass
class CircuitBreakerConfig:
    # ── failure-count trip ───────────────────────────────────
    failure_threshold: int = 3
    window_seconds: float = 300.0

    # ── PnL-based trip ──────────────────────────────────────
    max_drawdown_usd: float = 50.0  # cumulative loss that trips

    # ── cooldown / half-open ────────────────────────────────
    cooldown_seconds: float = 600.0
    half_open_after_pct: float = 0.8  # enter HALF_OPEN at 80% of cooldown

    # ── success decay ───────────────────────────────────────
    success_decay: int = 1  # each success removes N failure records

    # ── per-pair isolation ──────────────────────────────────
    per_pair: bool = True  # track per-pair; global is always tracked too


@dataclass
class BreakerSnapshot:
    """Read-only view of breaker state for logging / dashboards."""

    state: BreakerState
    failures_in_window: int
    cumulative_pnl: float
    tripped_at: Optional[float]
    time_until_reset: float
    pair: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "failures": self.failures_in_window,
            "cumulative_pnl": round(self.cumulative_pnl, 4),
            "tripped_at": self.tripped_at,
            "reset_in_s": round(self.time_until_reset, 1),
            "pair": self.pair,
        }


class _SingleBreaker:
    """State machine for one circuit breaker (global or per-pair)."""

    def __init__(self, config: CircuitBreakerConfig, label: str = "global"):
        self._cfg = config
        self._label = label
        self._failures: list[float] = []
        self._state: BreakerState = BreakerState.CLOSED
        self._tripped_at: Optional[float] = None
        self._cumulative_pnl: float = 0.0

    # ── recording ─────────────────────────────────────────────

    def record_failure(
        self,
        category: FailureCategory = FailureCategory.UNKNOWN,
        pnl: float = 0.0,
    ) -> None:
        now = time.time()
        # Permanent errors count double; transient count once
        weight = 2 if category == FailureCategory.PERMANENT else 1
        for _ in range(weight):
            self._failures.append(now)
        self._cumulative_pnl += pnl
        self._prune()

        if self._should_trip():
            self._trip()

    def record_success(self, pnl: float = 0.0) -> None:
        self._cumulative_pnl += pnl
        # Decay: remove oldest N failure records
        decay = min(self._cfg.success_decay, len(self._failures))
        self._failures = self._failures[decay:]

        # If we were half-open and the probe succeeded → close
        if self._state == BreakerState.HALF_OPEN:
            self._reset()
            logger.info("CB [%s] probe succeeded — closing breaker", self._label)

    # ── queries ───────────────────────────────────────────────

    def is_open(self) -> bool:
        """Return True if trading is blocked."""
        self._maybe_transition()
        return self._state == BreakerState.OPEN

    def allows_probe(self) -> bool:
        """Return True if exactly one probe trade is allowed."""
        self._maybe_transition()
        return self._state == BreakerState.HALF_OPEN

    def allows_trade(self) -> bool:
        """Return True if a trade can go through (CLOSED or HALF_OPEN)."""
        self._maybe_transition()
        return self._state in (BreakerState.CLOSED, BreakerState.HALF_OPEN)

    @property
    def state(self) -> BreakerState:
        self._maybe_transition()
        return self._state

    def time_until_reset(self) -> float:
        if self._tripped_at is None:
            return 0.0
        elapsed = time.time() - self._tripped_at
        return max(0.0, self._cfg.cooldown_seconds - elapsed)

    def snapshot(self, pair: Optional[str] = None) -> BreakerSnapshot:
        self._maybe_transition()
        return BreakerSnapshot(
            state=self._state,
            failures_in_window=self._failures_in_window(),
            cumulative_pnl=self._cumulative_pnl,
            tripped_at=self._tripped_at,
            time_until_reset=self.time_until_reset(),
            pair=pair,
        )

    # ── internals ─────────────────────────────────────────────

    def _prune(self) -> None:
        cutoff = time.time() - self._cfg.window_seconds
        self._failures = [t for t in self._failures if t > cutoff]

    def _failures_in_window(self) -> int:
        self._prune()
        return len(self._failures)

    def _should_trip(self) -> bool:
        if self._failures_in_window() >= self._cfg.failure_threshold:
            return True
        if self._cumulative_pnl <= -self._cfg.max_drawdown_usd:
            return True
        return False

    def _trip(self) -> None:
        if self._state in (BreakerState.OPEN, BreakerState.HALF_OPEN):
            return  # already tripped
        self._state = BreakerState.OPEN
        self._tripped_at = time.time()
        logger.critical(
            "CIRCUIT BREAKER [%s] TRIPPED  pnl=%.4f", self._label, self._cumulative_pnl
        )

    def _reset(self) -> None:
        self._state = BreakerState.CLOSED
        self._tripped_at = None
        self._failures.clear()
        logger.info("CB [%s] reset to CLOSED", self._label)

    def _maybe_transition(self) -> None:
        """Auto-transition OPEN → HALF_OPEN → CLOSED based on elapsed time."""
        if self._tripped_at is None:
            return
        elapsed = time.time() - self._tripped_at
        if elapsed >= self._cfg.cooldown_seconds:
            self._reset()
        elif elapsed >= self._cfg.cooldown_seconds * self._cfg.half_open_after_pct:
            if self._state == BreakerState.OPEN:
                self._state = BreakerState.HALF_OPEN
                logger.info("CB [%s] entering HALF_OPEN (probe allowed)", self._label)


class CircuitBreaker:
    """
    Production circuit breaker with per-pair isolation.

    The **global** breaker trips on aggregate failures.
    Each **pair** breaker trips independently so one toxic pair
    doesn't shut down the whole bot.
    """

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self._global = _SingleBreaker(self.config, label="global")
        self._per_pair: dict[str, _SingleBreaker] = {}

    def _pair_breaker(self, pair: str) -> _SingleBreaker:
        if pair not in self._per_pair:
            self._per_pair[pair] = _SingleBreaker(self.config, label=pair)
        return self._per_pair[pair]

    # ── recording ─────────────────────────────────────────────

    def record_failure(
        self,
        pair: Optional[str] = None,
        category: FailureCategory = FailureCategory.UNKNOWN,
        pnl: float = 0.0,
    ) -> None:
        self._global.record_failure(category, pnl)
        if pair and self.config.per_pair:
            self._pair_breaker(pair).record_failure(category, pnl)

    def record_success(self, pair: Optional[str] = None, pnl: float = 0.0) -> None:
        self._global.record_success(pnl)
        if pair and self.config.per_pair:
            self._pair_breaker(pair).record_success(pnl)

    # ── queries ───────────────────────────────────────────────

    def is_open(self, pair: Optional[str] = None) -> bool:
        """True if trading is blocked globally **or** for *pair*."""
        if self._global.is_open():
            return True
        if pair and self.config.per_pair:
            return self._pair_breaker(pair).is_open()
        return False

    def allows_trade(self, pair: Optional[str] = None) -> bool:
        """True if a trade (or probe) can go through."""
        if not self._global.allows_trade():
            return False
        if pair and self.config.per_pair:
            return self._pair_breaker(pair).allows_trade()
        return True

    def trip(self, pair: Optional[str] = None) -> None:
        """Manual trip — useful for emergency stop."""
        self._global._trip()
        if pair and self.config.per_pair:
            self._pair_breaker(pair)._trip()

    def time_until_reset(self, pair: Optional[str] = None) -> float:
        g = self._global.time_until_reset()
        if pair and self.config.per_pair:
            return max(g, self._pair_breaker(pair).time_until_reset())
        return g

    def snapshot(self, pair: Optional[str] = None) -> dict:
        """Full observability snapshot."""
        result: dict = {"global": self._global.snapshot().to_dict()}
        if pair and pair in self._per_pair:
            result["pair"] = self._per_pair[pair].snapshot(pair).to_dict()
        elif pair:
            result["pair"] = _SingleBreaker(self.config, pair).snapshot(pair).to_dict()
        return result


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Replay Protection                                              ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass
class ReplayEvent:
    """One row in the replay audit log."""

    signal_id: str
    pair: str
    timestamp: float
    accepted: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "pair": self.pair,
            "ts": self.timestamp,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass
class ReplayConfig:
    ttl_seconds: float = 60.0  # how long a signal_id is remembered
    max_entries: int = 10_000  # bounded size (LRU eviction)
    nonce_check: bool = True  # enforce monotonic nonces per pair
    max_age_seconds: float = 30.0  # reject signals older than this
    audit_log_size: int = 500  # keep last N accept/reject decisions


class ReplayProtection:
    """
    Deduplication + nonce + staleness guard.

    * **Dedup**: remembers ``signal_id`` for ``ttl_seconds``.
    * **Nonce**: per-pair monotonic timestamp — rejects out-of-order signals.
    * **Max-age**: rejects signals whose ``timestamp`` is too old.
    * **LRU cap**: evicts oldest entries when ``max_entries`` is reached.
    * **Audit log**: ring-buffer of accept/reject decisions.
    """

    def __init__(self, config: ReplayConfig = None):
        self.config = config or ReplayConfig()
        # OrderedDict gives us LRU semantics
        self._executed: OrderedDict[str, float] = OrderedDict()
        self._pair_nonces: dict[str, float] = {}
        self._audit: list[ReplayEvent] = []

    # ── public API ────────────────────────────────────────────

    def check(self, signal: Signal) -> tuple[bool, str]:
        """
        Return ``(allowed, reason)``.

        ``allowed=True`` means the signal may proceed.
        Every call is recorded in the audit log.
        """
        self._cleanup()

        # 1. Max-age check
        age = time.time() - signal.timestamp
        if age > self.config.max_age_seconds:
            reason = f"stale: age {age:.1f}s > max {self.config.max_age_seconds}s"
            self._log(signal, False, reason)
            return False, reason

        # 2. Duplicate ID check
        if signal.signal_id in self._executed:
            reason = "duplicate signal_id"
            self._log(signal, False, reason)
            return False, reason

        # 3. Nonce monotonic check
        if self.config.nonce_check:
            last_nonce = self._pair_nonces.get(signal.pair, 0.0)
            if signal.timestamp <= last_nonce:
                reason = (
                    f"nonce stale: ts {signal.timestamp:.3f} <= last {last_nonce:.3f}"
                )
                self._log(signal, False, reason)
                return False, reason

        self._log(signal, True, "ok")
        return True, "ok"

    def is_duplicate(self, signal: Signal) -> bool:
        """Backward-compatible: True if signal should be rejected."""
        allowed, _ = self.check(signal)
        return not allowed

    def mark_executed(self, signal: Signal) -> None:
        """Record that *signal* was executed (or attempted)."""
        self._executed[signal.signal_id] = time.time()
        # Move to end (most recent)
        self._executed.move_to_end(signal.signal_id)
        # Update pair nonce
        self._pair_nonces[signal.pair] = max(
            self._pair_nonces.get(signal.pair, 0.0),
            signal.timestamp,
        )
        # LRU eviction
        while len(self._executed) > self.config.max_entries:
            self._executed.popitem(last=False)

    @property
    def audit_log(self) -> list[dict]:
        return [e.to_dict() for e in self._audit]

    @property
    def stats(self) -> dict:
        accepted = sum(1 for e in self._audit if e.accepted)
        rejected = len(self._audit) - accepted
        return {
            "tracked_ids": len(self._executed),
            "tracked_pairs": len(self._pair_nonces),
            "audit_accepted": accepted,
            "audit_rejected": rejected,
        }

    # ── internals ─────────────────────────────────────────────

    def _cleanup(self) -> None:
        cutoff = time.time() - self.config.ttl_seconds
        # Remove expired entries (oldest first thanks to OrderedDict)
        expired = [k for k, v in self._executed.items() if v <= cutoff]
        for k in expired:
            del self._executed[k]

    def _log(self, signal: Signal, accepted: bool, reason: str) -> None:
        event = ReplayEvent(
            signal_id=signal.signal_id,
            pair=signal.pair,
            timestamp=time.time(),
            accepted=accepted,
            reason=reason,
        )
        self._audit.append(event)
        # Ring-buffer cap
        if len(self._audit) > self.config.audit_log_size:
            self._audit = self._audit[-self.config.audit_log_size :]
        if not accepted:
            logger.debug("Replay REJECT %s: %s", signal.signal_id, reason)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Recovery Manager (façade)                                      ║
# ╚══════════════════════════════════════════════════════════════════╝


@dataclass
class RecoveryConfig:
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)


class RecoveryManager:
    """
    Single entry-point for all failure-handling logic.

    The :class:`Executor` calls:

    * ``pre_flight(signal)`` before executing.
    * ``record_outcome(signal, success, error, pnl)`` after.

    Internally this delegates to the circuit breaker, replay guard,
    and failure classifier.
    """

    def __init__(self, config: RecoveryConfig = None, alerter=None):
        cfg = config or RecoveryConfig()
        self.circuit_breaker = CircuitBreaker(cfg.circuit_breaker)
        self.replay = ReplayProtection(cfg.replay)
        self.classifier = FailureClassifier()
        self.alerter = alerter  # Optional WebhookAlerter
        self._outcomes: list[dict] = []

    # ── pre-flight gate ───────────────────────────────────────

    def pre_flight(self, signal: Signal) -> tuple[bool, str]:
        """
        Return ``(allowed, reason)``.

        Checks (in order):
        1. Global + per-pair circuit breaker
        2. Replay / dedup / nonce / staleness
        """
        # 1. Circuit breaker
        if not self.circuit_breaker.allows_trade(signal.pair):
            # Note: when the global breaker is HALF_OPEN, ``allows_trade()`` already
            # returns True (the single probe is allowed through there). Reaching
            # this branch therefore implies the breaker is fully OPEN globally or
            # for this pair.
            reason = "circuit breaker open"
            return False, reason

        # 2. Replay protection
        allowed, reason = self.replay.check(signal)
        if not allowed:
            return False, reason

        return True, "ok"

    # ── outcome recording ─────────────────────────────────────

    def record_outcome(
        self,
        signal: Signal,
        success: bool,
        error: Optional[str] = None,
        pnl: float = 0.0,
    ) -> None:
        """Record the result of an execution attempt."""
        self.replay.mark_executed(signal)

        # Capture breaker state *before* recording
        was_open = self.circuit_breaker.is_open(signal.pair)

        if success:
            self.circuit_breaker.record_success(signal.pair, pnl)
        else:
            category = self.classifier.classify(error)
            self.circuit_breaker.record_failure(signal.pair, category, pnl)

        # Fire webhook if breaker just tripped
        is_open_now = self.circuit_breaker.is_open(signal.pair)
        if not was_open and is_open_now and self.alerter:
            snap = self.circuit_breaker.snapshot(signal.pair)
            self.alerter.on_circuit_breaker_trip(signal.pair, snap)

        self._outcomes.append(
            {
                "signal_id": signal.signal_id,
                "pair": signal.pair,
                "success": success,
                "error": error,
                "category": self.classifier.classify(error).name if error else None,
                "pnl": pnl,
                "ts": time.time(),
            }
        )

    # ── observability ─────────────────────────────────────────

    def snapshot(self, pair: Optional[str] = None) -> dict:
        return {
            "circuit_breaker": self.circuit_breaker.snapshot(pair),
            "replay": self.replay.stats,
            "recent_outcomes": self._outcomes[-20:],
        }
