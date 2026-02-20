"""
Session-scoped stats (total PnL, trade count) for Telegram /stats and /status.

Updated when a trade completes (e.g. Double Limit SUCCESS); read by Telegram bot.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class SessionStats:
    """Mutable session stats; thread-safe for single process."""

    total_pnl_usd: float = 0.0
    trade_count: int = 0
    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_trade(self, profit_usd: float) -> None:
        with self._lock:
            self.trade_count += 1
            self.total_pnl_usd += float(profit_usd)

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_pnl_usd": round(self.total_pnl_usd, 4),
                "trade_count": self.trade_count,
                "started_at": self.started_at,
            }

    def format_for_telegram(self) -> str:
        s = self.summary()
        return f"Session PnL: ${s['total_pnl_usd']:.2f}  |  Trades: {s['trade_count']}"


# Singleton used by Double Limit (and optionally arb_bot) and Telegram bot
_session_stats = SessionStats()


def get_session_stats() -> SessionStats:
    return _session_stats


def record_trade(profit_usd: float) -> None:
    _session_stats.record_trade(profit_usd)
