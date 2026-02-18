from __future__ import annotations

"""
Absolute, non-configurable safety limits for live trading.

These values are intentionally hard-coded. They should not be changed at
runtime or overridden via environment variables or config files.
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple

# safety.py — DO NOT MODIFY THESE VALUES

ABSOLUTE_MAX_TRADE_USD = 25.0  # Hard ceiling on any single trade
ABSOLUTE_MAX_DAILY_LOSS = 20.0  # Hard ceiling on daily loss
ABSOLUTE_MIN_CAPITAL = 50.0  # Auto-stop if total capital < $50
ABSOLUTE_MAX_TRADES_PER_HOUR = 30  # Prevent runaway loops


def safety_check(
    trade_usd: float,
    daily_loss: float,
    total_capital: float,
    trades_this_hour: int,
) -> Tuple[bool, str]:
    """Final safety gate — runs AFTER all other checks."""
    if trade_usd > ABSOLUTE_MAX_TRADE_USD:
        return False, f"Trade ${trade_usd:.0f} exceeds absolute max"
    if daily_loss <= -ABSOLUTE_MAX_DAILY_LOSS:
        return False, "Absolute daily loss limit reached"
    if total_capital < ABSOLUTE_MIN_CAPITAL:
        return False, f"Capital ${total_capital:.0f} below minimum"
    if trades_this_hour >= ABSOLUTE_MAX_TRADES_PER_HOUR:
        return False, "Absolute hourly trade limit reached"
    return True, "OK"


# Kill switch — simple file-based emergency stop shared across bots.
# Use the OS temp directory so it works cross-platform.
KILL_SWITCH_FILE = str(Path(tempfile.gettempdir()) / "arb_bot_kill")


def is_kill_switch_active() -> bool:
    """Return True if the kill-switch file exists."""
    return os.path.exists(KILL_SWITCH_FILE)
