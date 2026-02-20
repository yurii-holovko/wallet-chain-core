#!/usr/bin/env python3
"""
Quick risk management demo script for oral defense.

Demonstrates:
1. Kill switch trigger
2. Circuit breaker settings
3. Risk limits
4. Daily loss limit behavior
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from executor.recovery import CircuitBreakerConfig  # noqa: E402
from safety import (  # noqa: E402
    ABSOLUTE_MAX_DAILY_LOSS,
    ABSOLUTE_MAX_TRADE_USD,
    ABSOLUTE_MAX_TRADES_PER_HOUR,
    ABSOLUTE_MIN_CAPITAL,
    KILL_SWITCH_FILE,
    is_kill_switch_active,
    safety_check,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_kill_switch():
    """Demonstrate kill switch."""
    print_section("1. Kill Switch")

    kill_file = Path(KILL_SWITCH_FILE)
    print("\nKill switch file location:")
    print(f"  {kill_file}")

    # Check current status
    active = is_kill_switch_active()
    print(f"\nCurrent status: {'ACTIVE' if active else 'INACTIVE'}")

    if not active:
        print("\nTo activate:")
        print(
            '  Windows PowerShell: New-Item -Path "$env:TEMP\\arb_bot_kill" '
            "-ItemType File -Force"
        )
        print(f"  Linux/Mac: touch {kill_file}")
        print(f"  Python: Path('{kill_file}').touch()")
    else:
        print("\n⚠️  Kill switch is ACTIVE - bot will not place trades!")
        print(
            '\nTo deactivate: Remove-Item "$env:TEMP\\arb_bot_kill" '
            "-ErrorAction SilentlyContinue"
        )


def demo_circuit_breaker():
    """Show circuit breaker settings."""
    print_section("2. Circuit Breaker Settings")

    cfg = CircuitBreakerConfig()

    print("\nConfiguration:")
    print(f"  Failure threshold:     {cfg.failure_threshold} failures")
    window_min = cfg.window_seconds / 60
    print(
        f"  Window:                {cfg.window_seconds:.0f} seconds "
        f"({window_min:.1f} minutes)"
    )
    print(f"  Max drawdown:          ${cfg.max_drawdown_usd:.2f}")
    cooldown_min = cfg.cooldown_seconds / 60
    print(
        f"  Cooldown:              {cfg.cooldown_seconds:.0f} seconds "
        f"({cooldown_min:.1f} minutes)"
    )
    half_open_min = cfg.cooldown_seconds * cfg.half_open_after_pct / 60
    print(
        f"  Half-open after:       {cfg.half_open_after_pct*100:.0f}% of cooldown "
        f"({half_open_min:.1f} min)"
    )
    print(
        f"  Success decay:         {cfg.success_decay} failure(s) removed per success"
    )
    print(f"  Per-pair isolation:    {cfg.per_pair}")

    print("\nTrip Conditions (OR logic):")
    print("  1. 3 failures within 5 minutes")
    print("  2. Cumulative PnL <= -$50")

    print("\nState Machine:")
    print("  CLOSED -> OPEN -> HALF_OPEN -> CLOSED")
    print("  - CLOSED: Normal operation")
    print("  - OPEN: Tripped, no trades")
    print("  - HALF_OPEN: After 8 min, allows 1 probe trade")
    print("  - CLOSED: If probe succeeds, reset immediately")


def demo_risk_limits():
    """Show risk limits."""
    print_section("3. Risk Limits (Absolute Safety Constants)")

    print("\nHard-coded limits (from src/safety.py):")
    print(f"  Max trade size:        ${ABSOLUTE_MAX_TRADE_USD:.2f}")
    print(f"  Max daily loss:        ${ABSOLUTE_MAX_DAILY_LOSS:.2f}")
    print(f"  Min capital:           ${ABSOLUTE_MIN_CAPITAL:.2f}")
    print(f"  Max trades/hour:       {ABSOLUTE_MAX_TRADES_PER_HOUR}")

    print("\nRationale (for ~$100 starting capital):")
    print(
        f"  - Max trade ${ABSOLUTE_MAX_TRADE_USD:.0f}: Single trade <= 25% of capital"
    )
    print(f"  - Daily loss ${ABSOLUTE_MAX_DAILY_LOSS:.0f}: Max 20% drawdown per day")
    print(
        f"  - Min capital ${ABSOLUTE_MIN_CAPITAL:.0f}: Halt at 50% of starting capital"
    )
    print(f"  - Max trades/hour {ABSOLUTE_MAX_TRADES_PER_HOUR}: Prevents runaway loops")

    print("\nSafety check order:")
    print("  1. Trade size <= $25?")
    print("  2. Daily loss >= -$20?")
    print("  3. Total capital >= $50?")
    print("  4. Trades/hour < 30?")


def demo_daily_loss_limit():
    """Demonstrate daily loss limit behavior."""
    print_section("4. Daily Loss Limit Behavior")

    print(f"\nDaily loss limit: ${ABSOLUTE_MAX_DAILY_LOSS:.2f}")
    print("Starting capital: $100.00")
    print("Trade size: $5.00")
    print("\nSimulating trades...\n")

    starting_capital = 100.0
    cumulative_pnl = 0.0
    trades = [
        -5.0,  # Trade 1: lose $5
        -8.0,  # Trade 2: lose $8
        -7.0,  # Trade 3: lose $7 (total: -$20, at limit)
        -1.0,  # Trade 4: would lose $1 (total: -$21, exceeds limit)
    ]

    for i, trade_pnl in enumerate(trades, 1):
        cumulative_pnl += trade_pnl
        daily_loss = min(0.0, cumulative_pnl)  # Only negative values
        total_capital = starting_capital + cumulative_pnl

        allowed, reason = safety_check(
            trade_usd=5.0,
            daily_loss=daily_loss,
            total_capital=total_capital,
            trades_this_hour=i,
        )

        status_icon = "[ALLOWED]" if allowed else "[BLOCKED]"
        print(f"Trade {i}:")
        print(f"  Trade PnL:        ${trade_pnl:+.2f}")
        print(f"  Cumulative PnL:   ${cumulative_pnl:+.2f}")
        print(f"  Daily Loss:       ${daily_loss:.2f}")
        print(f"  Total Capital:    ${total_capital:.2f}")
        print(f"  Status:           {status_icon}")
        if not allowed:
            print(f"  Reason:           {reason}")
            print("\n" + "=" * 70)
            print("RESULT: Daily loss limit reached. Bot will stop trading.")
            print("=" * 70)
            break
        print()

    print("\nKey points:")
    print("  - Daily loss accumulates across all pairs")
    print("  - At limit (-$20): Trade is still allowed")
    print("  - Beyond limit (-$21): Trade is blocked")
    print("  - No auto-recovery: Limit persists until manually reset")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  Risk Management Demo for Oral Defense")
    print("=" * 70)

    demo_kill_switch()
    demo_circuit_breaker()
    demo_risk_limits()
    demo_daily_loss_limit()

    print("\n" + "=" * 70)
    print("  Demo Complete")
    print("=" * 70)
    print("\nAll risk management features demonstrated.")
    print("Ready for questions!")


if __name__ == "__main__":
    main()
