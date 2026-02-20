#!/usr/bin/env python3
"""
Demonstrate daily loss limit behavior for oral defense.

Shows exactly what happens when ABSOLUTE_MAX_DAILY_LOSS is reached.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safety import ABSOLUTE_MAX_DAILY_LOSS, safety_check  # noqa: E402


def main():
    """Simulate trades until daily loss limit is hit."""
    print("=" * 70)
    print("Daily Loss Limit Demonstration")
    print("=" * 70)
    print("\nStarting capital: $100.00")
    print(f"Daily loss limit: ${ABSOLUTE_MAX_DAILY_LOSS:.2f}")
    print("Trade size: $5.00")
    print("\n" + "-" * 70)
    print("Simulating trades...\n")

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
            trades_this_hour=i,  # Incrementing count
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

    # Show what happens if we try one more trade
    if cumulative_pnl <= -ABSOLUTE_MAX_DAILY_LOSS:
        print("\n" + "-" * 70)
        print("Attempting one more trade after limit is hit...\n")
        cumulative_pnl += -2.0
        daily_loss = min(0.0, cumulative_pnl)
        total_capital = starting_capital + cumulative_pnl

        allowed, reason = safety_check(
            trade_usd=5.0,
            daily_loss=daily_loss,
            total_capital=total_capital,
            trades_this_hour=5,
        )

        print("Trade 5 (attempted):")
        print(f"  Daily Loss:       ${daily_loss:.2f}")
        print("  Status:           [BLOCKED]")
        print(f"  Reason:           {reason}")
        print("\n" + "=" * 70)
        print("Bot remains blocked until daily loss resets (next day).")
        print("=" * 70)


if __name__ == "__main__":
    main()
