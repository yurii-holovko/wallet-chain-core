from __future__ import annotations

"""
Test/preview all Telegram-delivered outputs without placing real trades.

What it exercises:
  - TelegramLogHandler forwarding (INFO/WARNING/ERROR)
  - CEX/DEX execution report formatting (Executor/ExecutionContext)
  - Double Limit execution report formatting
  - Kill-switch response messages (as they appear in chat)

Modes:
  - default (dry): does NOT call Telegram API; prints exactly what would be sent
  - --live: sends to Telegram using TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from env
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root and src/ are on sys.path so imports work from scripts/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exchange.mexc_client import MexcOrderStatus  # noqa: E402
from executor.double_limit_engine import DoubleLimitOpportunity  # noqa: E402
from executor.engine import ExecutionContext, ExecutorState  # noqa: E402
from executor.execution_report import (  # noqa: E402
    format_cex_dex_execution_report,
    format_double_limit_report,
)
from strategy.signal import Direction, Signal  # noqa: E402
from telegram_bot import (  # noqa: E402
    TelegramBot,
    TelegramBotConfig,
    TelegramLogHandler,
)


def _make_fake_signal() -> Signal:
    now = time.time()
    return Signal.create(
        pair="ETH/USDT",
        direction=Direction.BUY_CEX_SELL_DEX,
        cex_price=3000.0,
        dex_price=3012.0,
        spread_bps=40.0,
        size=0.01,
        expected_gross_pnl=1.20,
        expected_fees=0.45,
        expected_net_pnl=0.75,
        score=80.0,
        expiry=now + 30.0,
        inventory_ok=True,
        within_limits=True,
    )


def _make_fake_ctx_success() -> ExecutionContext:
    s = _make_fake_signal()
    ctx = ExecutionContext(signal=s)
    ctx.state = ExecutorState.DONE
    ctx.leg1_venue = "cex"
    ctx.leg1_order_id = "mexc_123456789"
    ctx.leg1_fill_price = 3001.0
    ctx.leg1_fill_size = 0.01
    ctx.leg2_venue = "dex"
    ctx.leg2_tx_hash = "0x" + "a" * 64
    ctx.leg2_fill_price = 3010.5
    ctx.leg2_fill_size = 0.01
    ctx.actual_net_pnl = 0.62
    ctx.finished_at = ctx.started_at + 0.8
    return ctx


def _make_fake_ctx_fail() -> ExecutionContext:
    s = _make_fake_signal()
    s.direction = Direction.BUY_DEX_SELL_CEX
    ctx = ExecutionContext(signal=s)
    ctx.state = ExecutorState.FAILED
    ctx.leg1_venue = "dex"
    ctx.leg2_venue = "cex"
    ctx.error = "DEX timeout"
    ctx.finished_at = ctx.started_at + 2.1
    return ctx


def _make_fake_double_limit() -> tuple[dict, DoubleLimitOpportunity]:
    opp = DoubleLimitOpportunity(
        token_symbol="ARB",
        token_address="0x912ce59144191c1204e64559fe8253a0e49e6548",
        mex_symbol="ARBUSDT",
        direction="mex_to_arb",
        mex_bid=1.2345,
        mex_ask=1.2350,
        odos_price=1.2400,
        gross_spread=0.0123,
        total_cost_usd=0.045,
        net_profit_usd=0.0165,
        net_profit_pct=0.0033,
        executable=True,
    )
    mex_order = MexcOrderStatus(
        order_id="987654321",
        symbol=opp.mex_symbol,
        side="BUY",
        status="FILLED",
        price=1.2340,
        orig_qty=10.0,
        executed_qty=10.0,
    )
    v3_status = {
        "pool": "0x" + "b" * 40,
        "fee_tier": 500,
        "in_range": False,
        "liquidity": 1,
        "is_executed": True,
    }
    result = {
        "status": "SUCCESS",
        "mex_order": mex_order,
        "v3_status": v3_status,
        "opportunity": opp,
    }
    return result, opp


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview Telegram bot outputs")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send to Telegram using TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID.",
    )
    args = parser.parse_args()

    # Windows consoles often default to non-UTF8 encodings; ensure preview prints.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )

    if args.live:
        bot = TelegramBot(TelegramBotConfig.from_env())
        if not bot.config.enabled:
            raise SystemExit(
                "Telegram bot not configured (missing TELEGRAM_BOT_TOKEN/CHAT_ID)."
            )

        def send_with_delay(text, reply_markup=None):
            """Send to Telegram with a small delay to avoid rate limits."""
            try:
                bot.send(text, reply_markup=reply_markup)
                time.sleep(0.5)  # Small delay between messages
            except Exception as exc:
                print(f"ERROR sending to Telegram: {exc}", file=sys.stderr)

        send = send_with_delay
        bot.start()
        print("Bot started. Sending preview messages to Telegram...")
        time.sleep(1)  # Give bot time to start polling
        # show buttons once
        bot.send_with_command_buttons("Preview: command buttons should appear below.")
        time.sleep(0.5)
    else:

        def send(text, reply_markup=None):
            print("\n=== TELEGRAM MESSAGE ===\n" + str(text) + "\n")

    # 1) Log forwarding preview (through TelegramLogHandler)
    print("1/4: Testing log forwarding (INFO/WARNING/ERROR)...")
    handler = TelegramLogHandler(lambda text: send(text), level=logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)

    logging.getLogger("test").info("INFO example: bot started (preview)")
    time.sleep(0.3)
    logging.getLogger("test").warning(
        "WARNING example: circuit breaker open for ETH/USDT"
    )
    time.sleep(0.3)
    try:
        raise RuntimeError("example exception")
    except Exception:
        logging.getLogger("test").exception("ERROR example: execution failed")
    time.sleep(0.5)

    # 2) CEX/DEX execution reports
    print("2/4: Testing CEX/DEX execution reports...")
    send(format_cex_dex_execution_report(_make_fake_ctx_success()))
    time.sleep(0.5)
    send(format_cex_dex_execution_report(_make_fake_ctx_fail()))
    time.sleep(0.5)

    # 3) Double Limit execution report
    print("3/4: Testing Double Limit execution report...")
    dl_result, dl_opp = _make_fake_double_limit()
    send(format_double_limit_report(dl_result, dl_opp))
    time.sleep(0.5)

    # 4) Kill-switch responses (as they appear)
    print("4/4: Testing kill-switch responses...")
    send("Kill switch ACTIVATED. Bots will stop shortly.")
    time.sleep(0.5)
    send("Kill switch status: ACTIVE")
    time.sleep(0.5)
    send("Kill switch CLEARED. You may restart bots.")
    time.sleep(0.5)

    if args.live:
        send("✅ Preview complete. Check your Telegram chat!")
        time.sleep(0.5)
        bot.stop()
        print("Done. Check your Telegram chat for all messages.")
    else:
        print("\n✅ Preview complete. Run with --live to send to Telegram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
