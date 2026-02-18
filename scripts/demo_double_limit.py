from __future__ import annotations

"""
Demo script for the Double Limit micro-arbitrage components.

This runs in *observation* mode only: it evaluates opportunities between
MEXC and Arbitrum via ODOS but does NOT place real orders.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure project root and src/ are on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
for path in (ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chain import ChainClient  # noqa: E402
from config import get_env  # noqa: E402
from config_tokens_arb_mex import TOKEN_MAPPINGS  # noqa: E402
from core.capital_manager import CapitalManager, CapitalManagerConfig  # noqa: E402
from core.wallet_manager import WalletManager  # noqa: E402
from exchange.mexc_client import MexcClient  # noqa: E402
from exchange.uniswap_v3_range import UniswapV3RangeOrderManager  # noqa: E402
from executor.double_limit_engine import (  # noqa: E402
    DoubleLimitArbitrageEngine,
    DoubleLimitConfig,
)
from executor.execution_report import format_double_limit_report  # noqa: E402
from pricing.odos_client import OdosClient  # noqa: E402
from safety import is_kill_switch_active  # noqa: E402
from telegram_bot import (  # noqa: E402
    TelegramBot,
    TelegramBotConfig,
    TelegramLogHandler,
    add_telegram_log_handler,
)


def _live_execution_enabled() -> bool:
    """True if ENABLE_LIVE_EXECUTION is set to 1 or true (case-insensitive)."""
    v = (get_env("ENABLE_LIVE_EXECUTION", "0") or "0").strip().lower()
    return v in ("1", "true", "yes")


def _root_has_telegram_handler() -> bool:
    for h in logging.getLogger().handlers:
        if isinstance(h, TelegramLogHandler):
            return True
    return False


async def main(
    trade_size_usd: float | None = None,
    enable_live_execution: bool | None = None,
    telegram_bot: TelegramBot | None = None,
) -> None:
    """
    Run Double Limit micro-arbitrage demo.

    By default runs in observation-only mode (evaluate + log, no orders).
    To place real orders (both legs: MEXC limit + Uniswap V3 range), set
    ENABLE_LIVE_EXECUTION=1 in .env or pass enable_live_execution=True.

    Args:
        trade_size_usd: Trade size in USD. If None, uses TRADE_SIZE_USD env or 5.0.
        enable_live_execution: If True, place real orders. If None, use env
            ENABLE_LIVE_EXECUTION (default False).
    """
    # Structured logging to both file and stdout
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    log_dir = _Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"double_limit_{_dt.now():%Y%m%d}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s |%(levelname)s |%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    # Wire logs to Telegram when standalone (arb_bot adds handler when run via it)
    report_sender = None
    _tg = None
    if not _root_has_telegram_handler():
        _tg = telegram_bot or TelegramBot(TelegramBotConfig.from_env())
        if _tg.config.enabled:
            add_telegram_log_handler(_tg)
            if telegram_bot is None:
                _tg.start()
            report_sender = _tg.send
    elif telegram_bot and telegram_bot.config.enabled:
        report_sender = telegram_bot.send

    mexc = MexcClient()
    odos = OdosClient()

    cap_cfg = CapitalManagerConfig()
    capital = CapitalManager(mexc, config=cap_cfg)

    # Determine trade size: command line arg > env var > default
    if trade_size_usd is None:
        trade_size_usd = float(get_env("TRADE_SIZE_USD", "5.0") or "5.0")

    logging.info("Using trade size: $%.2f USD", trade_size_usd)

    cfg = DoubleLimitConfig(
        trade_size_usd=trade_size_usd,
        # For demo purposes, default to looser thresholds; override via env in prod.
        min_spread_pct=float(get_env("MIN_SPREAD_PCT", "0.008") or "0.008"),
        min_profit_usd=float(get_env("MIN_PROFIT_USD", "0.005") or "0.005"),
        max_slippage_pct=float(get_env("MAX_SLIPPAGE_PCT", "0.5") or "0.5"),
    )

    # Optional: Uniswap V3 range-order manager for full Double Limit.
    # Requires ARBITRUM_RPC_HTTPS, PRIVATE_KEY, USDC_ADDRESS.
    #
    # Pools can be specified via TOKEN_MAPPINGS['v3_pool'] or auto-resolved
    # via Uniswap V3 factory using (USDC, token, fee_tier).
    range_manager = None
    try:
        rpc_https = get_env("ARBITRUM_RPC_HTTPS")
        usdc_address = get_env("USDC_ADDRESS")
        private_key = get_env("PRIVATE_KEY")
        if rpc_https and usdc_address and private_key:
            chain_client = ChainClient([rpc_https])
            wallet = WalletManager(private_key)
            # Build pool config map for all tokens in TOKEN_MAPPINGS
            pool_cfg: dict[str, dict] = {}
            for meta in TOKEN_MAPPINGS.values():
                token_addr = str(meta["address"]).lower()
                pool_cfg[token_addr] = {
                    "pool": meta.get("v3_pool"),
                    "fee_tier": meta.get("fee_tier", 3_000),
                }
            if pool_cfg:
                range_manager = UniswapV3RangeOrderManager(
                    client=chain_client,
                    wallet=wallet,
                    usdc_address=usdc_address,
                    pool_config=pool_cfg,
                )
                logging.info(
                    "Uniswap V3 range manager enabled for tokens: %s",
                    ", ".join(sorted(pool_cfg.keys())),
                )
            else:
                logging.info(
                    "Uniswap V3 range manager not enabled: no v3_pool in TOKEN_MAPPINGS"
                )
        else:
            logging.info(
                "Uniswap V3 range manager disabled (missing RPC/USDC/PRIVATE_KEY)"
            )
    except Exception as exc:
        logging.warning("Failed to initialize Uniswap V3 range manager: %s", exc)
        range_manager = None

    # Use only tokens that are active and ODOS-supported by default.
    active_tokens = {
        k: v
        for k, v in TOKEN_MAPPINGS.items()
        if v.get("active") and v.get("odos_supported")
    }

    # Or manually restrict to a test subset if desired.
    test_symbols = [
        "ARB",
        "GMX",
        "MAGIC",
        "GNS",
        "RDNT",
        "PENDLE",
        "LINK",
        "UNI",
        "AAVE",
        "CRV",
    ]
    token_universe = {
        k: active_tokens[k] for k in test_symbols if k in active_tokens
    } or active_tokens

    engine = DoubleLimitArbitrageEngine(
        mexc_client=mexc,
        odos_client=odos,
        token_mappings=token_universe,
        config=cfg,
        range_manager=range_manager,
        capital_manager=capital,
    )

    symbols = list(token_universe.keys())
    live = (
        enable_live_execution
        if enable_live_execution is not None
        else _live_execution_enabled()
    )
    if live:
        logging.warning(
            "LIVE EXECUTION ENABLED — real MEXC + Uniswap V3 orders will be placed."
        )
    else:
        logging.info("Observation mode — no orders will be placed.")
    logging.info("Starting Double Limit demo for tokens: %s", ", ".join(symbols))

    try:
        kill_logged = False
        while True:
            if is_kill_switch_active():
                if not kill_logged:
                    logging.warning("Kill switch active — Double Limit demo PAUSED.")
                    kill_logged = True
                await asyncio.sleep(1)
                continue
            else:
                if kill_logged:
                    logging.info("Kill switch cleared — Double Limit demo RESUMING.")
                    kill_logged = False

            now = time.strftime("%H:%M:%S")
            for key in symbols:
                opp = engine.evaluate_opportunity(key)
                if not opp:
                    continue
                status = "EXECUTABLE" if opp.executable else "SKIP"
                msg = (
                    "[%s] %s  mex_bid=%.4f mex_ask=%.4f arb=%.4f  "
                    "spread=%.2f%%  net=$%.4f (%+.2f%%)  %s"
                ) % (
                    now,
                    key,
                    opp.mex_bid,
                    opp.mex_ask,
                    opp.odos_price,
                    opp.gross_spread * 100,
                    opp.net_profit_usd,
                    opp.net_profit_pct * 100,
                    status,
                )
                if opp.executable:
                    logging.warning(msg)  # higher level so Telegram shows EXECUTABLE
                else:
                    logging.info(msg)
                if live and opp.executable:
                    logging.info("Executing double limit for %s (both legs)...", key)
                    try:
                        result = await engine.execute_double_limit(opp)
                        logging.info(
                            "Execute result: status=%s opportunity=%s",
                            result.get("status"),
                            key,
                        )
                        if result.get("status") == "SUCCESS":
                            logging.info(
                                "Both legs filled: mex_order=%s v3_status=%s",
                                result.get("mex_order"),
                                result.get("v3_status"),
                            )
                        if report_sender:
                            try:
                                report_sender(format_double_limit_report(result, opp))
                            except Exception as send_exc:
                                logging.warning(
                                    "Execution report send failed: %s", send_exc
                                )
                        # Cooldown after an execution to avoid hammering
                        await asyncio.sleep(30)
                    except Exception as exc:
                        logging.exception("Execution failed for %s: %s", key, exc)
            await asyncio.sleep(5)
    except KeyboardInterrupt:
        logging.info("Stopping demo.")


if __name__ == "__main__":
    asyncio.run(main())
