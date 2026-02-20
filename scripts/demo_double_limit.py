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

import requests

# Ensure project root and src/ are on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
for path in (ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chain import ChainClient  # noqa: E402
from config import get_env  # noqa: E402
from config_tokens_arb_mex import TOKEN_MAPPINGS  # noqa: E402
from core.base_types import Address, TokenAmount, TransactionRequest  # noqa: E402
from core.capital_manager import CapitalManager, CapitalManagerConfig  # noqa: E402
from core.wallet_manager import WalletManager  # noqa: E402
from exchange.dex_swap import DexSwapManager  # noqa: E402
from exchange.mexc_client import MexcClient  # noqa: E402
from executor.double_limit_engine import (  # noqa: E402
    DoubleLimitArbitrageEngine,
    DoubleLimitConfig,
    DoubleLimitOpportunity,
)
from executor.execution_report import format_double_limit_report  # noqa: E402
from executor.metrics import MetricsRegistry, MetricsServer  # noqa: E402
from executor.recovery import (  # noqa: E402
    CircuitBreaker,
    CircuitBreakerConfig,
    FailureClassifier,
)
from pricing.odos_client import OdosClient  # noqa: E402
from safety import is_kill_switch_active, safety_check  # noqa: E402
from session_stats import record_trade as record_session_trade  # noqa: E402
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


def _simulate_execution_enabled() -> bool:
    """True if SIMULATE_EXECUTION is set to 1 or true (case-insensitive)."""
    v = (get_env("SIMULATE_EXECUTION", "0") or "0").strip().lower()
    return v in ("1", "true", "yes")


def _resolve_eth_price_usd(default: float = 2600.0) -> float:
    """
    Best-effort ETH/USD resolver.

    Precedence:
      1) ETH_PRICE_USD env (exact override)
      2) Live fetch from a public API (CoinGecko)
      3) Static default (2600.0) on failure
    """
    raw = (get_env("ETH_PRICE_USD", "") or "").strip()
    if raw:
        try:
            val = float(raw)
            logging.info("Using ETH_PRICE_USD from env: $%.2f", val)
            return val
        except ValueError:
            logging.warning(
                "Invalid ETH_PRICE_USD value %r — falling back to live fetch", raw
            )

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "ethereum", "vs_currencies": "usd"},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        price = float(data["ethereum"]["usd"])
        logging.info("Fetched live ETH price from CoinGecko: $%.2f", price)
        return price
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        logging.warning(
            "Failed to fetch live ETH price, falling back to default $%.2f: %s",
            default,
            exc,
        )
        return default


def _root_has_telegram_handler() -> bool:
    for h in logging.getLogger().handlers:
        if isinstance(h, TelegramLogHandler):
            return True
    return False


# ── Balance verification (live mode only) ─────────────────────────────

# Arbitrum mainnet
ARBITRUM_CHAIN_ID = 42161


def _fetch_mexc_balances(mexc: MexcClient, base_symbol: str) -> dict[str, float]:
    """Return current MEXC free balances for USDT and base token."""
    account = mexc.get_account()
    out: dict[str, float] = {"USDT": 0.0, base_symbol: 0.0}
    for entry in account.get("balances", []):
        asset = entry.get("asset")
        if asset in out:
            out[asset] = float(entry.get("free", 0.0))
    return out


def _fetch_erc20_balance(
    chain_client: ChainClient,
    wallet_address: str,
    token_address: str,
    decimals: int,
    chain_id: int = ARBITRUM_CHAIN_ID,
) -> float:
    """Return ERC20 balance (human) for wallet at token contract."""
    selector = bytes.fromhex("70a08231")
    owner = Address.from_string(wallet_address).checksum
    calldata = selector + bytes.fromhex(owner[2:]).rjust(32, b"\x00")
    call = TransactionRequest(
        to=Address.from_string(token_address),
        value=TokenAmount(raw=0, decimals=18, symbol="ETH"),
        data=calldata,
        chain_id=chain_id,
    )
    raw = chain_client.call(call)
    amount_raw = int.from_bytes(raw, "big") if raw else 0
    return float(amount_raw) / (10**decimals)


def _fetch_arb_balances(
    chain_client: ChainClient,
    wallet: WalletManager,
    usdc_address: str,
    usdc_decimals: int,
    token_address: str,
    token_decimals: int,
) -> dict[str, float]:
    """Return current Arbitrum balances: USDC, ETH, and base token (human)."""
    addr = wallet.address
    eth_balance = chain_client.get_balance(Address.from_string(addr))
    return {
        "USDC": _fetch_erc20_balance(chain_client, addr, usdc_address, usdc_decimals),
        "ETH": float(eth_balance.human),
        "BASE": _fetch_erc20_balance(chain_client, addr, token_address, token_decimals),
    }


def _expected_balances_after_trade(
    opp: DoubleLimitOpportunity,
    mexc_before: dict[str, float],
    arb_before: dict[str, float],
    trade_size_usd: float,
    usdc_decimals: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute expected MEXC and Arbitrum balances after a filled double-limit trade.
    """
    trade_size_base = trade_size_usd / opp.odos_price
    base_symbol = opp.token_symbol
    if opp.direction == "mex_to_arb":
        mex_price = opp.mex_ask
        mexc_expected = {
            "USDT": mexc_before["USDT"] - trade_size_base * mex_price,
            base_symbol: mexc_before.get(base_symbol, 0.0) + trade_size_base,
        }
        arb_expected = {
            "USDC": arb_before["USDC"] - trade_size_usd,
            "ETH": arb_before["ETH"],
            "BASE": arb_before["BASE"] + trade_size_base,
        }
    else:
        mex_price = opp.mex_bid
        mexc_expected = {
            "USDT": mexc_before["USDT"] + trade_size_base * mex_price,
            base_symbol: mexc_before.get(base_symbol, 0.0) - trade_size_base,
        }
        arb_expected = {
            "USDC": arb_before["USDC"] + trade_size_usd,
            "ETH": arb_before["ETH"],
            "BASE": arb_before["BASE"] - trade_size_base,
        }
    return mexc_expected, arb_expected


def _fetch_double_limit_balances(
    mexc: MexcClient,
    chain_client: ChainClient,
    wallet: WalletManager,
    usdc_address: str,
    usdc_decimals: int,
    opp: DoubleLimitOpportunity,
    token_universe: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fetch MEXC and Arbitrum balances before a trade (for later verification)."""
    base_symbol = opp.token_symbol
    meta = token_universe.get(base_symbol, {})
    token_address = str(meta.get("address", opp.token_address))
    token_decimals = int(meta.get("decimals", 18))
    mexc_bal = _fetch_mexc_balances(mexc, base_symbol)
    arb_bal = _fetch_arb_balances(
        chain_client, wallet, usdc_address, usdc_decimals, token_address, token_decimals
    )
    return mexc_bal, arb_bal


def _verify_double_limit_balances(
    balances_before: tuple[dict[str, float], dict[str, float]],
    opp: DoubleLimitOpportunity,
    result: dict,
    mexc: MexcClient,
    chain_client: ChainClient,
    wallet: WalletManager,
    usdc_address: str,
    usdc_decimals: int,
    trade_size_usd: float,
    token_universe: dict,
) -> None:
    """
    After a successful Double Limit trade, verify actual balances match expected.
    On mismatch > tolerance: log critical and exit(1).
    """
    if result.get("status") != "SUCCESS":
        return
    mexc_before, arb_before = balances_before
    mexc_expected, arb_expected = _expected_balances_after_trade(
        opp, mexc_before, arb_before, trade_size_usd, usdc_decimals
    )
    base_symbol = opp.token_symbol
    meta = token_universe.get(base_symbol, {})
    token_address = str(meta.get("address", opp.token_address))
    token_decimals = int(meta.get("decimals", 18))

    mexc_actual = _fetch_mexc_balances(mexc, base_symbol)
    arb_actual = _fetch_arb_balances(
        chain_client, wallet, usdc_address, usdc_decimals, token_address, token_decimals
    )

    tolerance = 0.001
    mismatches: list[str] = []

    for asset in ("USDT", base_symbol):
        exp = mexc_expected.get(asset, 0.0)
        act = mexc_actual.get(asset, 0.0)
        if abs(act - exp) > tolerance:
            mismatches.append(f"MEXC {asset}: expected={exp:.6f} actual={act:.6f}")

    for key in ("USDC", "ETH", "BASE"):
        exp = arb_expected.get(key, 0.0)
        act = arb_actual.get(key, 0.0)
        if abs(act - exp) > tolerance:
            mismatches.append(f"Arb {key}: expected={exp:.6f} actual={act:.6f}")

    if mismatches:
        logging.critical(
            "BALANCE MISMATCH after Double Limit trade: %s",
            "; ".join(mismatches),
        )
        sys.exit(1)


async def main(
    trade_size_usd: float | None = None,
    enable_live_execution: bool | None = None,
    simulate_execution: bool | None = None,
    telegram_bot: TelegramBot | None = None,
    max_trades: int | None = None,
    tokens: list[str] | None = None,
) -> None:
    """
    Run Double Limit micro-arbitrage demo.

    By default runs in observation-only mode (evaluate + log, no orders).
    To place real orders (both legs: MEXC limit + ODOS DEX swap), set
    ENABLE_LIVE_EXECUTION=1 in .env or pass enable_live_execution=True.

    Args:
        trade_size_usd: Trade size in USD. If None, uses TRADE_SIZE_USD env or 5.0.
        enable_live_execution: If True, place real orders. If None, use env
            ENABLE_LIVE_EXECUTION (default False).
        max_trades: If set, stop after this many successful executions
            (e.g. 1 for one transaction).
        tokens: List of token symbols to track (e.g. ['LINK', 'ARB', 'GMX']).
            If None, uses all active and ODOS-supported tokens from TOKEN_MAPPINGS.
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

    # ── PRODUCTION flag — interpreted together with execution mode ─
    is_production = (get_env("PRODUCTION", "false") or "false").strip().lower() in (
        "1",
        "true",
        "yes",
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

    # ── Prometheus metrics + /health heartbeat endpoint ────────────
    metrics = MetricsRegistry()
    metrics_port = int(get_env("METRICS_PORT", "9090") or "9090")
    metrics_server = MetricsServer(metrics, port=metrics_port)
    try:
        metrics_server.start()
    except Exception as exc:
        logging.warning("Metrics server failed to start: %s", exc)

    # ── Circuit breaker (per-pair + global) ────────────────────────
    cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        window_seconds=300.0,
        max_drawdown_usd=float(get_env("CB_MAX_DRAWDOWN_USD", "50.0") or "50.0"),
        cooldown_seconds=600.0,
    )
    circuit_breaker = CircuitBreaker(cb_config)
    classifier = FailureClassifier()

    # ── Cumulative PnL tracker (for safety_check) ─────────────────
    cumulative_pnl: float = 0.0
    starting_capital_usd = float(get_env("STARTING_CAPITAL_USD", "100.0") or "100.0")
    trade_timestamps: list[float] = []

    def _trades_last_hour() -> int:
        now = time.time()
        cutoff = now - 3600.0
        while trade_timestamps and trade_timestamps[0] < cutoff:
            trade_timestamps.pop(0)
        return len(trade_timestamps)

    mexc = MexcClient()
    odos = OdosClient()

    cap_cfg = CapitalManagerConfig()
    capital = CapitalManager(mexc, config=cap_cfg)

    # Determine trade size: command line arg > env var > default
    if trade_size_usd is None:
        trade_size_usd = float(get_env("TRADE_SIZE_USD", "5.0") or "5.0")

    logging.info("Using trade size: $%.2f USD", trade_size_usd)

    live = (
        enable_live_execution
        if enable_live_execution is not None
        else _live_execution_enabled()
    )
    sim_exec = (
        simulate_execution
        if simulate_execution is not None
        else _simulate_execution_enabled()
    )
    simulation_mode = not (live and not sim_exec)

    # ── Mode banner: environment × execution ────────────────────────
    env_label = "*** PRODUCTION ***" if is_production else "TESTNET"
    if live and not sim_exec:
        exec_label = "LIVE EXECUTION"
    elif sim_exec:
        exec_label = "SIMULATED EXECUTION"
    else:
        exec_label = "OBSERVATION"

    logging.info("=" * 60)
    logging.info("  MODE: %s / %s", env_label, exec_label)
    logging.info("=" * 60)

    max_dex_gas = int(get_env("MAX_DEX_GAS_LIMIT", "1000000") or "1000000")
    eth_price = _resolve_eth_price_usd(default=2600.0)
    arb_gas_gwei = float(get_env("ARB_GAS_PRICE_GWEI", "0.1") or "0.1")

    cfg = DoubleLimitConfig(
        trade_size_usd=trade_size_usd,
        min_spread_pct=float(get_env("MIN_SPREAD_PCT", "0.004") or "0.004"),
        min_profit_usd=float(get_env("MIN_PROFIT_USD", "0.001") or "0.001"),
        max_slippage_pct=float(get_env("MAX_SLIPPAGE_PCT", "0.5") or "0.5"),
        simulation_mode=simulation_mode,
        max_dex_gas_limit=max_dex_gas,
        eth_price_usd=eth_price,
        arb_gas_price_gwei=arb_gas_gwei,
    )
    logging.info(
        "Gas model: max_gas=%d eth=$%.0f arb_gas=%.3f gwei -- "
        "200k gas ~ $%.4f, 500k gas ~ $%.4f, 1M gas ~ $%.4f",
        max_dex_gas,
        eth_price,
        arb_gas_gwei,
        cfg.estimate_gas_cost_usd(200_000),
        cfg.estimate_gas_cost_usd(500_000),
        cfg.estimate_gas_cost_usd(1_000_000),
    )

    # DEX swap manager: executes ODOS swaps on Arbitrum for the on-chain leg.
    # Requires ARBITRUM_RPC_HTTPS, PRIVATE_KEY, USDC_ADDRESS.
    dex_swap_mgr: DexSwapManager | None = None
    chain_client: ChainClient | None = None
    wallet: WalletManager | None = None
    usdc_address = get_env("USDC_ADDRESS")
    usdc_decimals = cfg.usdc_decimals
    try:
        rpc_https = get_env("ARBITRUM_RPC_HTTPS")
        private_key = get_env("PRIVATE_KEY")
        if rpc_https and usdc_address and private_key:
            chain_client = ChainClient([rpc_https])
            # Hard safety check: ensure the RPC is actually Arbitrum One.
            try:
                chain_id = chain_client.get_chain_id()
                if chain_id != ARBITRUM_CHAIN_ID:
                    raise RuntimeError(
                        f"Connected RPC chain_id={chain_id} but "
                        f"ARBITRUM_CHAIN_ID={ARBITRUM_CHAIN_ID}. "
                        "Refusing to trade on the wrong network."
                    )
                logging.info("Connected to Arbitrum One (chain_id=%d).", chain_id)
            except Exception as exc:
                logging.error("Failed to verify RPC chain id for Arbitrum: %s", exc)
                raise

            wallet = WalletManager(private_key)
            dex_swap_mgr = DexSwapManager(
                odos=odos,
                chain_client=chain_client,
                wallet=wallet,
                chain_id=ARBITRUM_CHAIN_ID,
                max_gas_limit=max_dex_gas,
            )
            logging.info("DexSwapManager enabled (ODOS aggregator on Arbitrum).")
        else:
            logging.info(
                "DexSwapManager disabled "
                "(missing ARBITRUM_RPC_HTTPS / USDC_ADDRESS / PRIVATE_KEY)"
            )
    except Exception as exc:
        logging.warning("Failed to initialize DexSwapManager: %s", exc)
        dex_swap_mgr = None

    # Use only tokens that are active and ODOS-supported by default.
    active_tokens = {
        k: v
        for k, v in TOKEN_MAPPINGS.items()
        if v.get("active") and v.get("odos_supported")
    }

    # Filter by user-provided token list (CLI arg) or env var, or use all active tokens.
    if tokens is None:
        env_tokens = get_env("TRACKED_TOKENS", None)
        if env_tokens:
            tokens = [t.strip().upper() for t in env_tokens.split(",") if t.strip()]

    if tokens:
        requested = [t.upper() for t in tokens]
        token_universe = {k: active_tokens[k] for k in requested if k in active_tokens}
        missing = [t for t in requested if t not in active_tokens]
        if missing:
            logging.warning(
                "Requested tokens not found or inactive/unsupported: %s. "
                "Available active tokens: %s",
                ", ".join(missing),
                ", ".join(sorted(active_tokens.keys())),
            )
        if not token_universe:
            logging.error(
                "No valid tokens selected. Available active tokens: %s",
                ", ".join(sorted(active_tokens.keys())),
            )
            return
        logging.info(
            "Tracking %d token(s): %s",
            len(token_universe),
            ", ".join(sorted(token_universe.keys())),
        )
    else:
        token_universe = active_tokens
        logging.info(
            "Tracking all %d active tokens: %s",
            len(token_universe),
            ", ".join(sorted(token_universe.keys())),
        )

    engine = DoubleLimitArbitrageEngine(
        mexc_client=mexc,
        odos_client=odos,
        token_mappings=token_universe,
        config=cfg,
        dex_swap_manager=dex_swap_mgr,
        capital_manager=capital,
    )

    symbols = list(token_universe.keys())
    if sim_exec:
        logging.warning(
            "SIMULATED EXECUTION ENABLED — no real orders; "
            "two-leg execution is simulated."
        )
    elif live:
        logging.warning(
            "LIVE EXECUTION ENABLED — real MEXC + ODOS DEX swap "
            "orders will be placed."
        )
    else:
        logging.info("Observation mode — no orders will be placed.")

    # Starting balances
    try:
        mexc_usdt = mexc.get_balance("USDT")
        logging.info(
            "Starting balances — MEXC USDT: %.2f",
            mexc_usdt,
        )
        if chain_client is not None and wallet is not None and usdc_address:
            arb_usdc = _fetch_erc20_balance(
                chain_client, wallet.address, usdc_address, usdc_decimals
            )
            eth_bal = chain_client.get_balance(Address.from_string(wallet.address))
            arb_eth = float(eth_bal.human)
            logging.info(
                "  Arbitrum USDC: %.2f, ETH: %.6f",
                arb_usdc,
                arb_eth,
            )
            if live and not sim_exec:
                if arb_usdc < trade_size_usd:
                    logging.warning(
                        "Arbitrum USDC (%.2f) < TRADE_SIZE_USD (%.2f): "
                        "first mex_to_arb will fail. "
                        "Deposit or swap USDT->USDC on Arbitrum.",
                        arb_usdc,
                        trade_size_usd,
                    )
                if arb_eth < 0.0001:
                    logging.warning(
                        "Arbitrum ETH (%.6f) very low — top up for gas.",
                        arb_eth,
                    )
        for sym in symbols:
            meta = token_universe.get(sym, {})
            if not meta:
                continue
            mexc_base = mexc.get_balance(sym)
            arb_base = 0.0
            if chain_client is not None and wallet is not None:
                arb_base = _fetch_erc20_balance(
                    chain_client,
                    wallet.address,
                    str(meta.get("address", "")),
                    int(meta.get("decimals", 18)),
                )
            logging.info(
                "  %s — MEXC: %.6f, Arbitrum: %.6f",
                sym,
                mexc_base,
                arb_base,
            )
    except Exception as e:
        logging.warning("Could not fetch starting balances: %s", e)

    # Scan interval: 2s for fast opportunity detection (was 5s).
    scan_interval = float(get_env("SCAN_INTERVAL_SECONDS", "2.0") or "2.0")

    logging.info(
        "Starting Double Limit demo for tokens: %s (scan every %.1fs, parallel quotes)",
        ", ".join(symbols),
        scan_interval,
    )
    if max_trades is not None:
        logging.info("Will stop after %d successful trade(s).", max_trades)

    # ── Alert: bot started ─────────────────────────────────────────
    start_msg = (
        f"Double Limit bot STARTED "
        f"({'PRODUCTION' if is_production else 'OBSERVATION'}, "
        f"live={live}, sim={sim_exec}, "
        f"trade_size=${trade_size_usd:.0f}, tokens={','.join(symbols)})"
    )
    logging.warning(start_msg)

    heartbeat_interval = 300  # log heartbeat every 5 minutes
    last_heartbeat = time.monotonic()

    trades_done = 0
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

            # ── Heartbeat — periodic liveness log ─────────────────
            now_mono = time.monotonic()
            if now_mono - last_heartbeat >= heartbeat_interval:
                from session_stats import get_session_stats

                stats = get_session_stats().summary()
                logging.info(
                    "HEARTBEAT | alive | trades=%d pnl=$%.4f uptime=%.0fs",
                    stats["trade_count"],
                    stats["total_pnl_usd"],
                    now_mono
                    - (last_heartbeat - heartbeat_interval + heartbeat_interval),
                )
                last_heartbeat = now_mono

            # ── Circuit breaker global check ──────────────────────
            if circuit_breaker.is_open():
                reset_in = circuit_breaker.time_until_reset()
                logging.warning(
                    "Circuit breaker OPEN — no new trades; reset in %.0fs", reset_in
                )
                metrics.cb_state.set(1)
                await asyncio.sleep(10)
                continue
            metrics.cb_state.set(0)

            scan_start = time.monotonic()
            now = time.strftime("%H:%M:%S")

            # Parallel evaluation of all tokens (ODOS + MEXC fetched concurrently)
            opportunities = await engine.evaluate_all(symbols)

            for key, opp in zip(symbols, opportunities):
                if not opp:
                    continue
                metrics.spread_bps.set(opp.gross_spread * 10_000, pair=key)
                metrics.signals_total.inc(pair=key, direction=opp.direction)
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
                    logging.warning(msg)
                else:
                    logging.info(msg)
                if (live or sim_exec) and opp.executable:
                    # ── Per-pair circuit breaker gate ──────────────
                    if circuit_breaker.is_open(key):
                        logging.info("CB open for %s — skipping execution", key)
                        continue

                    # ── Absolute safety gate ──────────────────────
                    daily_loss = min(0.0, cumulative_pnl)
                    total_capital = starting_capital_usd + cumulative_pnl
                    allowed, reason = safety_check(
                        trade_usd=trade_size_usd,
                        daily_loss=daily_loss,
                        total_capital=total_capital,
                        trades_this_hour=_trades_last_hour(),
                    )
                    if not allowed:
                        logging.warning(
                            "SAFETY CHECK blocked trade %s: %s", key, reason
                        )
                        continue

                    logging.info("Executing double limit for %s (both legs)...", key)
                    balances_before: tuple[dict, dict] | None = None
                    if (
                        live
                        and not sim_exec
                        and chain_client is not None
                        and wallet is not None
                        and usdc_address
                    ):
                        try:
                            balances_before = _fetch_double_limit_balances(
                                mexc,
                                chain_client,
                                wallet,
                                usdc_address,
                                usdc_decimals,
                                opp,
                                token_universe,
                            )
                        except Exception as e:
                            logging.warning(
                                "Failed to fetch balances before execution: %s", e
                            )
                    try:
                        result = await engine.execute_double_limit(opp)
                        exec_status = result.get("status", "UNKNOWN")
                        logging.info(
                            "Execute result: status=%s opportunity=%s",
                            exec_status,
                            key,
                        )
                        metrics.executions_total.inc(pair=key, state=exec_status)

                        if exec_status == "SUCCESS":
                            pnl = opp.net_profit_usd
                            circuit_breaker.record_success(key, pnl)
                            cumulative_pnl += pnl
                            metrics.pnl_total.set(cumulative_pnl)
                            trade_timestamps.append(time.time())
                            if result.get("both_legs"):
                                logging.info(
                                    "Both legs filled: mex_order=%s dex_tx=%s",
                                    result.get("mex_order"),
                                    result.get("dex_tx_hash"),
                                )
                            else:
                                logging.info(
                                    "MEXC leg filled (DEX not active): mex_order=%s",
                                    result.get("mex_order"),
                                )
                            try:
                                record_session_trade(pnl)
                            except Exception:
                                pass
                            trades_done += 1
                        elif exec_status == "TIMEOUT" and result.get(
                            "unwind_attempted"
                        ):
                            circuit_breaker.record_failure(
                                key, classifier.classify("timeout")
                            )
                            logging.warning(
                                "TIMEOUT: one leg filled; "
                                "unwind attempted=%s success=%s",
                                result.get("unwind_attempted"),
                                result.get("unwind_success"),
                            )
                            if (
                                live
                                and not sim_exec
                                and balances_before is not None
                                and chain_client is not None
                                and wallet is not None
                                and usdc_address
                            ):
                                _verify_double_limit_balances(
                                    balances_before,
                                    opp,
                                    result,
                                    mexc,
                                    chain_client,
                                    wallet,
                                    usdc_address,
                                    usdc_decimals,
                                    trade_size_usd,
                                    token_universe,
                                )
                        else:
                            err_msg = result.get("error", exec_status)
                            circuit_breaker.record_failure(
                                key, classifier.classify(err_msg)
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
                        if max_trades is not None and trades_done >= max_trades:
                            logging.info(
                                "Max trades reached (%d). Stopping.",
                                max_trades,
                            )
                            break
                    except SystemExit:
                        raise
                    except Exception as exc:
                        circuit_breaker.record_failure(
                            key, classifier.classify(str(exc))
                        )
                        logging.exception("Execution failed for %s: %s", key, exc)

            scan_elapsed = time.monotonic() - scan_start
            logging.debug("Scan cycle completed in %.1fs", scan_elapsed)

            if max_trades is not None and trades_done >= max_trades:
                break
            # Sleep remaining time to hit target scan interval
            sleep_time = max(0.5, scan_interval - scan_elapsed)
            await asyncio.sleep(sleep_time)
    except KeyboardInterrupt:
        logging.info("Stopping demo.")
    finally:
        # ── Alert: bot stopped ─────────────────────────────────
        from session_stats import get_session_stats as _final_stats

        stats = _final_stats().summary()
        stop_msg = (
            f"Double Limit bot STOPPED | trades={stats['trade_count']} "
            f"pnl=${stats['total_pnl_usd']:.4f}"
        )
        logging.warning(stop_msg)
        try:
            metrics_server.stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
