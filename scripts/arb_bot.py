"""
Arbitrage Bot — two modes in one script.

Modes:
  simulation  — simulated DEX prices (CEX mid × markup), frequent trades,
                full pipeline (scoring, inventory, circuit breaker, recovery).
                For testing bot logic.

  paper       — REAL CEX + DEX on-chain prices, simulated execution,
                live PnL dashboard with trade log.
                For monitoring real market conditions.

Usage:
  python scripts/arb_bot.py                  # default: simulation
  python scripts/arb_bot.py --mode simulation
  python scripts/arb_bot.py --mode paper
"""

# flake8: noqa

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# Ensure src/ is on sys.path so bare imports work from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_double_limit import main as double_limit_main  # noqa: E402

from chain import ChainClient  # noqa: E402
from core.base_types import Address, TokenAmount, TransactionRequest  # noqa: E402
from core.wallet_manager import WalletManager  # noqa: E402
from exchange.client import ExchangeClient  # noqa: E402
from executor.alerts import (  # noqa: E402
    Alert,
    AlertLevel,
    AlertType,
    WebhookAlerter,
    WebhookConfig,
)
from executor.engine import Executor, ExecutorConfig, ExecutorState  # noqa: E402
from executor.execution_report import format_cex_dex_execution_report  # noqa: E402
from executor.metrics import MetricsRegistry, MetricsServer  # noqa: E402
from executor.recovery import RecoveryConfig  # noqa: E402
from inventory.tracker import InventoryTracker, Venue  # noqa: E402
from pricing.dex_pricer import DexPricer  # noqa: E402
from safety import (  # noqa: E402
    ABSOLUTE_MIN_CAPITAL,
    is_kill_switch_active,
    safety_check,
)
from strategy.fees import FeeStructure  # noqa: E402
from strategy.generator import SignalGenerator  # noqa: E402
from strategy.priority_queue import (  # noqa: E402
    PriorityQueueConfig,
    SignalPriorityQueue,
)
from strategy.scorer import ScorerConfig, SignalScorer  # noqa: E402
from strategy.signal import Direction  # noqa: E402
from telegram_bot import (  # noqa: E402
    TelegramBot,
    TelegramBotConfig,
    add_telegram_log_handler,
)

logger = logging.getLogger(__name__)


# ── Paper-trade record ────────────────────────────────────────────


@dataclass
class PaperTrade:
    """One simulated trade with full details."""

    timestamp: float
    pair: str
    direction: Direction
    size: float
    cex_price: float
    dex_price: float
    spread_bps: float
    gross_pnl: float
    fees_usd: float
    net_pnl: float
    cumulative_pnl: float = 0.0

    @property
    def time_str(self) -> str:
        dt = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        return dt.strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════
#  ArbBot — full simulation pipeline
# ══════════════════════════════════════════════════════════════════


class ArbBot:
    """
    Full arbitrage bot with scoring, inventory, circuit breaker,
    recovery, webhooks, Prometheus metrics, and priority queue.

    DEX prices are simulated from CEX mid × markup → frequent trades.
    """

    def __init__(self, config: dict):
        self.exchange = ExchangeClient(
            {
                "apiKey": config["binance_key"],
                "secret": config["binance_secret"],
                "sandbox": True,
            }
        )
        self.inventory = InventoryTracker([Venue.BINANCE, Venue.WALLET])
        self.fees = FeeStructure(
            gas_cost_usd=config.get("gas_cost_usd", 0.50),
        )

        simulation_mode = config.get("simulation", True)
        self.simulation_mode = simulation_mode
        # Dry-run mode: run full pipeline but do NOT execute trades by default.
        self.dry_run = bool(config.get("dry_run", True))
        # Simulated CEX balances (e.g. MEXC $50 USDT, 0 ETH)
        self.sim_cex_eth = float(
            config.get("sim_cex_eth", os.getenv("SIM_CEX_ETH", "0"))
        )
        self.sim_cex_usdt = float(
            config.get("sim_cex_usdt", os.getenv("SIM_CEX_USDT", "50"))
        )
        # Simulated wallet balances (e.g. Arbitrum $45 USDT + ~$5 ETH for gas)
        self.sim_wallet_eth = float(
            config.get("sim_wallet_eth", os.getenv("SIM_WALLET_ETH", "0.002"))
        )
        self.sim_wallet_usdt = float(
            config.get("sim_wallet_usdt", os.getenv("SIM_WALLET_USDT", "45"))
        )
        self.wallet: WalletManager | None = None
        self.chain_client: ChainClient | None = None
        self.dex_pricer: DexPricer | None = None
        self.dex_quote_token = config.get("dex_quote_token_address") or os.getenv(
            "DEX_QUOTE_TOKEN_ADDRESS"
        )
        self.dex_quote_decimals = int(
            config.get("dex_quote_decimals", os.getenv("DEX_QUOTE_TOKEN_DECIMALS", "6"))
        )
        self.dex_chain_id = int(config.get("dex_chain_id", 11155111))

        # Absolute safety accounting (approximate, for hard-stop gates).
        self._starting_capital_usd = float(
            config.get("starting_capital_usd", max(ABSOLUTE_MIN_CAPITAL, 100.0))
        )
        self._trade_timestamps: list[float] = []

        # ── Chain client (read-only for DEX pricing) ──────────
        # Explicit None in config means "no DEX pricing" — skip fallbacks.
        if "dex_pricing_rpc_url" in config and config["dex_pricing_rpc_url"] is None:
            pricing_rpc = None
        else:
            pricing_rpc = (
                config.get("dex_pricing_rpc_url")
                or os.getenv("DEX_PRICING_RPC_URL")
                or config.get("dex_rpc_url")
                or os.getenv("ETH_RPC_URL")
            )
        rpc_url = config.get("dex_rpc_url") or os.getenv("SEPOLIA_RPC_URL")
        pool_address = config.get("dex_pool_address") or os.getenv("DEX_POOL_ADDRESS")
        weth_address = config.get("dex_weth_address") or os.getenv("DEX_WETH_ADDRESS")

        if pricing_rpc and pool_address and weth_address:
            pricing_client = ChainClient([pricing_rpc])
            self.dex_pricer = DexPricer(
                pricing_client,
                pool_address,
                weth_address,
            )
            logging.info(
                "DEX pricer: real on-chain prices from pool %s",
                pool_address,
            )
        else:
            logging.warning(
                "DEX pricer: simulated prices (set DEX_POOL_ADDRESS + "
                "DEX_WETH_ADDRESS + ETH_RPC_URL for real quotes)"
            )

        # ── Wallet (only needed for real execution) ───────────
        if not simulation_mode:
            if not rpc_url:
                raise ValueError("SEPOLIA_RPC_URL is required when simulation=False")
            private_key = config.get("dex_private_key") or os.getenv("PRIVATE_KEY")
            if not private_key:
                raise ValueError("PRIVATE_KEY is required when simulation=False")
            self.wallet = WalletManager(private_key)
            if self.chain_client is None:
                self.chain_client = ChainClient([rpc_url])

        self.generator = SignalGenerator(
            self.exchange,
            self.dex_pricer,  # None → simulated prices, DexPricer → real on-chain
            self.inventory,
            self.fees,
            config.get("signal_config", {}),
        )
        scorer_cfg = ScorerConfig(
            min_score=config.get("min_score", 55.0),
        )
        self.scorer = SignalScorer(scorer_cfg)

        # ── Stretch Goal 1: Webhook Alerts ────────────────────
        webhook_cfg = WebhookConfig.from_env()
        self.alerter = WebhookAlerter(webhook_cfg)

        # ── Stretch Goal 4: Prometheus Metrics ────────────────
        self.metrics = MetricsRegistry()
        metrics_port = int(
            config.get("metrics_port", os.getenv("METRICS_PORT", "9090"))
        )
        self.metrics_server = MetricsServer(self.metrics, port=metrics_port)

        self.executor = Executor(
            self.exchange,
            None,
            self.inventory,
            ExecutorConfig(
                simulation_mode=simulation_mode,
                dex_chain_id=self.dex_chain_id,
                dex_rpc_url=config.get("dex_rpc_url"),
                dex_private_key=config.get("dex_private_key"),
                dex_router_address=config.get("dex_router_address"),
                dex_weth_address=config.get("dex_weth_address"),
                dex_quote_token_address=config.get("dex_quote_token_address"),
            ),
            recovery_config=RecoveryConfig(),
        )
        self.executor.recovery.alerter = self.alerter

        # ── Stretch Goal 3: Priority Queue ────────────────────
        pq_cfg = PriorityQueueConfig(
            max_depth=int(config.get("pq_max_depth", 50)),
            max_per_pair=int(config.get("pq_max_per_pair", 1)),
            min_score=self.scorer.config.min_score,
        )
        self.priority_queue = SignalPriorityQueue(
            config=pq_cfg,
            decay_fn=self.scorer.apply_decay,
        )

        self.pairs = config.get("pairs", ["ETH/USDT"])
        self.trade_size = config.get("trade_size", 0.1)
        self.running = False

    def set_execution_report_sender(self, send_fn):
        """Optional: callable(text) to send full execution report (e.g. tg.send)."""
        self._execution_report_sender = send_fn

    async def run(self):
        self.running = True
        exec_mode = "SIMULATION" if self.simulation_mode else "LIVE"
        price_mode = "REAL on-chain" if self.dex_pricer else "SIMULATED"
        logging.info("Bot starting... [exec=%s, dex_prices=%s]", exec_mode, price_mode)
        if self.simulation_mode:
            logging.info(
                "Simulated balances — CEX: %.4f ETH / %.2f USDT; wallet: %.4f ETH / %.2f USDT",
                self.sim_cex_eth,
                self.sim_cex_usdt,
                self.sim_wallet_eth,
                self.sim_wallet_usdt,
            )

        self.alerter.start()
        self._execution_report_sender = (
            None  # optional: send full execution report (e.g. Telegram)
        )
        # Alert: bot started
        self.alerter.send(
            Alert(
                alert_type=AlertType.CUSTOM,
                level=AlertLevel.INFO,
                pair=None,
                message=f"ArbBot started (mode={exec_mode}, dry_run={self.dry_run})",
            )
        )
        self.metrics_server.start()

        await self._sync_balances()

        kill_logged = False

        while self.running:
            try:
                if is_kill_switch_active():
                    if not kill_logged:
                        logging.warning("Kill switch active — ArbBot PAUSED.")
                        self.alerter.send(
                            Alert(
                                alert_type=AlertType.CUSTOM,
                                level=AlertLevel.CRITICAL,
                                pair=None,
                                message="Kill switch activated — ArbBot paused (no new trades).",
                            )
                        )
                        kill_logged = True
                    await asyncio.sleep(1)
                    continue
                else:
                    if kill_logged:
                        logging.info("Kill switch cleared — ArbBot RESUMING.")
                        self.alerter.send(
                            Alert(
                                alert_type=AlertType.CUSTOM,
                                level=AlertLevel.INFO,
                                pair=None,
                                message="Kill switch cleared — ArbBot resuming.",
                            )
                        )
                        kill_logged = False
                await self._tick()
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"Tick error: {e}")
                await asyncio.sleep(5)

    async def _tick(self):
        recovery = self.executor.recovery

        if self.executor.circuit_breaker.is_open():
            reset_in = self.executor.circuit_breaker.time_until_reset()
            logging.info("Circuit breaker open — reset in %.0fs", reset_in)
            return

        # ── Phase 1: Collect signals into priority queue ──────
        self.priority_queue.clear()

        for pair in self.pairs:
            if self.executor.circuit_breaker.is_open(pair):
                logging.info("CB open for %s — skipping", pair)
                continue

            signal = self.generator.generate(pair, self.trade_size)
            if signal is None:
                continue

            skews = self._get_inventory_skews(pair)
            signal.score = self.scorer.score(signal, skews)

            self.metrics.signals_total.inc(pair=pair, direction=signal.direction.name)
            self.metrics.spread_bps.set(signal.spread_bps, pair=pair)
            self.metrics.score.set(signal.score, pair=pair)

            dex_src = "on-chain" if self.dex_pricer else "sim"
            logging.info(
                "Signal: %s spread=%.1fbps score=%d  " "CEX=%.2f  DEX=%.2f [%s]",
                pair,
                signal.spread_bps,
                int(round(signal.score)),
                signal.cex_price,
                signal.dex_price,
                dex_src,
            )

            if signal.score < self.scorer.config.min_score:
                logging.info(
                    "Skipped: score below threshold (%.1f < %.1f)",
                    signal.score,
                    self.scorer.config.min_score,
                )
                continue

            self.priority_queue.push(signal)

        self.metrics.queue_depth.set(self.priority_queue.size)

        # ── Phase 2: Execute signals in priority order ────────
        for signal in self.priority_queue.drain():
            pair = signal.pair

            trade_value_usd = signal.size * signal.cex_price
            stats = self.executor.stats
            cumulative_pnl = float(stats.get("total_pnl", 0.0))
            daily_loss = min(0.0, cumulative_pnl)
            total_capital = self._starting_capital_usd + cumulative_pnl
            trades_this_hour = self._trades_last_hour()

            allowed, reason = safety_check(
                trade_usd=trade_value_usd,
                daily_loss=daily_loss,
                total_capital=total_capital,
                trades_this_hour=trades_this_hour,
            )
            if not allowed:
                logging.warning("Safety check blocked trade %s: %s", pair, reason)
                # Alert: absolute safety limit hit (e.g. daily loss)
                self.alerter.send(
                    Alert(
                        alert_type=AlertType.CUSTOM,
                        level=AlertLevel.CRITICAL,
                        pair=pair,
                        message=f"Safety check blocked trade: {reason}",
                    )
                )
                continue

            if self.dry_run:
                logging.info(
                    "DRY RUN | Would trade: %s %s size=%.4f spread=%.1fbps "
                    "expected_pnl=$%.2f",
                    pair,
                    signal.direction.value,
                    signal.size,
                    signal.spread_bps,
                    signal.expected_net_pnl,
                )
                continue

            logging.info(
                "Executing: %s %.4g %s",
                signal.direction.name,
                signal.size,
                self._base_asset(pair),
            )

            ctx = await self.executor.execute(signal)

            if not self.simulation_mode and self.wallet and self.chain_client:
                await self._verify_balances(ctx)

            if self._execution_report_sender:
                try:
                    self._execution_report_sender(format_cex_dex_execution_report(ctx))
                except Exception as exc:
                    logging.warning("Execution report send failed: %s", exc)

            self.metrics.executions_total.inc(pair=pair, state=ctx.state.name)
            if ctx.metrics.leg1_latency_ms:
                self.metrics.execution_latency.observe(
                    ctx.metrics.leg1_latency_ms, pair=pair, leg="leg1"
                )
            if ctx.metrics.leg2_latency_ms:
                self.metrics.execution_latency.observe(
                    ctx.metrics.leg2_latency_ms, pair=pair, leg="leg2"
                )
            if ctx.metrics.unwind_attempted:
                self.metrics.unwinds_total.inc(
                    pair=pair,
                    success=str(ctx.metrics.unwind_success or False),
                )

            success = ctx.state == ExecutorState.DONE
            self.scorer.record_result(pair, success)

            if success:
                logging.info("SUCCESS: PnL=$%.4f", ctx.actual_net_pnl or 0)
                self.metrics.pnl_total.inc(ctx.actual_net_pnl or 0)
            else:
                unwind_note = " - unwound" if ctx.metrics.unwind_success else ""
                logging.warning(
                    "FAILED: %s%s", ctx.error or "execution failed", unwind_note
                )
                self._log_circuit_breaker_status(pair, recovery.snapshot(pair))

                self.alerter.on_execution_failure(
                    pair, ctx.error or "unknown", bool(ctx.metrics.unwind_success)
                )

            cb_val = 0
            if self.executor.circuit_breaker.is_open(pair):
                cb_val = 1
            self.metrics.cb_state.set(cb_val, pair=pair)

            await self._sync_balances()

            # Count successful executions for hourly safety limits.
            if ctx.state == ExecutorState.DONE:
                self._record_trade_execution()
                # Alert: trade completed with PnL
                self.alerter.send(
                    Alert(
                        alert_type=AlertType.CUSTOM,
                        level=AlertLevel.INFO,
                        pair=pair,
                        message=(
                            f"Trade completed: pair={pair} "
                            f"net_pnl=${(ctx.actual_net_pnl or 0):.4f}"
                        ),
                        details={"state": ctx.state.name},
                    )
                )

    async def _sync_balances(self):
        if self.simulation_mode:
            # In simulation mode, use configurable fake balances (e.g. CEX $50 USDT, wallet $45 USDT + $5 ETH)
            sim_cex = {
                "ETH": {
                    "free": str(self.sim_cex_eth),
                    "locked": "0",
                    "total": str(self.sim_cex_eth),
                },
                "USDT": {
                    "free": str(self.sim_cex_usdt),
                    "locked": "0",
                    "total": str(self.sim_cex_usdt),
                },
            }
            self.inventory.update_from_cex(Venue.BINANCE, sim_cex)
            self.inventory.update_from_wallet(
                Venue.WALLET,
                {"ETH": str(self.sim_wallet_eth), "USDT": str(self.sim_wallet_usdt)},
            )
            return

        balances = self.exchange.fetch_balance()
        self.inventory.update_from_cex(Venue.BINANCE, balances)

        if self.chain_client is None or self.wallet is None:
            return

        eth_balance = self.chain_client.get_balance(
            Address.from_string(self.wallet.address)
        )
        wallet_balances = {"ETH": str(eth_balance.human)}
        if self.dex_quote_token:
            wallet_balances["USDT"] = self._fetch_erc20_balance(
                token=self.dex_quote_token, decimals=self.dex_quote_decimals
            )
        self.inventory.update_from_wallet(Venue.WALLET, wallet_balances)

        for pair in self.pairs:
            base = self._base_asset(pair)
            try:
                skew = self.inventory.skew(base)
                for venue_name, venue_data in skew.get("venues", {}).items():
                    self.metrics.inventory_skew.set(
                        venue_data.get("deviation_pct", 0.0),
                        pair=pair,
                        venue=venue_name,
                    )
            except Exception:
                pass

    def stop(self):
        self.running = False
        self.alerter.stop()
        self.metrics_server.stop()

    def _get_inventory_skews(self, pair: str) -> list[dict]:
        try:
            base, quote = pair.split("/")
            return [
                self.inventory.skew(base),
                self.inventory.skew(quote),
            ]
        except Exception:
            return []

    def _fetch_erc20_balance(self, token: str, decimals: int) -> str:
        assert self.chain_client is not None
        assert self.wallet is not None
        selector = bytes.fromhex("70a08231")  # balanceOf(address)
        owner = Address.from_string(self.wallet.address).checksum
        calldata = selector + bytes.fromhex(owner[2:]).rjust(32, b"\x00")
        call = TransactionRequest(
            to=Address.from_string(token),
            value=TokenAmount(raw=0, decimals=18, symbol="ETH"),
            data=calldata,
            chain_id=self.dex_chain_id,
        )
        raw = self.chain_client.call(call)
        amount_raw = int.from_bytes(raw, "big") if raw else 0
        human = Decimal(amount_raw) / Decimal(10**decimals)
        return str(human)

    async def _verify_balances(self, ctx):
        """
        After a trade, check actual CEX and wallet balances match expected
        (inventory + trade effect). If mismatch > 0.001, log critical and stop.
        """
        signal = ctx.signal
        pair = signal.pair
        base = self._base_asset(pair)
        quote = "USDT" if "USDT" in pair else pair.split("/")[1]

        snap = self.inventory.snapshot()
        venues = snap.get("venues", {})

        def _venue_total(venue_key: str, asset: str) -> Decimal:
            v = venues.get(venue_key, {})
            a = v.get(asset, {})
            if isinstance(a, dict):
                return self.inventory._to_decimal(a.get("total", 0))
            return Decimal("0")

        expected_cex_base = _venue_total("binance", base)
        expected_cex_quote = _venue_total("binance", quote)
        expected_wallet_base = _venue_total("wallet", base)
        expected_wallet_quote = _venue_total("wallet", quote)

        if (
            ctx.state == ExecutorState.DONE
            and (ctx.leg1_fill_size or 0)
            and (ctx.leg2_fill_size or 0)
        ):
            size1 = Decimal(str(ctx.leg1_fill_size or 0))
            size2 = Decimal(str(ctx.leg2_fill_size or 0))
            price1 = Decimal(str(ctx.leg1_fill_price or 0))
            price2 = Decimal(str(ctx.leg2_fill_price or 0))
            if ctx.leg1_venue == "cex":
                expected_cex_base += size1
                expected_cex_quote -= size1 * price1
                expected_wallet_base -= size2
                expected_wallet_quote += size2 * price2
            else:
                expected_wallet_base += size1
                expected_wallet_quote -= size1 * price1
                expected_cex_base -= size2
                expected_cex_quote += size2 * price2

        try:
            cex_balances = self.exchange.fetch_balance()
        except Exception as e:
            logging.critical(
                "Balance verification failed: could not fetch CEX balances: %s", e
            )
            self.stop()
            return

        def _cex_total(asset: str) -> Decimal:
            entry = cex_balances.get(asset, {})
            if isinstance(entry, dict):
                return self.inventory._to_decimal(entry.get("total", 0))
            return Decimal("0")

        actual_cex_base = _cex_total(base)
        actual_cex_quote = _cex_total(quote)

        eth_balance = self.chain_client.get_balance(
            Address.from_string(self.wallet.address)
        )
        actual_wallet_base = Decimal(str(eth_balance.human))
        actual_wallet_quote = (
            Decimal(
                self._fetch_erc20_balance(
                    token=self.dex_quote_token, decimals=self.dex_quote_decimals
                )
            )
            if self.dex_quote_token
            else Decimal("0")
        )

        tolerance = Decimal("0.001")
        cex_base_diff = abs(actual_cex_base - expected_cex_base)
        cex_quote_diff = abs(actual_cex_quote - expected_cex_quote)
        wallet_base_diff = abs(actual_wallet_base - expected_wallet_base)
        wallet_quote_diff = abs(actual_wallet_quote - expected_wallet_quote)

        if (
            cex_base_diff > tolerance
            or cex_quote_diff > tolerance
            or wallet_base_diff > tolerance
            or wallet_quote_diff > tolerance
        ):
            logging.critical(
                "BALANCE MISMATCH! CEX %s diff=%s, CEX %s diff=%s; wallet %s diff=%s, wallet %s diff=%s",
                base,
                cex_base_diff,
                quote,
                cex_quote_diff,
                base,
                wallet_base_diff,
                quote,
                wallet_quote_diff,
            )
            self.stop()

    @staticmethod
    def _base_asset(pair: str) -> str:
        return pair.split("/")[0] if "/" in pair else pair

    def _trades_last_hour(self) -> int:
        now = time.time()
        cutoff = now - 3600.0
        self._trade_timestamps = [t for t in self._trade_timestamps if t >= cutoff]
        return len(self._trade_timestamps)

    def _record_trade_execution(self) -> None:
        self._trade_timestamps.append(time.time())

    def _log_circuit_breaker_status(self, pair: str, snapshot: dict) -> None:
        cb = snapshot.get("circuit_breaker", {})
        pair_snap = cb.get("pair")
        global_snap = cb.get("global", {})

        active = pair_snap if pair_snap else global_snap
        failures = int(active.get("failures", 0))
        threshold = int(self.executor.circuit_breaker.config.failure_threshold)
        logging.warning(
            "Circuit breaker: %d/%d failures (%s)", failures, threshold, pair
        )


# ══════════════════════════════════════════════════════════════════
#  PaperBot — real prices, simulated execution, PnL dashboard
# ══════════════════════════════════════════════════════════════════


class PaperBot:
    """
    Fetch REAL prices from CEX (Binance) + DEX (Uniswap V2 on-chain),
    simulate trades, track PnL with a live dashboard.

    No real orders are placed.  No wallet or private key needed.
    """

    def __init__(self, config: dict):
        # ── CEX client (for order book) ───────────────────────
        self.exchange = ExchangeClient(
            {
                "apiKey": config["binance_key"],
                "secret": config["binance_secret"],
                "sandbox": config.get("binance_sandbox", True),
            }
        )

        # ── DEX pricer (on-chain reads, no gas) ───────────────
        pricing_rpc = (
            config.get("dex_pricing_rpc_url")
            or os.getenv("DEX_PRICING_RPC_URL")
            or os.getenv("ETH_RPC_URL")
        )
        pool_address = config.get("dex_pool_address") or os.getenv(
            "DEX_POOL_ADDRESS",
            "0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852",  # WETH/USDT V2
        )
        weth_address = config.get("dex_weth_address") or os.getenv(
            "DEX_WETH_ADDRESS",
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # Mainnet WETH
        )

        if not pricing_rpc:
            raise ValueError(
                "ETH_RPC_URL is required for paper mode (real DEX prices). "
                "Set it to an Ethereum mainnet RPC (e.g. Infura/Alchemy)."
            )

        chain_client = ChainClient([pricing_rpc])
        self.dex_pricer = DexPricer(chain_client, pool_address, weth_address)

        # ── Fee model ─────────────────────────────────────────
        self.fees = FeeStructure(
            gas_cost_usd=config.get("gas_cost_usd", 0.50),
            cex_taker_bps=config.get("cex_taker_bps", 10.0),
            dex_swap_bps=config.get("dex_swap_bps", 30.0),
            slippage_bps=config.get("slippage_bps", 5.0),
        )

        # ── Strategy config ───────────────────────────────────
        self.pairs = config.get("pairs", ["ETH/USDT"])
        self.trade_size = config.get("trade_size", 0.05)
        self.min_spread_bps = config.get("min_spread_bps", 10)
        self.min_profit_usd = config.get("min_profit_usd", 0.01)
        self.tick_interval = config.get("tick_interval", 5.0)

        # ── Paper trading state ───────────────────────────────
        self.trades: list[PaperTrade] = []
        self.cumulative_pnl: float = 0.0
        self.total_ticks: int = 0
        self.start_time: float = 0.0
        self.running = False

    # ── Main loop ─────────────────────────────────────────────

    async def run(self):
        self.running = True
        self.start_time = time.time()

        self._print_header()

        while self.running:
            try:
                if is_kill_switch_active():
                    logger.warning("Kill switch active — stopping PaperBot.")
                    self.stop()
                    break
                await self._tick()
                await asyncio.sleep(self.tick_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Tick error: %s", e)
                await asyncio.sleep(5)

        self._print_summary()

    async def _tick(self):
        self.total_ticks += 1

        for pair in self.pairs:
            # ── 1. Fetch real CEX prices ──────────────────────
            try:
                ob = self.exchange.fetch_order_book(pair)
            except Exception as exc:
                logger.warning("CEX fetch failed: %s", exc)
                continue

            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if not bids or not asks:
                continue

            cex_bid = float(bids[0][0])
            cex_ask = float(asks[0][0])
            cex_mid = (cex_bid + cex_ask) / 2

            # ── 2. Fetch real DEX prices ──────────────────────
            dex_quote = self.dex_pricer.get_quote(pair, self.trade_size)
            if dex_quote is None:
                logger.warning("DEX quote failed for %s", pair)
                continue

            dex_buy = dex_quote["buy"]
            dex_sell = dex_quote["sell"]

            # ── 3. Compute spreads in both directions ─────────
            spread_a = (dex_sell - cex_ask) / cex_ask * 10_000
            spread_b = (cex_bid - dex_buy) / dex_buy * 10_000

            if spread_a >= spread_b:
                direction = Direction.BUY_CEX_SELL_DEX
                spread = spread_a
                cex_price = cex_ask
                dex_price = dex_sell
            else:
                direction = Direction.BUY_DEX_SELL_CEX
                spread = spread_b
                cex_price = cex_bid
                dex_price = dex_buy

            # ── 4. Economics ──────────────────────────────────
            trade_value = self.trade_size * cex_mid
            gross_pnl = (spread / 10_000) * trade_value
            total_fee_bps = self.fees.total_fee_bps(trade_value)
            fees_usd = (total_fee_bps / 10_000) * trade_value
            net_pnl = gross_pnl - fees_usd

            # ── 5. Print live prices ──────────────────────────
            now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
            spread_icon = "🟢" if spread >= self.min_spread_bps else "⚪"

            print(
                f"  {now_str}  {pair}  "
                f"CEX bid={cex_bid:>10.2f}  ask={cex_ask:>10.2f}  │  "
                f"DEX buy={dex_buy:>10.2f}  sell={dex_sell:>10.2f}  │  "
                f"{spread_icon} spread={spread:>+7.1f}bps  "
                f"net=${net_pnl:>+.4f}  "
                f"fees={total_fee_bps:.0f}bps",
            )

            # ── 6. Simulate trade if profitable ──────────────
            if spread < self.min_spread_bps:
                continue
            if net_pnl < self.min_profit_usd:
                continue

            self.cumulative_pnl += net_pnl
            trade = PaperTrade(
                timestamp=time.time(),
                pair=pair,
                direction=direction,
                size=self.trade_size,
                cex_price=cex_price,
                dex_price=dex_price,
                spread_bps=round(spread, 1),
                gross_pnl=round(gross_pnl, 4),
                fees_usd=round(fees_usd, 4),
                net_pnl=round(net_pnl, 4),
                cumulative_pnl=round(self.cumulative_pnl, 4),
            )
            self.trades.append(trade)

            dir_label = (
                "CEX->DEX" if direction == Direction.BUY_CEX_SELL_DEX else "DEX->CEX"
            )
            pnl_icon = "+" if net_pnl > 0 else "-"
            print(
                f"  {pnl_icon} PAPER TRADE #{len(self.trades):>3}  "
                f"{dir_label}  {self.trade_size} {pair.split('/')[0]}  "
                f"spread={spread:+.1f}bps  "
                f"gross=${gross_pnl:+.4f}  fees=${fees_usd:.4f}  "
                f"net=${net_pnl:+.4f}  "
                f"cumPnL=${self.cumulative_pnl:+.4f}"
            )

    # ── Display helpers ───────────────────────────────────────

    def _print_header(self):
        print("\n" + "=" * 100)
        print("  📄 PAPER MODE — Real Prices, Simulated Execution")
        print("=" * 100)
        print(f"  Pairs:      {', '.join(self.pairs)}")
        print(f"  Trade size: {self.trade_size}")
        print(f"  Min spread: {self.min_spread_bps} bps")
        print(f"  Min profit: ${self.min_profit_usd}")
        print(
            f"  Fees model: CEX={self.fees.cex_taker_bps}bps  "
            f"DEX={self.fees.dex_swap_bps}bps  "
            f"gas=${self.fees.gas_cost_usd}  "
            f"slippage={self.fees.slippage_bps}bps"
        )
        print(f"  Tick:       every {self.tick_interval}s")
        print("-" * 100)
        print("  Press Ctrl+C to stop and see summary\n")

    def _print_summary(self):
        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60

        print("\n" + "=" * 100)
        print("  📊 PAPER TRADING SUMMARY")
        print("=" * 100)
        print(f"  Duration:       {elapsed_min:.1f} minutes ({self.total_ticks} ticks)")
        print(f"  Total trades:   {len(self.trades)}")

        if not self.trades:
            print("  No trades executed.")
            print("=" * 100)
            return

        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]
        win_rate = len(wins) / len(self.trades) * 100

        total_gross = sum(t.gross_pnl for t in self.trades)
        total_fees = sum(t.fees_usd for t in self.trades)
        total_net = sum(t.net_pnl for t in self.trades)
        avg_spread = sum(t.spread_bps for t in self.trades) / len(self.trades)
        best_trade = max(self.trades, key=lambda t: t.net_pnl)
        worst_trade = min(self.trades, key=lambda t: t.net_pnl)

        print(f"  Win rate:       {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"  Gross PnL:      ${total_gross:+.4f}")
        print(f"  Total fees:     ${total_fees:.4f}")
        print(f"  Net PnL:        ${total_net:+.4f}")
        print(f"  Avg spread:     {avg_spread:.1f} bps")
        print(
            f"  Best trade:     ${best_trade.net_pnl:+.4f} ({best_trade.spread_bps}bps)"
        )
        print(
            f"  Worst trade:    ${worst_trade.net_pnl:+.4f} ({worst_trade.spread_bps}bps)"
        )

        if elapsed_min > 0:
            trades_per_hour = len(self.trades) / (elapsed_min / 60)
            pnl_per_hour = total_net / (elapsed_min / 60)
            print(f"  Trades/hour:    {trades_per_hour:.1f}")
            print(f"  PnL/hour:       ${pnl_per_hour:+.4f}")

        # ── Last 10 trades table ──────────────────────────────
        print("\n  Last 10 trades:")
        print(
            f"  {'#':>4}  {'Time':>8}  {'Dir':>7}  {'Size':>6}  "
            f"{'CEX':>10}  {'DEX':>10}  {'Spread':>8}  "
            f"{'Net PnL':>9}  {'CumPnL':>9}"
        )
        print("  " + "-" * 90)

        for i, t in enumerate(self.trades[-10:], start=max(1, len(self.trades) - 9)):
            dir_label = (
                "CEX->DEX" if t.direction == Direction.BUY_CEX_SELL_DEX else "DEX->CEX"
            )
            print(
                f"  {i:>4}  {t.time_str:>8}  {dir_label:>7}  {t.size:>6.3f}  "
                f"{t.cex_price:>10.2f}  {t.dex_price:>10.2f}  "
                f"{t.spread_bps:>+7.1f}  "
                f"${t.net_pnl:>+8.4f}  ${t.cumulative_pnl:>+8.4f}"
            )

        print("=" * 100 + "\n")

    def stop(self):
        self.running = False


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════


def _build_simulation_config() -> dict:
    """Config for simulation mode (fake DEX prices, frequent trades)."""
    return {
        "binance_key": os.getenv("BINANCE_TESTNET_API_KEY"),
        "binance_secret": os.getenv("BINANCE_TESTNET_SECRET"),
        "pairs": ["ETH/USDT"],
        "trade_size": 0.05,
        "simulation": True,
        "sim_cex_eth": float(os.getenv("SIM_CEX_ETH", "0")),
        "sim_cex_usdt": float(os.getenv("SIM_CEX_USDT", "50")),
        "sim_wallet_eth": float(os.getenv("SIM_WALLET_ETH", "0.002")),
        "sim_wallet_usdt": float(os.getenv("SIM_WALLET_USDT", "45")),
        "gas_cost_usd": 0.10,
        "min_score": 30.0,
        "signal_config": {
            "min_profit_usd": 0.10,
            "min_spread_bps": 30,
        },
        # Simulation uses fake DEX prices — no RPC needed
        "dex_pricing_rpc_url": None,
        "dex_rpc_url": os.getenv("SEPOLIA_RPC_URL"),
        "dex_pool_address": os.getenv(
            "DEX_POOL_ADDRESS",
            "0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852",
        ),
        "dex_weth_address": os.getenv(
            "DEX_WETH_ADDRESS",
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        ),
        "dex_router_address": os.getenv("DEX_ROUTER_ADDRESS"),
        "dex_quote_token_address": os.getenv("DEX_QUOTE_TOKEN_ADDRESS"),
    }


def _build_paper_config() -> dict:
    """Config for paper mode (real CEX + DEX prices, simulated execution)."""
    return {
        "binance_key": os.getenv("BINANCE_TESTNET_API_KEY"),
        "binance_secret": os.getenv("BINANCE_TESTNET_SECRET"),
        "binance_sandbox": True,
        # Real on-chain DEX prices (eth_call is free)
        "dex_pricing_rpc_url": os.getenv("ETH_RPC_URL"),
        "dex_pool_address": os.getenv(
            "DEX_POOL_ADDRESS",
            "0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852",
        ),
        "dex_weth_address": os.getenv(
            "DEX_WETH_ADDRESS",
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        ),
        # Strategy
        "pairs": ["ETH/USDT"],
        "trade_size": 0.05,
        "min_spread_bps": 10,
        "min_profit_usd": 0.01,
        "tick_interval": 5.0,
        # Fee model
        "gas_cost_usd": 0.50,
        "cex_taker_bps": 10.0,
        "dex_swap_bps": 30.0,
        "slippage_bps": 5.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CEX↔DEX Arbitrage Bot",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "paper", "mexc_v3", "double_limit"],
        default="simulation",
        help=(
            "simulation   — fake DEX prices, frequent trades, full pipeline\n"
            "               (scoring, inventory, circuit breaker, recovery)\n"
            "paper        — REAL CEX + DEX prices, simulated execution,\n"
            "               live PnL dashboard with trade log\n"
            "mexc_v3     — MEXC + Uniswap V3 arb dry-run (formerly double_limit)\n"
            "               (same behavior as scripts/demo_double_limit.py)\n"
            "double_limit — Deprecated alias for mexc_v3"
        ),
    )
    parser.add_argument(
        "--trade-size",
        type=float,
        default=None,
        help=(
            "Trade size in USD (for mexc_v3 / double_limit mode). "
            "Default: 5.0 (from TRADE_SIZE_USD env var or config). "
            "Common values: 5.0, 10.0"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Place real orders (both legs: MEXC limit + Uniswap V3 range). "
            "Only for mexc_v3 / double_limit mode. Default: observation only."
        ),
    )
    parser.add_argument(
        "--simulate-execution",
        action="store_true",
        help=(
            "Simulate MEXC+V3 execution (two legs, fills/timeouts) without real orders. "
            "Only for mexc_v3 / double_limit mode."
        ),
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=None,
        metavar="N",
        help=(
            "MEXC+V3 arb only: stop after N successful trades (e.g. 1 for one transaction). "
            "Default: run until Ctrl+C."
        ),
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        metavar="TOKEN1,TOKEN2,...",
        help=(
            "MEXC+V3 arb only: comma-separated list of token symbols to track "
            "(e.g. 'LINK,ARB,GMX'). Can also be set via TRACKED_TOKENS env var. "
            "Default: all active and ODOS-supported tokens. "
            "Available tokens: ARB, GMX, MAGIC, GNS, RDNT, PENDLE, LINK, UNI, BAL, STG, etc."
        ),
    )
    args = parser.parse_args()

    # Structured logging to both file and stdout
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    log_dir = _Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"bot_{_dt.now():%Y%m%d}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s |%(levelname)s |%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    # Suppress noisy third-party logs
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Shared Telegram bot for all modes (if configured)
    tg = TelegramBot(TelegramBotConfig.from_env())
    tg.start()
    # Send logs to Telegram (level from TELEGRAM_LOG_LEVEL, default WARNING)
    add_telegram_log_handler(tg)

    try:
        if args.mode == "simulation":
            config = _build_simulation_config()
            bot = ArbBot(config)
            if tg.config.enabled:
                bot.set_execution_report_sender(tg.send)
            try:
                asyncio.run(bot.run())
            except KeyboardInterrupt:
                bot.stop()

        elif args.mode == "paper":
            config = _build_paper_config()
            bot = PaperBot(config)
            try:
                asyncio.run(bot.run())
            except KeyboardInterrupt:
                bot.stop()
                bot._print_summary()

        elif args.mode in ("mexc_v3", "double_limit"):
            # Run the MEXC + Uniswap V3 arb demo (formerly Double Limit).
            try:
                trade_size = args.trade_size if args.trade_size is not None else None
                tokens = None
                if args.tokens:
                    tokens = [
                        t.strip().upper() for t in args.tokens.split(",") if t.strip()
                    ]
                asyncio.run(
                    double_limit_main(
                        trade_size_usd=trade_size,
                        enable_live_execution=args.execute,
                        simulate_execution=args.simulate_execution,
                        telegram_bot=tg if tg.config.enabled else None,
                        max_trades=args.max_trades,
                        tokens=tokens,
                    )
                )
            except KeyboardInterrupt:
                logging.info("Stopping Double Limit mode.")
    finally:
        tg.stop()
