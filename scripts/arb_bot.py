import asyncio
import logging
import os
import sys
from decimal import Decimal
from pathlib import Path

# Ensure src/ is on sys.path so bare imports work from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chain import ChainClient  # noqa: E402
from core.base_types import Address, TokenAmount, TransactionRequest  # noqa: E402
from core.wallet_manager import WalletManager  # noqa: E402
from exchange.client import ExchangeClient  # noqa: E402
from executor.engine import Executor, ExecutorConfig, ExecutorState  # noqa: E402
from executor.recovery import RecoveryConfig  # noqa: E402
from inventory.tracker import InventoryTracker, Venue  # noqa: E402
from strategy.fees import FeeStructure  # noqa: E402
from strategy.generator import SignalGenerator  # noqa: E402
from strategy.scorer import SignalScorer  # noqa: E402


class ArbBot:
    def __init__(self, config: dict):
        # Initialize all modules from Weeks 1-4
        self.exchange = ExchangeClient(
            {
                "apiKey": config["binance_key"],
                "secret": config["binance_secret"],
                "sandbox": True,
            }
        )
        self.inventory = InventoryTracker([Venue.BINANCE, Venue.WALLET])
        self.fees = FeeStructure()
        self.generator = SignalGenerator(
            self.exchange,
            None,
            self.inventory,
            self.fees,
            config.get("signal_config", {}),
        )
        self.scorer = SignalScorer()
        simulation_mode = config.get("simulation", False)
        self.wallet: WalletManager | None = None
        self.chain_client: ChainClient | None = None
        self.dex_quote_token = config.get("dex_quote_token_address") or os.getenv(
            "DEX_QUOTE_TOKEN_ADDRESS"
        )
        self.dex_quote_decimals = int(
            config.get("dex_quote_decimals", os.getenv("DEX_QUOTE_TOKEN_DECIMALS", "6"))
        )
        self.dex_chain_id = int(config.get("dex_chain_id", 11155111))

        if not simulation_mode:
            rpc_url = config.get("dex_rpc_url") or os.getenv("SEPOLIA_RPC_URL")
            if not rpc_url:
                raise ValueError("SEPOLIA_RPC_URL is required when simulation=False")
            private_key = config.get("dex_private_key") or os.getenv("PRIVATE_KEY")
            if not private_key:
                raise ValueError("PRIVATE_KEY is required when simulation=False")
            self.wallet = WalletManager(private_key)
            self.chain_client = ChainClient([rpc_url])

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

        self.pairs = config.get("pairs", ["ETH/USDT"])
        self.trade_size = config.get("trade_size", 0.1)
        self.running = False

    async def run(self):
        self.running = True
        logging.info("Bot starting...")
        await self._sync_balances()

        while self.running:
            try:
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

        for pair in self.pairs:
            # Per-pair circuit breaker check
            if self.executor.circuit_breaker.is_open(pair):
                logging.info("CB open for %s — skipping", pair)
                continue

            signal = self.generator.generate(pair, self.trade_size)
            if signal is None:
                continue

            # Score signal with real inventory skew data
            skews = self._get_inventory_skews(pair)
            signal.score = self.scorer.score(signal, skews)

            if signal.score < self.scorer.config.min_score:
                continue

            logging.info(
                f"Signal: {pair} spread={signal.spread_bps:.1f}bps score={signal.score}"
            )

            # Execute (pre-flight + replay handled inside executor)
            ctx = await self.executor.execute(signal)

            # Record result for scorer history
            success = ctx.state == ExecutorState.DONE
            self.scorer.record_result(pair, success)

            if success:
                logging.info(
                    "SUCCESS %s  PnL=$%.4f  duration=%.0fms  retries=%d/%d",
                    pair,
                    ctx.actual_net_pnl or 0,
                    ctx.duration_ms or 0,
                    ctx.metrics.leg1_retries,
                    ctx.metrics.leg2_retries,
                )
            else:
                logging.warning(
                    "FAILED %s  error=%s  unwind=%s  cb=%s",
                    pair,
                    ctx.error,
                    ctx.metrics.unwind_success,
                    recovery.snapshot(pair)["circuit_breaker"],
                )

            await self._sync_balances()

    async def _sync_balances(self):
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

    def stop(self):
        self.running = False

    def _get_inventory_skews(self, pair: str) -> list[dict]:
        """Build skew dicts for the base and quote assets of *pair*."""
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


# Entry point
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "binance_key": os.getenv("BINANCE_TESTNET_API_KEY"),
        "binance_secret": os.getenv("BINANCE_TESTNET_SECRET"),
        "pairs": ["ETH/USDT"],
        "trade_size": 0.1,
        "simulation": False,
        "dex_rpc_url": os.getenv("SEPOLIA_RPC_URL"),
        "dex_router_address": os.getenv("DEX_ROUTER_ADDRESS"),
        "dex_weth_address": os.getenv("DEX_WETH_ADDRESS"),
        "dex_quote_token_address": os.getenv("DEX_QUOTE_TOKEN_ADDRESS"),
    }
    bot = ArbBot(config)
    asyncio.run(bot.run())
