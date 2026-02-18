from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import get_env
from exchange.mexc_client import MexcApiError, MexcClient, MexcOrderStatus
from pricing.odos_client import OdosClient

logger = logging.getLogger(__name__)


@dataclass
class DoubleLimitConfig:
    """
    Configuration for the Double Limit micro-arbitrage engine.
    """

    trade_size_usd: float = 5.0
    # Legacy flat thresholds (kept as sensible defaults / fallback).
    # NOTE: dynamic per-fee-tier thresholds are applied via ``min_spread_by_tier``.
    min_spread_pct: float = 0.005  # 0.5% minimum gross spread (fallback)
    min_profit_usd: float = 0.001  # do not bother for < $0.001
    max_slippage_pct: float = 0.5
    # Cost model (conservative defaults for Arbitrum micro-trades)
    gas_cost_usd: float = 0.03
    # ODOS "surplus" fee is small but non-zero (0.01% = 1 bp)
    odos_fee_pct: float = 0.0001
    # Bridge amortization: do not assume infinite trades; use at least N.
    min_trades_for_bridge_amortization: int = 5
    position_ttl_seconds: int = 600  # 10 minutes
    monitor_interval_seconds: float = 1.0
    usdc_decimals: int = 6
    simulation_mode: bool = True

    # Dynamic spread thresholds by Uniswap V3 fee tier.
    # 500   → 0.05% pool fee  → require 0.7% spread
    # 3000  → 0.30% pool fee  → require 1.0% spread
    # 10000 → 1.00% pool fee  → require 1.5% spread
    min_spread_by_tier: Dict[int, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.min_spread_by_tier is None:
            self.min_spread_by_tier = {
                500: 0.007,  # 0.7% for 0.05% fee tier (ARB, GMX, LINK, UNI, BAL)
                3_000: 0.010,  # 1.0% for 0.3% fee tier (MAGIC, GNS, RDNT, etc.)
                10_000: 0.015,  # 1.5% for 1% fee-tier pools (safety margin)
            }

    def get_min_spread(self, fee_tier: int) -> float:
        """
        Return the minimum required gross spread for a given V3 fee tier.

        Falls back to ``min_spread_pct`` if tier is unknown.
        """
        return float(self.min_spread_by_tier.get(int(fee_tier), self.min_spread_pct))


@dataclass
class DoubleLimitOpportunity:
    """
    Evaluated opportunity between MEXC and Arbitrum for a single token.
    """

    token_symbol: str
    token_address: str
    mex_symbol: str
    direction: str
    mex_bid: float
    mex_ask: float
    odos_price: float
    gross_spread: float
    total_cost_usd: float
    net_profit_usd: float
    net_profit_pct: float
    executable: bool


class DoubleLimitArbitrageEngine:
    """
    High-level coordination for the Double Limit strategy.

    This component is intentionally decoupled from the existing ``Executor``
    state machine so that micro-arbitrage can evolve independently while
    still sharing lower-level primitives (MEXC client, chain client, ODOS).

    Notes:
      * The on-chain leg (Uniswap V3 range order) is injected as ``range_manager``.
      * Capital/bridging policy is delegated to a separate capital manager.
    """

    def __init__(
        self,
        mexc_client: MexcClient,
        odos_client: OdosClient,
        token_mappings: Dict[str, Dict[str, Any]],
        config: Optional[DoubleLimitConfig] = None,
        range_manager: Any | None = None,
        capital_manager: Any | None = None,
    ) -> None:
        self.mexc = mexc_client
        self.odos = odos_client
        self.tokens = token_mappings
        self.config = config or DoubleLimitConfig()
        self.range_manager = range_manager
        self.capital_manager = capital_manager

        # Environment-derived addresses
        usdc = get_env("USDC_ADDRESS", required=True)
        user = get_env("ARBITRUM_WALLET_ADDRESS", required=True)
        assert usdc is not None
        assert user is not None
        self.usdc_address: str = usdc
        self.user_address: str = user

    # ── Opportunity evaluation ─────────────────────────────────────

    def evaluate_opportunity(self, key: str) -> Optional[DoubleLimitOpportunity]:
        """
        Compute the net economics for a token given current MEXC and ODOS prices.

        ``key`` should refer to an entry in ``token_mappings`` such as 'ARB'.
        """
        cfg = self.tokens.get(key)
        if not cfg:
            logger.warning("No token mapping for key=%s", key)
            return None

        # Skip tokens explicitly marked as inactive
        if not cfg.get("active", True):
            logger.debug("Token %s marked inactive, skipping", key)
            return None

        # Skip tokens that are not supported by ODOS (unless a future
        # alternative DEX integration is provided).
        if not cfg.get("odos_supported", True):
            logger.debug("Token %s not supported by ODOS, skipping", key)
            return None

        token_symbol = key
        token_address = cfg["address"]
        token_decimals = int(cfg.get("decimals", 18))
        mex_symbol = cfg["mex_symbol"]

        # 1. MEXC top-of-book
        try:
            book = self.mexc.get_order_book(mex_symbol, limit=5)
        except MexcApiError as exc:
            logger.warning("MEXC order book failed for %s: %s", mex_symbol, exc)
            return None
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            logger.debug("Empty MEXC order book for %s", mex_symbol)
            return None

        mex_bid = bids[0][0]
        mex_ask = asks[0][0]

        # 2. ODOS quote for $5 USDC -> token
        amount_usdc_raw = int(
            self.config.trade_size_usd * (10**self.config.usdc_decimals)
        )
        try:
            quote = self.odos.quote(
                input_token=self.usdc_address,
                output_token=token_address,
                amount_in=amount_usdc_raw,
                user_address=self.user_address,
                slippage_percent=self.config.max_slippage_pct,
            )
        except Exception as exc:
            logger.warning("ODOS quote failed for %s: %s", token_symbol, exc)
            return None

        # Convert ODOS raw output amount into an effective USD price per token.
        # amount_out is in token_out raw units (10**token_decimals).
        token_out_human = quote.amount_out / float(10**token_decimals)
        if token_out_human <= 0:
            return None
        # We send trade_size_usd worth of USDC and receive token_out_human units.
        arbitrum_price = self.config.trade_size_usd / token_out_human

        # 3. Spreads
        spread_mex_cheaper = (arbitrum_price - mex_ask) / mex_ask  # buy MEXC, sell Arb
        spread_arb_cheaper = (
            mex_bid - arbitrum_price
        ) / arbitrum_price  # buy Arb, sell MEXC

        best_spread = max(spread_mex_cheaper, spread_arb_cheaper)
        if best_spread <= 0:
            return None

        direction = (
            "mex_to_arb" if spread_mex_cheaper >= spread_arb_cheaper else "arb_to_mex"
        )

        # 4. Approximate fixed costs (gas, LP, bridge amortization)
        # Cost breakdown:
        #   - LP fee: Uniswap V3 fee (0.05% for tier 500, 0.3% for tier 3000)
        #     Applied once on DEX side (mex_to_arb: sell on Arb, arb_to_mex: buy on Arb)
        #   - ODOS fee: Aggregator surplus fee (~0.01% = 1 bp), conservative estimate
        #   - Gas cost: Fixed per-trade cost (configurable, default $0.03)
        #   - Bridge cost: Amortized withdrawal ($0.05 / min_trades, default 5)
        #   - MEXC fee: 0% for post-only limit orders (maker fee)
        fee_tier = int(cfg.get("fee_tier", 3_000))
        lp_fee_pct = self._lp_fee_pct(fee_tier)
        lp_fee_usd = lp_fee_pct * self.config.trade_size_usd

        # ODOS aggregator fee (small but non-zero, ~0.01% = 1 bp)
        odos_fee_usd = float(self.config.odos_fee_pct) * self.config.trade_size_usd

        # Bridge amortization: use bridge_fixed_cost_usd and amortize across a
        # conservative minimum number of trades if we haven't recorded any yet.
        bridge_amortized = 0.0
        if self.capital_manager is not None:
            fixed = float(
                getattr(
                    getattr(self.capital_manager, "config", None),
                    "bridge_fixed_cost_usd",
                    0.0,
                )
                or 0.0
            )
            trades = int(
                getattr(self.capital_manager, "trade_count_since_bridge", 0) or 0
            )
            denom = max(trades, int(self.config.min_trades_for_bridge_amortization))
            bridge_amortized = fixed / float(denom) if fixed > 0 else 0.0

        gas_cost_usd = float(self.config.gas_cost_usd)
        total_cost = gas_cost_usd + lp_fee_usd + odos_fee_usd + bridge_amortized

        gross_profit_usd = best_spread * self.config.trade_size_usd
        net_profit_usd = gross_profit_usd - total_cost
        net_profit_pct = net_profit_usd / self.config.trade_size_usd

        # Dynamic spread threshold based on pool fee tier.
        min_spread_required = self.config.get_min_spread(fee_tier)

        executable = (
            best_spread >= min_spread_required
            and net_profit_usd >= self.config.min_profit_usd
        )

        return DoubleLimitOpportunity(
            token_symbol=token_symbol,
            token_address=token_address,
            mex_symbol=mex_symbol,
            direction=direction,
            mex_bid=mex_bid,
            mex_ask=mex_ask,
            odos_price=arbitrum_price,
            gross_spread=best_spread,
            total_cost_usd=total_cost,
            net_profit_usd=net_profit_usd,
            net_profit_pct=net_profit_pct,
            executable=executable,
        )

    @staticmethod
    def _lp_fee_pct(fee_tier: int) -> float:
        """
        Convert Uniswap V3 fee tier (uint24) to LP fee percent.

        fee_tier values:
          100   = 0.01%
          500   = 0.05%
          3000  = 0.30%
          10000 = 1.00%
        """
        return {
            100: 0.0001,
            500: 0.0005,
            3_000: 0.003,
            10_000: 0.01,
        }.get(int(fee_tier), 0.003)

    # ── Execution ─────────────────────────────────────────────────

    async def execute_double_limit(self, opp: DoubleLimitOpportunity) -> Dict[str, Any]:
        """
        Place the two limit legs (MEXC + Arbitrum range order) and monitor them.

        The on-chain leg is optional; when ``range_manager`` is not provided we
        still place and monitor the MEXC leg, which is useful for dry-runs.
        """
        if not opp.executable:
            return {"status": "SKIPPED", "reason": "not executable", "opportunity": opp}

        if opp.direction == "mex_to_arb":
            mex_side = "BUY"
            mex_price = opp.mex_ask * 0.9995  # small discount to ensure maker
        else:
            mex_side = "SELL"
            mex_price = opp.mex_bid * 1.0005

        trade_size_base = self.config.trade_size_usd / opp.odos_price

        # Side A: post-only limit on MEXC
        try:
            mex_order = self.mexc.place_limit_order(
                symbol=opp.mex_symbol,
                side=mex_side,
                quantity=trade_size_base,
                price=mex_price,
                post_only=True,
            )
        except Exception as exc:
            logger.warning("Failed to place MEXC limit order: %s", exc)
            return {"status": "FAILED", "error": str(exc), "opportunity": opp}

        # Side B: Uniswap V3 range order (optional)
        v3_position_id: Optional[int] = None
        if self.range_manager is not None:
            try:
                if opp.direction == "mex_to_arb":
                    v3_result = self.range_manager.place_limit_sell_order(
                        token_in=self.usdc_address,
                        token_out=opp.token_address,
                        amount_in=int(
                            self.config.trade_size_usd * (10**self.config.usdc_decimals)
                        ),
                    )
                else:
                    v3_result = self.range_manager.place_limit_buy_order(
                        token_in=opp.token_address,
                        token_out=self.usdc_address,
                        amount_in=int(
                            self.config.trade_size_usd * (10**self.config.usdc_decimals)
                        ),
                    )
                v3_position_id = int(v3_result.get("token_id"))
            except Exception as exc:
                logger.warning("Failed to place V3 range order: %s", exc)

        result = await self._monitor_positions(
            mex_order=mex_order,
            v3_position_id=v3_position_id,
            opportunity=opp,
        )
        return result

    async def _monitor_positions(
        self,
        mex_order: MexcOrderStatus,
        v3_position_id: Optional[int],
        opportunity: DoubleLimitOpportunity,
    ) -> Dict[str, Any]:
        """
        Monitor both legs for up to ``position_ttl_seconds``.

        This is intentionally conservative: if only one side fills by the
        deadline we cancel the open leg (if possible) and return a structured
        result. Unwind logic will be layered on top by a higher-level manager.
        """
        start = time.time()

        while time.time() - start < self.config.position_ttl_seconds:
            try:
                mex_status = self.mexc.get_order_status(
                    symbol=mex_order.symbol, order_id=mex_order.order_id
                )
            except Exception as exc:
                logger.warning("MEXC status check failed: %s", exc)
                mex_status = mex_order

            v3_status: Dict[str, Any] = {}
            v3_executed = False
            if self.range_manager is not None and v3_position_id is not None:
                try:
                    v3_status = self.range_manager.check_position_status(v3_position_id)
                    v3_executed = bool(v3_status.get("is_executed"))
                except Exception as exc:
                    logger.warning("V3 status check failed: %s", exc)

            both_filled = mex_status.is_filled and (
                v3_executed or v3_position_id is None
            )

            if both_filled:
                if self.range_manager is not None and v3_position_id is not None:
                    try:
                        self.range_manager.withdraw_executed_position(v3_position_id)
                    except Exception as exc:
                        logger.warning("V3 withdraw failed: %s", exc)

                # Profit attribution and compounding are delegated to capital manager.
                if self.capital_manager is not None:
                    try:
                        self.capital_manager.record_trade(opportunity.net_profit_usd)
                    except Exception:
                        logger.exception("Capital manager record_trade failed")

                return {
                    "status": "SUCCESS",
                    "mex_order": mex_status,
                    "v3_status": v3_status,
                    "opportunity": opportunity,
                }

            await asyncio.sleep(self.config.monitor_interval_seconds)

        # Timeout: cancel remaining legs
        try:
            if mex_order.is_active:
                self.mexc.cancel_order(
                    symbol=mex_order.symbol, order_id=mex_order.order_id
                )
        except Exception:
            logger.exception(
                "Failed to cancel expired MEXC order %s", mex_order.order_id
            )

        if self.range_manager is not None and v3_position_id is not None:
            try:
                self.range_manager.withdraw_executed_position(v3_position_id)
            except Exception:
                logger.exception("Failed to withdraw V3 position %s", v3_position_id)

        return {
            "status": "TIMEOUT",
            "mex_order": mex_order,
            "v3_position_id": v3_position_id,
            "opportunity": opportunity,
        }
