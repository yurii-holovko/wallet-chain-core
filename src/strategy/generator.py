"""
Signal Generator — the brain of the arbitrage system.

Responsibilities:
  1. Fetch CEX prices (order book) and DEX prices (simulated / pricing module)
  2. Compute spread in both directions
  3. Estimate economics: gross PnL, fees, net PnL
  4. Validate against inventory (can we actually execute both legs?)
  5. Validate against position limits
  6. Emit a Signal or return None (with logged reason)
"""

import logging
import time
from decimal import Decimal
from typing import Optional

from strategy.fees import FeeStructure
from strategy.signal import Direction, Signal

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Detect arbitrage opportunities, validate against inventory, emit signals.

    Parameters
    ----------
    exchange_client : ExchangeClient
        Used to fetch CEX order book (and optionally live fees).
    pricing_module : object or None
        Future DEX pricing source.  When None, DEX prices are simulated
        from the CEX mid-price.
    inventory_tracker : InventoryTracker
        Pre-flight balance checks across venues.
    fee_structure : FeeStructure
        Cost model (CEX fee, DEX fee, gas, slippage).
    config : dict
        Tunable thresholds — see defaults below.
    """

    def __init__(
        self,
        exchange_client,
        pricing_module,
        inventory_tracker,
        fee_structure: FeeStructure,
        config: dict,
    ):
        self.exchange = exchange_client
        self.pricing = pricing_module
        self.inventory = inventory_tracker
        self.fees = fee_structure

        # ── tunables ────────────────────────────────────────────
        self.min_spread_bps: float = config.get("min_spread_bps", 50)
        self.min_profit_usd: float = config.get("min_profit_usd", 5.0)
        self.max_position_usd: float = config.get("max_position_usd", 10_000)
        self.signal_ttl: float = config.get("signal_ttl_seconds", 5)
        self.cooldown: float = config.get("cooldown_seconds", 2)
        self.dex_buy_markup: float = config.get("dex_buy_markup", 1.005)
        self.dex_sell_markup: float = config.get("dex_sell_markup", 1.008)

        # ── internal state ──────────────────────────────────────
        self.last_signal_time: dict[str, float] = {}

    # ── public API ──────────────────────────────────────────────

    def generate(self, pair: str, size: float) -> Optional[Signal]:
        """
        Main entry point.  Attempt to detect an arb opportunity for *pair*
        at trade *size*.

        Returns a Signal if an opportunity passes every gate, None otherwise.
        Every rejection is logged so you can trace exactly why a tick
        produced nothing.
        """
        # Gate 1: cooldown — avoid hammering the same pair
        if self._in_cooldown(pair):
            logger.debug("%s  skip: cooldown active", pair)
            return None

        # Gate 2: fetch live prices
        prices = self._fetch_prices(pair, size)
        if prices is None:
            return None  # _fetch_prices already logs the reason

        # Gate 3: compute spreads in both directions
        direction, spread, cex_price, dex_price = self._pick_direction(prices, pair)
        if direction is None:
            return None  # _pick_direction already logs

        # Gate 4: economics — is the trade profitable after all costs?
        trade_value = size * cex_price
        gross_pnl = (spread / 10_000) * trade_value
        total_fee_bps = self.fees.total_fee_bps(trade_value)
        fees = (total_fee_bps / 10_000) * trade_value
        net_pnl = gross_pnl - fees

        if net_pnl < self.min_profit_usd:
            logger.debug(
                "%s  skip: net_pnl $%.2f < min $%.2f  "
                "(spread=%.1f bps, fees=%.1f bps)",
                pair,
                net_pnl,
                self.min_profit_usd,
                spread,
                total_fee_bps,
            )
            return None

        # Gate 5: inventory — do we have the balances on both venues?
        inventory_ok, inv_reason = self._check_inventory(
            pair, direction, size, cex_price
        )

        # Gate 6: position limit
        within_limits = trade_value <= self.max_position_usd

        rejection_reasons: list[str] = []
        if not inventory_ok:
            rejection_reasons.append(f"inventory: {inv_reason}")
        if not within_limits:
            rejection_reasons.append(
                f"position_limit: ${trade_value:,.0f} > "
                f"${self.max_position_usd:,.0f}"
            )

        # ── Build signal ────────────────────────────────────────
        signal = Signal.create(
            pair=pair,
            direction=direction,
            cex_price=cex_price,
            dex_price=dex_price,
            spread_bps=spread,
            size=size,
            expected_gross_pnl=round(gross_pnl, 4),
            expected_fees=round(fees, 4),
            expected_net_pnl=round(net_pnl, 4),
            score=0,  # scorer fills this in later
            expiry=time.time() + self.signal_ttl,
            inventory_ok=inventory_ok,
            within_limits=within_limits,
            rejection_reasons=rejection_reasons,
            meta={
                "cex_bid": prices["cex_bid"],
                "cex_ask": prices["cex_ask"],
                "dex_buy": prices["dex_buy"],
                "dex_sell": prices["dex_sell"],
                "breakeven_bps": round(total_fee_bps, 2),
            },
        )

        if rejection_reasons:
            logger.info(
                "%s  signal BLOCKED: %s  (spread=%.1f bps, net=$%.2f)",
                pair,
                "; ".join(rejection_reasons),
                spread,
                net_pnl,
            )
        else:
            logger.info(
                "%s  signal EMITTED: %s spread=%.1f bps net=$%.2f size=%.4f",
                pair,
                direction.value,
                spread,
                net_pnl,
                size,
            )

        self.last_signal_time[pair] = time.time()
        return signal

    # ── private helpers ─────────────────────────────────────────

    def _in_cooldown(self, pair: str) -> bool:
        elapsed = time.time() - self.last_signal_time.get(pair, 0)
        return elapsed < self.cooldown

    def _fetch_prices(self, pair: str, size: float) -> Optional[dict]:
        """
        Fetch CEX order book and derive DEX prices.

        Returns dict with float prices:
          cex_bid, cex_ask, dex_buy, dex_sell
        or None on failure.
        """
        try:
            ob = self.exchange.fetch_order_book(pair)
        except Exception:
            logger.warning("%s  price fetch failed (exchange error)", pair)
            return None

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            logger.warning("%s  empty order book", pair)
            return None

        # ExchangeClient returns (Decimal, Decimal) tuples — cast to float
        cex_bid = float(bids[0][0])
        cex_ask = float(asks[0][0])

        if cex_bid <= 0 or cex_ask <= 0:
            logger.warning("%s  invalid prices bid=%s ask=%s", pair, cex_bid, cex_ask)
            return None

        # DEX prices — use pricing module when available, else simulate
        if self.pricing is not None:
            dex_prices = self._fetch_dex_prices(pair, size)
            if dex_prices is not None:
                return {
                    "cex_bid": cex_bid,
                    "cex_ask": cex_ask,
                    "dex_buy": dex_prices["buy"],
                    "dex_sell": dex_prices["sell"],
                }

        # Simulated DEX prices based on CEX mid
        mid = (cex_bid + cex_ask) / 2
        return {
            "cex_bid": cex_bid,
            "cex_ask": cex_ask,
            "dex_buy": mid * self.dex_buy_markup,
            "dex_sell": mid * self.dex_sell_markup,
        }

    def _fetch_dex_prices(self, pair: str, size: float) -> Optional[dict]:
        """Fetch real DEX quote via pricing module (future integration)."""
        try:
            return self.pricing.get_quote(pair, size)
        except Exception:
            logger.warning("%s  DEX pricing failed, falling back to sim", pair)
            return None

    def _pick_direction(
        self, prices: dict, pair: str
    ) -> tuple[Optional[Direction], float, float, float]:
        """
        Compare spreads in both directions, return the better one
        if it exceeds min_spread_bps.

        Direction A  (BUY_CEX_SELL_DEX):
          Buy on CEX at ask → sell on DEX at dex_sell
          spread = (dex_sell − cex_ask) / cex_ask × 10 000

        Direction B  (BUY_DEX_SELL_CEX):
          Buy on DEX at dex_buy → sell on CEX at bid
          spread = (cex_bid − dex_buy) / dex_buy × 10 000
        """
        cex_ask = prices["cex_ask"]
        cex_bid = prices["cex_bid"]
        dex_sell = prices["dex_sell"]
        dex_buy = prices["dex_buy"]

        spread_a = (dex_sell - cex_ask) / cex_ask * 10_000
        spread_b = (cex_bid - dex_buy) / dex_buy * 10_000

        # Pick the wider spread
        if spread_a >= spread_b and spread_a >= self.min_spread_bps:
            return Direction.BUY_CEX_SELL_DEX, spread_a, cex_ask, dex_sell
        if spread_b >= self.min_spread_bps:
            return Direction.BUY_DEX_SELL_CEX, spread_b, cex_bid, dex_buy

        logger.debug(
            "%s  skip: no spread above min  " "A=%.1f bps  B=%.1f bps  (min=%s)",
            pair,
            spread_a,
            spread_b,
            self.min_spread_bps,
        )
        return None, 0.0, 0.0, 0.0

    def _check_inventory(
        self,
        pair: str,
        direction: Direction,
        size: float,
        cex_price: float,
    ) -> tuple[bool, Optional[str]]:
        """
        Use InventoryTracker.can_execute() for a pre-flight balance check.

        BUY_CEX_SELL_DEX means:
          - We spend USDT on Binance (buy leg)  → need quote on CEX
          - We sell ETH on DEX (sell leg)        → need base on wallet

        BUY_DEX_SELL_CEX means:
          - We spend USDT on DEX (buy leg)       → need quote on wallet
          - We sell ETH on Binance (sell leg)     → need base on CEX
        """
        if self.inventory is None:
            return True, None

        base, quote = pair.split("/")
        cost = Decimal(str(size)) * Decimal(str(cex_price)) * Decimal("1.01")
        qty = Decimal(str(size))

        try:
            if direction == Direction.BUY_CEX_SELL_DEX:
                from inventory.tracker import Venue

                result = self.inventory.can_execute(
                    buy_venue=Venue.BINANCE,
                    buy_asset=quote,
                    buy_amount=cost,
                    sell_venue=Venue.WALLET,
                    sell_asset=base,
                    sell_amount=qty,
                )
            else:
                from inventory.tracker import Venue

                result = self.inventory.can_execute(
                    buy_venue=Venue.WALLET,
                    buy_asset=quote,
                    buy_amount=cost,
                    sell_venue=Venue.BINANCE,
                    sell_asset=base,
                    sell_amount=qty,
                )

            return result["can_execute"], result.get("reason")
        except Exception as exc:
            logger.warning("%s  inventory check failed: %s", pair, exc)
            return False, str(exc)
