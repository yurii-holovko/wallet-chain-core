"""Tests for strategy.generator — SignalGenerator."""

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from inventory.tracker import InventoryTracker, Venue
from strategy.fees import FeeStructure
from strategy.generator import SignalGenerator
from strategy.signal import Direction

# ── helpers ─────────────────────────────────────────────────────


def _mock_exchange(bid=2000.0, ask=2001.0):
    """Return a mock ExchangeClient with a configurable order book."""
    exchange = MagicMock()
    exchange.fetch_order_book.return_value = {
        "bids": [(Decimal(str(bid)), Decimal("10"))],
        "asks": [(Decimal(str(ask)), Decimal("10"))],
        "best_bid": (Decimal(str(bid)), Decimal("10")),
        "best_ask": (Decimal(str(ask)), Decimal("10")),
    }
    return exchange


def _funded_inventory(
    base="ETH",
    quote="USDT",
    cex_base="5",
    cex_quote="50000",
    wallet_base="5",
    wallet_quote="50000",
):
    """Build an InventoryTracker with balances on both venues."""
    tracker = InventoryTracker([Venue.BINANCE, Venue.WALLET])
    tracker.update_from_cex(
        Venue.BINANCE,
        {
            base: {"free": Decimal(cex_base), "locked": Decimal("0")},
            quote: {"free": Decimal(cex_quote), "locked": Decimal("0")},
        },
    )
    tracker.update_from_wallet(
        Venue.WALLET,
        {base: wallet_base, quote: wallet_quote},
    )
    return tracker


def _make_generator(
    exchange=None,
    inventory=None,
    fees=None,
    config=None,
):
    """Build a SignalGenerator with sensible test defaults."""
    default_config = {
        "min_spread_bps": 30,
        "min_profit_usd": 1.0,
        "cooldown_seconds": 0,
    }
    if config is not None:
        default_config.update(config)
    return SignalGenerator(
        exchange_client=exchange or _mock_exchange(),
        pricing_module=None,
        inventory_tracker=inventory or _funded_inventory(),
        fee_structure=fees or FeeStructure(),
        config=default_config,
    )


# ── Signal emission ─────────────────────────────────────────────


class TestSignalEmission:
    def test_emits_signal_when_spread_sufficient(self):
        # dex_sell = mid * 1.008 → ~80 bps above ask → should emit
        gen = _make_generator()
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert sig.pair == "ETH/USDT"
        assert sig.direction in (
            Direction.BUY_CEX_SELL_DEX,
            Direction.BUY_DEX_SELL_CEX,
        )

    def test_signal_has_positive_net_pnl(self):
        gen = _make_generator()
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert sig.expected_net_pnl > 0

    def test_signal_populates_meta(self):
        gen = _make_generator()
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert "cex_bid" in sig.meta
        assert "breakeven_bps" in sig.meta

    def test_signal_id_format(self):
        gen = _make_generator()
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert sig.signal_id.startswith("ETHUSDT_")

    def test_signal_expiry_in_future(self):
        gen = _make_generator()
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert sig.expiry > time.time()


# ── Direction picking ───────────────────────────────────────────


class TestDirectionPicking:
    def test_buy_cex_sell_dex_when_dex_higher(self):
        # Default mock: dex_sell > cex_ask → BUY_CEX_SELL_DEX
        gen = _make_generator()
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert sig.direction == Direction.BUY_CEX_SELL_DEX

    def test_picks_wider_spread(self):
        # Both directions have spread, pick the wider one
        gen = _make_generator(exchange=_mock_exchange(bid=2000, ask=2001))
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        # dex_sell markup (1.008) > dex_buy markup (1.005)
        # so spread_a (sell on DEX) should be wider
        assert sig.direction == Direction.BUY_CEX_SELL_DEX


# ── Rejection gates ─────────────────────────────────────────────


class TestRejectionGates:
    def test_returns_none_when_spread_too_small(self):
        gen = _make_generator(config={"min_spread_bps": 500, "cooldown_seconds": 0})
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is None

    def test_returns_none_when_net_pnl_below_min(self):
        gen = _make_generator(
            config={
                "min_spread_bps": 30,
                "min_profit_usd": 999_999,
                "cooldown_seconds": 0,
            }
        )
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is None

    def test_returns_none_when_exchange_errors(self):
        exchange = MagicMock()
        exchange.fetch_order_book.side_effect = RuntimeError("timeout")
        gen = _make_generator(exchange=exchange)
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is None

    def test_returns_none_when_orderbook_empty(self):
        exchange = MagicMock()
        exchange.fetch_order_book.return_value = {"bids": [], "asks": []}
        gen = _make_generator(exchange=exchange)
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is None

    def test_cooldown_blocks_rapid_signals(self):
        gen = _make_generator(config={"min_spread_bps": 30, "cooldown_seconds": 60})
        sig1 = gen.generate("ETH/USDT", 1.0)
        assert sig1 is not None
        sig2 = gen.generate("ETH/USDT", 1.0)
        assert sig2 is None  # blocked by cooldown

    def test_cooldown_per_pair(self):
        gen = _make_generator(config={"min_spread_bps": 30, "cooldown_seconds": 60})
        sig1 = gen.generate("ETH/USDT", 1.0)
        assert sig1 is not None
        # Different pair should not be blocked by ETH/USDT cooldown
        sig2 = gen.generate("BTC/USDT", 1.0)
        assert sig2 is not None


# ── Inventory validation ────────────────────────────────────────


class TestInventoryValidation:
    def test_inventory_ok_when_funded(self):
        gen = _make_generator(inventory=_funded_inventory())
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert sig.inventory_ok is True
        assert sig.rejection_reasons == []

    def test_inventory_blocked_when_no_quote(self):
        inv = _funded_inventory(cex_quote="0", wallet_quote="0")
        gen = _make_generator(inventory=inv)
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert sig.inventory_ok is False
        assert any("inventory" in r for r in sig.rejection_reasons)

    def test_inventory_blocked_when_no_base_on_wallet(self):
        inv = _funded_inventory(wallet_base="0")
        gen = _make_generator(inventory=inv)
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        # BUY_CEX_SELL_DEX needs base on wallet to sell
        assert sig.inventory_ok is False

    def test_signal_still_emitted_when_blocked(self):
        """Blocked signals are still returned (with inventory_ok=False)
        so the caller can log / learn from them."""
        inv = _funded_inventory(cex_quote="0")
        gen = _make_generator(inventory=inv)
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert not sig.is_valid()

    def test_none_inventory_passes_check(self):
        gen = _make_generator(inventory=None)
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert sig.inventory_ok is True


# ── Position limits ─────────────────────────────────────────────


class TestPositionLimits:
    def test_within_limits(self):
        gen = _make_generator(
            config={
                "min_spread_bps": 30,
                "max_position_usd": 50_000,
                "cooldown_seconds": 0,
            }
        )
        sig = gen.generate("ETH/USDT", 1.0)  # ~$2000
        assert sig is not None
        assert sig.within_limits is True

    def test_exceeds_limits(self):
        gen = _make_generator(
            config={
                "min_spread_bps": 30,
                "max_position_usd": 100,
                "cooldown_seconds": 0,
            }
        )
        sig = gen.generate("ETH/USDT", 1.0)  # ~$2000 > $100
        assert sig is not None
        assert sig.within_limits is False
        assert any("position_limit" in r for r in sig.rejection_reasons)


# ── Fee economics ───────────────────────────────────────────────


class TestFeeEconomics:
    def test_fees_deducted_from_gross(self):
        gen = _make_generator()
        sig = gen.generate("ETH/USDT", 1.0)
        assert sig is not None
        assert sig.expected_fees > 0
        assert sig.expected_net_pnl < sig.expected_gross_pnl

    def test_higher_gas_reduces_net_pnl(self):
        low_gas = FeeStructure(gas_cost_usd=1.0)
        high_gas = FeeStructure(gas_cost_usd=20.0)
        gen_low = _make_generator(fees=low_gas)
        gen_high = _make_generator(fees=high_gas)
        sig_low = gen_low.generate("ETH/USDT", 1.0)
        sig_high = gen_high.generate("ETH/USDT", 1.0)
        # high gas might kill the signal entirely
        if sig_high is not None and sig_low is not None:
            assert sig_low.expected_net_pnl > sig_high.expected_net_pnl


# ── FeeStructure unit tests ────────────────────────────────────


class TestFeeStructure:
    def test_total_fee_includes_all_components(self):
        f = FeeStructure(
            cex_taker_bps=10, dex_swap_bps=30, gas_cost_usd=5, slippage_bps=5
        )
        # $1000 trade: gas = 5/1000*10000 = 50 bps
        total = f.total_fee_bps(1000)
        assert total == pytest.approx(95.0)

    def test_gas_bps_scales_with_trade_size(self):
        f = FeeStructure(gas_cost_usd=10)
        small = f.total_fee_bps(100)  # gas = 1000 bps
        large = f.total_fee_bps(10_000)  # gas = 10 bps
        assert small > large

    def test_breakeven_equals_total_fee(self):
        f = FeeStructure()
        assert f.breakeven_spread_bps(1000) == f.total_fee_bps(1000)

    def test_net_profit_positive_above_breakeven(self):
        f = FeeStructure()
        be = f.breakeven_spread_bps(1000)
        assert f.net_profit_usd(be + 10, 1000) > 0

    def test_net_profit_negative_below_breakeven(self):
        f = FeeStructure()
        be = f.breakeven_spread_bps(1000)
        assert f.net_profit_usd(be - 10, 1000) < 0

    def test_zero_trade_value_returns_inf(self):
        f = FeeStructure()
        assert f.total_fee_bps(0) == float("inf")
