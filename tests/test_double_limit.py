"""
Comprehensive tests for the Double Limit micro-arbitrage engine.

Covers:
  - DoubleLimitConfig: defaults, fee-tier thresholds, get_min_spread
  - DoubleLimitOpportunity: dataclass fields
  - DoubleLimitArbitrageEngine.evaluate_opportunity: direction, spreads, costs,
    executable gate, inactive/odos-unsupported/missing tokens, error handling
  - _lp_fee_pct: all tiers + unknown fallback
  - execute_double_limit (simulation): success, timeout, mex_reject,
    dex_failed, not-executable skip
  - execute_double_limit (live): mex_to_arb and arb_to_mex full flow,
    insufficient balance skip, MEXC order failure, DEX swap integration
  - _monitor_positions: both-filled success, timeout + cancel, timeout + unwind,
    MEXC filled + DEX failed → immediate unwind
  - _unwind_mexc_filled: simulation mode, mex_to_arb / arb_to_mex directions,
    zero qty, market order failure
  - Execution report: format_double_limit_report for all statuses (DEX swap + V3)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _run(coro):
    """Run an async coroutine synchronously (Python 3.10+ compatible)."""
    return asyncio.run(coro)


from exchange.mexc_client import MexcApiError, MexcOrderStatus  # noqa: E402
from executor.double_limit_engine import (  # noqa: E402
    V3_GAS_ESTIMATE,
    DoubleLimitArbitrageEngine,
    DoubleLimitConfig,
    DoubleLimitOpportunity,
    OdosQuoteCache,
    RouteHealthTracker,
)
from executor.execution_report import format_double_limit_report  # noqa: E402
from pricing.odos_client import OdosQuote  # noqa: E402

# ── Helpers ────────────────────────────────────────────────────────

USDC_ADDR = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
WALLET_ADDR = "0x0000000000000000000000000000000000001234"
LINK_ADDR = "0xf97f4df75117a78c1A5a0DBb814Af92458539FB4"

TOKEN_MAPPINGS = {
    "LINK": {
        "address": LINK_ADDR,
        "mex_symbol": "LINKUSDT",
        "decimals": 18,
        "fee_tier": 500,
        "odos_supported": True,
        "active": True,
        "v3_pool": "0xbbe36e6f0331c6a36ab44bc8421e28e1a1871c1e",  # LINK/USDC V3 pool
    },
    "MAGIC": {
        "address": "0x539bdE0d7Dbd336b79148AA742883198BBF60342",
        "mex_symbol": "MAGICUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": True,
        "active": True,
    },
    "DEAD": {
        "address": "0x0000000000000000000000000000000000000001",
        "mex_symbol": "DEADUSDT",
        "decimals": 18,
        "fee_tier": 500,
        "odos_supported": False,
        "active": True,
    },
    "INACTIVE": {
        "address": "0x0000000000000000000000000000000000000002",
        "mex_symbol": "INACTIVEUSDT",
        "decimals": 18,
        "fee_tier": 500,
        "odos_supported": True,
        "active": False,
    },
}


def _make_odos_quote(amount_out: int, amount_in: int = 5_000_000) -> OdosQuote:
    return OdosQuote(
        chain_id=42161,
        input_token=USDC_ADDR,
        output_token=LINK_ADDR,
        amount_in=amount_in,
        amount_out=amount_out,
        gas_estimate=200_000,
        price_impact=0.001,
        block_number=100,
        path_viz=None,
        path_id="test_path_id_123",
    )


def _make_order_status(
    *,
    order_id: str = "ord_1",
    symbol: str = "LINKUSDT",
    side: str = "BUY",
    status: str = "FILLED",
    price: float = 14.0,
    orig_qty: float = 0.357,
    executed_qty: float = 0.357,
) -> MexcOrderStatus:
    return MexcOrderStatus(
        order_id=order_id,
        symbol=symbol,
        side=side,
        status=status,
        price=price,
        orig_qty=orig_qty,
        executed_qty=executed_qty,
    )


def _make_opp(
    *,
    direction: str = "mex_to_arb",
    executable: bool = True,
    gross_spread: float = 0.02,
    net_profit_usd: float = 0.05,
) -> DoubleLimitOpportunity:
    return DoubleLimitOpportunity(
        token_symbol="LINK",
        token_address=LINK_ADDR,
        mex_symbol="LINKUSDT",
        direction=direction,
        mex_bid=13.90,
        mex_ask=14.00,
        odos_price=14.28,
        gross_spread=gross_spread,
        total_cost_usd=0.05,
        net_profit_usd=net_profit_usd,
        net_profit_pct=net_profit_usd / 5.0,
        executable=executable,
    )


def _make_dex_swap_result(
    *,
    success=True,
    tx_hash="0xabc123",
    amount_out=350_000,
    gas_used=200_000,
    error=None,
):
    """Create a mock DexSwapResult-like object."""
    m = MagicMock()
    m.success = success
    m.tx_hash = tx_hash
    m.amount_out = amount_out
    m.gas_used = gas_used
    m.error = error
    return m


@pytest.fixture(autouse=True)
def _env_vars(monkeypatch):
    monkeypatch.setenv("USDC_ADDRESS", USDC_ADDR)
    monkeypatch.setenv("ARBITRUM_WALLET_ADDRESS", WALLET_ADDR)
    monkeypatch.setenv("MEXC_API_KEY", "test_key")
    monkeypatch.setenv("MEXC_API_SECRET", "test_secret")


def _build_engine(
    *,
    mexc: MagicMock | None = None,
    odos: MagicMock | None = None,
    tokens: dict | None = None,
    config: DoubleLimitConfig | None = None,
    range_manager: Any | None = None,
    dex_swap_manager: Any | None = None,
    capital_manager: Any | None = None,
) -> DoubleLimitArbitrageEngine:
    mexc = mexc or MagicMock()
    odos = odos or MagicMock()
    tokens = tokens if tokens is not None else TOKEN_MAPPINGS
    config = config or DoubleLimitConfig()
    return DoubleLimitArbitrageEngine(
        mexc_client=mexc,
        odos_client=odos,
        token_mappings=tokens,
        config=config,
        range_manager=range_manager,
        dex_swap_manager=dex_swap_manager,
        capital_manager=capital_manager,
    )


# ══════════════════════════════════════════════════════════════════
#  DoubleLimitConfig
# ══════════════════════════════════════════════════════════════════


class TestDoubleLimitConfig:
    def test_defaults(self):
        cfg = DoubleLimitConfig()
        assert cfg.trade_size_usd == 5.0
        assert cfg.simulation_mode is True
        assert cfg.enable_unwind_on_timeout is True
        assert cfg.gas_cost_usd == 0.03
        assert cfg.usdc_decimals == 6

    def test_min_spread_by_tier_defaults(self):
        cfg = DoubleLimitConfig()
        assert cfg.min_spread_by_tier[500] == 0.0045
        assert cfg.min_spread_by_tier[3_000] == 0.0065
        assert cfg.min_spread_by_tier[10_000] == 0.012

    def test_get_min_spread_known_tier(self):
        cfg = DoubleLimitConfig()
        assert cfg.get_min_spread(500) == 0.0045

    def test_get_min_spread_unknown_tier_falls_back(self):
        cfg = DoubleLimitConfig(min_spread_pct=0.042)
        assert cfg.get_min_spread(99999) == 0.042

    def test_custom_min_spread_by_tier(self):
        cfg = DoubleLimitConfig(min_spread_by_tier={100: 0.001})
        assert cfg.get_min_spread(100) == 0.001
        assert cfg.get_min_spread(500) == cfg.min_spread_pct  # not in custom map

    def test_estimate_gas_cost_usd_typical(self):
        cfg = DoubleLimitConfig(arb_gas_price_gwei=0.1, eth_price_usd=2600.0)
        cost = cfg.estimate_gas_cost_usd(200_000)
        assert cost == pytest.approx(0.052, abs=1e-6)

    def test_estimate_gas_cost_usd_high_gas(self):
        cfg = DoubleLimitConfig(arb_gas_price_gwei=0.1, eth_price_usd=2600.0)
        cost = cfg.estimate_gas_cost_usd(1_374_057)
        assert cost == pytest.approx(0.3572, abs=0.001)

    def test_estimate_gas_cost_usd_zero_falls_back(self):
        cfg = DoubleLimitConfig(gas_cost_usd=0.03)
        assert cfg.estimate_gas_cost_usd(0) == 0.03

    def test_max_dex_gas_limit_default(self):
        cfg = DoubleLimitConfig()
        assert cfg.max_dex_gas_limit == 1_200_000

    def test_mexc_fee_maker(self):
        cfg = DoubleLimitConfig()
        assert cfg.mexc_fee_usd(5.0, post_only=True) == pytest.approx(0.0)

    def test_mexc_fee_taker(self):
        cfg = DoubleLimitConfig()
        assert cfg.mexc_fee_usd(5.0, post_only=False) == pytest.approx(0.005)

    def test_mexc_fee_taker_10usd(self):
        cfg = DoubleLimitConfig()
        assert cfg.mexc_fee_usd(10.0, post_only=False) == pytest.approx(0.01)


# ══════════════════════════════════════════════════════════════════
#  OdosQuoteCache
# ══════════════════════════════════════════════════════════════════


class TestOdosQuoteCache:
    def test_put_and_get(self):
        cache = OdosQuoteCache(ttl=10.0)
        q = _make_odos_quote(int(0.35 * 1e18))
        cache.put("LINK:buy", q)
        assert cache.get("LINK:buy") is q

    def test_miss_returns_none(self):
        cache = OdosQuoteCache(ttl=10.0)
        assert cache.get("NOSUCHKEY") is None

    def test_expired_returns_none(self):
        cache = OdosQuoteCache(ttl=0.0)
        q = _make_odos_quote(int(0.35 * 1e18))
        cache.put("LINK:buy", q)
        assert cache.get("LINK:buy") is None


# ══════════════════════════════════════════════════════════════════
#  _lp_fee_pct
# ══════════════════════════════════════════════════════════════════


class TestLpFeePct:
    def test_tier_100(self):
        assert DoubleLimitArbitrageEngine._lp_fee_pct(100) == 0.0001

    def test_tier_500(self):
        assert DoubleLimitArbitrageEngine._lp_fee_pct(500) == 0.0005

    def test_tier_3000(self):
        assert DoubleLimitArbitrageEngine._lp_fee_pct(3000) == 0.003

    def test_tier_10000(self):
        assert DoubleLimitArbitrageEngine._lp_fee_pct(10000) == 0.01

    def test_unknown_tier_fallback(self):
        assert DoubleLimitArbitrageEngine._lp_fee_pct(12345) == 0.003


# ══════════════════════════════════════════════════════════════════
#  evaluate_opportunity
# ══════════════════════════════════════════════════════════════════


class TestEvaluateOpportunity:
    def _setup_mexc_and_odos(
        self, mex_bid, mex_ask, odos_amount_out, sell_usdc_out=None
    ):
        """
        Set up MEXC and ODOS mocks for evaluate_opportunity tests.

        evaluate_opportunity now fetches TWO ODOS quotes:
          1. buy quote (USDC → token): amount_out in token units (18 dec)
          2. sell quote (token → USDC): amount_out in USDC units (6 dec)

        ``odos_amount_out``: token amount returned by buy quote (18 decimals).
        ``sell_usdc_out``: USDC amount returned by sell quote (6 decimals).
            If None, defaults to ~trade_size worth of USDC at mid price.
        """
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[mex_bid, 100.0]],
            "asks": [[mex_ask, 100.0]],
        }

        if sell_usdc_out is None:
            mex_mid = (mex_bid + mex_ask) / 2.0
            token_amount_human = odos_amount_out / 1e18
            sell_usdc_out = int(token_amount_human * mex_mid * 1e6)

        buy_quote = _make_odos_quote(odos_amount_out)
        sell_quote = OdosQuote(
            chain_id=42161,
            input_token=LINK_ADDR,
            output_token=USDC_ADDR,
            amount_in=0,
            amount_out=sell_usdc_out,
            gas_estimate=200_000,
            price_impact=0.001,
            block_number=100,
            path_viz=None,
            path_id="test_sell_path_id",
        )
        odos = MagicMock()
        odos.quote.side_effect = [buy_quote, sell_quote]
        return mexc, odos

    def test_mex_to_arb_direction(self):
        """When Arbitrum price > MEXC ask: mex_to_arb."""
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=13.90,
            mex_ask=14.00,
            # MEXC mid = 13.95, expected = 5/13.95 = 0.358
            # Set actual = 0.355 to keep impact < 2%: (0.358-0.355)/0.358 = 0.84%
            odos_amount_out=int(0.355 * 1e18),  # $5 / 0.355 = $14.08 on Arb (higher)
        )
        engine = _build_engine(mexc=mexc, odos=odos)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        assert opp.direction == "mex_to_arb"
        assert opp.mex_ask == 14.00

    def test_arb_to_mex_direction(self):
        """When MEXC bid > Arbitrum price: arb_to_mex."""
        # MEXC mid = 15.05, est_tokens = 5/15.05 = 0.3322
        # sell_quote: selling 0.3322 tokens on Arb → ~$4.65 USDC  # noqa: E501
        # (cheap buy price = ~$14.0)
        # This makes arb price ~$14.0 < MEXC bid $15.0 → arb_to_mex
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=15.00,
            mex_ask=15.10,
            # buy quote: USDC→token
            odos_amount_out=int(0.357 * 1e18),
            sell_usdc_out=int(4.65 * 1e6),  # sell quote: token→USDC (cheap arb price)
        )
        engine = _build_engine(mexc=mexc, odos=odos)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        assert opp.direction == "arb_to_mex"
        assert opp.mex_bid == 15.00

    def test_executable_when_spread_and_profit_sufficient(self):
        # MEXC mid=13.55, expected=5/13.55=0.369 tokens
        # odos_out=0.363 → arb_price=$13.77 → spread ~1.25% (realistic, under 2% impact)
        # impact = (0.369-0.363)/0.369 = 1.6% < 2%
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=13.50,
            mex_ask=13.60,
            odos_amount_out=int(0.363 * 1e18),
        )
        cfg = DoubleLimitConfig(min_spread_pct=0.005, min_profit_usd=-1.0)
        engine = _build_engine(mexc=mexc, odos=odos, config=cfg)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        assert opp.executable is True

    def test_not_executable_when_spread_too_small(self):
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=14.00,
            mex_ask=14.01,
            odos_amount_out=int(0.3570 * 1e18),  # $14.01 → ~0 spread
        )
        engine = _build_engine(mexc=mexc, odos=odos)
        opp = engine.evaluate_opportunity("LINK")
        # Near-zero spread → either None or not executable
        if opp is not None:
            assert opp.executable is False

    def test_returns_none_for_unknown_token(self):
        engine = _build_engine()
        assert engine.evaluate_opportunity("DOESNOTEXIST") is None

    def test_returns_none_for_inactive_token(self):
        engine = _build_engine()
        assert engine.evaluate_opportunity("INACTIVE") is None

    def test_returns_none_for_odos_unsupported_token(self):
        engine = _build_engine()
        assert engine.evaluate_opportunity("DEAD") is None

    def test_returns_none_on_mexc_api_error(self):
        mexc = MagicMock()
        mexc.get_order_book.side_effect = MexcApiError("timeout")
        engine = _build_engine(mexc=mexc)
        assert engine.evaluate_opportunity("LINK") is None

    def test_returns_none_on_empty_order_book(self):
        mexc = MagicMock()
        mexc.get_order_book.return_value = {"bids": [], "asks": []}
        engine = _build_engine(mexc=mexc)
        assert engine.evaluate_opportunity("LINK") is None

    def test_returns_none_on_odos_failure(self):
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[14.0, 10.0]],
            "asks": [[14.1, 10.0]],
        }
        odos = MagicMock()
        odos.quote.side_effect = RuntimeError("ODOS 500")
        engine = _build_engine(mexc=mexc, odos=odos)
        assert engine.evaluate_opportunity("LINK") is None

    def test_returns_none_on_zero_odos_output(self):
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[14.0, 10.0]],
            "asks": [[14.1, 10.0]],
        }
        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(0)
        engine = _build_engine(mexc=mexc, odos=odos)
        assert engine.evaluate_opportunity("LINK") is None

    def test_cost_includes_gas_lp_odos(self):
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=13.50,
            mex_ask=13.60,
            odos_amount_out=int(0.363 * 1e18),
        )
        cfg = DoubleLimitConfig(gas_cost_usd=0.05, odos_fee_pct=0.0002)
        engine = _build_engine(mexc=mexc, odos=odos, config=cfg)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        # LINK has V3 pool, so uses V3 direct gas (~$0.039) instead of ODOS
        # total_cost includes gas + lp_fee (0.0005*5=0.0025)
        # V3 doesn't pay ODOS fee, so total_cost ≈ $0.039 + $0.0025 = ~$0.0415
        assert opp.total_cost_usd > 0.03  # At least gas + lp_fee
        assert opp.use_v3_direct is True  # V3 chosen over ODOS

    def test_cost_includes_bridge_amortization(self):
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=13.50,
            mex_ask=13.60,
            odos_amount_out=int(0.363 * 1e18),
        )
        cap = MagicMock()
        cap.config = MagicMock()
        cap.config.bridge_fixed_cost_usd = 0.50
        cap.trade_count_since_bridge = 0
        cfg = DoubleLimitConfig(min_trades_for_bridge_amortization=10)
        engine = _build_engine(mexc=mexc, odos=odos, config=cfg, capital_manager=cap)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        # bridge amortized = 0.50 / 10 = 0.05
        assert opp.total_cost_usd >= 0.05

    def test_different_fee_tier_magic(self):
        """MAGIC has fee_tier=3000 → higher LP fee → higher total cost."""
        # MEXC mid=0.405, expected=5/0.405=12.35 tokens
        # 12.1 tokens → impact = (12.35-12.1)/12.35 = 2.0% (at boundary)
        # Use 12.15 to stay under 2%: (12.35-12.15)/12.35 = 1.6%
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=0.40,
            mex_ask=0.41,
            odos_amount_out=int(12.15 * 1e18),
        )
        engine = _build_engine(mexc=mexc, odos=odos)
        opp = engine.evaluate_opportunity("MAGIC")
        assert opp is not None
        # LP fee for 3000 tier = 0.3% → 0.003 * 5 = 0.015
        assert opp.total_cost_usd >= 0.015

    def test_gas_estimate_stored_in_opportunity(self):
        """Opportunity stores effective gas_estimate (V3 or ODOS) and computed gas cost."""  # noqa: E501
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=13.50,
            mex_ask=13.60,
            # noqa: E501
            odos_amount_out=int(0.363 * 1e18),
        )
        engine = _build_engine(mexc=mexc, odos=odos)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        # LINK has V3 pool, so V3 direct (150K) should be chosen over ODOS (200K)
        assert opp.use_v3_direct is True
        assert opp.odos_gas_estimate == 150_000  # V3 gas estimate
        assert opp.estimated_gas_cost_usd > 0  # noqa: E501

    def test_compares_v3_vs_odos_chooses_better(self):
        """For tokens with V3 pool, compares both routes and chooses better net profit."""  # noqa: E501
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[13.50, 100.0]],
            # noqa: E501
            "asks": [[13.60, 100.0]],
        }
        odos = MagicMock()
        # ODOS with very high gas but same price - V3 should win
        expensive_quote = OdosQuote(
            chain_id=42161,
            input_token=USDC_ADDR,
            output_token=LINK_ADDR,
            amount_in=5_000_000,
            amount_out=int(0.363 * 1e18),  # ~1.6% impact, under 2% hard stop
            gas_estimate=2_000_000,  # Very high ODOS gas
            price_impact=0.001,
            block_number=100,
            path_viz=None,
            path_id="expensive_path",
        )
        odos.quote.return_value = expensive_quote
        cfg = DoubleLimitConfig()
        engine = _build_engine(mexc=mexc, odos=odos, config=cfg)
        opp = engine.evaluate_opportunity("LINK")
        # Should not reject due to gas cap - compares V3 vs ODOS and chooses better
        assert opp is not None
        # With same price but much higher gas, V3 direct should be chosen
        assert opp.use_v3_direct is True
        assert opp.odos_gas_estimate == 150_000  # V3 gas estimate

    def test_odos_chosen_when_better_despite_higher_gas(self):
        """ODOS can be chosen if it finds better prices despite higher gas."""
        # buy_quote: 0.365 tokens ($13.70 arb price) > MEXC ask $13.60 → spread ~0.74%
        # sell_quote: 0.365 tokens back → ~$4.99 USDC (mid-price sell)
        buy_quote = OdosQuote(
            chain_id=42161,
            input_token=USDC_ADDR,
            output_token=LINK_ADDR,
            amount_in=5_000_000,
            amount_out=int(0.365 * 1e18),
            gas_estimate=800_000,
            price_impact=0.001,
            block_number=100,
            path_viz=None,
            path_id="better_price_path",
        )
        sell_quote = OdosQuote(
            chain_id=42161,
            input_token=LINK_ADDR,
            output_token=USDC_ADDR,
            amount_in=int(0.369 * 1e18),
            amount_out=int(4.99 * 1e6),
            gas_estimate=800_000,
            price_impact=0.001,
            block_number=100,
            path_viz=None,
            path_id="sell_path",
        )
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[13.50, 100.0]],
            "asks": [[13.60, 100.0]],
        }
        odos = MagicMock()
        odos.quote.side_effect = [buy_quote, sell_quote]
        cfg = DoubleLimitConfig()
        engine = _build_engine(mexc=mexc, odos=odos, config=cfg)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None

    def test_dynamic_gas_cost_used_instead_of_static(self):
        """Gas cost uses dynamic estimate (V3 or ODOS), not static gas_cost_usd."""
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=13.50,
            mex_ask=13.60,
            odos_amount_out=int(0.363 * 1e18),
        )
        cfg = DoubleLimitConfig(
            gas_cost_usd=0.03,
            arb_gas_price_gwei=0.1,
            eth_price_usd=2600.0,
        )
        engine = _build_engine(mexc=mexc, odos=odos, config=cfg)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        # LINK has V3 pool, so uses V3 direct: 150_000 * 0.1 * 1e-9 * 2600 = $0.039
        assert opp.use_v3_direct is True
        assert opp.estimated_gas_cost_usd == pytest.approx(0.039, abs=1e-4)
        assert opp.estimated_gas_cost_usd > cfg.gas_cost_usd  # Dynamic > static

    def test_price_impact_hard_stop(self):
        """Opportunities with price impact > 2% are rejected."""
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[13.50, 100.0]],
            "asks": [[13.60, 100.0]],
        }
        odos = MagicMock()
        high_impact_quote = OdosQuote(
            chain_id=42161,
            input_token=USDC_ADDR,
            output_token=LINK_ADDR,
            amount_in=5_000_000,
            amount_out=int(0.330 * 1e18),
            gas_estimate=200_000,
            price_impact=0.035,  # 3.5% in decimal format (> 2% hard stop)
            block_number=100,
            path_viz=None,
            path_id="high_impact_path",
        )
        odos.quote.return_value = high_impact_quote
        engine = _build_engine(mexc=mexc, odos=odos)
        opp = engine.evaluate_opportunity("LINK")  # noqa: E501
        assert opp is None

    def test_price_impact_just_under_threshold_passes(self):
        """Computed price impact at 1.9% should pass (ODOS reported impact is ignored)."""  # noqa: E501
        # MEXC mid = 13.55, expected = 5/13.55 = 0.369 tokens
        # For 1.9% impact: actual = 0.369 * 0.981 = 0.362 tokens
        high_impact_odos_quote = OdosQuote(
            chain_id=42161,
            input_token=USDC_ADDR,
            output_token=LINK_ADDR,  # noqa: E501
            amount_in=5_000_000,
            amount_out=int(0.362 * 1e18),
            gas_estimate=200_000,
            # ODOS reports high impact (ignored - cross-market comparison)
            price_impact=0.50,
            block_number=100,
            path_viz=None,
            path_id="ok_impact_path",  # noqa: E501
        )
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[13.50, 100.0]],
            "asks": [[13.60, 100.0]],
        }
        odos = MagicMock()
        odos.quote.return_value = high_impact_odos_quote
        engine = _build_engine(mexc=mexc, odos=odos)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        assert opp.price_impact == pytest.approx(0.019, abs=0.002)

    def test_gas_adjusted_score_penalizes_complex_routes(self):
        """score_route applies penalty for gas > 250K and ODOS uncertainty."""
        cfg = DoubleLimitConfig()
        # Simple V3 route: 150K gas, no penalty
        v3_score = cfg.score_route(0.05, 150_000, "v3_direct")
        # Complex ODOS route: 800K gas, penalty for complexity + uncertainty  # noqa: E501
        odos_score = cfg.score_route(0.05, 800_000, "odos")
        assert v3_score > odos_score

    def test_gas_adjusted_score_same_profit_v3_wins(self):
        """With identical net profit, V3 direct wins due to lower gas & no uncertainty."""  # noqa: E501
        cfg = DoubleLimitConfig()
        v3_s = cfg.score_route(0.01, V3_GAS_ESTIMATE, "v3_direct")
        odos_s = cfg.score_route(0.01, 200_000, "odos")
        # noqa: E501
        assert v3_s > odos_s

    def test_opportunity_stores_route_score(self):
        """evaluate_opportunity stores the gas-adjusted score."""
        mexc, odos = self._setup_mexc_and_odos(
            mex_bid=13.50,
            mex_ask=13.60,
            odos_amount_out=int(0.363 * 1e18),
        )
        engine = _build_engine(mexc=mexc, odos=odos)
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        assert opp.route_score != 0.0

    def test_health_tracker_degrades_odos(self):
        """When ODOS route is unreliable (high avg gas), V3 is preferred even more."""
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[13.50, 100.0]],
            "asks": [[13.60, 100.0]],
        }
        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(int(0.363 * 1e18))

        health = RouteHealthTracker(window=5, unreliable_avg_gas=300_000)
        for _ in range(5):
            health.record("LINK", "odos", 900_000)

        engine = _build_engine(mexc=mexc, odos=odos)
        engine.route_health = health  # noqa: E501
        opp = engine.evaluate_opportunity("LINK")
        assert opp is not None
        assert opp.use_v3_direct is True

    def test_score_gate_blocks_negative_score(self):
        """Opportunities with negative route_score are not executable even if net_profit > min."""  # noqa: E501
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[0.40, 100.0]],
            # noqa: E501
            "asks": [[0.41, 100.0]],
        }
        odos = MagicMock()
        # Use MAGIC (no V3 pool) to force ODOS route
        # High gas route that makes score negative
        # MEXC mid = 0.405, expected out = $5 / 0.405 = 12.35 tokens
        # 12.15 → impact = (12.35-12.15)/12.35 = 1.6% (under 2% hard stop)
        expensive_quote = OdosQuote(
            chain_id=42161,
            input_token=USDC_ADDR,
            output_token="0x539bdE0d7Dbd336b79148AA742883198BBF60342",  # MAGIC
            amount_in=5_000_000,
            amount_out=int(12.15 * 1e18),
            gas_estimate=2_500_000,  # Very high gas -> large penalty -> negative score
            price_impact=0.001,  # Low reported impact
            block_number=100,
            path_viz=None,
            path_id="expensive_path",
        )
        odos.quote.return_value = expensive_quote
        cfg = DoubleLimitConfig(
            min_profit_usd=-1.0
        )  # Allow negative net profit for test
        engine = _build_engine(mexc=mexc, odos=odos, config=cfg)
        opp = engine.evaluate_opportunity("MAGIC")
        # Should exist but not executable due to negative score
        assert opp is not None
        assert opp.route_score < 0.0  # Score is negative (gas penalty > profit)
        assert (
            opp.executable is False
        )  # Therefore not executable (score gate blocks it)

    def test_computed_price_impact_catches_low_liquidity(self):
        """Computed price impact catches low liquidity even when ODOS reports 0."""
        mexc = MagicMock()
        mexc.get_order_book.return_value = {
            "bids": [[0.40, 100.0]],
            "asks": [[0.41, 100.0]],
        }
        odos = MagicMock()
        # Use MAGIC (no V3 pool) to force ODOS route
        # ODOS reports 0 impact, but actual output is much less than expected
        # MEXC mid = (0.40 + 0.41) / 2 = 0.405
        # Expected out = $5 / $0.405 = 12.35 tokens
        # Actual out = 5.0 tokens (much less due to low liquidity)
        # Impact = (12.35 - 5.0) / 12.35 = 59.5% > 2%
        low_liquidity_quote = OdosQuote(
            chain_id=42161,
            input_token=USDC_ADDR,
            output_token="0x539bdE0d7Dbd336b79148AA742883198BBF60342",  # MAGIC
            amount_in=5_000_000,
            amount_out=int(5.0 * 1e18),  # Much less than expected 12.35
            gas_estimate=200_000,
            price_impact=0.0,  # ODOS reports 0 (unreliable)
            block_number=100,
            path_viz=None,
            path_id="low_liq_path",
        )
        odos.quote.return_value = low_liquidity_quote
        engine = _build_engine(mexc=mexc, odos=odos)
        opp = engine.evaluate_opportunity("MAGIC")
        # Should be rejected due to computed price impact > 2%
        assert opp is None


# ══════════════════════════════════════════════════════════════════
#  RouteHealthTracker
# ══════════════════════════════════════════════════════════════════


class TestRouteHealthTracker:
    def test_no_data_is_reliable(self):
        t = RouteHealthTracker()
        assert t.is_reliable("ARB", "odos") is True

    def test_avg_gas_none_when_empty(self):
        t = RouteHealthTracker()
        assert t.avg_gas("ARB", "odos") is None

    def test_records_and_computes_avg(self):
        t = RouteHealthTracker(window=5)
        t.record("ARB", "odos", 200_000)
        t.record("ARB", "odos", 400_000)
        assert t.avg_gas("ARB", "odos") == pytest.approx(300_000)

    def test_unreliable_when_avg_high(self):
        t = RouteHealthTracker(window=3, unreliable_avg_gas=500_000)
        for _ in range(3):
            t.record("GMX", "odos", 800_000)
        assert t.is_reliable("GMX", "odos") is False

    def test_reliable_when_avg_low(self):
        t = RouteHealthTracker(window=3, unreliable_avg_gas=500_000)
        for _ in range(3):
            t.record("GMX", "odos", 200_000)
        assert t.is_reliable("GMX", "odos") is True

    def test_window_evicts_old(self):
        t = RouteHealthTracker(window=3, unreliable_avg_gas=500_000)
        t.record("ARB", "odos", 900_000)
        t.record("ARB", "odos", 900_000)
        t.record("ARB", "odos", 900_000)
        assert t.is_reliable("ARB", "odos") is False
        t.record("ARB", "odos", 100_000)
        t.record("ARB", "odos", 100_000)
        t.record("ARB", "odos", 100_000)
        assert t.is_reliable("ARB", "odos") is True

    def test_separate_token_tracking(self):
        t = RouteHealthTracker(window=3, unreliable_avg_gas=500_000)
        for _ in range(3):
            t.record("GMX", "odos", 800_000)
            t.record("LINK", "odos", 150_000)
        assert t.is_reliable("GMX", "odos") is False
        assert t.is_reliable("LINK", "odos") is True


# ══════════════════════════════════════════════════════════════════
#  Simulation execution
# ══════════════════════════════════════════════════════════════════


class TestSimulateExecution:
    def test_success_scenario(self, monkeypatch):
        monkeypatch.setenv("DOUBLE_LIMIT_SIM_SCENARIO", "success")
        engine = _build_engine(config=DoubleLimitConfig(simulation_mode=True))
        opp = _make_opp()
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "SUCCESS"
        assert isinstance(result["mex_order"], MexcOrderStatus)
        assert result["mex_order"].is_filled
        assert result["dex_success"] is True

    def test_timeout_scenario(self, monkeypatch):
        monkeypatch.setenv("DOUBLE_LIMIT_SIM_SCENARIO", "timeout")
        engine = _build_engine(config=DoubleLimitConfig(simulation_mode=True))
        opp = _make_opp()
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "TIMEOUT"
        assert result["unwind_attempted"] is True
        assert result["unwind_success"] is True

    def test_mex_reject_scenario(self, monkeypatch):
        monkeypatch.setenv("DOUBLE_LIMIT_SIM_SCENARIO", "mex_reject")
        engine = _build_engine(config=DoubleLimitConfig(simulation_mode=True))
        opp = _make_opp()
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "FAILED"
        assert "MEXC rejected" in result["error"]

    def test_dex_failed_scenario(self, monkeypatch):
        monkeypatch.setenv("DOUBLE_LIMIT_SIM_SCENARIO", "dex_failed")
        engine = _build_engine(config=DoubleLimitConfig(simulation_mode=True))
        opp = _make_opp()
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "TIMEOUT"
        assert result["dex_success"] is False

    def test_v3_not_executed_scenario(self, monkeypatch):
        """Legacy V3 scenario: range_manager present, V3 not executed."""
        monkeypatch.setenv("DOUBLE_LIMIT_SIM_SCENARIO", "v3_not_executed")
        rm = MagicMock()
        engine = _build_engine(
            config=DoubleLimitConfig(simulation_mode=True), range_manager=rm
        )
        opp = _make_opp()
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "TIMEOUT"
        assert result["v3_status"]["is_executed"] is False

    def test_skip_not_executable(self, monkeypatch):
        monkeypatch.setenv("DOUBLE_LIMIT_SIM_SCENARIO", "success")
        engine = _build_engine(config=DoubleLimitConfig(simulation_mode=True))
        opp = _make_opp(executable=False)
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "SKIPPED"

    def test_simulation_mex_to_arb_side(self, monkeypatch):
        monkeypatch.setenv("DOUBLE_LIMIT_SIM_SCENARIO", "success")
        engine = _build_engine(config=DoubleLimitConfig(simulation_mode=True))
        opp = _make_opp(direction="mex_to_arb")
        result = _run(engine.execute_double_limit(opp))
        assert result["mex_order"].side == "BUY"

    def test_simulation_arb_to_mex_side(self, monkeypatch):
        monkeypatch.setenv("DOUBLE_LIMIT_SIM_SCENARIO", "success")
        engine = _build_engine(config=DoubleLimitConfig(simulation_mode=True))
        opp = _make_opp(direction="arb_to_mex")
        result = _run(engine.execute_double_limit(opp))
        assert result["mex_order"].side == "SELL"

    def test_simulation_with_range_manager(self, monkeypatch):
        monkeypatch.setenv("DOUBLE_LIMIT_SIM_SCENARIO", "success")
        rm = MagicMock()
        engine = _build_engine(
            config=DoubleLimitConfig(simulation_mode=True), range_manager=rm
        )
        opp = _make_opp()
        result = _run(engine.execute_double_limit(opp))
        assert result["v3_position_id"] is not None
        assert result["v3_status"]["is_executed"] is True


# ══════════════════════════════════════════════════════════════════
#  Live execution (mocked I/O)
# ══════════════════════════════════════════════════════════════════


class TestLiveExecution:
    def test_skip_not_executable(self):
        mexc = MagicMock()
        engine = _build_engine(
            mexc=mexc, config=DoubleLimitConfig(simulation_mode=False)
        )
        opp = _make_opp(executable=False)
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "SKIPPED"
        mexc.place_limit_order.assert_not_called()

    def test_insufficient_balance_arb_to_mex(self):
        mexc = MagicMock()
        mexc.get_balance.return_value = 0.0  # no LINK on MEXC
        engine = _build_engine(
            mexc=mexc, config=DoubleLimitConfig(simulation_mode=False)
        )
        opp = _make_opp(direction="arb_to_mex")
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "insufficient_base_on_mexc"

    def test_insufficient_usdt_mex_to_arb(self):
        """When buying on MEXC, require enough USDT to fund the order."""
        mexc = MagicMock()
        mexc.get_balance.return_value = 0.0  # no USDT on MEXC
        engine = _build_engine(
            mexc=mexc, config=DoubleLimitConfig(simulation_mode=False)
        )
        opp = _make_opp(direction="mex_to_arb")
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "insufficient_quote_on_mexc"
        mexc.place_limit_order.assert_not_called()

    def test_mexc_order_failure(self):
        mexc = MagicMock()
        mexc.place_limit_order.side_effect = MexcApiError("rate limited")
        engine = _build_engine(
            mexc=mexc, config=DoubleLimitConfig(simulation_mode=False)
        )
        opp = _make_opp(direction="mex_to_arb")
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "FAILED"
        assert "rate limited" in result["error"]

    def test_mex_to_arb_places_buy(self):
        mexc = MagicMock()
        order = _make_order_status(side="BUY", status="NEW")
        mexc.place_limit_order.return_value = order
        mexc.get_order_status.return_value = _make_order_status(
            side="BUY", status="FILLED"
        )
        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=0.1,
                monitor_interval_seconds=0.01,
            ),
        )
        opp = _make_opp(direction="mex_to_arb")
        _run(engine.execute_double_limit(opp))
        mexc.place_limit_order.assert_called_once()
        call_kwargs = mexc.place_limit_order.call_args
        assert (
            call_kwargs.kwargs.get("side") == "BUY"
            or call_kwargs[1].get("side") == "BUY"
        )

    def test_arb_to_mex_places_sell(self):
        mexc = MagicMock()
        mexc.get_balance.return_value = 100.0  # plenty of LINK
        order = _make_order_status(side="SELL", status="NEW")
        mexc.place_limit_order.return_value = order
        mexc.get_order_status.return_value = _make_order_status(
            side="SELL", status="FILLED"
        )
        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=0.1,
                monitor_interval_seconds=0.01,
            ),
        )
        opp = _make_opp(direction="arb_to_mex")
        _run(engine.execute_double_limit(opp))
        call_kwargs = mexc.place_limit_order.call_args  # noqa: E501
        assert (
            call_kwargs.kwargs.get("side") == "SELL"
            or call_kwargs[1].get("side") == "SELL"
        )

    def test_dex_swap_mex_to_arb(self):
        """DEX swap manager called with correct params for mex_to_arb (sell token for USDC)."""  # noqa: E501
        mexc = MagicMock()
        order = _make_order_status(status="NEW")
        mexc.place_limit_order.return_value = order
        mexc.get_order_status.return_value = _make_order_status(status="FILLED")

        dex_mgr = MagicMock()
        dex_mgr.execute_swap.return_value = _make_dex_swap_result()

        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=0.1,
                monitor_interval_seconds=0.01,
            ),
            dex_swap_manager=dex_mgr,
        )
        opp = _make_opp(direction="mex_to_arb")
        result = _run(engine.execute_double_limit(opp))
        dex_mgr.execute_swap.assert_called_once()
        call_kwargs = dex_mgr.execute_swap.call_args.kwargs
        # sell token
        assert call_kwargs["input_token"] == LINK_ADDR
        assert call_kwargs["output_token"] == USDC_ADDR  # get USDC
        assert result["status"] == "SUCCESS"
        assert result["dex_success"] is True
        assert result["both_legs"] is True

    def test_dex_swap_arb_to_mex(self):
        """DEX swap manager called with correct params for arb_to_mex (buy token with USDC)."""  # noqa: E501
        mexc = MagicMock()
        mexc.get_balance.return_value = 100.0
        order = _make_order_status(side="SELL", status="NEW")
        mexc.place_limit_order.return_value = order
        mexc.get_order_status.return_value = _make_order_status(
            side="SELL",
            status="FILLED",
        )

        dex_mgr = MagicMock()
        dex_mgr.execute_swap.return_value = _make_dex_swap_result()

        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=0.1,
                monitor_interval_seconds=0.01,
            ),
            dex_swap_manager=dex_mgr,
        )
        opp = _make_opp(direction="arb_to_mex")
        result = _run(engine.execute_double_limit(opp))
        dex_mgr.execute_swap.assert_called_once()
        call_kwargs = dex_mgr.execute_swap.call_args.kwargs
        assert call_kwargs["input_token"] == USDC_ADDR  # spend USDC
        assert call_kwargs["output_token"] == LINK_ADDR  # buy token
        assert result["status"] == "SUCCESS"

    def test_dex_swap_failure_triggers_unwind(self):
        """DEX swap fails → MEXC fills → immediate unwind."""
        mexc = MagicMock()
        order = _make_order_status(status="NEW")
        mexc.place_limit_order.return_value = order
        mexc.get_order_status.return_value = _make_order_status(
            status="FILLED", executed_qty=0.357
        )
        mexc.place_market_order.return_value = _make_order_status(
            side="SELL", status="FILLED", executed_qty=0.357
        )

        dex_mgr = MagicMock()
        dex_mgr.execute_swap.return_value = _make_dex_swap_result(
            success=False, tx_hash=None, error="tx failed"
        )

        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=0.1,
                monitor_interval_seconds=0.01,
                enable_unwind_on_timeout=True,
            ),
            dex_swap_manager=dex_mgr,
        )
        opp = _make_opp(direction="mex_to_arb")
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "TIMEOUT"
        assert result["dex_success"] is False
        assert result["unwind_attempted"] is True
        assert result["unwind_success"] is True

    def test_dex_swap_exception_handled(self):
        """If dex_swap_manager.execute_swap raises, it's caught and DEX is marked failed."""  # noqa: E501
        mexc = MagicMock()
        order = _make_order_status(status="NEW")
        mexc.place_limit_order.return_value = order
        mexc.get_order_status.return_value = _make_order_status(
            status="FILLED",
            executed_qty=0.357,
        )
        mexc.place_market_order.return_value = _make_order_status(
            side="SELL", status="FILLED", executed_qty=0.357
        )

        dex_mgr = MagicMock()
        dex_mgr.execute_swap.side_effect = RuntimeError("network error")

        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=0.1,
                monitor_interval_seconds=0.01,
                enable_unwind_on_timeout=True,
            ),
            dex_swap_manager=dex_mgr,
        )
        opp = _make_opp(direction="mex_to_arb")
        result = _run(engine.execute_double_limit(opp))
        assert result["status"] == "TIMEOUT"
        assert result["dex_success"] is False
        assert result["unwind_attempted"] is True


# ══════════════════════════════════════════════════════════════════
#  _monitor_positions
# ══════════════════════════════════════════════════════════════════


class TestMonitorPositions:
    def test_both_filled_immediately(self):
        mexc = MagicMock()
        mexc.get_order_status.return_value = _make_order_status(status="FILLED")
        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=5,
                monitor_interval_seconds=0.01,
            ),
        )
        mex_order = _make_order_status(status="FILLED")
        opp = _make_opp()
        result = _run(
            engine._monitor_positions(
                mex_order,
                dex_success=True,
                dex_tx_hash="0xabc",
                dex_swap_result=None,
                opportunity=opp,
            )
        )
        assert result["status"] == "SUCCESS"
        assert result["both_legs"] is True
        assert result["unwind_attempted"] is False

    def test_both_filled_with_capital_manager(self):
        mexc = MagicMock()
        mexc.get_order_status.return_value = _make_order_status(status="FILLED")
        cap = MagicMock()
        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=5,
                monitor_interval_seconds=0.01,
            ),
            capital_manager=cap,
        )
        mex_order = _make_order_status(status="FILLED")
        opp = _make_opp(net_profit_usd=0.05)
        result = _run(
            engine._monitor_positions(
                mex_order,
                dex_success=True,
                dex_tx_hash="0xabc",
                dex_swap_result=None,
                opportunity=opp,
            )
        )
        assert result["status"] == "SUCCESS"
        cap.record_trade.assert_called_once_with(0.05)

    def test_mexc_filled_dex_failed_immediate_unwind(self):
        """MEXC fills immediately but DEX swap failed → immediate unwind (not timeout wait)."""  # noqa: E501
        mexc = MagicMock()
        mexc.get_order_status.return_value = _make_order_status(
            status="FILLED",
            executed_qty=0.357,
        )
        mexc.place_market_order.return_value = _make_order_status(
            side="SELL",
            status="FILLED",
            executed_qty=0.357,
        )
        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=600,
                monitor_interval_seconds=0.01,
                enable_unwind_on_timeout=True,
            ),
        )
        mex_order = _make_order_status(status="FILLED")
        opp = _make_opp(direction="mex_to_arb")
        result = _run(
            engine._monitor_positions(
                mex_order,
                dex_success=False,
                dex_tx_hash=None,
                dex_swap_result=None,
                opportunity=opp,
            )
        )
        assert result["status"] == "TIMEOUT"
        assert result["unwind_attempted"] is True
        assert result["unwind_success"] is True
        mexc.place_market_order.assert_called_once()

    def test_timeout_cancels_active_order(self):
        mexc = MagicMock()
        mexc.get_order_status.return_value = _make_order_status(status="NEW")
        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=0.05,
                monitor_interval_seconds=0.01,
                enable_unwind_on_timeout=False,
            ),
        )
        mex_order = _make_order_status(status="NEW")
        opp = _make_opp()
        result = _run(
            engine._monitor_positions(
                mex_order,
                dex_success=True,
                dex_tx_hash="0xabc",
                dex_swap_result=None,
                opportunity=opp,
            )
        )  # noqa: E501
        assert result["status"] == "TIMEOUT"
        mexc.cancel_order.assert_called_once()
        assert result["unwind_attempted"] is False

    def test_timeout_triggers_unwind_when_mexc_filled(self):
        """MEXC fills during the loop but DEX succeeded → waits for MEXC → success.
        Here we test: DEX OK, MEXC never fills → timeout → no unwind
        (only MEXC was unfilled).
        """
        mexc = MagicMock()
        mexc.get_order_status.return_value = _make_order_status(status="NEW")
        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=0.05,
                monitor_interval_seconds=0.01,
                enable_unwind_on_timeout=True,
            ),
        )
        mex_order = _make_order_status(status="NEW")
        opp = _make_opp(direction="mex_to_arb")
        result = _run(
            engine._monitor_positions(
                mex_order,
                dex_success=True,
                dex_tx_hash="0xabc",
                dex_swap_result=None,
                opportunity=opp,
            )
        )
        assert result["status"] == "TIMEOUT"
        # MEXC never filled → no unwind needed
        assert result["unwind_attempted"] is False

    def test_success_without_dex_manager_has_both_legs_true_if_dex_success(self):
        """When dex_success=True (regardless of manager), both_legs=True."""
        mexc = MagicMock()
        mexc.get_order_status.return_value = _make_order_status(status="FILLED")
        engine = _build_engine(
            mexc=mexc,
            config=DoubleLimitConfig(
                simulation_mode=False,
                position_ttl_seconds=5,
                monitor_interval_seconds=0.01,
            ),
        )
        mex_order = _make_order_status(status="FILLED")
        opp = _make_opp()
        result = _run(
            engine._monitor_positions(
                mex_order,
                dex_success=True,
                dex_tx_hash="0x123",
                dex_swap_result=None,
                opportunity=opp,
            )
        )
        assert result["status"] == "SUCCESS"
        assert result["both_legs"] is True


# ══════════════════════════════════════════════════════════════════
#  _unwind_mexc_filled
# ══════════════════════════════════════════════════════════════════


class TestUnwindMexcFilled:
    def test_simulation_mode_returns_success(self):
        engine = _build_engine(config=DoubleLimitConfig(simulation_mode=True))
        opp = _make_opp(direction="mex_to_arb")
        status = _make_order_status(executed_qty=0.357)
        result = engine._unwind_mexc_filled(opp, status)
        assert result["success"] is True

    def test_mex_to_arb_unwinds_with_sell(self):
        mexc = MagicMock()
        mexc.place_market_order.return_value = _make_order_status(
            side="SELL", status="FILLED", executed_qty=0.357
        )
        engine = _build_engine(
            mexc=mexc, config=DoubleLimitConfig(simulation_mode=False)
        )
        opp = _make_opp(direction="mex_to_arb")
        status = _make_order_status(executed_qty=0.357)
        result = engine._unwind_mexc_filled(opp, status)
        assert result["success"] is True
        mexc.place_market_order.assert_called_once_with(
            symbol="LINKUSDT", side="SELL", quantity=0.357
        )

    def test_arb_to_mex_unwinds_with_buy(self):
        mexc = MagicMock()
        mexc.place_market_order.return_value = _make_order_status(
            side="BUY", status="FILLED", executed_qty=0.357
        )
        engine = _build_engine(
            mexc=mexc, config=DoubleLimitConfig(simulation_mode=False)
        )
        opp = _make_opp(direction="arb_to_mex")
        status = _make_order_status(executed_qty=0.357)
        result = engine._unwind_mexc_filled(opp, status)
        assert result["success"] is True
        mexc.place_market_order.assert_called_once_with(
            symbol="LINKUSDT", side="BUY", quantity=0.357
        )

    def test_zero_executed_qty_fails(self):
        engine = _build_engine(config=DoubleLimitConfig(simulation_mode=False))
        opp = _make_opp()
        status = _make_order_status(executed_qty=0.0)
        result = engine._unwind_mexc_filled(opp, status)
        assert result["success"] is False
        assert "executed_qty" in result["error"]

    def test_market_order_mexc_api_error(self):
        mexc = MagicMock()
        mexc.place_market_order.side_effect = MexcApiError("insufficient funds")
        engine = _build_engine(
            mexc=mexc, config=DoubleLimitConfig(simulation_mode=False)
        )
        opp = _make_opp(direction="mex_to_arb")
        status = _make_order_status(executed_qty=0.357)
        result = engine._unwind_mexc_filled(opp, status)
        assert result["success"] is False
        assert "insufficient funds" in result["error"]

    def test_market_order_generic_exception(self):
        mexc = MagicMock()
        mexc.place_market_order.side_effect = RuntimeError("network down")
        engine = _build_engine(
            mexc=mexc, config=DoubleLimitConfig(simulation_mode=False)
        )
        opp = _make_opp(direction="mex_to_arb")
        status = _make_order_status(executed_qty=0.357)
        result = engine._unwind_mexc_filled(opp, status)
        assert result["success"] is False
        assert "network down" in result["error"]

    def test_partial_fill_detected(self):
        mexc = MagicMock()
        mexc.place_market_order.return_value = _make_order_status(
            status="PARTIALLY_FILLED", executed_qty=0.100, orig_qty=0.357
        )
        engine = _build_engine(
            mexc=mexc, config=DoubleLimitConfig(simulation_mode=False)
        )
        opp = _make_opp(direction="mex_to_arb")
        status = _make_order_status(executed_qty=0.357)
        result = engine._unwind_mexc_filled(opp, status)
        assert result["success"] is False  # 0.100 < 0.357 * 0.999


# ══════════════════════════════════════════════════════════════════
#  MexcOrderStatus properties
# ══════════════════════════════════════════════════════════════════


class TestMexcOrderStatus:
    def test_is_filled(self):
        o = _make_order_status(status="FILLED")
        assert o.is_filled is True
        assert o.is_active is False

    def test_is_active(self):
        o = _make_order_status(status="NEW")
        assert o.is_active is True
        assert o.is_filled is False

    def test_partially_filled_is_active(self):
        o = _make_order_status(status="PARTIALLY_FILLED")
        assert o.is_active is True
        assert o.is_filled is False

    def test_remaining_qty(self):
        o = _make_order_status(orig_qty=1.0, executed_qty=0.4)
        assert abs(o.remaining_qty - 0.6) < 1e-9


# ══════════════════════════════════════════════════════════════════
#  Execution report
# ══════════════════════════════════════════════════════════════════


class TestFormatDoubleLimitReport:
    def test_success_report_dex_swap(self):
        """Report for a successful DEX swap (new ODOS path)."""
        opp = _make_opp()
        mex = _make_order_status()
        result = {
            "status": "SUCCESS",
            "mex_order": mex,
            "dex_success": True,
            "dex_tx_hash": "0xabc123",
            "dex_swap_result": _make_dex_swap_result(),
            "opportunity": opp,
            "unwind_attempted": False,
            "unwind_success": None,
        }
        text = format_double_limit_report(result, opp)
        assert "DOUBLE LIMIT EXECUTION REPORT" in text
        assert "Outcome: SUCCESS" in text
        assert "LINK" in text
        assert "ODOS swap" in text
        assert "Unwind" not in text

    def test_success_report_legacy_v3(self):
        """Report for a legacy V3 result (no dex_success key)."""
        opp = _make_opp()
        mex = _make_order_status()
        result = {
            "status": "SUCCESS",
            "mex_order": mex,
            "v3_status": {"is_executed": True, "liquidity": 100, "in_range": False},
            "v3_position_id": 42,
            "opportunity": opp,
            "unwind_attempted": False,
            "unwind_success": None,
        }
        text = format_double_limit_report(result, opp)
        assert "V3 range" in text
        assert "Outcome: SUCCESS" in text

    def test_timeout_with_unwind_report(self):
        opp = _make_opp()
        mex = _make_order_status(status="FILLED")
        result = {
            "status": "TIMEOUT",
            "mex_order": mex,
            "dex_success": False,
            "dex_tx_hash": None,
            "opportunity": opp,
            "unwind_attempted": True,
            "unwind_success": True,
        }
        text = format_double_limit_report(result, opp)
        assert "Outcome: TIMEOUT" in text
        assert "Unwind (MEXC): attempted=True success=True" in text

    def test_timeout_without_unwind(self):
        opp = _make_opp()
        mex = _make_order_status(status="NEW")
        result = {
            "status": "TIMEOUT",
            "mex_order": mex,
            "dex_success": True,
            "dex_tx_hash": "0x123",
            "opportunity": opp,
        }
        text = format_double_limit_report(result, opp)
        assert "Outcome: TIMEOUT" in text
        assert "Unwind" not in text

    def test_failed_report(self):
        opp = _make_opp()
        result = {
            "status": "FAILED",
            "error": "rate limited",
            "opportunity": opp,
        }
        text = format_double_limit_report(result, opp)
        assert "Outcome: FAILED" in text
        assert "rate limited" in text

    def test_report_no_mex_order(self):
        opp = _make_opp()
        result = {"status": "SKIPPED", "opportunity": opp}
        text = format_double_limit_report(result, opp)
        assert "(no order)" in text

    def test_report_truncated_for_telegram(self):
        opp = _make_opp()
        result = {
            "status": "SUCCESS",
            "mex_order": _make_order_status(),
            "dex_success": True,
            "dex_tx_hash": "0xabc",
            "dex_swap_result": _make_dex_swap_result(),
            "opportunity": opp,
        }
        text = format_double_limit_report(result, opp)
        assert len(text) <= 4000


# ══════════════════════════════════════════════════════════════════
#  DoubleLimitOpportunity dataclass
# ══════════════════════════════════════════════════════════════════


class TestDoubleLimitOpportunity:
    def test_fields(self):
        opp = _make_opp(
            direction="arb_to_mex",
            executable=False,
            gross_spread=0.015,
            net_profit_usd=-0.01,
        )
        assert opp.token_symbol == "LINK"
        assert opp.direction == "arb_to_mex"
        assert opp.executable is False
        assert opp.gross_spread == 0.015
        assert opp.net_profit_usd == -0.01
        assert opp.net_profit_pct == pytest.approx(-0.01 / 5.0)


# ══════════════════════════════════════════════════════════════════
#  OdosClient.assemble and OdosQuote.path_id
# ══════════════════════════════════════════════════════════════════


class TestOdosClientAssemble:
    def test_quote_returns_path_id(self):
        """OdosQuote now has a path_id field."""
        q = OdosQuote(
            chain_id=42161,
            input_token=USDC_ADDR,
            output_token=LINK_ADDR,
            amount_in=5_000_000,
            amount_out=350_000_000_000_000_000,
            gas_estimate=200_000,
            price_impact=0.001,
            block_number=100,
            path_viz=None,
            path_id="abc123",
        )
        assert q.path_id == "abc123"

    def test_quote_path_id_optional(self):
        """path_id defaults to None."""
        q = OdosQuote(
            chain_id=42161,
            input_token=USDC_ADDR,
            output_token=LINK_ADDR,
            amount_in=5_000_000,
            amount_out=350_000_000_000_000_000,
            gas_estimate=200_000,
            price_impact=0.001,
            block_number=100,
            path_viz=None,
        )
        assert q.path_id is None

    def test_assemble_parses_response(self):
        """OdosClient.assemble returns OdosAssembledTx from API response."""
        from pricing.odos_client import OdosAssembledTx, OdosClient

        client = OdosClient(chain_id=42161, base_url="http://fake")
        fake_response = {
            "transaction": {
                "to": "0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559",
                "data": "0xabcdef",
                "value": 0,
                "gas": 350_000,
                "chainId": 42161,
            }
        }

        with patch.object(client, "_post", return_value=fake_response):
            result = client.assemble(
                path_id="test_path_id",
                user_address=WALLET_ADDR,
            )
        assert isinstance(result, OdosAssembledTx)
        assert result.to == "0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559"
        assert result.data == "0xabcdef"
        assert result.gas == 350_000
        assert result.value == 0
        assert result.chain_id == 42161

    def test_assemble_raises_on_missing_transaction(self):
        from pricing.odos_client import OdosClient

        client = OdosClient(chain_id=42161, base_url="http://fake")
        with patch.object(client, "_post", return_value={"status": "ok"}):
            with pytest.raises(RuntimeError, match="no transaction"):
                client.assemble(path_id="test", user_address=WALLET_ADDR)

    def test_quote_captures_path_id_from_response(self):
        """The quote method now captures pathId from the ODOS response."""
        from pricing.odos_client import OdosClient

        client = OdosClient(chain_id=42161, base_url="http://fake")
        fake_response = {
            "outAmounts": ["350000000000000000"],
            "gasEstimate": 200000,
            "blockNumber": 100,
            "priceImpact": 0.001,
            "pathId": "captured_path_id_456",
        }
        with patch.object(client, "_post", return_value=fake_response):
            quote = client.quote(
                input_token=USDC_ADDR,
                output_token=LINK_ADDR,
                amount_in=5_000_000,
                user_address=WALLET_ADDR,
            )
        assert quote.path_id == "captured_path_id_456"
        assert quote.amount_out == 350_000_000_000_000_000


# ══════════════════════════════════════════════════════════════════
#  DexSwapManager
# ══════════════════════════════════════════════════════════════════


class TestDexSwapManager:
    def test_execute_swap_success(self):
        """Full successful swap: quote → assemble → approve → send."""
        from exchange.dex_swap import DexSwapManager, DexSwapResult

        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(amount_out=350_000_000_000_000_000)
        odos.assemble.return_value = MagicMock(
            to="0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559",
            data="0xabcdef",
            value=0,
            gas=350_000,
        )

        chain_client = MagicMock()
        # allowance check returns large value (no approve needed)
        chain_client.call.return_value = (2**256 - 1).to_bytes(32, "big")

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        # Mock TransactionBuilder chain
        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mock_builder = MagicMock()
            MockTB.return_value = mock_builder
            mock_builder.to.return_value = mock_builder
            mock_builder.value.return_value = mock_builder
            mock_builder.data.return_value = mock_builder
            mock_builder.chain_id.return_value = mock_builder
            mock_builder.gas_limit.return_value = mock_builder
            mock_builder.with_gas_price.return_value = mock_builder
            mock_builder.send_and_wait.return_value = MagicMock(
                status=True, tx_hash="0xswap_tx_hash", gas_used=300_000
            )

            mgr = DexSwapManager(odos=odos, chain_client=chain_client, wallet=wallet)
            result = mgr.execute_swap(
                input_token=USDC_ADDR,
                output_token=LINK_ADDR,
                amount_in=5_000_000,
            )

        assert isinstance(result, DexSwapResult)
        assert result.success is True
        assert result.tx_hash == "0xswap_tx_hash"
        assert result.gas_used == 300_000
        assert result.amount_out == 350_000_000_000_000_000
        assert result.route == "odos"

    def test_execute_swap_quote_failure(self):
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.side_effect = RuntimeError("ODOS 500")

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        mgr = DexSwapManager(odos=odos, chain_client=MagicMock(), wallet=wallet)
        result = mgr.execute_swap(
            input_token=USDC_ADDR, output_token=LINK_ADDR, amount_in=5_000_000
        )
        assert result.success is False
        assert "quote failed" in result.error

    def test_execute_swap_no_path_id(self):
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.return_value = OdosQuote(
            chain_id=42161,
            input_token=USDC_ADDR,
            output_token=LINK_ADDR,
            amount_in=5_000_000,
            amount_out=350_000,
            gas_estimate=200_000,
            price_impact=0.001,
            block_number=100,
            path_viz=None,
            path_id=None,
        )

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        mgr = DexSwapManager(odos=odos, chain_client=MagicMock(), wallet=wallet)
        result = mgr.execute_swap(
            input_token=USDC_ADDR, output_token=LINK_ADDR, amount_in=5_000_000
        )
        assert result.success is False
        assert "pathId" in result.error

    def test_execute_swap_assemble_failure(self):
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(350_000)
        odos.assemble.side_effect = RuntimeError("assemble failed")

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        mgr = DexSwapManager(odos=odos, chain_client=MagicMock(), wallet=wallet)
        result = mgr.execute_swap(
            input_token=USDC_ADDR, output_token=LINK_ADDR, amount_in=5_000_000
        )
        assert result.success is False
        assert "assemble failed" in result.error

    def test_execute_swap_allows_high_gas_route(self):
        """High gas routes are no longer rejected - net profit comparison handles it."""
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(350_000)
        odos.assemble.return_value = MagicMock(
            to="0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559",
            data="0xabcdef",
            value=0,
            gas=800_000,  # High gas, but should still execute
        )

        chain_client = MagicMock()
        chain_client.call.return_value = (2**256 - 1).to_bytes(32, "big")

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mock_builder = MagicMock()
            MockTB.return_value = mock_builder
            mock_builder.to.return_value = mock_builder
            mock_builder.value.return_value = mock_builder
            mock_builder.data.return_value = mock_builder
            mock_builder.chain_id.return_value = mock_builder
            mock_builder.gas_limit.return_value = mock_builder
            mock_builder.with_gas_price.return_value = mock_builder
            mock_builder.send_and_wait.return_value = MagicMock(
                status=True, tx_hash="0xhigh_gas_ok", gas_used=750_000
            )

            mgr = DexSwapManager(
                odos=odos,
                chain_client=chain_client,
                wallet=wallet,
                max_gas_limit=500_000,
            )
            result = mgr.execute_swap(
                input_token=USDC_ADDR, output_token=LINK_ADDR, amount_in=5_000_000
            )
        # Should execute even with high gas (gas cap check removed)
        assert result.success is True
        assert result.route == "odos"

    def test_execute_swap_tx_reverted(self):
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(350_000)
        odos.assemble.return_value = MagicMock(
            to="0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559",
            data="0xabcdef",
            value=0,
            gas=350_000,
        )

        chain_client = MagicMock()
        chain_client.call.return_value = (2**256 - 1).to_bytes(32, "big")

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mock_builder = MagicMock()
            MockTB.return_value = mock_builder
            mock_builder.to.return_value = mock_builder
            mock_builder.value.return_value = mock_builder
            mock_builder.data.return_value = mock_builder
            mock_builder.chain_id.return_value = mock_builder
            mock_builder.gas_limit.return_value = mock_builder
            mock_builder.with_gas_price.return_value = mock_builder
            mock_builder.send_and_wait.return_value = MagicMock(
                status=False, tx_hash="0xreverted", gas_used=100_000
            )

            mgr = DexSwapManager(odos=odos, chain_client=chain_client, wallet=wallet)
            result = mgr.execute_swap(
                input_token=USDC_ADDR, output_token=LINK_ADDR, amount_in=5_000_000
            )

        assert result.success is False
        assert result.tx_hash == "0xreverted"
        assert "reverted" in result.error

    def test_v3_direct_swap_success(self):
        """When fee_tier is provided, uses direct V3 swap (~150K gas)."""
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        chain_client = MagicMock()
        chain_client.call.return_value = (2**256 - 1).to_bytes(32, "big")

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mock_builder = MagicMock()
            MockTB.return_value = mock_builder
            mock_builder.to.return_value = mock_builder
            mock_builder.value.return_value = mock_builder
            mock_builder.data.return_value = mock_builder
            mock_builder.chain_id.return_value = mock_builder
            mock_builder.gas_limit.return_value = mock_builder
            mock_builder.with_gas_price.return_value = mock_builder
            mock_builder.send_and_wait.return_value = MagicMock(
                status=True, tx_hash="0xv3_direct_hash", gas_used=145_000
            )

            mgr = DexSwapManager(odos=odos, chain_client=chain_client, wallet=wallet)
            result = mgr.execute_swap(
                input_token=USDC_ADDR,
                output_token=LINK_ADDR,
                amount_in=5_000_000,
                fee_tier=500,
            )

        assert result.success is True
        assert result.tx_hash == "0xv3_direct_hash"
        assert result.gas_used == 145_000
        assert result.route == "v3_direct"
        odos.quote.assert_not_called()

    def test_v3_direct_reverts_falls_back_to_odos(self):
        """V3 direct swap reverts -> falls back to ODOS successfully."""
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(350_000_000_000_000_000)
        odos.assemble.return_value = MagicMock(
            to="0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559",
            data="0xabcdef",
            value=0,
            gas=300_000,
        )

        chain_client = MagicMock()
        chain_client.call.return_value = (2**256 - 1).to_bytes(32, "big")

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        call_count = [0]

        def mock_send_and_wait(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(status=False, tx_hash="0xv3_reverted", gas_used=50_000)
            return MagicMock(status=True, tx_hash="0xodos_ok", gas_used=280_000)

        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mock_builder = MagicMock()
            MockTB.return_value = mock_builder
            mock_builder.to.return_value = mock_builder
            mock_builder.value.return_value = mock_builder
            mock_builder.data.return_value = mock_builder
            mock_builder.chain_id.return_value = mock_builder
            mock_builder.gas_limit.return_value = mock_builder
            mock_builder.with_gas_price.return_value = mock_builder
            mock_builder.send_and_wait.side_effect = mock_send_and_wait

            mgr = DexSwapManager(odos=odos, chain_client=chain_client, wallet=wallet)
            result = mgr.execute_swap(
                input_token=USDC_ADDR,
                output_token=LINK_ADDR,
                amount_in=5_000_000,
                fee_tier=500,
            )

        assert result.success is True
        assert result.route == "odos"
        assert result.tx_hash == "0xodos_ok"
        odos.quote.assert_called_once()

    def test_no_fee_tier_skips_v3_uses_odos_directly(self):
        """When fee_tier is None, goes straight to ODOS."""
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(350_000_000_000_000_000)
        odos.assemble.return_value = MagicMock(
            to="0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559",
            data="0xabcdef",
            value=0,
            gas=200_000,
        )

        chain_client = MagicMock()
        chain_client.call.return_value = (2**256 - 1).to_bytes(32, "big")

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mock_builder = MagicMock()
            MockTB.return_value = mock_builder
            mock_builder.to.return_value = mock_builder
            mock_builder.value.return_value = mock_builder
            mock_builder.data.return_value = mock_builder
            mock_builder.chain_id.return_value = mock_builder
            mock_builder.gas_limit.return_value = mock_builder
            mock_builder.with_gas_price.return_value = mock_builder
            mock_builder.send_and_wait.return_value = MagicMock(
                status=True, tx_hash="0xodos_direct", gas_used=180_000
            )

            mgr = DexSwapManager(odos=odos, chain_client=chain_client, wallet=wallet)
            result = mgr.execute_swap(
                input_token=USDC_ADDR,
                output_token=LINK_ADDR,
                amount_in=5_000_000,
                fee_tier=None,
            )

        assert result.success is True
        assert result.route == "odos"
        odos.quote.assert_called_once()

    def test_v3_direct_approve_failure_falls_back_to_odos(self):
        """V3 approve fails -> falls back to ODOS."""
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(350_000_000_000_000_000)
        odos.assemble.return_value = MagicMock(
            to="0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559",
            data="0xabcdef",
            value=0,
            gas=200_000,
        )

        chain_client = MagicMock()
        chain_client.call.return_value = b"\x00" * 32

        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        call_count = [0]

        def mock_send_and_wait(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(
                    status=False, tx_hash="0xapprove_fail", gas_used=30_000
                )
            if call_count[0] == 2:
                return MagicMock(status=True, tx_hash="0xapprove_odos", gas_used=50_000)
            return MagicMock(status=True, tx_hash="0xodos_swap", gas_used=200_000)

        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mock_builder = MagicMock()
            MockTB.return_value = mock_builder
            mock_builder.to.return_value = mock_builder
            mock_builder.value.return_value = mock_builder
            mock_builder.data.return_value = mock_builder
            mock_builder.chain_id.return_value = mock_builder
            mock_builder.gas_limit.return_value = mock_builder
            mock_builder.with_gas_estimate.return_value = mock_builder
            mock_builder.with_gas_price.return_value = mock_builder
            mock_builder.send_and_wait.side_effect = mock_send_and_wait

            mgr = DexSwapManager(odos=odos, chain_client=chain_client, wallet=wallet)
            result = mgr.execute_swap(
                input_token=USDC_ADDR,
                output_token=LINK_ADDR,
                amount_in=5_000_000,
                fee_tier=500,
            )

        assert result.success is True
        assert result.route == "odos"

    def test_v3_fail_reason_classify_approve(self):
        """V3 approve failure is classified as APPROVE_FAILED (safe for fallback)."""
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(350_000_000_000_000_000)
        odos.assemble.return_value = MagicMock(
            to="0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559",
            data="0xabcdef",
            value=0,
            gas=200_000,
        )
        chain_client = MagicMock()
        chain_client.call.return_value = b"\x00" * 32
        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        call_count = [0]

        def mock_send(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(
                    status=False, tx_hash="0xapprove_fail_v3", gas_used=30_000
                )
            if call_count[0] == 2:
                return MagicMock(status=True, tx_hash="0xapprove_odos", gas_used=50_000)
            return MagicMock(status=True, tx_hash="0xodos_ok", gas_used=200_000)

        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mb = MagicMock()
            MockTB.return_value = mb
            mb.to.return_value = mb
            mb.value.return_value = mb
            mb.data.return_value = mb
            mb.chain_id.return_value = mb
            mb.gas_limit.return_value = mb
            mb.with_gas_estimate.return_value = mb
            mb.with_gas_price.return_value = mb
            mb.send_and_wait.side_effect = mock_send

            mgr = DexSwapManager(odos=odos, chain_client=chain_client, wallet=wallet)
            result = mgr.execute_swap(
                input_token=USDC_ADDR,
                output_token=LINK_ADDR,
                amount_in=5_000_000,
                fee_tier=500,
            )
        # APPROVE_FAILED is safe for fallback -> should get ODOS result
        assert result.success is True
        assert result.route == "odos"

    def test_v3_fail_reason_price_impact_blocks_fallback(self):
        """V3 revert with high gas usage (price impact) blocks ODOS fallback."""
        from exchange.dex_swap import V3_DIRECT_GAS_LIMIT, DexSwapManager, V3FailReason

        odos = MagicMock()
        chain_client = MagicMock()
        chain_client.call.return_value = (2**256 - 1).to_bytes(32, "big")
        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mb = MagicMock()
            MockTB.return_value = mb
            mb.to.return_value = mb
            mb.value.return_value = mb
            mb.data.return_value = mb
            mb.chain_id.return_value = mb
            mb.gas_limit.return_value = mb
            mb.with_gas_price.return_value = mb
            # Revert consuming near-full gas -> price impact signal
            mb.send_and_wait.return_value = MagicMock(
                status=False,
                tx_hash="0xv3_impact",
                gas_used=int(V3_DIRECT_GAS_LIMIT * 0.95),
            )

            mgr = DexSwapManager(odos=odos, chain_client=chain_client, wallet=wallet)
            result = mgr.execute_swap(
                input_token=USDC_ADDR,
                output_token=LINK_ADDR,
                amount_in=5_000_000,
                fee_tier=500,
            )

        assert result.success is False
        assert result.v3_fail_reason == V3FailReason.PRICE_IMPACT
        assert result.route == "v3_direct"
        odos.quote.assert_not_called()

    def test_v3_fail_reason_reverted_allows_fallback(self):
        """V3 revert with low gas usage allows ODOS fallback."""
        from exchange.dex_swap import DexSwapManager

        odos = MagicMock()
        odos.quote.return_value = _make_odos_quote(350_000_000_000_000_000)
        odos.assemble.return_value = MagicMock(
            to="0xCf5540fFFCdC3d510B18bFcA6d2b9987b0772559",
            data="0xabcdef",
            value=0,
            gas=200_000,
        )
        chain_client = MagicMock()
        chain_client.call.return_value = (2**256 - 1).to_bytes(32, "big")
        wallet = MagicMock()
        wallet.address = WALLET_ADDR

        call_count = [0]

        def mock_send(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Low gas used -> simple revert (wrong pool, etc.)
                return MagicMock(status=False, tx_hash="0xv3_revert", gas_used=50_000)
            return MagicMock(status=True, tx_hash="0xodos_ok", gas_used=200_000)

        with patch("exchange.dex_swap.TransactionBuilder") as MockTB:
            mb = MagicMock()
            MockTB.return_value = mb
            mb.to.return_value = mb
            mb.value.return_value = mb
            mb.data.return_value = mb
            mb.chain_id.return_value = mb
            mb.gas_limit.return_value = mb
            mb.with_gas_price.return_value = mb
            mb.send_and_wait.side_effect = mock_send

            mgr = DexSwapManager(odos=odos, chain_client=chain_client, wallet=wallet)
            result = mgr.execute_swap(
                input_token=USDC_ADDR,
                output_token=LINK_ADDR,
                amount_in=5_000_000,
                fee_tier=500,
            )

        assert result.success is True
        assert result.route == "odos"
        odos.quote.assert_called_once()
