from decimal import Decimal

from core.base_types import Address
from pricing.price_impact_analyzer import PriceImpactAnalyzer
from pricing.uniswap_v2_pair import Token, UniswapV2Pair

ETH = Token(Address("0x0000000000000000000000000000000000000100"), "ETH", 18)
USDC = Token(Address("0x0000000000000000000000000000000000000200"), "USDC", 6)


def _pair(reserve0: int, reserve1: int) -> UniswapV2Pair:
    return UniswapV2Pair(
        address=Address("0x0000000000000000000000000000000000000300"),
        token0=ETH,
        token1=USDC,
        reserve0=reserve0,
        reserve1=reserve1,
        fee_bps=30,
    )


def test_price_impact_increases_with_size():
    pair = _pair(1000 * 10**18, 2_000_000 * 10**6)
    analyzer = PriceImpactAnalyzer(pair)

    small = analyzer.generate_impact_table(USDC, [1_000 * 10**6])[0]
    large = analyzer.generate_impact_table(USDC, [100_000 * 10**6])[0]

    assert small["price_impact_pct"] < large["price_impact_pct"]


def test_find_max_size_for_impact():
    pair = _pair(1000 * 10**18, 2_000_000 * 10**6)
    analyzer = PriceImpactAnalyzer(pair)

    max_size = analyzer.find_max_size_for_impact(USDC, Decimal("1.0"))
    assert max_size > 0


def test_execution_price_above_spot_for_buy():
    pair = _pair(1000 * 10**18, 2_000_000 * 10**6)
    analyzer = PriceImpactAnalyzer(pair)
    row = analyzer.generate_impact_table(USDC, [10_000 * 10**6])[0]

    assert row["execution_price"] > row["spot_price"]
