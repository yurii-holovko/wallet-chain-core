import pytest

from core.base_types import Address
from pricing.route import Route, RouteFinder
from pricing.uniswap_v2_pair import Token, UniswapV2Pair

SHIB = Token(Address("0x00000000000000000000000000000000000000a1"), "SHIB", 18)
ETH = Token(Address("0x00000000000000000000000000000000000000b2"), "ETH", 18)
USDC = Token(Address("0x00000000000000000000000000000000000000c3"), "USDC", 6)


def _pair(token0: Token, token1: Token, reserve0: int, reserve1: int) -> UniswapV2Pair:
    return UniswapV2Pair(
        address=Address("0x00000000000000000000000000000000000000f0"),
        token0=token0,
        token1=token1,
        reserve0=reserve0,
        reserve1=reserve1,
        fee_bps=30,
    )


def _setup_pools_for_routing() -> tuple[UniswapV2Pair, UniswapV2Pair, UniswapV2Pair]:
    # SHIB -> ETH (direct) pool (worse price)
    shib_eth = _pair(
        SHIB,
        ETH,
        reserve0=1_000_000_000 * 10**18,
        reserve1=300 * 10**18,
    )
    # SHIB -> USDC pool
    shib_usdc = _pair(
        SHIB,
        USDC,
        reserve0=1_000_000_000 * 10**18,
        reserve1=400_000 * 10**6,
    )
    # ETH -> USDC pool
    eth_usdc = _pair(
        ETH,
        USDC,
        reserve0=4_000 * 10**18,
        reserve1=5_000_000 * 10**6,
    )
    return shib_eth, shib_usdc, eth_usdc


def test_direct_vs_multihop():
    """Sometimes multi-hop is better despite gas."""
    shib_eth, shib_usdc, eth_usdc = _setup_pools_for_routing()
    finder = RouteFinder([shib_eth, shib_usdc, eth_usdc])

    amount_in = 1_000_000 * 10**18
    best_route, _ = finder.find_best_route(SHIB, ETH, amount_in, gas_price_gwei=1)

    assert best_route.num_hops == 2


def test_gas_makes_direct_better():
    """At high gas prices, fewer hops win."""
    shib_eth, shib_usdc, eth_usdc = _setup_pools_for_routing()
    finder = RouteFinder([shib_eth, shib_usdc, eth_usdc])

    amount_in = 1_000_000 * 10**18
    best_route, _ = finder.find_best_route(SHIB, ETH, amount_in, gas_price_gwei=250)

    assert best_route.num_hops == 1


def test_no_route_exists():
    """Handle disconnected tokens gracefully."""
    only_shib_eth = _pair(
        SHIB,
        ETH,
        reserve0=1_000_000_000 * 10**18,
        reserve1=300 * 10**18,
    )
    finder = RouteFinder([only_shib_eth])
    with pytest.raises(ValueError, match="No route found"):
        finder.find_best_route(SHIB, USDC, 1_000_000 * 10**18, gas_price_gwei=1)


def test_route_output_matches_sequential_swaps():
    """Route simulation equals doing swaps one by one."""
    shib_eth, shib_usdc, eth_usdc = _setup_pools_for_routing()
    route = Route(pools=[shib_usdc, eth_usdc], path=[SHIB, USDC, ETH])

    amount_in = 1_000_000 * 10**18
    out_route = route.get_output(amount_in)

    out_first = shib_usdc.get_amount_out(amount_in, SHIB)
    out_second = eth_usdc.get_amount_out(out_first, USDC)

    assert out_route == out_second


def test_route_finder_reports_all_routes():
    shib_eth, shib_usdc, eth_usdc = _setup_pools_for_routing()
    finder = RouteFinder([shib_eth, shib_usdc, eth_usdc])

    routes = finder.find_all_routes(SHIB, ETH, max_hops=3)
    hop_counts = sorted(route.num_hops for route in routes)
    assert hop_counts == [1, 2]
