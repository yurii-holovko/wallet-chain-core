from core.base_types import Address
from pricing.route import RouteFinder
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


def main() -> None:
    # Same pools used in tests: multihop beats direct at low gas, flips at high gas.
    shib_eth = _pair(
        SHIB,
        ETH,
        reserve0=1_000_000_000 * 10**18,
        reserve1=300 * 10**18,
    )
    shib_usdc = _pair(
        SHIB,
        USDC,
        reserve0=1_000_000_000 * 10**18,
        reserve1=400_000 * 10**6,
    )
    eth_usdc = _pair(
        ETH,
        USDC,
        reserve0=4_000 * 10**18,
        reserve1=5_000_000 * 10**6,
    )

    finder = RouteFinder([shib_eth, shib_usdc, eth_usdc])
    amount_in = 1_000_000 * 10**18

    for gas_price in (1, 250):
        route, net_output = finder.find_best_route(
            SHIB, ETH, amount_in, gas_price_gwei=gas_price
        )
        hop_symbols = " -> ".join(token.symbol for token in route.path)
        print(f"Gas: {gas_price} gwei | Hops: {route.num_hops} | Path: {hop_symbols}")
        print(f"Net output (wei): {net_output}")


if __name__ == "__main__":
    main()
