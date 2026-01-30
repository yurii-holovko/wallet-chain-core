from core.base_types import Address
from pricing.uniswap_v2_pair import Token, UniswapV2Pair

ETH = Token(Address("0x0000000000000000000000000000000000000001"), "ETH", 18)
USDC = Token(Address("0x0000000000000000000000000000000000000002"), "USDC", 6)
PAIR = Address("0x0000000000000000000000000000000000000003")


def _make_pair(reserve0: int, reserve1: int) -> UniswapV2Pair:
    return UniswapV2Pair(
        address=PAIR,
        token0=ETH,
        token1=USDC,
        reserve0=reserve0,
        reserve1=reserve1,
        fee_bps=30,
    )


def test_get_amount_out_basic():
    """1000 ETH / 2M USDC pool, buy 1 ETH worth."""
    pair = _make_pair(
        reserve0=1000 * 10**18,  # 1000 ETH
        reserve1=2_000_000 * 10**6,  # 2M USDC
    )

    usdc_in = 2000 * 10**6  # 2000 USDC
    eth_out = pair.get_amount_out(usdc_in, USDC)

    # Should get slightly less than 1 ETH due to fee + impact.
    assert eth_out < 1 * 10**18
    assert eth_out > int(0.99 * 10**18)


def test_get_amount_out_matches_solidity():
    """Compare against known on-chain-style result (same formula)."""
    pair = _make_pair(
        reserve0=1000 * 10**18,
        reserve1=2_000_000 * 10**6,
    )

    usdc_in = 2000 * 10**6
    eth_out = pair.get_amount_out(usdc_in, USDC)

    assert eth_out == 996006981039903216


def test_integer_math_no_floats():
    """Verify no floating point used."""
    pair = _make_pair(reserve0=10**30, reserve1=10**30)
    out = pair.get_amount_out(10**25, ETH)
    assert isinstance(out, int)


def test_fee_calculation_matches_formula():
    pair = _make_pair(reserve0=1000 * 10**18, reserve1=2_000_000 * 10**6)
    amount_in = 2000 * 10**6
    reserve_in = pair.reserve1
    reserve_out = pair.reserve0
    amount_in_with_fee = amount_in * (10000 - pair.fee_bps)
    numerator = amount_in_with_fee * reserve_out
    denominator = reserve_in * 10000 + amount_in_with_fee
    expected = numerator // denominator

    assert pair.get_amount_out(amount_in, USDC) == expected


def test_known_onchain_style_result():
    pair = _make_pair(reserve0=1000 * 10**18, reserve1=2_000_000 * 10**6)
    amount_in = 2000 * 10**6
    assert pair.get_amount_out(amount_in, USDC) == 996006981039903216


def test_swap_is_immutable():
    """simulate_swap doesn't modify original."""
    pair = _make_pair(reserve0=1000 * 10**18, reserve1=2_000_000 * 10**6)
    original_reserve0 = pair.reserve0
    original_reserve1 = pair.reserve1

    amount_in = 1 * 10**18
    amount_out = pair.get_amount_out(amount_in, ETH)
    new_pair = pair.simulate_swap(amount_in, ETH)

    assert pair.reserve0 == original_reserve0
    assert pair.reserve1 == original_reserve1

    assert new_pair.reserve0 == original_reserve0 + amount_in
    assert new_pair.reserve1 == original_reserve1 - amount_out
