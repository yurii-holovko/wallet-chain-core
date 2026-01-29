from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from chain.client import ChainClient
from core.base_types import Address


@dataclass(frozen=True)
class Token:
    address: Address
    symbol: str
    decimals: int


class UniswapV2Pair:
    """
    Represents a Uniswap V2 liquidity pair.
    All math uses integers only — no floats anywhere.
    """

    def __init__(
        self,
        address: Address,
        token0: Token,
        token1: Token,
        reserve0: int,
        reserve1: int,
        fee_bps: int = 30,  # 0.30% = 30 basis points
    ): ...

    def get_amount_out(self, amount_in: int, token_in: Token) -> int:
        """
        Calculate output amount for a given input.
        Must match Solidity exactly:

        amount_in_with_fee = amount_in * (10000 - fee_bps)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in * 10000 + amount_in_with_fee
        amount_out = numerator // denominator
        """
        ...

    def get_amount_in(self, amount_out: int, token_out: Token) -> int:
        """
        Calculate required input for desired output.
        (Inverse of get_amount_out)
        """
        ...

    def get_spot_price(self, token_in: Token) -> Decimal:
        """
        Returns spot price (for display only, not calculations).
        """
        ...

    def get_execution_price(self, amount_in: int, token_in: Token) -> Decimal:
        """
        Returns actual execution price for given trade size.
        """
        ...

    def get_price_impact(self, amount_in: int, token_in: Token) -> Decimal:
        """
        Returns price impact as a decimal (0.01 = 1%).
        """
        ...

    def simulate_swap(self, amount_in: int, token_in: Token) -> "UniswapV2Pair":
        """
        Returns a NEW pair with updated reserves after the swap.
        (Useful for multi-hop simulation)
        """
        ...

    @classmethod
    def from_chain(cls, address: Address, client: ChainClient) -> "UniswapV2Pair":
        """
        Fetch pair data from on-chain.
        """
        ...
