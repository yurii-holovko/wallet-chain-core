from __future__ import annotations

from decimal import Decimal

from uniswap_v2_pair import Token, UniswapV2Pair


class PriceImpactAnalyzer:
    """
    Analyzes price impact across different trade sizes.
    """

    def __init__(self, pair: UniswapV2Pair):
        self.pair = pair

    def generate_impact_table(
        self, token_in: Token, sizes: list[int]  # List of input amounts to analyze
    ) -> list[dict]:
        """
        Returns list of:
        {
            'amount_in': int,
            'amount_out': int,
            'spot_price': Decimal,
            'execution_price': Decimal,
            'price_impact_pct': Decimal,
        }
        """
        ...

    def find_max_size_for_impact(self, token_in: Token, max_impact_pct: Decimal) -> int:
        """
        Binary search to find largest trade with impact <= max_impact_pct.
        """
        ...

    def estimate_true_cost(
        self,
        amount_in: int,
        token_in: Token,
        gas_price_gwei: int,
        gas_estimate: int = 150000,
    ) -> dict:
        """
        Returns total cost including gas:
        {
            'gross_output': int,
            'gas_cost_eth': int,
            'gas_cost_in_output_token': int,
            'net_output': int,
            'effective_price': Decimal,
        }
        """
        ...
