from uniswap_v2_pair import Token, UniswapV2Pair


class Route:
    """Represents a swap route through one or more pools."""

    def __init__(self, pools: list[UniswapV2Pair], path: list[Token]):
        self.pools = pools
        self.path = path  # token_in → intermediate... → token_out

    @property
    def num_hops(self) -> int:
        return len(self.pools)

    def get_output(self, amount_in: int) -> int:
        """Simulate full route, return final output."""
        ...

    def get_intermediate_amounts(self, amount_in: int) -> list[int]:
        """Return amount at each step: [input, after_hop1, after_hop2, ...]"""
        ...

    def estimate_gas(self) -> int:
        """Estimate gas: ~150k base + ~100k per hop."""
        ...


class RouteFinder:
    """
    Finds optimal routes between tokens.
    """

    def __init__(self, pools: list[UniswapV2Pair]):
        self.pools = pools
        self.graph = self._build_graph()

    def _build_graph(self) -> dict:
        """
        Build adjacency graph: token → [(pool, other_token), ...]
        """
        ...

    def find_all_routes(
        self, token_in: Token, token_out: Token, max_hops: int = 3
    ) -> list[Route]:
        """
        Find all possible routes up to max_hops.
        """
        ...

    def find_best_route(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: int,
        gas_price_gwei: int,
        max_hops: int = 3,
    ) -> tuple[Route, int]:
        """
        Find route that maximizes NET output (after gas).
        Returns (best_route, net_output).
        """
        ...

    def compare_routes(
        self, token_in: Token, token_out: Token, amount_in: int, gas_price_gwei: int
    ) -> list[dict]:
        """
        Compare all routes with detailed breakdown:
        {
            'route': Route,
            'gross_output': int,
            'gas_estimate': int,
            'gas_cost': int,
            'net_output': int,
        }
        """
        ...
