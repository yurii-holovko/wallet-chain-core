from .uniswap_v2_pair import Token, UniswapV2Pair


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
        if len(self.path) != len(self.pools) + 1:
            raise ValueError("path length must be pools + 1")
        if amount_in <= 0:
            raise ValueError("amount_in must be positive")

        amount = amount_in
        for idx, pool in enumerate(self.pools):
            token_in = self.path[idx]
            amount = pool.get_amount_out(amount, token_in)
        return amount

    def get_intermediate_amounts(self, amount_in: int) -> list[int]:
        """Return amount at each step: [input, after_hop1, after_hop2, ...]"""
        if len(self.path) != len(self.pools) + 1:
            raise ValueError("path length must be pools + 1")
        if amount_in <= 0:
            raise ValueError("amount_in must be positive")

        amounts = [amount_in]
        current = amount_in
        for idx, pool in enumerate(self.pools):
            token_in = self.path[idx]
            current = pool.get_amount_out(current, token_in)
            amounts.append(current)
        return amounts

    def estimate_gas(self) -> int:
        """Estimate gas: ~150k base + ~100k per hop."""
        return 150_000 + 100_000 * self.num_hops


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
        graph: dict[str, list[tuple[UniswapV2Pair, Token]]] = {}
        for pool in self.pools:
            token0 = pool.token0
            token1 = pool.token1
            graph.setdefault(token0.address.checksum, []).append((pool, token1))
            graph.setdefault(token1.address.checksum, []).append((pool, token0))
        return graph

    def find_all_routes(
        self, token_in: Token, token_out: Token, max_hops: int = 3
    ) -> list[Route]:
        """
        Find all possible routes up to max_hops.
        """
        if max_hops <= 0:
            return []

        routes: list[Route] = []

        def dfs(
            current: Token,
            target: Token,
            pools_used: list[UniswapV2Pair],
            path_tokens: list[Token],
            visited_tokens: set[str],
        ) -> None:
            if current.address == target.address:
                routes.append(Route(pools=list(pools_used), path=list(path_tokens)))
                return
            if len(pools_used) >= max_hops:
                return

            for pool, other_token in self.graph.get(current.address.checksum, []):
                if pool in pools_used:
                    continue
                other_key = other_token.address.checksum
                if other_key in visited_tokens:
                    continue
                pools_used.append(pool)
                path_tokens.append(other_token)
                visited_tokens.add(other_key)
                dfs(other_token, target, pools_used, path_tokens, visited_tokens)
                visited_tokens.remove(other_key)
                path_tokens.pop()
                pools_used.pop()

        dfs(token_in, token_out, [], [token_in], {token_in.address.checksum})
        return routes

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
        routes = self.find_all_routes(token_in, token_out, max_hops=max_hops)
        if not routes:
            raise ValueError("No route found")

        best_route = routes[0]
        best_net = -1

        for route in routes:
            gross_output = route.get_output(amount_in)
            gas_estimate = route.estimate_gas()
            net_output = self._apply_gas_cost(
                gross_output, gas_estimate, gas_price_gwei, token_out
            )
            if net_output > best_net:
                best_net = net_output
                best_route = route

        return best_route, best_net

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
        routes = self.find_all_routes(token_in, token_out)
        comparisons: list[dict] = []
        for route in routes:
            gross_output = route.get_output(amount_in)
            gas_estimate = route.estimate_gas()
            gas_cost = gas_estimate * gas_price_gwei * 10**9
            net_output = self._apply_gas_cost(
                gross_output, gas_estimate, gas_price_gwei, token_out
            )
            comparisons.append(
                {
                    "route": route,
                    "gross_output": gross_output,
                    "gas_estimate": gas_estimate,
                    "gas_cost": gas_cost,
                    "net_output": net_output,
                }
            )
        return comparisons

    @staticmethod
    def _apply_gas_cost(
        gross_output: int, gas_estimate: int, gas_price_gwei: int, token_out: Token
    ) -> int:
        gas_cost_wei = gas_estimate * gas_price_gwei * 10**9
        if token_out.symbol.upper() == "ETH" and token_out.decimals == 18:
            net = gross_output - gas_cost_wei
            return net if net > 0 else 0
        return gross_output
