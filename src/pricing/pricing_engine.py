from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from chain.client import ChainClient
from core.base_types import Address

from .fork_simulator import ForkSimulator
from .mempool_monitor import MempoolMonitor, ParsedSwap
from .route import Route, RouteFinder
from .uniswap_v2_pair import Token, UniswapV2Pair


class PricingEngine:
    """
    Main interface for the pricing module.
    Integrates AMM math, routing, simulation, and mempool monitoring.
    """

    def __init__(
        self, chain_client: ChainClient, fork_url: str, ws_url: str  # From Week 1
    ):
        self.client = chain_client
        self.simulator = ForkSimulator(fork_url)
        self.monitor = MempoolMonitor(
            ws_url, self._on_mempool_swap, quote_fn=self._quote_for_swap
        )
        self.pools: dict[Address, UniswapV2Pair] = {}
        self._pool_index: dict[tuple[str, str], UniswapV2Pair] = {}
        self._token_to_pools: dict[str, set[Address]] = {}
        self.router: Optional[RouteFinder] = None

    def load_pools(self, pool_addresses: list[Address]):
        """Load pool data from chain."""
        for addr in pool_addresses:
            self.pools[addr] = UniswapV2Pair.from_chain(addr, self.client)
        self._rebuild_indices()
        self.router = RouteFinder(list(self.pools.values()))

    def refresh_pool(self, address: Address):
        """Refresh single pool's reserves."""
        self.pools[address] = UniswapV2Pair.from_chain(address, self.client)
        self._rebuild_indices()
        self.router = RouteFinder(list(self.pools.values()))

    def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: int,
        gas_price_gwei: int,
        sender: Address,
    ) -> Quote:
        """
        Get best quote for a swap.
        """
        if self.router is None:
            raise QuoteError("Router not initialized. Call load_pools first.")

        route, net_output = self.router.find_best_route(
            token_in, token_out, amount_in, gas_price_gwei
        )

        # Verify with simulation
        sim_result = self.simulator.simulate_route(route, amount_in, sender)

        if not sim_result.success:
            raise QuoteError(f"Simulation failed: {sim_result.error}")

        return Quote(
            route=route,
            amount_in=amount_in,
            expected_output=net_output,
            simulated_output=sim_result.amount_out,
            gas_estimate=sim_result.gas_used,
            timestamp=time.time(),
        )

    def _on_mempool_swap(self, swap: ParsedSwap):
        """Handle detected mempool swap."""
        if not swap.token_in or not swap.token_out:
            return
        affected: set[Address] = set()
        for token in swap.path or [swap.token_in, swap.token_out]:
            pools = self._token_to_pools.get(token.checksum, set())
            affected.update(pools)
        for pool_addr in affected:
            try:
                self.refresh_pool(pool_addr)
            except Exception:
                continue

    async def start_monitoring(self) -> None:
        await self.monitor.start()

    def _rebuild_indices(self) -> None:
        self._pool_index.clear()
        self._token_to_pools.clear()
        for addr, pool in self.pools.items():
            key = _pair_key(pool.token0.address, pool.token1.address)
            self._pool_index[key] = pool
            for token in (pool.token0, pool.token1):
                self._token_to_pools.setdefault(token.address.checksum, set()).add(addr)

    def _find_pool(self, token_a: Address, token_b: Address) -> UniswapV2Pair | None:
        return self._pool_index.get(_pair_key(token_a, token_b))

    def _find_token(self, address: Address) -> Token | None:
        for pool in self.pools.values():
            if pool.token0.address == address:
                return pool.token0
            if pool.token1.address == address:
                return pool.token1
        return None

    def _quote_for_swap(self, swap: ParsedSwap) -> int | None:
        if self.router is None or not swap.token_in or not swap.token_out:
            return None
        if swap.amount_in <= 0:
            return None

        if swap.path:
            amount = swap.amount_in
            for i in range(len(swap.path) - 1):
                token_in_addr = swap.path[i]
                token_out_addr = swap.path[i + 1]
                pool = self._find_pool(token_in_addr, token_out_addr)
                if pool is None:
                    return None
                token_in = (
                    pool.token0 if pool.token0.address == token_in_addr else pool.token1
                )
                amount = pool.get_amount_out(amount, token_in)
            return amount

        token_in = self._find_token(swap.token_in)
        token_out = self._find_token(swap.token_out)
        if token_in is None or token_out is None:
            return None
        route, expected = self.router.find_best_route(
            token_in, token_out, swap.amount_in, gas_price_gwei=0
        )
        return expected


def _pair_key(token_a: Address, token_b: Address) -> tuple[str, str]:
    a = token_a.checksum
    b = token_b.checksum
    return (a, b) if a < b else (b, a)


@dataclass
class Quote:
    route: Route
    amount_in: int
    expected_output: int
    simulated_output: int
    gas_estimate: int
    timestamp: float

    @property
    def is_valid(self) -> bool:
        """Quote valid if simulation matches expectation within tolerance."""
        tolerance = 0.001  # 0.1%
        diff = abs(self.expected_output - self.simulated_output) / self.expected_output
        return diff < tolerance


class QuoteError(RuntimeError):
    """Raised when a quote cannot be produced."""
