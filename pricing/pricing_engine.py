from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from fork_simulator import ForkSimulator
from mempool_monitor import MempoolMonitor, ParsedSwap
from route import Route, RouteFinder
from uniswap_v2_pair import Token, UniswapV2Pair

from chain.client import ChainClient
from core.base_types import Address


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
        self.monitor = MempoolMonitor(ws_url, self._on_mempool_swap)
        self.pools: dict[Address, UniswapV2Pair] = {}
        self.router: Optional[RouteFinder] = None

    def load_pools(self, pool_addresses: list[Address]):
        """Load pool data from chain."""
        for addr in pool_addresses:
            self.pools[addr] = UniswapV2Pair.from_chain(addr, self.client)
        self.router = RouteFinder(list(self.pools.values()))

    def refresh_pool(self, address: Address):
        """Refresh single pool's reserves."""
        ...

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
        # Check if it affects any of our pools
        # Could trigger re-quote or alert
        ...


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
