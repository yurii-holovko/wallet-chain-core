from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from route import Route
from uniswap_v2_pair import Token, UniswapV2Pair
from web3 import Web3

from core.base_types import Address


class ForkSimulator:
    """
    Simulates transactions on a local fork.
    """

    def __init__(self, fork_url: str):
        """
        fork_url: Local Anvil/Hardhat fork RPC
        """
        self.w3 = Web3(Web3.HTTPProvider(fork_url))

    def simulate_swap(
        self, router: Address, swap_params: dict, sender: Address
    ) -> SimulationResult:
        """
        Simulate a swap and return detailed results.
        """
        ...

    def simulate_route(
        self, route: Route, amount_in: int, sender: Address
    ) -> SimulationResult:
        """
        Simulate a multi-hop route.
        """
        ...

    def compare_simulation_vs_calculation(
        self,
        pair: UniswapV2Pair,
        amount_in: int,
        token_in: Token,
        router: Address,
        swap_params: dict,
        sender: Address,
    ) -> dict:
        """
        Compare our AMM math vs actual fork simulation.
        Useful for validation.
        """
        calculated = pair.get_amount_out(amount_in, token_in)
        simulated = self.simulate_swap(router, swap_params, sender=sender)

        return {
            "calculated": calculated,
            "simulated": simulated.amount_out,
            "difference": abs(calculated - simulated.amount_out),
            "match": calculated == simulated.amount_out,
        }


@dataclass
class SimulationResult:
    success: bool
    amount_out: int
    gas_used: int
    error: Optional[str]
    logs: list  # Decoded events
