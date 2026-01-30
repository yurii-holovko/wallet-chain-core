from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from eth_abi.abi import decode
from eth_utils.crypto import keccak
from web3 import Web3
from web3.types import TxParams

from core.base_types import Address

from .route import Route
from .uniswap_v2_pair import Token, UniswapV2Pair


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
        tx = _build_tx(router, swap_params, sender)
        try:
            tx_hash = self.w3.eth.send_transaction(tx)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        except Exception as exc:  # noqa: BLE001
            return SimulationResult(
                success=False,
                amount_out=0,
                gas_used=0,
                error=str(exc),
                logs=[],
            )

        logs = list(receipt["logs"])
        amount_out = _extract_amount_out(logs)
        return SimulationResult(
            success=bool(receipt["status"]),
            amount_out=amount_out,
            gas_used=int(receipt["gasUsed"]),
            error=None if receipt["status"] else "Transaction failed",
            logs=logs,
        )

    def simulate_route(
        self, route: Route, amount_in: int, sender: Address
    ) -> SimulationResult:
        """
        Simulate a multi-hop route.
        """
        try:
            amount_out = route.get_output(amount_in)
            gas_used = route.estimate_gas()
        except Exception as exc:  # noqa: BLE001
            return SimulationResult(
                success=False,
                amount_out=0,
                gas_used=0,
                error=str(exc),
                logs=[],
            )

        return SimulationResult(
            success=True,
            amount_out=amount_out,
            gas_used=gas_used,
            error=None,
            logs=[],
        )

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


SWAP_TOPIC = (
    f"0x{keccak(text='Swap(address,uint256,uint256,uint256,uint256,address)').hex()}"
)


def _hex_to_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 16) if value.startswith("0x") else int(value)
    return 0


def _build_tx(router: Address, swap_params: dict, sender: Address) -> TxParams:
    data = swap_params.get("data")
    if not isinstance(data, str) or not data.startswith("0x"):
        raise ValueError("swap_params.data must be 0x-prefixed hex calldata")
    tx: dict[str, object] = {
        "from": sender.checksum,
        "to": router.checksum,
        "data": data,
    }
    value = swap_params.get("value")
    if value is not None:
        tx["value"] = _hex_to_int(value)
    gas = swap_params.get("gas")
    if gas is not None:
        tx["gas"] = _hex_to_int(gas)
    gas_price = swap_params.get("gasPrice")
    if gas_price is not None:
        tx["gasPrice"] = _hex_to_int(gas_price)
    return tx  # type: ignore[return-value]


def _decode_swap_amounts(log) -> tuple[int, int]:
    if not log.get("data"):
        return 0, 0
    amounts = decode(["uint256", "uint256", "uint256", "uint256"], log["data"])
    amount0_out = int(amounts[2])
    amount1_out = int(amounts[3])
    return amount0_out, amount1_out


def _is_swap_log(log) -> bool:
    topics = log.get("topics") or []
    return bool(topics and topics[0] == SWAP_TOPIC)


def _extract_amount_out(logs) -> int:
    amount_out = 0
    for log in logs or []:
        if _is_swap_log(log):
            amount0_out, amount1_out = _decode_swap_amounts(log)
            amount_out = max(amount_out, amount0_out, amount1_out)
    return amount_out
