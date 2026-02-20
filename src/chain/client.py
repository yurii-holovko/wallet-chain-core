"""Ethereum JSON-RPC client with retries and error classification."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from core.base_types import Address, TokenAmount, TransactionReceipt, TransactionRequest

from .errors import (
    ChainError,
    InsufficientFunds,
    NonceTooLow,
    ReplacementUnderpriced,
    RPCError,
    TransactionFailed,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GasPrice:
    """Current gas price information."""

    base_fee: int
    priority_fee_low: int
    priority_fee_medium: int
    priority_fee_high: int

    def get_max_fee(self, priority: str = "medium", buffer: float = 1.2) -> int:
        """Calculate maxFeePerGas with buffer for base fee increase."""
        if buffer <= 0:
            raise ValueError("buffer must be positive")
        priority_fee = {
            "low": self.priority_fee_low,
            "medium": self.priority_fee_medium,
            "high": self.priority_fee_high,
        }.get(priority)
        if priority_fee is None:
            raise ValueError("priority must be low, medium, or high")
        return int(self.base_fee * buffer) + priority_fee


class ChainClient:
    """
    Ethereum RPC client with reliability features.

    Features:
    - Automatic retry with exponential backoff
    - Multiple RPC endpoint fallback
    - Request timing/logging
    - Proper error classification
    """

    def __init__(
        self,
        rpc_urls: list[str],
        timeout: int = 30,
        max_retries: int = 3,
    ):
        if not rpc_urls:
            raise ValueError("rpc_urls must not be empty")
        self._rpc_urls = rpc_urls
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = requests.Session()

    def get_balance(self, address: Address) -> TokenAmount:
        balance_hex = self._rpc_call("eth_getBalance", [address.checksum, "latest"])
        return TokenAmount(
            raw=_hex_to_int(balance_hex),
            decimals=18,
            symbol="ETH",
        )

    def get_nonce(self, address: Address, block: str = "pending") -> int:
        nonce_hex = self._rpc_call("eth_getTransactionCount", [address.checksum, block])
        return _hex_to_int(nonce_hex)

    def get_gas_price(self) -> GasPrice:
        block = self._rpc_call("eth_getBlockByNumber", ["latest", False])
        base_fee = _hex_to_int(block.get("baseFeePerGas", "0x0"))
        priority = self._rpc_call("eth_maxPriorityFeePerGas", [])
        priority_fee = _hex_to_int(priority)
        return GasPrice(
            base_fee=base_fee,
            priority_fee_low=priority_fee,
            priority_fee_medium=priority_fee,
            priority_fee_high=int(priority_fee * 1.5),
        )

    def estimate_gas(self, tx: TransactionRequest) -> int:
        gas_hex = self._rpc_call("eth_estimateGas", [tx.to_dict()])
        return _hex_to_int(gas_hex)

    def send_transaction(self, signed_tx: bytes) -> str:
        tx_hash = self._rpc_call("eth_sendRawTransaction", [f"0x{signed_tx.hex()}"])
        return str(tx_hash)

    def wait_for_receipt(
        self,
        tx_hash: str,
        timeout: int = 120,
        poll_interval: float = 1.0,
    ) -> TransactionReceipt:
        start = time.time()
        while time.time() - start < timeout:
            receipt = self.get_receipt(tx_hash)
            if receipt is not None:
                if receipt.status is False:
                    raise TransactionFailed(tx_hash, receipt)
                return receipt
            time.sleep(poll_interval)
        raise TimeoutError(f"Timed out waiting for receipt {tx_hash}")

    def get_transaction(self, tx_hash: str) -> dict:
        return self._rpc_call("eth_getTransactionByHash", [tx_hash])

    def get_receipt(self, tx_hash: str) -> Optional[TransactionReceipt]:
        data = self._rpc_call("eth_getTransactionReceipt", [tx_hash])
        if data is None:
            return None
        return TransactionReceipt.from_web3(data)

    def get_chain_id(self) -> int:
        """Return the numeric chain ID reported by the RPC node."""
        chain_hex = self._rpc_call("eth_chainId", [])
        return _hex_to_int(chain_hex)

    def get_block(self, block: str, full: bool = False) -> dict:
        return self._rpc_call("eth_getBlockByNumber", [block, full])

    def call(self, tx: TransactionRequest, block: str = "latest") -> bytes:
        result = self._rpc_call("eth_call", [tx.to_dict(), block])
        return _hex_to_bytes(result)

    def _rpc_call(self, method: str, params: list[Any]) -> Any:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        last_error: Optional[Exception] = None
        for url in self._rpc_urls:
            for attempt in range(self._max_retries):
                start = time.perf_counter()
                try:
                    response = self._session.post(
                        url,
                        json=payload,
                        timeout=self._timeout,
                    )
                    elapsed = time.perf_counter() - start
                    logger.info("rpc %s in %.3fs", method, elapsed)
                    if response.status_code >= 400:
                        raise RPCError(f"HTTP {response.status_code} from {url}")
                    data = response.json()
                    if "error" in data:
                        self._raise_rpc_error(data["error"])
                    return data.get("result")
                except (requests.Timeout, requests.ConnectionError) as exc:
                    last_error = exc
                    self._sleep_backoff(attempt)
                except RPCError as exc:
                    raise exc
                except json.JSONDecodeError as exc:
                    last_error = exc
                    self._sleep_backoff(attempt)
        raise ChainError("RPC request failed") from last_error

    def _rpc_batch(self, calls: list[tuple[str, list[Any]]]) -> list[Any]:
        payload = [
            {"jsonrpc": "2.0", "id": idx + 1, "method": method, "params": params}
            for idx, (method, params) in enumerate(calls)
        ]
        last_error: Optional[Exception] = None
        for url in self._rpc_urls:
            for attempt in range(self._max_retries):
                start = time.perf_counter()
                try:
                    response = self._session.post(
                        url,
                        json=payload,
                        timeout=self._timeout,
                    )
                    elapsed = time.perf_counter() - start
                    logger.info("rpc batch (%d calls) in %.3fs", len(calls), elapsed)
                    if response.status_code >= 400:
                        raise RPCError(f"HTTP {response.status_code} from {url}")
                    data = response.json()
                    if not isinstance(data, list):
                        raise RPCError("Invalid batch response")
                    results: dict[int, Any] = {}
                    for entry in data:
                        if "error" in entry:
                            self._raise_rpc_error(entry["error"])
                        results[int(entry["id"])] = entry.get("result")
                    return [results[idx + 1] for idx in range(len(calls))]
                except (requests.Timeout, requests.ConnectionError) as exc:
                    last_error = exc
                    self._sleep_backoff(attempt)
                except RPCError:
                    raise
                except json.JSONDecodeError as exc:
                    last_error = exc
                    self._sleep_backoff(attempt)
        raise ChainError("RPC request failed") from last_error

    def _sleep_backoff(self, attempt: int) -> None:
        delay = 0.5 * (2**attempt)
        time.sleep(delay)

    def _raise_rpc_error(self, error: dict) -> None:
        message = str(error.get("message", "RPC error"))
        code = error.get("code")
        data = error.get("data")
        lowered = message.lower()
        if "insufficient funds" in lowered:
            raise InsufficientFunds(message)
        if "nonce too low" in lowered:
            raise NonceTooLow(message)
        if "replacement transaction underpriced" in lowered:
            raise ReplacementUnderpriced(message)
        raise RPCError(message, code=code, data=data)


def _hex_to_int(value: str) -> int:
    if not isinstance(value, str):
        raise RPCError("Expected hex string result")
    return int(value, 16)


def _hex_to_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        raise RPCError("Expected hex string result")
    normalized = value[2:] if value.startswith("0x") else value
    if normalized == "":
        return b""
    return bytes.fromhex(normalized)
