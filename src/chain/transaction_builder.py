"""Fluent transaction builder for signing and sending."""

from __future__ import annotations

from dataclasses import dataclass

from eth_account.datastructures import SignedTransaction

from core.base_types import Address, TokenAmount, TransactionReceipt, TransactionRequest
from core.wallet_manager import WalletManager

from .client import ChainClient


@dataclass
class _TxState:
    to: Address | None = None
    value: TokenAmount | None = None
    data: bytes | None = None
    nonce: int | None = None
    gas_limit: int | None = None
    max_fee_per_gas: int | None = None
    max_priority_fee: int | None = None
    chain_id: int = 1


class TransactionBuilder:
    """
    Fluent builder for transactions.

    Usage:
        tx = (TransactionBuilder(client, wallet)
            .to(recipient)
            .value(TokenAmount.from_human("0.1", 18))
            .data(calldata)
            .with_gas_estimate()
            .with_gas_price("high")
            .build())
    """

    def __init__(self, client: ChainClient, wallet: WalletManager):
        self._client = client
        self._wallet = wallet
        self._state = _TxState()

    def to(self, address: Address) -> "TransactionBuilder":
        self._state.to = address
        return self

    def value(self, amount: TokenAmount) -> "TransactionBuilder":
        self._state.value = amount
        return self

    def data(self, calldata: bytes) -> "TransactionBuilder":
        self._state.data = calldata
        return self

    def nonce(self, nonce: int) -> "TransactionBuilder":
        """Explicit nonce (for replacement or batch)."""
        self._state.nonce = nonce
        return self

    def gas_limit(self, limit: int) -> "TransactionBuilder":
        self._state.gas_limit = limit
        return self

    def chain_id(self, chain_id: int) -> "TransactionBuilder":
        """Set EVM chain id for signing."""
        if chain_id <= 0:
            raise ValueError("chain_id must be positive")
        self._state.chain_id = chain_id
        return self

    def with_gas_estimate(self, buffer: float = 1.2) -> "TransactionBuilder":
        """Estimate gas and set limit with buffer."""
        if buffer <= 0:
            raise ValueError("buffer must be positive")
        estimate = self._client.estimate_gas(self._build_request(require_fee=False))
        self._state.gas_limit = int(estimate * buffer)
        return self

    def with_gas_price(self, priority: str = "medium") -> "TransactionBuilder":
        """Set gas price based on current network conditions."""
        gas = self._client.get_gas_price()
        self._state.max_priority_fee = {
            "low": gas.priority_fee_low,
            "medium": gas.priority_fee_medium,
            "high": gas.priority_fee_high,
        }.get(priority)
        if self._state.max_priority_fee is None:
            raise ValueError("priority must be low, medium, or high")
        self._state.max_fee_per_gas = gas.get_max_fee(priority)
        return self

    def build(self) -> TransactionRequest:
        """Validate and return transaction request."""
        return self._build_request(require_fee=True)

    def build_and_sign(self) -> SignedTransaction:
        """Build, sign, and return ready-to-send transaction."""
        request = self.build()
        return self._wallet.sign_transaction(request.to_dict())

    def send(self) -> str:
        """Build, sign, send, return tx hash."""
        signed = self.build_and_sign()
        return self._client.send_transaction(signed.raw_transaction)

    def send_and_wait(self, timeout: int = 120) -> TransactionReceipt:
        """Build, sign, send, wait for confirmation."""
        tx_hash = self.send()
        return self._client.wait_for_receipt(tx_hash, timeout=timeout)

    def _build_request(self, require_fee: bool) -> TransactionRequest:
        if self._state.to is None:
            raise ValueError("to address is required")
        if self._state.value is None:
            raise ValueError("value is required")
        if self._state.data is None:
            self._state.data = b""
        if self._state.gas_limit is None:
            if require_fee:
                raise ValueError("gas_limit is required (call with_gas_estimate)")
            self._state.gas_limit = 0

        if self._state.nonce is None:
            sender = Address.from_string(self._wallet.address)
            self._state.nonce = self._client.get_nonce(sender)

        if require_fee:
            if self._state.max_fee_per_gas is None:
                raise ValueError("max_fee_per_gas is required (call with_gas_price)")
            if self._state.max_priority_fee is None:
                raise ValueError("max_priority_fee is required (call with_gas_price)")

        return TransactionRequest(
            to=self._state.to,
            value=self._state.value,
            data=self._state.data,
            from_address=Address.from_string(self._wallet.address),
            nonce=self._state.nonce,
            gas_limit=self._state.gas_limit,
            max_fee_per_gas=self._state.max_fee_per_gas,
            max_priority_fee=self._state.max_priority_fee,
            chain_id=self._state.chain_id,
        )
