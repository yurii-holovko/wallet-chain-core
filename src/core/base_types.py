"""Core type definitions for wallet-chain modules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from eth_utils.address import is_address, to_checksum_address


@dataclass(frozen=True)
class Address:
    """Ethereum address with validation and checksumming."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("Address value must be a string")
        if not is_address(self.value):
            raise ValueError("Invalid Ethereum address")
        object.__setattr__(self, "value", to_checksum_address(self.value))

    @classmethod
    def from_string(cls, s: str) -> "Address":
        return cls(s)

    @property
    def checksum(self) -> str:
        return self.value

    @property
    def lower(self) -> str:
        return self.value.lower()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Address):
            return self.lower == other.lower
        if isinstance(other, str):
            return self.lower == other.lower()
        return False


@dataclass(frozen=True)
class TokenAmount:
    """
    Represents a token amount with proper decimal handling.

    Internally stores raw integer (wei-equivalent).
    Provides human-readable formatting.
    """

    raw: int
    decimals: int
    symbol: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.raw, int):
            raise TypeError("raw must be an int")
        if not isinstance(self.decimals, int) or self.decimals < 0:
            raise ValueError("decimals must be a non-negative integer")

    @classmethod
    def from_human(
        cls, amount: str | Decimal, decimals: int, symbol: str | None = None
    ) -> "TokenAmount":
        """Create from human-readable amount (e.g., '1.5' ETH)."""
        if isinstance(amount, float):
            raise TypeError("amount must be a string or Decimal, not float")
        if isinstance(amount, str):
            decimal_amount = Decimal(amount)
        elif isinstance(amount, Decimal):
            decimal_amount = amount
        else:
            raise TypeError("amount must be a string or Decimal")

        scale = Decimal(10) ** Decimal(decimals)
        raw_decimal = decimal_amount * scale
        if raw_decimal != raw_decimal.to_integral_value():
            raise ValueError("amount has more precision than decimals allow")
        return cls(raw=int(raw_decimal), decimals=decimals, symbol=symbol)

    @property
    def human(self) -> Decimal:
        """Returns human-readable decimal."""
        scale = Decimal(10) ** Decimal(self.decimals)
        return Decimal(self.raw) / scale

    def __add__(self, other: "TokenAmount") -> "TokenAmount":
        if not isinstance(other, TokenAmount):
            return NotImplemented
        if self.decimals != other.decimals:
            raise ValueError("TokenAmount decimals must match")
        symbol = self.symbol
        if self.symbol != other.symbol:
            symbol = self.symbol or other.symbol
        return TokenAmount(self.raw + other.raw, self.decimals, symbol)

    def __mul__(self, factor: int | Decimal) -> "TokenAmount":
        if isinstance(factor, float):
            raise TypeError("factor must be int or Decimal, not float")
        if isinstance(factor, int):
            return TokenAmount(self.raw * factor, self.decimals, self.symbol)
        if isinstance(factor, Decimal):
            raw_decimal = Decimal(self.raw) * factor
            if raw_decimal != raw_decimal.to_integral_value():
                raise ValueError("factor results in fractional base units")
            return TokenAmount(int(raw_decimal), self.decimals, self.symbol)
        raise TypeError("factor must be int or Decimal")

    def __str__(self) -> str:
        return f"{self.human} {self.symbol or ''}".strip()


@dataclass
class TransactionRequest:
    """A transaction ready to be signed."""

    to: Address
    value: TokenAmount
    data: bytes
    from_address: Optional[Address] = None
    nonce: Optional[int] = None
    gas_limit: Optional[int] = None
    max_fee_per_gas: Optional[int] = None
    max_priority_fee: Optional[int] = None
    chain_id: int = 1

    def to_dict(self) -> dict:
        """Convert to web3-compatible dict with hex-encoded numerics for JSON-RPC."""
        payload: dict[str, object] = {
            "to": self.to.checksum,
            "value": hex(self.value.raw) if self.value.raw >= 0 else "0x0",
            "data": f"0x{self.data.hex()}",
        }
        if self.from_address is not None:
            payload["from"] = self.from_address.checksum
        if self.chain_id:
            payload["chainId"] = hex(self.chain_id)
        if self.nonce is not None:
            payload["nonce"] = hex(self.nonce)
        if self.gas_limit is not None:
            payload["gas"] = hex(self.gas_limit)
        if self.max_fee_per_gas is not None:
            payload["maxFeePerGas"] = hex(self.max_fee_per_gas)
        if self.max_priority_fee is not None:
            payload["maxPriorityFeePerGas"] = hex(self.max_priority_fee)
        return payload


@dataclass
class TransactionReceipt:
    """Parsed transaction receipt."""

    tx_hash: str
    block_number: int
    status: bool
    gas_used: int
    effective_gas_price: int
    logs: list

    @property
    def tx_fee(self) -> TokenAmount:
        """Returns transaction fee as TokenAmount."""
        return TokenAmount(
            raw=self.gas_used * self.effective_gas_price,
            decimals=18,
            symbol="ETH",
        )

    @classmethod
    def from_web3(cls, receipt: dict) -> "TransactionReceipt":
        """Parse from web3 receipt dict."""
        tx_hash = receipt.get("transactionHash")
        if hasattr(tx_hash, "hex"):
            tx_hash_value = tx_hash.hex()
        else:
            tx_hash_value = str(tx_hash)

        status_value = receipt.get("status")
        if isinstance(status_value, bool):
            status = status_value
        elif isinstance(status_value, int):
            status = status_value == 1
        elif isinstance(status_value, str):
            if status_value.startswith("0x"):
                status = int(status_value, 16) == 1
            else:
                status = int(status_value) == 1
        else:
            raise ValueError("Invalid status in receipt")

        return cls(
            tx_hash=tx_hash_value,
            block_number=_to_int(receipt.get("blockNumber")),
            status=status,
            gas_used=_to_int(receipt.get("gasUsed")),
            effective_gas_price=_to_int(receipt.get("effectiveGasPrice")),
            logs=receipt.get("logs", []),
        )


def _to_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 16) if value.startswith("0x") else int(value)
    raise ValueError("Expected integer-like value")
