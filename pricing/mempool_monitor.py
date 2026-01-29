from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional

from core.base_types import Address


class MempoolMonitor:
    """
    Monitors pending transactions for swap activity.
    """

    # Known DEX router selectors
    SWAP_SELECTORS = {
        "0x38ed1739": ("UniswapV2", "swapExactTokensForTokens"),
        "0x7ff36ab5": ("UniswapV2", "swapExactETHForTokens"),
        "0x18cbafe5": ("UniswapV2", "swapExactTokensForETH"),
        "0x5ae401dc": ("UniswapV3", "multicall"),
        # Add more...
    }

    def __init__(self, ws_url: str, callback: Callable):
        """
        callback receives ParsedSwap objects for each detected swap.
        """
        self.ws_url = ws_url
        self.callback = callback

    async def start(self):
        """Start monitoring pending transactions."""
        ...

    def parse_transaction(self, tx: dict) -> Optional["ParsedSwap"]:
        """
        Parse transaction to extract swap details.
        Returns None if not a swap.
        """
        ...

    def decode_swap_params(self, selector: str, data: bytes) -> dict:
        """
        Decode swap parameters from calldata.
        """
        ...


@dataclass
class ParsedSwap:
    """Parsed swap transaction from mempool."""

    tx_hash: str
    router: str
    dex: str
    method: str
    token_in: Optional[Address]
    token_out: Optional[Address]
    amount_in: int
    min_amount_out: int
    deadline: int
    sender: Address
    gas_price: int

    @property
    def slippage_tolerance(self) -> Decimal:
        """Calculate implied slippage tolerance."""
        ...
