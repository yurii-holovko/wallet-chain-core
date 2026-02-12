"""Arbitrage signal dataclass and direction enum."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class Direction(Enum):
    BUY_CEX_SELL_DEX = "buy_cex_sell_dex"
    BUY_DEX_SELL_CEX = "buy_dex_sell_cex"


@dataclass
class Signal:
    """A validated arbitrage opportunity ready for execution."""

    signal_id: str
    pair: str
    direction: Direction

    cex_price: float
    dex_price: float
    spread_bps: float
    size: float

    expected_gross_pnl: float
    expected_fees: float
    expected_net_pnl: float

    score: float
    timestamp: float
    expiry: float

    inventory_ok: bool
    within_limits: bool

    # Optional metadata for debugging / audit trail
    rejection_reasons: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @classmethod
    def create(cls, pair: str, direction: Direction, **kwargs) -> "Signal":
        return cls(
            signal_id=f"{pair.replace('/', '')}_{uuid.uuid4().hex[:8]}",
            pair=pair,
            direction=direction,
            timestamp=time.time(),
            **kwargs,
        )

    def is_valid(self) -> bool:
        return (
            time.time() < self.expiry
            and self.inventory_ok
            and self.within_limits
            and self.expected_net_pnl > 0
            and self.score > 0
        )

    def age_seconds(self) -> float:
        return time.time() - self.timestamp
