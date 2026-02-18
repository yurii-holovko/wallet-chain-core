from __future__ import annotations

import math
from dataclasses import dataclass

# Tick spacing per fee tier (matches Uniswap V3 conventions)
TICK_SPACING: dict[int, int] = {
    100: 1,  # 0.01%
    500: 10,  # 0.05%
    3_000: 60,  # 0.3%
    10_000: 200,  # 1%
}


@dataclass(frozen=True)
class TickRange:
    tick_lower: int
    tick_upper: int


def nearest_usable_tick(tick: int, fee_tier: int) -> int:
    """
    Round an arbitrary tick to the nearest usable tick for the given fee tier.
    """
    spacing = TICK_SPACING.get(fee_tier)
    if spacing is None:
        raise ValueError(f"Unsupported fee tier for V3 tick spacing: {fee_tier}")
    # Floor towards negative infinity to stay consistent with solidity math
    return math.floor(tick / spacing) * spacing


def price_to_tick(price: float) -> int:
    """
    Convert a raw price (token1/token0) to the closest Uniswap V3 tick.

    This ignores token decimals; callers should adjust the price into the
    pool's internal representation before calling when necessary.
    """
    if price <= 0:
        raise ValueError("price must be positive")
    return int(math.floor(math.log(price) / math.log(1.0001)))


def tick_to_price(
    tick: int,
    token0_decimals: int = 6,
    token1_decimals: int = 18,
) -> float:
    """
    Convert a tick index into a human-readable price.

    The raw pool price is 1.0001 ** tick, then adjusted for token decimals.
    """
    raw_price = 1.0001**tick
    decimal_adjustment = 10 ** (token1_decimals - token0_decimals)
    return raw_price / decimal_adjustment


def single_tick_range(
    current_tick: int,
    fee_tier: int,
    direction: str = "above",
) -> TickRange:
    """
    Compute a single-tick range for a range order.

    direction:
      * 'above' → [current, current + spacing]  (limit sell at a higher price)
      * 'below' → [current - spacing, current]  (limit buy at a lower price)
    """
    spacing = TICK_SPACING.get(fee_tier)
    if spacing is None:
        raise ValueError(f"Unsupported fee tier for V3 tick spacing: {fee_tier}")

    current_usable = nearest_usable_tick(current_tick, fee_tier)
    direction_norm = direction.lower()

    if direction_norm == "above":
        tick_lower = current_usable
        tick_upper = current_usable + spacing
    elif direction_norm == "below":
        tick_lower = current_usable - spacing
        tick_upper = current_usable
    else:
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")

    return TickRange(tick_lower=tick_lower, tick_upper=tick_upper)
