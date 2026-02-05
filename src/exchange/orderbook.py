from __future__ import annotations

import argparse
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from config import BINANCE_CONFIG
from exchange.client import ExchangeClient


class OrderBookAnalyzer:
    """
    Analyze order book snapshots for trading decisions.
    """

    def __init__(self, orderbook: dict):
        """
        Initialize with order book from ExchangeClient.fetch_order_book().
        """
        self._orderbook = orderbook
        self._bids: list[tuple[Decimal, Decimal]] = orderbook.get("bids", [])
        self._asks: list[tuple[Decimal, Decimal]] = orderbook.get("asks", [])
        self._best_bid = orderbook.get("best_bid")
        self._best_ask = orderbook.get("best_ask")
        self._mid_price = orderbook.get("mid_price")

    @staticmethod
    def _to_decimal(value: float | str | Decimal) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _levels_for_side(self, side: str) -> list[tuple[Decimal, Decimal]]:
        normalized = side.lower()
        if normalized == "buy":
            return list(self._asks)
        if normalized == "sell":
            return list(self._bids)
        raise ValueError("side must be 'buy' or 'sell'")

    def walk_the_book(
        self,
        side: str,  # "buy" (walk asks) or "sell" (walk bids)
        qty: float,  # Amount of base asset
    ) -> dict:
        """
        Simulate filling `qty` against the order book.

        Returns:
        {
            'avg_price': Decimal,
            'total_cost': Decimal,     # In quote currency
            'slippage_bps': Decimal,   # vs best price
            'levels_consumed': int,    # How deep we went
            'fully_filled': bool,
            'fills': [
                {'price': Decimal, 'qty': Decimal, 'cost': Decimal},
                ...
            ]
        }

        If insufficient liquidity, fully_filled=False and fills show what IS available.
        """
        remaining = self._to_decimal(qty)
        if remaining <= 0:
            raise ValueError("qty must be positive")

        levels = self._levels_for_side(side)
        fills: list[dict] = []
        total_cost = Decimal("0")
        total_filled = Decimal("0")
        levels_consumed = 0

        for price, level_qty in levels:
            if remaining <= 0:
                break
            take_qty = min(remaining, level_qty)
            cost = take_qty * price
            fills.append({"price": price, "qty": take_qty, "cost": cost})
            total_cost += cost
            total_filled += take_qty
            remaining -= take_qty
            levels_consumed += 1

        fully_filled = remaining <= 0
        avg_price = total_cost / total_filled if total_filled > 0 else Decimal("0")
        best_price = None
        if side.lower() == "buy" and self._best_ask:
            best_price = self._best_ask[0]
        if side.lower() == "sell" and self._best_bid:
            best_price = self._best_bid[0]

        slippage_bps = Decimal("0")
        if best_price and best_price != 0 and total_filled > 0:
            if side.lower() == "buy":
                slippage_bps = (avg_price - best_price) / best_price * Decimal("10000")
            else:
                slippage_bps = (best_price - avg_price) / best_price * Decimal("10000")

        return {
            "avg_price": avg_price,
            "total_cost": total_cost,
            "slippage_bps": slippage_bps,
            "levels_consumed": levels_consumed,
            "fully_filled": fully_filled,
            "fills": fills,
        }

    def depth_at_bps(
        self,
        side: str,  # "bid" or "ask"
        bps: float,  # How deep (e.g., 10 = within 10 bps of best)
    ) -> Decimal:
        """
        Total quantity available within `bps` basis points of best price.
        Measures how much you can trade without moving price beyond threshold.
        """
        normalized = side.lower()
        bps_value = self._to_decimal(bps)
        if bps_value < 0:
            raise ValueError("bps must be non-negative")

        if normalized == "bid":
            if not self._best_bid:
                return Decimal("0")
            best_price = self._best_bid[0]
            min_price = best_price * (Decimal("1") - bps_value / Decimal("10000"))
            return sum(
                (qty for price, qty in self._bids if price >= min_price),
                Decimal("0"),
            )

        if normalized == "ask":
            if not self._best_ask:
                return Decimal("0")
            best_price = self._best_ask[0]
            max_price = best_price * (Decimal("1") + bps_value / Decimal("10000"))
            return sum(
                (qty for price, qty in self._asks if price <= max_price),
                Decimal("0"),
            )

        raise ValueError("side must be 'bid' or 'ask'")

    def imbalance(self, levels: int = 10) -> float:
        """
        Order book imbalance ratio.
        Returns [-1.0, +1.0] where:
          +1.0 = all bids (buy pressure)
          -1.0 = all asks (sell pressure)
        """
        if levels <= 0:
            raise ValueError("levels must be positive")
        bid_qty = sum(qty for _, qty in self._bids[:levels])
        ask_qty = sum(qty for _, qty in self._asks[:levels])
        total = bid_qty + ask_qty
        if total == 0:
            return 0.0
        imbalance = (bid_qty - ask_qty) / total
        return float(imbalance)

    def effective_spread(self, qty: float) -> Decimal:
        """
        Effective spread for a round-trip of size `qty`.
        = (avg_ask_fill - avg_bid_fill) / mid_price * 10000 (bps)

        This is the TRUE cost of immediacy for your trade size.
        Different from quoted spread which only considers best levels.
        """
        if not self._mid_price or self._mid_price == 0:
            return Decimal("0")
        buy = self.walk_the_book("buy", qty)
        sell = self.walk_the_book("sell", qty)
        if buy["avg_price"] == 0 or sell["avg_price"] == 0:
            return Decimal("0")
        return (
            (buy["avg_price"] - sell["avg_price"]) / self._mid_price * Decimal("10000")
        )


def _format_decimal(value: Decimal, places: int = 2) -> str:
    quantize_value = Decimal(f"1e-{places}")
    return format(
        value.quantize(quantize_value, rounding=ROUND_HALF_UP), f",.{places}f"
    )


def _format_qty(value: Decimal) -> str:
    return _format_decimal(value, 2).rstrip("0").rstrip(".")


def _format_usd(value: Decimal) -> str:
    return f"${_format_decimal(value, 2)}"


def _summarize_depth(
    levels: list[tuple[Decimal, Decimal]],
    best_price: Decimal,
    bps: Decimal,
    side: str,
) -> tuple[Decimal, Decimal]:
    if bps < 0:
        raise ValueError("bps must be non-negative")
    if best_price == 0:
        return Decimal("0"), Decimal("0")
    if side == "bid":
        min_price = best_price * (Decimal("1") - bps / Decimal("10000"))
        filtered = [(price, qty) for price, qty in levels if price >= min_price]
    else:
        max_price = best_price * (Decimal("1") + bps / Decimal("10000"))
        filtered = [(price, qty) for price, qty in levels if price <= max_price]
    total_qty = sum((qty for _, qty in filtered), Decimal("0"))
    total_notional = sum((price * qty for price, qty in filtered), Decimal("0"))
    return total_qty, total_notional


def _imbalance_label(value: float) -> str:
    if value > 0.1:
        return "slight buy pressure"
    if value < -0.1:
        return "slight sell pressure"
    return "balanced"


def _build_box(lines: list[str]) -> str:
    width = max(len(line) for line in lines) if lines else 0
    top = "+" + "-" * (width + 2) + "+"
    body = [f"| {line.ljust(width)} |" for line in lines]
    return "\n".join([top, *body, top])


def main() -> None:
    parser = argparse.ArgumentParser(description="Order book analyzer")
    parser.add_argument("symbol", help="Trading pair like ETH/USDT")
    parser.add_argument("--depth", type=int, default=20, help="Order book depth")
    parser.add_argument("--depth-bps", type=float, default=10, help="Depth band in bps")
    parser.add_argument(
        "--walk-sizes",
        default="2,10",
        help="Comma-separated sizes for walk-the-book (base asset)",
    )
    parser.add_argument(
        "--imbalance-levels",
        type=int,
        default=10,
        help="Levels used for imbalance calculation",
    )
    args = parser.parse_args()

    client = ExchangeClient(BINANCE_CONFIG)
    orderbook = client.fetch_order_book(args.symbol, limit=args.depth)
    analyzer = OrderBookAnalyzer(orderbook)

    best_bid = orderbook.get("best_bid")
    best_ask = orderbook.get("best_ask")
    mid_price = orderbook.get("mid_price") or Decimal("0")
    spread_bps = orderbook.get("spread_bps") or Decimal("0")
    spread_abs = (best_ask[0] - best_bid[0]) if best_bid and best_ask else Decimal("0")

    timestamp = orderbook.get("timestamp")
    if timestamp:
        ts_value = datetime.utcfromtimestamp(timestamp / 1000.0).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    else:
        ts_value = "N/A"

    depth_bps = Decimal(str(args.depth_bps))
    bid_qty, bid_notional = _summarize_depth(
        orderbook.get("bids", []),
        best_bid[0] if best_bid else Decimal("0"),
        depth_bps,
        "bid",
    )
    ask_qty, ask_notional = _summarize_depth(
        orderbook.get("asks", []),
        best_ask[0] if best_ask else Decimal("0"),
        depth_bps,
        "ask",
    )

    imbalance = analyzer.imbalance(levels=args.imbalance_levels)
    imbalance_text = _imbalance_label(imbalance)

    walk_sizes = [
        Decimal(size.strip()) for size in args.walk_sizes.split(",") if size.strip()
    ]

    lines: list[str] = []
    lines.append(f" {args.symbol} Order Book Analysis")
    lines.append(f" Timestamp: {ts_value}")
    lines.append("-" * 50)
    if best_bid:
        lines.append(
            f" Best Bid:    {_format_usd(best_bid[0])} x "
            f"{_format_qty(best_bid[1])} {args.symbol.split('/')[0]}"
        )
    else:
        lines.append(" Best Bid:    N/A")
    if best_ask:
        lines.append(
            f" Best Ask:    {_format_usd(best_ask[0])} x "
            f"{_format_qty(best_ask[1])} {args.symbol.split('/')[0]}"
        )
    else:
        lines.append(" Best Ask:    N/A")
    lines.append(f" Mid Price:   {_format_usd(mid_price)}")
    lines.append(
        f" Spread:      {_format_usd(spread_abs)} "
        f"({_format_decimal(spread_bps, 2)} bps)"
    )
    lines.append("-" * 50)
    lines.append(f" Depth (within {_format_decimal(depth_bps, 2)} bps):")
    lines.append(
        f"   Bids: {_format_qty(bid_qty)} {args.symbol.split('/')[0]} "
        f"({_format_usd(bid_notional)})"
    )
    lines.append(
        f"   Asks: {_format_qty(ask_qty)} {args.symbol.split('/')[0]} "
        f"({_format_usd(ask_notional)})"
    )
    lines.append(f" Imbalance: {imbalance:+.2f} ({imbalance_text})")
    lines.append("-" * 50)
    for size in walk_sizes:
        impact = analyzer.walk_the_book("buy", float(size))
        lines.append(
            f" Walk-the-book ({_format_qty(size)} {args.symbol.split('/')[0]} buy):"
        )
        lines.append(f"   Avg price:  {_format_usd(impact['avg_price'])}")
        lines.append(f"   Slippage:   {_format_decimal(impact['slippage_bps'], 2)} bps")
        lines.append(f"   Levels:     {impact['levels_consumed']}")
    lines.append("-" * 50)
    if walk_sizes:
        effective = analyzer.effective_spread(float(walk_sizes[0]))
        lines.append(
            f" Effective spread ({_format_qty(walk_sizes[0])} "
            f"{args.symbol.split('/')[0]} round-trip): "
            f"{_format_decimal(effective, 2)} bps"
        )

    print(_build_box(lines))


if __name__ == "__main__":
    main()
