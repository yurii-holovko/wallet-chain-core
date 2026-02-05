from decimal import Decimal

from exchange.orderbook import OrderBookAnalyzer


def _sample_orderbook() -> dict:
    bids = [
        (Decimal("100"), Decimal("1")),
        (Decimal("99"), Decimal("2")),
        (Decimal("98"), Decimal("3")),
    ]
    asks = [
        (Decimal("101"), Decimal("1.5")),
        (Decimal("102"), Decimal("2")),
        (Decimal("103"), Decimal("3")),
    ]
    best_bid = bids[0]
    best_ask = asks[0]
    mid = (best_bid[0] + best_ask[0]) / Decimal("2")
    spread_bps = (best_ask[0] - best_bid[0]) / mid * Decimal("10000")
    return {
        "symbol": "TEST/USDT",
        "timestamp": 0,
        "bids": bids,
        "asks": asks,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread_bps": spread_bps,
    }


def test_walk_the_book_exact_fill():
    analyzer = OrderBookAnalyzer(_sample_orderbook())
    result = analyzer.walk_the_book("buy", 1.5)
    assert result["fully_filled"] is True
    assert result["levels_consumed"] == 1
    assert result["avg_price"] == Decimal("101")
    assert result["total_cost"] == Decimal("151.5")


def test_walk_the_book_multiple_levels():
    analyzer = OrderBookAnalyzer(_sample_orderbook())
    result = analyzer.walk_the_book("buy", 2.5)
    assert result["fully_filled"] is True
    assert result["levels_consumed"] == 2
    expected_cost = Decimal("1.5") * Decimal("101") + Decimal("1") * Decimal("102")
    expected_avg = expected_cost / Decimal("2.5")
    assert result["avg_price"] == expected_avg


def test_walk_the_book_insufficient_liquidity():
    analyzer = OrderBookAnalyzer(_sample_orderbook())
    result = analyzer.walk_the_book("buy", 20)
    assert result["fully_filled"] is False
    assert result["levels_consumed"] == 3


def test_depth_at_bps_correct():
    analyzer = OrderBookAnalyzer(_sample_orderbook())
    depth = analyzer.depth_at_bps("bid", 200)
    assert depth == Decimal("6")


def test_imbalance_range():
    analyzer = OrderBookAnalyzer(_sample_orderbook())
    imbalance = analyzer.imbalance(levels=3)
    assert -1.0 <= imbalance <= 1.0


def test_effective_spread_greater_than_quoted():
    analyzer = OrderBookAnalyzer(_sample_orderbook())
    quoted_bps = _sample_orderbook()["spread_bps"]
    effective = analyzer.effective_spread(2)
    assert effective >= quoted_bps
