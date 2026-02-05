from __future__ import annotations

from decimal import Decimal

import ccxt
import pytest

from exchange.client import ExchangeClient, RateLimiter


class FakeExchange:
    def __init__(self):
        self.has = {"fetchTime": True, "fetchTradingFee": True}
        self.sandbox = False

    def set_sandbox_mode(self, enabled: bool) -> None:
        self.sandbox = enabled

    def fetch_time(self) -> int:
        return 123

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        return {
            "timestamp": 111,
            "bids": [[100, 1], [101, 0.5], [99, 2]],
            "asks": [[102, 1], [101.5, 0.2], [103, 1.1]],
        }

    def fetch_balance(self) -> dict:
        return {
            "free": {"ETH": "1", "USDT": "0"},
            "used": {"ETH": "0.1", "USDT": "0"},
            "total": {"ETH": "1.1", "USDT": "0"},
        }

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        return {
            "id": "abc",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "timeInForce": (params or {}).get("timeInForce"),
            "amount": amount,
            "filled": 0.5,
            "average": 100,
            "fee": {"cost": "0.1", "currency": "USDT"},
            "timestamp": 999,
        }

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        return {
            "id": order_id,
            "symbol": symbol,
            "side": "buy",
            "type": "limit",
            "amount": 1,
            "filled": 0,
            "average": None,
            "fee": None,
            "timestamp": 1000,
        }

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        return {
            "id": order_id,
            "symbol": symbol,
            "side": "buy",
            "type": "limit",
            "amount": 1,
            "filled": 1,
            "average": 105,
            "fee": {"cost": "0.05", "currency": "USDT"},
            "timestamp": 1001,
        }

    def fetch_trading_fee(self, symbol: str) -> dict:
        return {"maker": "0.001", "taker": "0.002"}


@pytest.fixture()
def client(monkeypatch) -> ExchangeClient:
    monkeypatch.setattr(ccxt, "binance", lambda config: FakeExchange())
    return ExchangeClient({"sandbox": True, "max_weight_per_minute": 50})


def test_fetch_order_book_structure(client: ExchangeClient) -> None:
    orderbook = client.fetch_order_book("ETH/USDT", limit=5)
    assert orderbook["symbol"] == "ETH/USDT"
    assert orderbook["timestamp"] == 111
    assert orderbook["bids"]
    assert orderbook["asks"]
    assert orderbook["best_bid"]
    assert orderbook["best_ask"]
    assert isinstance(orderbook["mid_price"], Decimal)
    assert isinstance(orderbook["spread_bps"], Decimal)


def test_order_book_bids_descending(client: ExchangeClient) -> None:
    orderbook = client.fetch_order_book("ETH/USDT", limit=5)
    bid_prices = [price for price, _ in orderbook["bids"]]
    assert bid_prices == sorted(bid_prices, reverse=True)


def test_order_book_asks_ascending(client: ExchangeClient) -> None:
    orderbook = client.fetch_order_book("ETH/USDT", limit=5)
    ask_prices = [price for price, _ in orderbook["asks"]]
    assert ask_prices == sorted(ask_prices)


def test_spread_calculation(client: ExchangeClient) -> None:
    orderbook = client.fetch_order_book("ETH/USDT", limit=5)
    best_bid = orderbook["best_bid"][0]
    best_ask = orderbook["best_ask"][0]
    mid = (best_bid + best_ask) / Decimal("2")
    expected = (best_ask - best_bid) / mid * Decimal("10000")
    assert orderbook["spread_bps"] == expected


def test_fetch_balance_filters_zeros(client: ExchangeClient) -> None:
    balances = client.fetch_balance()
    assert "ETH" in balances
    assert "USDT" not in balances
    assert balances["ETH"]["total"] == Decimal("1.1")


def test_limit_ioc_returns_fill_info(client: ExchangeClient) -> None:
    order = client.create_limit_ioc_order("ETH/USDT", "buy", 1, 100)
    assert order["amount_requested"] == Decimal("1")
    assert order["amount_filled"] == Decimal("0.5")
    assert order["avg_fill_price"] == Decimal("100")
    assert order["fee"] == Decimal("0.1")
    assert order["fee_asset"] == "USDT"


def test_rate_limiter_blocks_when_exhausted() -> None:
    current = [0.0]
    sleeps: list[float] = []

    def time_fn() -> float:
        return current[0]

    def sleep_fn(seconds: float) -> None:
        sleeps.append(seconds)
        current[0] += seconds

    limiter = RateLimiter(
        max_weight=3, window_seconds=1.0, time_fn=time_fn, sleep_fn=sleep_fn
    )
    limiter.acquire(2)
    limiter.acquire(2)
    assert sleeps and sleeps[0] >= 1.0
