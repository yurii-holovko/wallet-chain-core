# exchange/client.py

from __future__ import annotations

import logging
import time
from collections import deque
from decimal import Decimal
from typing import Any, Callable, cast

import ccxt


class RateLimiter:
    def __init__(
        self,
        max_weight: int,
        window_seconds: float = 60.0,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._max_weight = max_weight
        self._window_seconds = window_seconds
        self._events: deque[tuple[float, int]] = deque()
        self._time_fn = time_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep

    def acquire(self, weight: int) -> None:
        while True:
            now = self._time_fn()
            self._expire_old(now)
            current_weight = sum(event_weight for _, event_weight in self._events)
            if current_weight + weight <= self._max_weight:
                self._events.append((now, weight))
                return
            sleep_for = (self._events[0][0] + self._window_seconds) - now
            if sleep_for > 0:
                self._sleep_fn(sleep_for)

    def _expire_old(self, now: float) -> None:
        while self._events and (now - self._events[0][0]) >= self._window_seconds:
            self._events.popleft()


class ExchangeClient:
    """
    Wrapper around ccxt for Binance testnet.
    Handles rate limiting, error handling, and response normalization.
    """

    _RETRYABLE_ERRORS = (
        ccxt.DDoSProtection,
        ccxt.ExchangeNotAvailable,
        ccxt.InvalidNonce,
        ccxt.NetworkError,
        ccxt.RateLimitExceeded,
        ccxt.RequestTimeout,
    )

    def __init__(self, config: dict[str, Any]):
        """
        Initialize with config dict containing apiKey, secret, sandbox flag.
        Validates connection on init (fetch server time or status).
        """
        self._logger = logging.getLogger(__name__)
        self._exchange = ccxt.binance(cast(Any, config))
        if config.get("sandbox"):
            self._exchange.set_sandbox_mode(True)
        self._max_retries = int(config.get("max_retries", 3))
        self._backoff_base = float(config.get("backoff_base", 0.5))
        max_weight = int(config.get("max_weight_per_minute", 1200))
        window_seconds = float(config.get("weight_window_seconds", 60.0))
        self._rate_limiter = RateLimiter(max_weight, window_seconds)
        self._sync_server_time()
        self._validate_connection()

    def _sync_server_time(self) -> None:
        """
        Measure the offset between local clock and Binance server clock,
        then tell ccxt to adjust all request timestamps accordingly.

        Binance rejects signed requests when the timestamp differs by more
        than ``recvWindow`` ms (error code -1021 / InvalidNonce).
        """
        try:
            server_ts = self._exchange.fetch_time()  # epoch-ms from Binance
            local_ts = int(time.time() * 1000)
            offset_ms = server_ts - local_ts
            self._exchange.options["adjustForTimeDifference"] = True
            self._exchange.options["timeDifference"] = offset_ms
            self._logger.info("Clock sync: server-local offset = %+d ms", offset_ms)
        except Exception as exc:
            self._logger.warning(
                "Clock sync failed, continuing with local time: %s", exc
            )

    def _validate_connection(self) -> None:
        try:
            if self._exchange.has.get("fetchTime"):
                self._request_with_retries(self._exchange.fetch_time)
            elif self._exchange.has.get("fetchStatus"):
                self._request_with_retries(self._exchange.fetch_status)
            else:
                self._request_with_retries(self._exchange.load_markets)
        except Exception as exc:
            raise RuntimeError("Failed to initialize exchange client") from exc

    def _request_with_retries(
        self, func: Callable[..., Any], *args: Any, weight: int = 1, **kwargs: Any
    ) -> Any:
        attempt = 0
        request_name = getattr(func, "__name__", str(func))
        while True:
            try:
                self._rate_limiter.acquire(weight)
                self._logger.debug(
                    "ccxt request: %s args=%s kwargs=%s",
                    request_name,
                    self._summarize_request(args),
                    self._summarize_request(kwargs),
                )
                result = func(*args, **kwargs)
                self._logger.debug(
                    "ccxt response: %s summary=%s",
                    request_name,
                    self._summarize_response(result),
                )
                return result
            except self._RETRYABLE_ERRORS as exc:
                attempt += 1
                if attempt > self._max_retries:
                    raise self._wrap_ccxt_error(exc) from exc
                sleep_for = self._backoff_base * (2 ** (attempt - 1))
                self._logger.warning(
                    "ccxt retry %s/%s after %s: %s",
                    attempt,
                    self._max_retries,
                    sleep_for,
                    exc.__class__.__name__,
                )
                # Re-sync clock on timestamp drift errors
                if isinstance(exc, ccxt.InvalidNonce):
                    self._sync_server_time()
                time.sleep(sleep_for)
            except ccxt.PermissionDenied as exc:
                raise RuntimeError("Permission denied") from exc
            except ccxt.AuthenticationError as exc:
                raise RuntimeError("Authentication failed") from exc
            except ccxt.ExchangeError as exc:
                raise RuntimeError("Exchange error") from exc

    def _wrap_ccxt_error(self, exc: Exception) -> RuntimeError:
        if isinstance(exc, ccxt.RateLimitExceeded):
            return RuntimeError("Rate limit exceeded")
        if isinstance(exc, (ccxt.NetworkError, ccxt.RequestTimeout)):
            return RuntimeError("Network error")
        if isinstance(exc, ccxt.ExchangeNotAvailable):
            return RuntimeError("Exchange not available")
        if isinstance(exc, ccxt.DDoSProtection):
            return RuntimeError("Exchange under protection")
        return RuntimeError("Request failed")

    @staticmethod
    def _summarize_request(value: Any) -> Any:
        if isinstance(value, dict):
            return {"keys": list(value.keys())}
        if isinstance(value, (list, tuple)):
            return {"len": len(value)}
        return value

    @staticmethod
    def _summarize_response(value: Any) -> Any:
        if isinstance(value, dict):
            return {"keys": list(value.keys())}
        if isinstance(value, (list, tuple)):
            return {"len": len(value)}
        return type(value).__name__

    @staticmethod
    def _to_decimal(value: Any, default: str = "0") -> Decimal:
        if value is None:
            return Decimal(default)
        return Decimal(str(value))

    def _normalize_orderbook(self, raw: dict, symbol: str) -> dict:
        bids = [
            (self._to_decimal(price), self._to_decimal(qty))
            for price, qty in (raw.get("bids") or [])
        ]
        asks = [
            (self._to_decimal(price), self._to_decimal(qty))
            for price, qty in (raw.get("asks") or [])
        ]
        bids.sort(key=lambda item: item[0], reverse=True)
        asks.sort(key=lambda item: item[0])
        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None
        mid_price = None
        spread_bps = None
        last_update_id = raw.get("nonce") or raw.get("lastUpdateId")
        if best_bid and best_ask:
            mid_price = (best_bid[0] + best_ask[0]) / Decimal("2")
            if mid_price != 0:
                spread_bps = (best_ask[0] - best_bid[0]) / mid_price * Decimal("10000")
        timestamp = raw.get("timestamp")
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "last_update_id": last_update_id,
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread_bps": spread_bps,
        }

    def _normalize_order(self, order: dict) -> dict:
        amount_requested = self._to_decimal(order.get("amount"))
        amount_filled = self._to_decimal(order.get("filled"))
        avg_price = order.get("average")
        if avg_price is None:
            cost = order.get("cost")
            if cost is not None and amount_filled != 0:
                avg_price = self._to_decimal(cost) / amount_filled
        avg_price = self._to_decimal(avg_price)
        fee = order.get("fee") or {}
        fee_cost = self._to_decimal(fee.get("cost"))
        fee_asset = fee.get("currency")
        status = self._normalize_status(amount_requested, amount_filled)
        return {
            "id": order.get("id"),
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "type": order.get("type"),
            "time_in_force": (order.get("timeInForce") or order.get("time_in_force")),
            "amount_requested": amount_requested,
            "amount_filled": amount_filled,
            "avg_fill_price": avg_price,
            "fee": fee_cost,
            "fee_asset": fee_asset,
            "status": status,
            "timestamp": order.get("timestamp"),
        }

    @staticmethod
    def _normalize_status(amount_requested: Decimal, amount_filled: Decimal) -> str:
        if amount_requested > 0 and amount_filled >= amount_requested:
            return "filled"
        if amount_filled > 0:
            return "partially_filled"
        return "expired"

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """
        Fetch L2 order book snapshot.
        """
        raw = self._request_with_retries(
            self._exchange.fetch_order_book, symbol, limit, weight=5
        )
        return self._normalize_orderbook(raw, symbol)

    def fetch_balance(self) -> dict[str, dict]:
        """
        Fetch account balances.
        """
        raw = self._request_with_retries(self._exchange.fetch_balance, weight=10)
        free = raw.get("free") or {}
        used = raw.get("used") or {}
        total = raw.get("total") or {}
        normalized: dict[str, dict] = {}
        for asset, free_value in free.items():
            free_dec = self._to_decimal(free_value)
            locked_dec = self._to_decimal(used.get(asset))
            total_value = total.get(asset)
            total_dec = (
                self._to_decimal(total_value)
                if total_value is not None
                else free_dec + locked_dec
            )
            if total_dec == 0:
                continue
            normalized[asset] = {
                "free": free_dec,
                "locked": locked_dec,
                "total": total_dec,
            }
        return normalized

    def create_limit_ioc_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> dict:
        """
        Place a LIMIT IOC (Immediate Or Cancel) order.
        """
        order = self._request_with_retries(
            self._exchange.create_order,
            symbol,
            "limit",
            side,
            amount,
            price,
            {"timeInForce": "IOC"},
            weight=1,
        )
        return self._normalize_order(order)

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """
        Place a market order. Same return format as create_limit_ioc_order.
        """
        order = self._request_with_retries(
            self._exchange.create_order, symbol, "market", side, amount, weight=1
        )
        return self._normalize_order(order)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an open order. Returns order status after cancel."""
        order = self._request_with_retries(
            self._exchange.cancel_order, order_id, symbol, weight=1
        )
        return self._normalize_order(order)

    def fetch_order_status(self, order_id: str, symbol: str) -> dict:
        """Check current status of an order."""
        order = self._request_with_retries(
            self._exchange.fetch_order, order_id, symbol, weight=1
        )
        return self._normalize_order(order)

    def get_trading_fees(self, symbol: str) -> dict:
        """
        Returns fee structure:
        {'maker': Decimal('0.001'), 'taker': Decimal('0.001')}
        """
        fee_info = None
        if self._exchange.has.get("fetchTradingFee"):
            fee_info = self._request_with_retries(
                self._exchange.fetch_trading_fee, symbol, weight=5
            )
        elif self._exchange.has.get("fetchTradingFees"):
            fees = self._request_with_retries(
                self._exchange.fetch_trading_fees, weight=5
            )
            fee_info = fees.get(symbol)
        maker = self._to_decimal((fee_info or {}).get("maker"))
        taker = self._to_decimal((fee_info or {}).get("taker"))
        return {"maker": maker, "taker": taker}
