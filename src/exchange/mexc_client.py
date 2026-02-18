from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

import requests

from config import get_env

logger = logging.getLogger(__name__)


class MexcApiError(RuntimeError):
    """Raised when MEXC responds with an error payload or HTTP status."""


@dataclass
class MexcOrderStatus:
    """Normalized view of a MEXC order used by higher-level components."""

    order_id: str
    symbol: str
    side: str
    status: str
    price: float
    orig_qty: float
    executed_qty: float

    @property
    def remaining_qty(self) -> float:
        return max(self.orig_qty - self.executed_qty, 0.0)

    @property
    def is_filled(self) -> bool:
        return self.status == "FILLED"

    @property
    def is_active(self) -> bool:
        return self.status in {"NEW", "PARTIALLY_FILLED"}


class MexcClient:
    """
    Lightweight MEXC REST client focused on:

      * Top-of-book order book snapshots
      * Post-only style limit orders for 0% maker fee
      * Basic balance / withdrawal data for capital management

    This intentionally mirrors the ergonomics of ``ExchangeClient`` but is kept
    separate so we can tune it specifically for MEXC + micro-arbitrage flows.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str | None = None,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        # Strip whitespace from API credentials to avoid signature errors
        raw_key = api_key or get_env("MEXC_API_KEY", required=True)
        raw_secret = api_secret or get_env("MEXC_API_SECRET", required=True)
        self._api_key = raw_key.strip() if isinstance(raw_key, str) else raw_key
        self._api_secret = (
            raw_secret.strip() if isinstance(raw_secret, str) else raw_secret
        )
        self._base_url = base_url or get_env("MEXC_BASE_URL", "https://api.mexc.com")
        self._timeout = timeout_seconds
        self._session = requests.Session()
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    # ── low-level helpers ──────────────────────────────────────────

    def _sign(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC-SHA256 signature for MEXC API request.

        MEXC signature format:
        - Sort parameters by key (alphabetically)
        - Exclude 'signature' parameter from signing
        - Convert all values to strings
        - Join as 'key=value&key=value'
        - HMAC-SHA256 with secret key
        - Return lowercase hex digest
        """
        query_parts = []
        for k, v in sorted(params.items()):
            if k == "signature":
                continue  # Exclude signature from the string being signed
            # Convert all values to strings (MEXC expects this)
            # Ensure None values are handled
            if v is None:
                continue
            query_parts.append(f"{k}={v}")
        query = "&".join(query_parts)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        params = dict(params or {})

        if signed:
            # Generate timestamp in milliseconds (epoch * 1000) as Long/int
            # This must match exactly what MEXC expects
            now = time.time()
            req_time = int(now * 1000)
            params["timestamp"] = req_time
            params.setdefault("recvWindow", 5_000)

            # Generate signature BEFORE adding it to params
            # Signature is computed from query string WITHOUT the signature field
            signature = self._sign(params)
            params["signature"] = signature

            # For GET requests, manually construct query string to ensure exact match
            # between what we sign and what we send
            query_for_sign = None
            if method == "GET":
                # Build query string manually to match signature exactly
                query_parts = []
                for k, v in sorted(params.items()):
                    if v is None:
                        continue
                    query_parts.append(f"{k}={v}")
                query_string = "&".join(query_parts)
                url = f"{url}?{query_string}"
                params = {}  # Clear params since we're using query string in URL
                query_for_sign = query_string.split("&signature=")[0]
            else:
                # For POST/DELETE, build query string for logging
                query_for_sign = "&".join(
                    f"{k}={v}"
                    for k, v in sorted(params.items())
                    if k != "signature" and v is not None
                )

            # Debug logging for signature generation
            sig_short = signature[:16] + "..." if len(signature) > 16 else signature
            logger.debug(
                "MEXC signed: endpoint=%s method=%s query=%s ts=%d sig=%s",
                endpoint,
                method,
                query_for_sign,
                req_time,
                sig_short,
            )

        headers = {
            "X-MEXC-APIKEY": self._api_key,
        }

        attempt = 0
        while True:
            attempt += 1
            try:
                if method == "GET":
                    resp = self._session.get(
                        url, params=params, headers=headers, timeout=self._timeout
                    )
                elif method == "DELETE":
                    resp = self._session.delete(
                        url, params=params, headers=headers, timeout=self._timeout
                    )
                else:
                    # For MEXC spot we send params in query as well
                    resp = self._session.post(
                        url, params=params, headers=headers, timeout=self._timeout
                    )
            except requests.RequestException as exc:
                if attempt > self._max_retries:
                    raise MexcApiError(f"Network error talking to MEXC: {exc}") from exc
                sleep_for = self._backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "MEXC request %s %s failed (%s), retrying in %.1fs",
                    method,
                    endpoint,
                    exc.__class__.__name__,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue

            if resp.status_code >= 400:
                # MEXC errors are JSON when possible
                try:
                    payload = resp.json()
                except Exception:
                    payload = {"message": resp.text}
                message = payload.get("msg") or payload.get("message") or str(payload)
                code = payload.get("code")
                # Log detailed error info for signature errors
                if resp.status_code == 400 and (
                    "signature" in str(message).lower() or code in (602, 700002)
                ):
                    query_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
                    logger.error(
                        "MEXC signature error: endpoint=%s status=%d code=%s "
                        "msg=%s query=%s",
                        endpoint,
                        resp.status_code,
                        code,
                        message,
                        query_str,
                    )
                raise MexcApiError(
                    f"MEXC HTTP {resp.status_code} for {endpoint}: {message}"
                )

            try:
                data = resp.json()
            except ValueError as exc:  # pragma: no cover - defensive
                raise MexcApiError(
                    f"Invalid JSON from MEXC for {endpoint}: {resp.text!r}"
                ) from exc
            return data

    @staticmethod
    def _to_decimal(value: Any, default: str = "0") -> Decimal:
        if value is None:
            return Decimal(default)
        return Decimal(str(value))

    # ── public API: market data ────────────────────────────────────

    def get_order_book(
        self, symbol: str, limit: int = 5
    ) -> Dict[str, List[List[float]]]:
        """
        Fetch L2 order book snapshot for *symbol*.

        Returns a dict with numeric ``bids`` / ``asks`` suitable for spread calcs:

            {
                "bids": [[price, qty], ...],
                "asks": [[price, qty], ...],
            }
        """
        data = self._request(
            "GET",
            "/api/v3/depth",
            params={"symbol": symbol, "limit": limit},
            signed=False,
        )
        return {
            "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
            "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
        }

    # ── public API: account / balances ─────────────────────────────

    def get_account(self) -> Dict[str, Any]:
        """Return the raw account payload from MEXC."""
        return self._request("GET", "/api/v3/account", signed=True)

    def get_balance(self, asset: str) -> float:
        """Return free balance for *asset* (e.g. ``USDT``)."""
        account = self.get_account()
        for entry in account.get("balances", []):
            if entry.get("asset") == asset:
                return float(entry.get("free", 0.0))
        return 0.0

    # ── public API: trading ────────────────────────────────────────

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool = True,
    ) -> MexcOrderStatus:
        """
        Place a LIMIT order.

        When ``post_only`` is True we enforce maker-only behaviour at the
        application level: if the order is immediately executed (takes
        liquidity), we cancel it and raise ``MexcApiError``.
        """
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "quantity": quantity,
            "price": price,
            "timeInForce": "GTC",
        }
        raw = self._request("POST", "/api/v3/order", params=params, signed=True)

        status = self._normalize_order(raw)

        if post_only and status.executed_qty > 0:
            # Order would have taken liquidity; cancel and surface error.
            logger.warning(
                "MEXC post-only order for %s filled immediately; cancelling", symbol
            )
            try:
                self.cancel_order(symbol, status.order_id)
            except Exception:
                logger.exception("Failed to cancel immediately-filled MEXC order")
            raise MexcApiError("Post-only order would take liquidity; cancelled")

        return status

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> MexcOrderStatus:
        """
        Place a MARKET order.

        This should only be used for emergency unwinds; market orders incur
        taker fees and are not suitable for micro-arbitrage.
        """
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
        }
        raw = self._request("POST", "/api/v3/order", params=params, signed=True)
        return self._normalize_order(raw)

    def get_order_status(self, symbol: str, order_id: str) -> MexcOrderStatus:
        raw = self._request(
            "GET",
            "/api/v3/order",
            params={"symbol": symbol, "orderId": order_id},
            signed=True,
        )
        return self._normalize_order(raw)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self._request(
                "DELETE",
                "/api/v3/order",
                params={"symbol": symbol, "orderId": order_id},
                signed=True,
            )
            return True
        except MexcApiError:
            logger.exception("Failed to cancel MEXC order %s %s", symbol, order_id)
            return False

    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        try:
            self._request(
                "DELETE",
                "/api/v3/openOrders",
                params=params,
                signed=True,
            )
            return True
        except MexcApiError:
            logger.exception("Failed to cancel all MEXC orders for %s", symbol or "*")
            return False

    # ── public API: withdrawals (for rebalancing) ──────────────────

    def withdraw(
        self,
        coin: str,
        amount: float,
        address: str,
        network: str = "Arbitrum One",
    ) -> str:
        """
        Submit a withdrawal request.

        This is only used by higher-level capital management once batching
        thresholds are met. It is guarded by configuration flags at the
        call site — this method assumes the caller has already decided that
        a withdrawal is safe.
        """
        enable_withdrawal = get_env("MEXC_ENABLE_WITHDRAWAL", "false") or "false"
        if enable_withdrawal.lower() not in {"1", "true", "yes", "y"}:
            raise MexcApiError("MEXC withdrawals are disabled by configuration")

        params: Dict[str, Any] = {
            "coin": coin,
            "amount": amount,
            "address": address,
            "network": network,
        }
        data = self._request(
            "POST",
            "/api/v3/capital/withdraw/apply",
            params=params,
            signed=True,
        )
        # MEXC returns an id for the withdrawal request
        withdrawal_id = str(data.get("id", ""))
        if not withdrawal_id:
            raise MexcApiError(f"Unexpected withdraw response: {data}")
        return withdrawal_id

    # ── normalization helpers ──────────────────────────────────────

    def _normalize_order(self, data: Dict[str, Any]) -> MexcOrderStatus:
        price = float(data.get("price", 0.0))
        orig_qty = float(data.get("origQty", data.get("quantity", 0.0)))
        executed_qty = float(data.get("executedQty", 0.0))
        return MexcOrderStatus(
            order_id=str(data.get("orderId")),
            symbol=str(data.get("symbol")),
            side=str(data.get("side", "")).upper(),
            status=str(data.get("status", "")),
            price=price,
            orig_qty=orig_qty,
            executed_qty=executed_qty,
        )
