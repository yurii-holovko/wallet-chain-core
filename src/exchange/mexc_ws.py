from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, List

import websockets

from config import get_env

logger = logging.getLogger(__name__)


@dataclass
class MexcOrderBookView:
    """Top-of-book snapshot used for spread calculations."""

    symbol: str
    bids: List[List[float]]
    asks: List[List[float]]
    timestamp_ms: int

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None


class MexcOrderBookWebSocket:
    """
    Minimal MEXC spot order book stream.

    This maintains only the current top-of-book snapshot per message and is
    designed for micro-arbitrage spread detection where we primarily care
    about the best bid/ask rather than a full depth book.
    """

    def __init__(self, symbol: str, depth: int = 5) -> None:
        # MEXC uses e.g. "arb_usdt" for ARB/USDT
        self._symbol = symbol
        self._symbol_stream = symbol.replace("/", "").lower()
        self._depth = depth
        default_ws = "wss://wbs.mexc.com/ws"
        self._ws_url = get_env("MEXC_WS_URL", default_ws) or default_ws

    async def stream(self) -> AsyncIterator[MexcOrderBookView]:
        """
        Yield ``MexcOrderBookView`` snapshots until the connection closes.

        Callers are responsible for reconnect/backoff policies.
        """
        subscribe_payload = {
            "method": "SUBSCRIPTION",
            "params": [
                f"spot@public.limit.depth.v3.api@{self._symbol_stream}@{self._depth}"
            ],
        }

        async with websockets.connect(
            self._ws_url, ping_interval=None, ping_timeout=None
        ) as ws:
            await ws.send(json.dumps(subscribe_payload))
            logger.info("MEXC WS subscribed to %s depth=%d", self._symbol, self._depth)

            async for raw in ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("MEXC WS non-JSON message: %r", raw)
                    continue

                # Depth updates carry bids/asks under "d"
                payload = message.get("d")
                if not isinstance(payload, dict):
                    continue

                bids_raw = payload.get("bids") or []
                asks_raw = payload.get("asks") or []
                bids = [[float(p), float(q)] for p, q in bids_raw]
                asks = [[float(p), float(q)] for p, q in asks_raw]
                ts = int(message.get("t") or 0)

                # Keep bids sorted descending, asks ascending
                bids.sort(key=lambda item: item[0], reverse=True)
                asks.sort(key=lambda item: item[0])

                yield MexcOrderBookView(
                    symbol=self._symbol,
                    bids=bids,
                    asks=asks,
                    timestamp_ms=ts,
                )
