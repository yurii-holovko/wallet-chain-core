from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

import requests
import websockets
from eth_abi import decode

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

    def __init__(
        self,
        ws_url: str,
        callback: Callable[[ParsedSwap], object],
        quote_fn: Callable[[ParsedSwap], Awaitable[int] | int] | None = None,
        quote_timeout: float = 1.0,
        subscription_method: str = "newPendingTransactions",
        subscription_params: list | dict | None = None,
        rpc_url: str | None = None,
    ):
        """
        callback receives ParsedSwap objects for each detected swap.
        """
        self.ws_url = ws_url
        self.callback = callback
        self.quote_fn = quote_fn
        self.quote_timeout = quote_timeout
        self.subscription_method = subscription_method
        self.subscription_params = subscription_params
        self.rpc_url = rpc_url or _ws_to_http(ws_url)
        self._max_inflight = 50

    async def start(self):
        """Start monitoring pending transactions."""
        async with websockets.connect(self.ws_url, ping_interval=20) as ws:
            sub_id = await self._subscribe(ws)
            inflight: set[asyncio.Task] = set()

            while True:
                message = await ws.recv()
                data = json.loads(message)
                params = data.get("params", {})
                if params.get("subscription") != sub_id:
                    continue
                result = params.get("result")
                if isinstance(result, dict):
                    await self._handle_tx(result)
                    continue
                if not isinstance(result, str):
                    continue
                tx_hash = result

                if len(inflight) >= self._max_inflight:
                    done, inflight = await asyncio.wait(
                        inflight, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        task.result()

                task = asyncio.create_task(self._handle_tx_hash(ws, tx_hash))
                inflight.add(task)
                task.add_done_callback(inflight.discard)

    def parse_transaction(self, tx: dict) -> Optional["ParsedSwap"]:
        """
        Parse transaction to extract swap details.
        Returns None if not a swap.
        """
        if not isinstance(tx, dict):
            return None

        input_data = tx.get("input")
        if not isinstance(input_data, str) or not input_data.startswith("0x"):
            return None
        if len(input_data) < 10:
            return None

        selector = input_data[:10]
        if selector not in self.SWAP_SELECTORS:
            return None

        dex, method = self.SWAP_SELECTORS[selector]
        params = tx.get("_override_params")
        if not params:
            params = self.decode_swap_params(selector, _hex_to_bytes(input_data[10:]))
        if not params:
            return None

        path = params.get("path", [])
        path_addrs = [Address.from_string(addr) for addr in path]
        token_in = path_addrs[0] if path_addrs else None
        token_out = path_addrs[-1] if path_addrs else None

        tx_hash = tx.get("hash")
        sender = tx.get("from")
        router = tx.get("to")
        if not (tx_hash and sender and router):
            return None

        gas_price = _hex_to_int(tx.get("gasPrice")) or _hex_to_int(
            tx.get("maxFeePerGas")
        )

        return ParsedSwap(
            tx_hash=tx_hash,
            router=router,
            dex=dex,
            method=method,
            token_in=token_in,
            token_out=token_out,
            path=path_addrs,
            amount_in=params.get("amount_in", 0),
            min_amount_out=params.get("amount_out_min", 0),
            deadline=params.get("deadline", 0),
            sender=Address.from_string(sender),
            gas_price=gas_price,
        )

    def decode_swap_params(self, selector: str, data: bytes) -> dict:
        """
        Decode swap parameters from calldata.
        """
        if selector == "0x38ed1739":  # swapExactTokensForTokens
            amount_in, amount_out_min, path, _, deadline = decode(
                ["uint256", "uint256", "address[]", "address", "uint256"], data
            )
            return {
                "amount_in": int(amount_in),
                "amount_out_min": int(amount_out_min),
                "path": list(path),
                "deadline": int(deadline),
            }
        if selector == "0x18cbafe5":  # swapExactTokensForETH
            amount_in, amount_out_min, path, _, deadline = decode(
                ["uint256", "uint256", "address[]", "address", "uint256"], data
            )
            return {
                "amount_in": int(amount_in),
                "amount_out_min": int(amount_out_min),
                "path": list(path),
                "deadline": int(deadline),
            }
        if selector == "0x7ff36ab5":  # swapExactETHForTokens
            amount_out_min, path, _, deadline = decode(
                ["uint256", "address[]", "address", "uint256"], data
            )
            return {
                "amount_in": 0,
                "amount_out_min": int(amount_out_min),
                "path": list(path),
                "deadline": int(deadline),
            }
        return {}

    async def _subscribe(self, ws) -> str:
        params = (
            [self.subscription_method, self.subscription_params]
            if self.subscription_params is not None
            else [self.subscription_method]
        )
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_subscribe",
            "params": params,
        }
        await ws.send(json.dumps(payload))
        response = json.loads(await ws.recv())
        if "error" in response:
            raise RuntimeError(f"Subscription failed: {response['error']}")
        return response["result"]

    async def _handle_tx_hash(self, ws, tx_hash: str) -> None:
        tx = await self._fetch_tx(ws, tx_hash)
        if tx is None:
            return

        if tx.get("input", "").startswith("0x7ff36ab5"):
            tx["value"] = tx.get("value", "0x0")
            params = self.decode_swap_params(
                "0x7ff36ab5", _hex_to_bytes(tx["input"][10:])
            )
            params["amount_in"] = _hex_to_int(tx.get("value"))
            tx["_override_params"] = params

        parsed = self.parse_transaction(tx)
        if parsed is None:
            return

        if self.quote_fn is not None:
            expected = await self._safe_quote(parsed)
            if expected is not None:
                parsed.expected_amount_out = expected

        if asyncio.iscoroutinefunction(self.callback):
            await self.callback(parsed)
        else:
            await asyncio.to_thread(self.callback, parsed)

    async def _handle_tx(self, tx: dict) -> None:
        if tx.get("input", "").startswith("0x7ff36ab5"):
            tx["value"] = tx.get("value", "0x0")
            params = self.decode_swap_params(
                "0x7ff36ab5", _hex_to_bytes(tx["input"][10:])
            )
            params["amount_in"] = _hex_to_int(tx.get("value"))
            tx["_override_params"] = params

        parsed = self.parse_transaction(tx)
        if parsed is None:
            return

        if self.quote_fn is not None:
            expected = await self._safe_quote(parsed)
            if expected is not None:
                parsed.expected_amount_out = expected

        if asyncio.iscoroutinefunction(self.callback):
            await self.callback(parsed)
        else:
            await asyncio.to_thread(self.callback, parsed)

    async def _fetch_tx(self, ws, tx_hash: str) -> Optional[dict]:
        if not self.rpc_url:
            return None
        return await asyncio.to_thread(self._rpc_get_tx, tx_hash)

    def _rpc_get_tx(self, tx_hash: str) -> Optional[dict]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getTransactionByHash",
            "params": [tx_hash],
        }
        try:
            response = requests.post(self.rpc_url, json=payload, timeout=5)
            response.raise_for_status()
            data = response.json()
        except Exception:
            return None
        return data.get("result")

    async def _safe_quote(self, swap: "ParsedSwap") -> Optional[int]:
        try:
            if asyncio.iscoroutinefunction(self.quote_fn):
                return await asyncio.wait_for(
                    self.quote_fn(swap), timeout=self.quote_timeout
                )
            return await asyncio.wait_for(
                asyncio.to_thread(self.quote_fn, swap), timeout=self.quote_timeout
            )
        except Exception:
            return None


@dataclass
class ParsedSwap:
    """Parsed swap transaction from mempool."""

    tx_hash: str
    router: str
    dex: str
    method: str
    token_in: Optional[Address]
    token_out: Optional[Address]
    path: list[Address]
    amount_in: int
    min_amount_out: int
    deadline: int
    sender: Address
    gas_price: int
    expected_amount_out: Optional[int] = None

    @property
    def slippage_tolerance(self) -> Decimal:
        """Calculate implied slippage tolerance."""
        if self.expected_amount_out and self.expected_amount_out > 0:
            return Decimal(self.expected_amount_out - self.min_amount_out) / Decimal(
                self.expected_amount_out
            )
        if self.amount_in <= 0 or self.min_amount_out <= 0:
            return Decimal(0)
        # Without a price quote, we can only express the min-out ratio.
        return Decimal(self.min_amount_out) / Decimal(self.amount_in)


def _hex_to_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 16) if value.startswith("0x") else int(value)
    return 0


def _hex_to_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        return b""
    normalized = value[2:] if value.startswith("0x") else value
    if normalized == "":
        return b""
    return bytes.fromhex(normalized)


def _ws_to_http(ws_url: str) -> str | None:
    try:
        parsed = urlparse(ws_url)
    except Exception:
        return None
    if parsed.scheme == "ws":
        return parsed._replace(scheme="http").geturl()
    if parsed.scheme == "wss":
        return parsed._replace(scheme="https").geturl()
    if parsed.scheme in ("http", "https"):
        return parsed.geturl()
    return None
