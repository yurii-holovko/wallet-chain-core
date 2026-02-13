"""
Lightweight DEX pricer — reads Uniswap V2 reserves on-chain and
returns buy/sell prices without requiring a full PricingEngine.

Only needs an RPC URL (read-only, no private key).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from eth_abi import decode
from eth_utils.crypto import keccak

from chain.client import ChainClient
from core.base_types import Address, TokenAmount, TransactionRequest

logger = logging.getLogger(__name__)


@dataclass
class DexQuote:
    """Result of a DEX price query."""

    dex_buy: float  # price to BUY base on DEX (you pay quote, receive base)
    dex_sell: float  # price to SELL base on DEX (you pay base, receive quote)
    reserve_base: int  # raw reserve of base token
    reserve_quote: int  # raw reserve of quote token
    base_decimals: int
    quote_decimals: int
    pool_address: str
    timestamp: float


class DexPricer:
    """
    Fetches real Uniswap V2 prices from on-chain reserves.

    Usage::

        pricer = DexPricer(chain_client, pool_address, weth_address)
        quote = pricer.get_quote("ETH/USDT", size=0.05)
        # quote.dex_buy  = how much USDT per ETH when buying ETH
        # quote.dex_sell = how much USDT per ETH when selling ETH
    """

    # Cache reserves to avoid spamming RPC — V2 reserves only change on swaps
    _CACHE_TTL = 5.0  # seconds

    def __init__(
        self,
        chain_client: ChainClient,
        pool_address: str,
        weth_address: str,
        fee_bps: int = 30,  # Uniswap V2 = 0.30%
    ):
        self._client = chain_client
        self._pool = Address.from_string(pool_address)
        self._weth = weth_address.lower()
        self._fee_bps = fee_bps

        # Pool metadata (fetched once)
        self._token0_addr: Optional[str] = None
        self._token1_addr: Optional[str] = None
        self._token0_decimals: Optional[int] = None
        self._token1_decimals: Optional[int] = None
        self._token0_symbol: Optional[str] = None
        self._token1_symbol: Optional[str] = None
        self._base_is_token0: Optional[bool] = None  # True if WETH is token0

        # Reserve cache
        self._cached_reserves: Optional[tuple[int, int]] = None
        self._cache_ts: float = 0.0

        # Load pool metadata once
        self._load_metadata()

    # ── public API ────────────────────────────────────────────

    def get_quote(self, pair: str, size: float) -> Optional[dict]:
        """
        Return DEX buy/sell prices for *pair* at trade *size*.

        Returns dict compatible with SignalGenerator._fetch_dex_prices():
            {"buy": float, "sell": float}
        or None on failure.
        """
        try:
            reserve_base, reserve_quote = self._get_reserves()
            if reserve_base <= 0 or reserve_quote <= 0:
                logger.warning("DEX pool has zero reserves")
                return None

            base_dec = self._base_decimals()
            quote_dec = self._quote_decimals()

            # Convert trade size to raw base amount
            size_raw = int(Decimal(str(size)) * Decimal(10**base_dec))

            # ── SELL base on DEX: base → quote ──────────────────
            # You give `size` base tokens, receive quote tokens
            quote_out = self._get_amount_out(size_raw, reserve_base, reserve_quote)
            # Price = quote_received / base_sold  (in human units)
            dex_sell = (Decimal(quote_out) / Decimal(10**quote_dec)) / (
                Decimal(size_raw) / Decimal(10**base_dec)
            )

            # ── BUY base on DEX: quote → base ──────────────────
            # You want `size` base tokens — how much quote do you need?
            quote_in = self._get_amount_in(size_raw, reserve_quote, reserve_base)
            # Price = quote_spent / base_received  (in human units)
            dex_buy = (Decimal(quote_in) / Decimal(10**quote_dec)) / (
                Decimal(size_raw) / Decimal(10**base_dec)
            )

            logger.debug(
                "DEX quote: sell=%.2f  buy=%.2f  reserves=%d/%d",
                dex_sell,
                dex_buy,
                reserve_base,
                reserve_quote,
            )

            return {"buy": float(dex_buy), "sell": float(dex_sell)}

        except Exception as exc:
            logger.warning("DEX pricing failed: %s", exc)
            return None

    # ── AMM math (matches Solidity exactly) ───────────────────

    def _get_amount_out(self, amount_in: int, reserve_in: int, reserve_out: int) -> int:
        """Uniswap V2 getAmountOut."""
        amount_in_with_fee = amount_in * (10000 - self._fee_bps)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in * 10000 + amount_in_with_fee
        return numerator // denominator

    def _get_amount_in(self, amount_out: int, reserve_in: int, reserve_out: int) -> int:
        """Uniswap V2 getAmountIn."""
        if amount_out >= reserve_out:
            raise ValueError("amount_out exceeds reserve")
        numerator = amount_out * reserve_in * 10000
        denominator = (reserve_out - amount_out) * (10000 - self._fee_bps)
        return numerator // denominator + 1

    # ── on-chain reads ────────────────────────────────────────

    def _load_metadata(self) -> None:
        """Fetch token0/token1 addresses, decimals, symbols from the pool."""
        try:
            self._token0_addr = self._call_address("token0()").lower()
            self._token1_addr = self._call_address("token1()").lower()
            self._token0_decimals = self._call_uint("decimals()", self._token0_addr)
            self._token1_decimals = self._call_uint("decimals()", self._token1_addr)
            self._token0_symbol = self._call_string("symbol()", self._token0_addr)
            self._token1_symbol = self._call_string("symbol()", self._token1_addr)

            self._base_is_token0 = self._token0_addr == self._weth

            logger.info(
                "DEX pool loaded: %s (%s/%s) "
                "token0=%s(%d) token1=%s(%d) base_is_token0=%s",
                self._pool.checksum,
                self._token0_symbol,
                self._token1_symbol,
                self._token0_addr[:10],
                self._token0_decimals,
                self._token1_addr[:10],
                self._token1_decimals,
                self._base_is_token0,
            )
        except Exception as exc:
            logger.error("Failed to load DEX pool metadata: %s", exc)
            raise

    def _get_reserves(self) -> tuple[int, int]:
        """Fetch reserves, with short cache to avoid RPC spam."""
        now = time.time()
        if self._cached_reserves and (now - self._cache_ts) < self._CACHE_TTL:
            return self._cached_reserves

        selector = keccak(text="getReserves()")[:4].hex()
        raw = self._eth_call(self._pool.checksum, selector)
        r0, r1, _ = decode(["uint112", "uint112", "uint32"], raw)

        # Order: (base_reserve, quote_reserve)
        if self._base_is_token0:
            result = (int(r0), int(r1))
        else:
            result = (int(r1), int(r0))

        self._cached_reserves = result
        self._cache_ts = now
        return result

    def _base_decimals(self) -> int:
        if self._base_is_token0:
            return self._token0_decimals or 18
        return self._token1_decimals or 18

    def _quote_decimals(self) -> int:
        if self._base_is_token0:
            return self._token1_decimals or 6
        return self._token0_decimals or 6

    # ── low-level RPC helpers ─────────────────────────────────

    def _eth_call(self, to: str, selector_hex: str, data_hex: str = "") -> bytes:
        """Execute eth_call to a contract."""
        calldata = bytes.fromhex(selector_hex + data_hex)
        tx = TransactionRequest(
            to=Address.from_string(to),
            value=TokenAmount(raw=0, decimals=18, symbol="ETH"),
            data=calldata,
            chain_id=0,
        )
        return self._client.call(tx)

    def _call_address(self, signature: str) -> str:
        """Call a view function on the pool that returns an address."""
        selector = keccak(text=signature)[:4].hex()
        raw = self._eth_call(self._pool.checksum, selector)
        (addr,) = decode(["address"], raw)
        return addr

    def _call_uint(self, signature: str, token_addr: str) -> int:
        """Call decimals() on a token contract."""
        selector = keccak(text=signature)[:4].hex()
        raw = self._eth_call(token_addr, selector)
        (val,) = decode(["uint8"], raw)
        return int(val)

    def _call_string(self, signature: str, token_addr: str) -> str:
        """Call symbol() on a token contract."""
        selector = keccak(text=signature)[:4].hex()
        raw = self._eth_call(token_addr, selector)
        if len(raw) == 32:
            return raw.rstrip(b"\x00").decode("utf-8", errors="ignore")
        (val,) = decode(["string"], raw)
        return str(val)
