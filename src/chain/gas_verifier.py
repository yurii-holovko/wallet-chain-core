from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional

import requests

from .client import ChainClient, GasPrice


@dataclass(frozen=True)
class GasEstimate:
    """Human-friendly view of a gas estimate in both ETH and USD."""

    estimated_gas: int
    gas_price: GasPrice
    priority_level: str
    total_gas_eth: float
    gas_cost_usd: float


class GasVerifier:
    """
    Conservative verifier for the gas component of the Double Limit strategy.

    The goal is not to model every nuance of Uniswap V3 range orders, but to
    ensure that the configured ``gas_cost_usd`` used in the cost model is
    *not* wildly optimistic given current Arbitrum L2 fees.

    Implementation notes:
      * We approximate mint gas with a fixed unit cost (default 350k), which is
        in line with observed Uniswap V3 position mints.
      * We use the current ``baseFeePerGas`` + ``maxPriorityFee`` from the
        JSON-RPC node to turn this into an ETH and USD amount.
      * ETH/USD is fetched from a public API (CoinGecko) the first time and
        then cached on the instance.
    """

    def __init__(
        self,
        client: ChainClient,
        eth_price_usd: Optional[float] = None,
        priority_level: str = "medium",
    ) -> None:
        self._client = client
        self._eth_price_usd = eth_price_usd
        self._priority_level = priority_level

    # ── public API ──────────────────────────────────────────────

    def estimate_typical_v3_mint_gas(
        self,
        typical_gas_units: int = 350_000,
    ) -> GasEstimate:
        """
        Approximate the cost of a typical Uniswap V3 mint on Arbitrum.

        ``typical_gas_units`` can be overridden if you have better empirical
        data from your own transaction history.
        """
        if typical_gas_units <= 0:
            raise ValueError("typical_gas_units must be positive")

        gas_price = self._client.get_gas_price()
        priority = self._priority_level
        max_fee_per_gas = gas_price.get_max_fee(priority)

        total_gas_wei = typical_gas_units * max_fee_per_gas
        total_gas_eth = float(Decimal(total_gas_wei) / Decimal(10**18))
        eth_price = self.get_eth_price()
        gas_cost_usd = total_gas_eth * eth_price

        return GasEstimate(
            estimated_gas=typical_gas_units,
            gas_price=gas_price,
            priority_level=priority,
            total_gas_eth=total_gas_eth,
            gas_cost_usd=gas_cost_usd,
        )

    def verify_against_config(
        self,
        config_gas_cost_usd: float,
        typical_gas_units: int = 350_000,
        tolerance_factor: float = 1.2,
    ) -> Dict[str, Any]:
        """
        Compare live gas conditions against the configured ``gas_cost_usd``.

        Returns a dict with a boolean ``ok`` flag plus diagnostic fields.
        ``tolerance_factor`` controls how much higher the live estimate can be
        before we flag the config as too low (default: 20%).
        """
        estimate = self.estimate_typical_v3_mint_gas(typical_gas_units)
        config_value = float(config_gas_cost_usd)

        ok = estimate.gas_cost_usd <= config_value * tolerance_factor
        return {
            "ok": ok,
            "config_gas_cost_usd": config_value,
            "estimated_gas_cost_usd": estimate.gas_cost_usd,
            "estimated_gas_units": estimate.estimated_gas,
            "priority_level": estimate.priority_level,
            "base_fee_gwei": self._wei_to_gwei(estimate.gas_price.base_fee),
            "priority_fee_gwei": self._priority_fee_gwei(estimate.gas_price),
            "eth_price_usd": self.get_eth_price(),
            "tolerance_factor": tolerance_factor,
        }

    # ── helpers ─────────────────────────────────────────────────

    def get_eth_price(self) -> float:
        """
        Fetch (and cache) the current ETH price in USD.

        Uses CoinGecko's simple price API; callers can also inject a fixed
        price via the constructor for deterministic tests.
        """
        if self._eth_price_usd is not None:
            return self._eth_price_usd

        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "ethereum", "vs_currencies": "usd"},
                timeout=5,
            )
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()
            price = float(data.get("ethereum", {}).get("usd", 0.0) or 0.0)
            if price <= 0:
                raise ValueError("ETH price from API is non-positive")
            self._eth_price_usd = price
            return price
        except Exception:
            # Fallback to a conservative hard-coded value if the API fails.
            # This errs on the side of *over*estimating gas cost.
            if self._eth_price_usd is None:
                self._eth_price_usd = 2000.0
            return self._eth_price_usd

    @staticmethod
    def _wei_to_gwei(value_wei: int) -> float:
        return float(Decimal(value_wei) / Decimal(10**9))

    @staticmethod
    def _priority_fee_gwei(gas_price: GasPrice) -> float:
        return float(Decimal(gas_price.priority_fee_medium) / Decimal(10**9))
