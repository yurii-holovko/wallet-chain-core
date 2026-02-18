from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .mexc_client import MexcApiError, MexcClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BridgeFeeInfo:
    """Snapshot of MEXC withdrawal fee configuration for a given asset/network."""

    asset: str
    network: str
    withdrawal_fee: float
    min_withdrawal: float
    max_withdrawal: Optional[float]
    fee_coin: str


class MEXCBridgeVerifier:
    """
    Helper for inspecting real bridge / withdrawal costs on MEXC.

    This reads the same configuration that governs actual withdrawals, so the
    numbers here can be used directly for bridge cost modelling and
    amortisation across trades.
    """

    def __init__(self, mexc_client: MexcClient) -> None:
        self._mexc = mexc_client

    def check_api_key_restrictions(self) -> Dict[str, Any]:
        """
        Check if API key has symbol restrictions that might affect capital endpoints.

        Returns dict: has_restrictions (bool), allowed_symbols (list or None).
        """
        try:
            data = self._mexc._request(  # type: ignore[attr-defined]
                "GET",
                "/api/v3/selfSymbols",
                params={},
                signed=True,
            )
            # If selfSymbols returns a list, the key has symbol restrictions
            if isinstance(data, dict) and "data" in data:
                symbols = data.get("data", [])
                return {
                    "has_restrictions": len(symbols) > 0,
                    "allowed_symbols": symbols if symbols else None,
                }
            return {"has_restrictions": False, "allowed_symbols": None}
        except Exception as exc:
            logger.warning("Failed to check API key symbol restrictions: %s", exc)
            return {
                "has_restrictions": None,
                "allowed_symbols": None,
                "error": str(exc),
            }

    def _fetch_raw_config(self) -> list[Dict[str, Any]]:
        """
        Fetch raw capital configuration from MEXC.

        API reference (subject to change by MEXC):
          GET /api/v3/capital/config/getall

        Note: This endpoint requires valid API credentials. If the request fails
        due to authentication issues, the method will raise an exception that
        callers should handle gracefully (e.g., fallback to config defaults).
        """
        try:
            data = self._mexc._request(  # type: ignore[attr-defined]
                "GET",
                "/api/v3/capital/config/getall",
                params={},
                signed=True,
            )
            if not isinstance(data, list):
                raise RuntimeError(
                    f"Unexpected MEXC config payload type: {type(data)!r}"
                )
            return data
        except MexcApiError as exc:
            # Re-raise with more context about what might be wrong
            error_msg = str(exc)
            if "Signature" in error_msg or "signature" in error_msg.lower():
                # Check for symbol restrictions that might affect access
                restrictions = self.check_api_key_restrictions()
                restriction_hint = ""
                if restrictions.get("has_restrictions"):
                    syms = restrictions.get("allowed_symbols")
                    restriction_hint = (
                        f" NOTE: API key has symbol restrictions: {syms}. "
                        "Consider removing or ensuring full access."
                    )
                else:
                    restriction_hint = ""
                raise RuntimeError(
                    "MEXC API signature error for /api/v3/capital/config/getall. "
                    "Please verify:\n"
                    "  1. MEXC_API_KEY and MEXC_API_SECRET in .env are correct\n"
                    "  2. API key has 'Withdrawal' permission (not just View)\n"
                    "  3. IP matches bound IP (if IP binding enabled)\n"
                    f"{restriction_hint}\n"
                    f"Original error: {error_msg}"
                ) from exc
            raise RuntimeError(
                f"MEXC API error fetching bridge config: {error_msg}"
            ) from exc

    def get_actual_bridge_cost(
        self,
        asset: str = "USDT",
        network: str = "Arbitrum One",
    ) -> BridgeFeeInfo:
        """
        Return the current withdrawal fee configuration for *asset* on *network*.

        Values are denominated in the fee coin (usually the same as *asset*).
        """
        raw = self._fetch_raw_config()

        # Log available coins and networks for debugging
        available_coins = set()
        available_networks_by_coin = {}

        for coin_info in raw:
            coin_name = str(coin_info.get("coin", ""))
            available_coins.add(coin_name)
            if coin_name == asset:
                networks = []
                for network_info in coin_info.get("networkList", []):
                    network_name = str(network_info.get("network", ""))
                    networks.append(network_name)
                available_networks_by_coin[coin_name] = networks
                logger.debug(
                    f"MEXC bridge config: Found {coin_name} with networks: {networks}"
                )

        # Try exact match first, then partial matches
        for coin_info in raw:
            if str(coin_info.get("coin")) != asset:
                continue
            for network_info in coin_info.get("networkList", []):
                network_name = str(network_info.get("network", ""))

                # Try exact match
                if network_name == network:
                    withdrawal_fee = float(network_info.get("withdrawFee", 0.0) or 0.0)
                    min_withdrawal = float(network_info.get("withdrawMin", 0.0) or 0.0)
                    max_raw = network_info.get("withdrawMax")
                    max_withdrawal = (
                        float(max_raw) if max_raw not in (None, "", "0") else None
                    )
                    fee_coin = str(network_info.get("feeCoin") or asset)
                    return BridgeFeeInfo(
                        asset=asset,
                        network=network,
                        withdrawal_fee=withdrawal_fee,
                        min_withdrawal=min_withdrawal,
                        max_withdrawal=max_withdrawal,
                        fee_coin=fee_coin,
                    )

                # Try case-insensitive exact match
                if network_name.lower() == network.lower():
                    logger.info(
                        "MEXC bridge: found '%s' (case-insensitive for '%s')",
                        network_name,
                        network,
                    )
                    withdrawal_fee = float(network_info.get("withdrawFee", 0.0) or 0.0)
                    min_withdrawal = float(network_info.get("withdrawMin", 0.0) or 0.0)
                    max_raw = network_info.get("withdrawMax")
                    max_withdrawal = (
                        float(max_raw) if max_raw not in (None, "", "0") else None
                    )
                    fee_coin = str(network_info.get("feeCoin") or asset)
                    return BridgeFeeInfo(
                        asset=asset,
                        network=network_name,  # actual name from API
                        withdrawal_fee=withdrawal_fee,
                        min_withdrawal=min_withdrawal,
                        max_withdrawal=max_withdrawal,
                        fee_coin=fee_coin,
                    )

                # Try partial match - network name starts with the requested network
                # This handles cases like "Arbitrum One(ARB)" matching "Arbitrum One"
                network_name_base = network_name.split("(")[
                    0
                ].strip()  # Remove anything in parentheses
                if network_name_base.lower() == network.lower():
                    logger.info(
                        "MEXC bridge: found '%s' (partial match for '%s')",
                        network_name,
                        network,
                    )
                    withdrawal_fee = float(network_info.get("withdrawFee", 0.0) or 0.0)
                    min_withdrawal = float(network_info.get("withdrawMin", 0.0) or 0.0)
                    max_raw = network_info.get("withdrawMax")
                    max_withdrawal = (
                        float(max_raw) if max_raw not in (None, "", "0") else None
                    )
                    fee_coin = str(network_info.get("feeCoin") or asset)
                    return BridgeFeeInfo(
                        asset=asset,
                        network=network_name,  # actual name from API
                        withdrawal_fee=withdrawal_fee,
                        min_withdrawal=min_withdrawal,
                        max_withdrawal=max_withdrawal,
                        fee_coin=fee_coin,
                    )

        # If not found, provide helpful error message
        error_msg = f"No bridge config found for {asset} on {network}\n"
        if asset in available_coins:
            nets = available_networks_by_coin.get(asset, [])
            error_msg += f"  Available networks for {asset}: {nets}\n"
        else:
            error_msg += f"  Available coins: {sorted(available_coins)}\n"
            if available_networks_by_coin:
                error_msg += "  Available networks by coin: "
                error_msg += f"{available_networks_by_coin}\n"

        raise RuntimeError(error_msg)

    def verify_bridge_amortization(
        self,
        trade_size_usd: float,
        expected_trades: int,
        asset: str = "USDT",
        network: str = "Arbitrum One",
    ) -> Dict[str, Any]:
        """
        Compute per-trade amortised bridge cost for a given trade size.

        ``expected_trades`` is the number of trades you expect to execute
        between CEX↔chain rebalances; conservatively low values (e.g. 5–10)
        will avoid underestimating bridge drag.

        Auth/signature problems are surfaced so they can be fixed at the root.
        """
        if expected_trades <= 0:
            raise ValueError("expected_trades must be positive")

        info = self.get_actual_bridge_cost(asset=asset, network=network)
        actual_fee = info.withdrawal_fee
        amortised_per_trade = actual_fee / float(expected_trades)
        pct_of_trade = (
            amortised_per_trade / float(trade_size_usd) if trade_size_usd > 0 else 0.0
        )

        return {
            "asset": info.asset,
            "network": info.network,
            "fee_coin": info.fee_coin,
            "actual_bridge_fee": actual_fee,
            "min_withdrawal": info.min_withdrawal,
            "max_withdrawal": info.max_withdrawal,
            "expected_trades": expected_trades,
            "amortized_per_trade": amortised_per_trade,
            "pct_of_trade": pct_of_trade,
            # Heuristic: bridge drag < 2% of notional per trade is OK for micro-arb.
            "is_realistic": pct_of_trade < 0.02,
        }
