from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class OdosQuote:
    """
    Lightweight representation of an ODOS quote response.

    All amounts are raw integer token amounts (respect pool decimals).
    """

    chain_id: int
    input_token: str
    output_token: str
    amount_in: int
    amount_out: int
    gas_estimate: int
    price_impact: float
    block_number: int
    path_viz: Optional[dict]

    @property
    def effective_price(self) -> float:
        """
        Effective output per unit input (token_out / token_in).
        """
        if self.amount_in == 0:
            return 0.0
        return self.amount_out / float(self.amount_in)


class OdosClient:
    """
    Minimal ODOS aggregator client for Arbitrum.

    We only use the SOR quote endpoint to obtain realistic execution prices
    and gas estimates for small notional swaps (e.g. $5 USDC → token).
    """

    def __init__(
        self,
        chain_id: int = 42161,
        base_url: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._chain_id = chain_id
        self._base_url = base_url or "https://api.odos.xyz"
        self._timeout = timeout_seconds
        self._session = requests.Session()

    def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        resp = self._session.post(url, json=json_body, timeout=self._timeout)
        try:
            resp.raise_for_status()
        except requests.RequestException as exc:
            # Include response text to make debugging 4xx/5xx errors easier.
            body = ""
            try:
                body = resp.text
            except Exception:  # pragma: no cover - defensive
                body = "<unavailable>"
            raise RuntimeError(f"ODOS request failed: {exc}  body={body!r}") from exc
        try:
            return resp.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Invalid JSON from ODOS: {resp.text!r}") from exc

    def quote(
        self,
        input_token: str,
        output_token: str,
        amount_in: int,
        user_address: str,
        slippage_percent: float = 0.5,
        compact: bool = True,
    ) -> OdosQuote:
        """
        Request a price quote for a single-input → single-output swap.

        ``amount_in`` is the raw token amount (respect input token decimals).
        """
        payload: Dict[str, Any] = {
            "chainId": self._chain_id,
            "inputTokens": [
                {
                    "tokenAddress": input_token,
                    "amount": str(amount_in),
                }
            ],
            "outputTokens": [
                {
                    "tokenAddress": output_token,
                    "proportion": 1,
                }
            ],
            "slippageLimitPercent": slippage_percent,
            "userAddr": user_address,
            "referralCode": 0,
            "compact": compact,
        }

        data = self._post("/sor/quote/v2", payload)

        try:
            out_amount = int(data["outAmounts"][0])
            gas_estimate = int(data.get("gasEstimate", 0))
            block_number = int(data.get("blockNumber", 0))
            price_impact = float(data.get("priceImpact", 0.0))
            path_viz = data.get("pathViz")
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Unexpected ODOS response schema: {data}") from exc

        quote = OdosQuote(
            chain_id=self._chain_id,
            input_token=input_token,
            output_token=output_token,
            amount_in=amount_in,
            amount_out=out_amount,
            gas_estimate=gas_estimate,
            price_impact=price_impact,
            block_number=block_number,
            path_viz=path_viz if isinstance(path_viz, dict) else None,
        )
        logger.debug(
            "ODOS quote: in=%s out=%s eff_price=%.6f gas=%d impact=%.4f",
            amount_in,
            out_amount,
            quote.effective_price,
            gas_estimate,
            price_impact,
        )
        return quote
