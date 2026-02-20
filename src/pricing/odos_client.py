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
    path_id: Optional[str] = None

    @property
    def effective_price(self) -> float:
        """
        Effective output per unit input (token_out / token_in).
        """
        if self.amount_in == 0:
            return 0.0
        return self.amount_out / float(self.amount_in)


@dataclass
class OdosAssembledTx:
    """Transaction payload returned by ODOS /sor/assemble."""

    to: str
    data: str
    value: int
    gas: int
    chain_id: int


class OdosClient:
    """
    ODOS aggregator client for Arbitrum.

    Supports both quoting (price discovery) and transaction assembly
    (for executing swaps on-chain via the ODOS router).
    """

    def __init__(
        self,
        chain_id: int = 42161,
        base_url: str | None = None,
        timeout_seconds: float = 10.0,
        pool_connections: int = 20,
        pool_maxsize: int = 20,
    ) -> None:
        self._chain_id = chain_id
        self._base_url = base_url or "https://api.odos.xyz"
        self._timeout = timeout_seconds
        # Increase connection pool for parallel batch quoting
        # (8 tokens × 2 quotes = 16 concurrent)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
        )
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        resp = self._session.post(url, json=json_body, timeout=self._timeout)
        try:
            resp.raise_for_status()
        except requests.RequestException as exc:
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
        Request a price quote for a single-input -> single-output swap.

        ``amount_in`` is the raw token amount (respect input token decimals).
        The returned ``OdosQuote.path_id`` can be passed to ``assemble()`` to
        obtain executable calldata (valid for 60 seconds).
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
            path_id = data.get("pathId")
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
            path_id=str(path_id) if path_id else None,
        )
        logger.debug(
            "ODOS quote: in=%s out=%s eff_price=%.6f gas=%d impact=%.4f pathId=%s",
            amount_in,
            out_amount,
            quote.effective_price,
            gas_estimate,
            price_impact,
            quote.path_id,
        )
        return quote

    def assemble(
        self,
        path_id: str,
        user_address: str,
        simulate: bool = False,
    ) -> OdosAssembledTx:
        """
        Assemble a previously quoted path into an executable transaction.

        Must be called within 60 seconds of the quote. Returns an
        ``OdosAssembledTx`` containing the router address, calldata, value,
        and gas limit needed to execute the swap on-chain.
        """
        payload: Dict[str, Any] = {
            "userAddr": user_address,
            "pathId": path_id,
            "simulate": simulate,
        }
        data = self._post("/sor/assemble", payload)

        tx = data.get("transaction")
        if not tx:
            raise RuntimeError(f"ODOS assemble returned no transaction: {data}")

        try:
            assembled = OdosAssembledTx(
                to=str(tx["to"]),
                data=str(tx["data"]),
                value=int(tx.get("value", 0)),
                gas=int(tx.get("gas", 500_000)),
                chain_id=int(tx.get("chainId", self._chain_id)),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected ODOS assemble transaction schema: {tx}"
            ) from exc

        logger.debug(
            "ODOS assemble: router=%s gas=%d value=%d",
            assembled.to,
            assembled.gas,
            assembled.value,
        )
        return assembled
