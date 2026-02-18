from __future__ import annotations

"""
Systematic verification script for TOKEN_MAPPINGS entries.

Run this before activating any new token:

  - Validates MEXC order book availability.
  - Validates ODOS routing (USDC → token) when enabled.
  - Produces a human-readable report and a suggested config patch
    for tokens that should be disabled.

Usage (from repo root, with .env configured for MEXC + Arbitrum/ODOS):

    python scripts/verify_tokens.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root and src/ are on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
for path in (ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import get_env  # noqa: E402
from config_tokens_arb_mex import TOKEN_MAPPINGS  # noqa: E402
from exchange.mexc_client import MexcClient  # noqa: E402
from pricing.odos_client import OdosClient  # noqa: E402

USDC_ADDRESS_ARB = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"


class TokenVerifier:
    def __init__(self, mexc: MexcClient, odos: OdosClient):
        self.mexc = mexc
        self.odos = odos
        self.results: List[Dict] = []

    async def verify_token(self, symbol: str, cfg: Dict) -> Dict:
        """Comprehensive token verification."""
        result: Dict = {
            "symbol": symbol,
            "address": cfg["address"],
            "mex_symbol": cfg["mex_symbol"],
            "checks": {},
            "status": "UNKNOWN",
            "recommendation": "",
        }

        # 1. Check MEXC order book
        try:
            book = self.mexc.get_order_book(cfg["mex_symbol"], limit=1)
            if book.get("bids") and book.get("asks"):
                result["checks"]["mexc"] = "PASS"
                bid = book["bids"][0][0]
                ask = book["asks"][0][0]
                result["checks"]["mexc_spread"] = (ask - bid) / bid if bid > 0 else None
            else:
                result["checks"]["mexc"] = "EMPTY_BOOK"
        except Exception as e:
            result["checks"]["mexc"] = f"FAIL: {str(e)[:50]}"

        # 2. Check ODOS routing (USDC → token)
        if cfg.get("odos_supported", False):
            try:
                user_addr = get_env("ARBITRUM_WALLET_ADDRESS") or USDC_ADDRESS_ARB
                quote = self.odos.quote(
                    input_token=USDC_ADDRESS_ARB,
                    output_token=cfg["address"],
                    amount_in=5_000_000,  # $5 with 6-decimal USDC
                    user_address=user_addr,
                )
                result["checks"]["odos"] = "PASS"
                result["checks"]["odos_output"] = quote.amount_out / (
                    10 ** int(cfg.get("decimals", 18))
                )
            except Exception as e:
                msg = f"FAIL: {str(e)[:50]}"
                result["checks"]["odos"] = msg
                # Critical: ODOS failure means we cannot use this token
                result["recommendation"] = "DISABLE - ODOS routing unavailable"
                result["status"] = "FAILED"
                return result

        # 3. Check V3 pool flag if provided (lightweight)
        if cfg.get("v3_pool"):
            # Record presence only; deeper validation elsewhere.
            result["checks"]["v3_pool"] = "PRESENT (validation pending)"

        # 4. Determine status
        if all(
            v == "PASS" for k, v in result["checks"].items() if k in {"mexc", "odos"}
        ):
            result["status"] = "ACTIVE"
            result["recommendation"] = "ENABLE for trading"
        elif str(result["checks"].get("odos", "")).startswith("FAIL"):
            result["status"] = "FAILED"
            result["recommendation"] = "DISABLE - No DEX routing"
        else:
            result["status"] = "INVESTIGATE"
            result["recommendation"] = "Manual review required"

        return result

    async def verify_all(self, tokens: Dict[str, Dict]) -> List[Dict]:
        """Verify all tokens in mapping.

        We skip tokens that are both inactive and explicitly ODOS-disabled,
        since they are clearly not candidates for activation.
        """
        tasks = []
        for symbol, cfg in tokens.items():
            if not cfg.get("active") and not cfg.get("odos_supported"):
                continue
            tasks.append(self.verify_token(symbol, cfg))

        self.results = await asyncio.gather(*tasks)
        return self.results

    def generate_config_patch(self) -> Dict[str, Dict]:
        """Generate updated token config based on verification.

        Only tokens that have definitively FAILED are included. For each, we
        recommend disabling ODOS and marking them inactive, with a notes field
        capturing the ODOS failure reason.
        """
        patch: Dict[str, Dict] = {}
        for r in self.results:
            if r.get("status") == "FAILED":
                patch[r["symbol"]] = {
                    "odos_supported": False,
                    "active": False,
                    "notes": f"Disabled: {r['checks'].get('odos', 'Unknown error')}",
                }
        return patch

    def print_report(self) -> None:
        """Print formatted verification report."""
        print("\n" + "=" * 80)
        print("TOKEN VERIFICATION REPORT")
        print("=" * 80)

        for r in sorted(self.results, key=lambda x: x.get("status", "")):
            status = r.get("status", "UNKNOWN")
            if status == "ACTIVE":
                status_icon = "✅"
            elif status == "FAILED":
                status_icon = "❌"
            else:
                status_icon = "⚠️"

            print(f"\n{status_icon} {r['symbol']} ({r['mex_symbol']})")
            print(f"   Status: {status}")
            print(f"   Checks: {r['checks']}")
            print(f"   Recommendation: {r['recommendation']}")


async def main() -> None:
    mexc = MexcClient()
    odos = OdosClient()

    verifier = TokenVerifier(mexc, odos)
    await verifier.verify_all(TOKEN_MAPPINGS)
    verifier.print_report()

    # Generate patch for failed tokens
    patch = verifier.generate_config_patch()
    print("\n" + "=" * 80)
    print("CONFIG PATCH FOR FAILED TOKENS:")
    print(json.dumps(patch, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
