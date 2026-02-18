from __future__ import annotations

"""
End-to-end cost verification for the Double Limit micro-arbitrage stack.

This script surfaces a *concrete* view of per-trade costs by combining:

  1. LP fee implied by Uniswap V3 fee tier for each active token.
  2. Live Arbitrum gas conditions (approximate V3 mint cost).
  3. Real MEXC bridge / withdrawal fees, amortised across trades.
  4. ODOS surplus fee from the DoubleLimitConfig.

Run from the project root with a configured .env:

    python scripts/verify_all_costs.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict

# Ensure project root and src/ are on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
for path in (ROOT, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chain import ChainClient  # noqa: E402
from chain.gas_verifier import GasVerifier  # noqa: E402
from config import get_env  # noqa: E402
from config_tokens_arb_mex import TOKEN_MAPPINGS  # noqa: E402
from exchange.mexc_bridge_verifier import MEXCBridgeVerifier  # noqa: E402
from exchange.mexc_client import MexcClient  # noqa: E402
from executor.double_limit_engine import DoubleLimitConfig  # noqa: E402


def _lp_fee_pct(fee_tier: int) -> float:
    """Mirror the internal DoubleLimit LP fee mapping."""
    mapping: Dict[int, float] = {
        100: 0.0001,
        500: 0.0005,
        3_000: 0.003,
        10_000: 0.01,
    }
    return mapping.get(int(fee_tier), 0.003)


async def main() -> None:
    print("=" * 60)
    print("COST VERIFICATION REPORT")
    print("=" * 60)

    # ── 0. Load config / env ────────────────────────────────────

    trade_size_usd = float(get_env("TRADE_SIZE_USD", "5.0") or "5.0")

    # Allow overriding gas cost via env while keeping sensible defaults.
    dl_defaults = DoubleLimitConfig()
    gas_cost_env = get_env("GAS_COST_USD")
    gas_cost_usd = (
        float(gas_cost_env)
        if gas_cost_env not in (None, "")
        else float(dl_defaults.gas_cost_usd)
    )

    dl_cfg = DoubleLimitConfig(
        trade_size_usd=trade_size_usd,
        min_spread_pct=float(
            get_env("MIN_SPREAD_PCT", str(dl_defaults.min_spread_pct))
            or dl_defaults.min_spread_pct
        ),
        min_profit_usd=float(
            get_env("MIN_PROFIT_USD", str(dl_defaults.min_profit_usd))
            or dl_defaults.min_profit_usd
        ),
        max_slippage_pct=float(
            get_env("MAX_SLIPPAGE_PCT", str(dl_defaults.max_slippage_pct))
            or dl_defaults.max_slippage_pct
        ),
        gas_cost_usd=gas_cost_usd,
        odos_fee_pct=dl_defaults.odos_fee_pct,
        min_trades_for_bridge_amortization=(
            dl_defaults.min_trades_for_bridge_amortization
        ),
        simulation_mode=True,
    )

    # ── 1. LP fee overview ───────────────────────────────────────

    print("\n1. LP FEE BY TOKEN (CONFIGURED)")
    for symbol, cfg in TOKEN_MAPPINGS.items():
        if not cfg.get("active"):
            continue
        fee_tier = int(cfg.get("fee_tier", 3_000))
        expected_pct = _lp_fee_pct(fee_tier)
        print(f"{symbol:>6}: fee_tier={fee_tier:5d} -> LP fee ~= {expected_pct:.2%}")

    # ── 2. Gas verification ──────────────────────────────────────

    print("\n2. GAS VERIFICATION (ARBITRUM)")
    rpc_https = get_env("ARBITRUM_RPC_HTTPS")
    gas_result = None
    if not rpc_https:
        print("Skipping gas verification: ARBITRUM_RPC_HTTPS not set in .env")
    else:
        try:
            client = ChainClient([rpc_https])
            gas_verifier = GasVerifier(client)
            gas_result = gas_verifier.verify_against_config(dl_cfg.gas_cost_usd)
            est = gas_result
            status = "OK" if est["ok"] else "WARNING"
            print(
                f"Config gas_cost_usd: ${est['config_gas_cost_usd']:.4f}  |  "
                f"Estimated: ${est['estimated_gas_cost_usd']:.4f} "
                f"(gas={est['estimated_gas_units']:,}, "
                f"base_fee={est['base_fee_gwei']:.3f} gwei, "
                f"priority~{est['priority_fee_gwei']:.3f} gwei, "
                f"ETH~${est['eth_price_usd']:.2f})  -> {status}"
            )
            if not est["ok"]:
                print(
                    f"  NOTE: live gas cost exceeds config by more than "
                    f"{int(est['tolerance_factor'] * 100 - 100)}%. "
                    "Consider increasing DoubleLimitConfig.gas_cost_usd."
                )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Gas verification failed: {exc}")

    # ── 3. Bridge verification ───────────────────────────────────

    print("\n3. BRIDGE VERIFICATION (MEXC -> ARBITRUM)")
    bridge_result = None
    try:
        mexc = MexcClient()
        bridge_verifier = MEXCBridgeVerifier(mexc)
        expected_trades = int(dl_cfg.min_trades_for_bridge_amortization)
        bridge_result = bridge_verifier.verify_bridge_amortization(
            trade_size_usd=trade_size_usd,
            expected_trades=expected_trades,
            asset="USDT",
            network="Arbitrum One",
        )
        print(
            f"Withdrawal fee: {bridge_result['actual_bridge_fee']} "
            f"{bridge_result['fee_coin']} on {bridge_result['network']}"
        )
        print(
            f"Amortized over {expected_trades} trades of ${trade_size_usd:.2f}: "
            f"{bridge_result['amortized_per_trade']:.4f} {bridge_result['fee_coin']} "
            f"({bridge_result['pct_of_trade']:.2%} of notional)"
        )
        print(f"Bridge cost realistic for micro-arb: {bridge_result['is_realistic']}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Bridge verification failed: {exc}")

    # ── 4. Total per-trade cost estimation ───────────────────────

    print("\n4. TOTAL COST ESTIMATION PER ACTIVE TOKEN")
    if bridge_result is None:
        bridge_per_trade = 0.0
    else:
        bridge_per_trade = float(bridge_result["amortized_per_trade"])

    if gas_result is None:
        gas_per_trade = float(dl_cfg.gas_cost_usd)
    else:
        gas_per_trade = float(gas_result["estimated_gas_cost_usd"])

    odos_per_trade = dl_cfg.odos_fee_pct * trade_size_usd

    print(
        f"Assumptions: trade_size=${trade_size_usd:.2f}, "
        f"gas~${gas_per_trade:.4f}, bridge~${bridge_per_trade:.4f}, "
        f"ODOS~${odos_per_trade:.4f}"
    )
    print("-" * 60)
    for symbol, cfg in TOKEN_MAPPINGS.items():
        if not cfg.get("active"):
            continue
        fee_tier = int(cfg.get("fee_tier", 3_000))
        lp_fee_usd = _lp_fee_pct(fee_tier) * trade_size_usd
        total = lp_fee_usd + gas_per_trade + bridge_per_trade + odos_per_trade
        pct_of_trade = total / trade_size_usd if trade_size_usd > 0 else 0.0
        print(
            f"{symbol:>6}: total=${total:.4f} ({pct_of_trade:.2%})  "
            f"[LP=${lp_fee_usd:.4f}, gas=${gas_per_trade:.4f}, "
            f"bridge=${bridge_per_trade:.4f}, ODOS=${odos_per_trade:.4f}]"
        )


if __name__ == "__main__":
    asyncio.run(main())
