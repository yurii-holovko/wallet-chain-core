from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from exchange.mexc_client import MexcClient


@dataclass
class CapitalManagerConfig:
    """
    Configuration for capital allocation and bridging policy.
    """

    starting_cex_usd: float = 50.0
    starting_chain_usd: float = 50.0
    bridge_threshold_usd: float = 20.0
    min_tradable_usd: float = 5.0
    bridge_fixed_cost_usd: float = (
        0.05  # Updated from 0.32 to match actual MEXC withdrawal fee
    )


class CapitalManager:
    """
    Tracks capital on MEXC vs Arbitrum and decides when bridging is rational.

    This is a deliberately high-level abstraction: it works in USD space and
    delegates concrete on-chain / CEX withdrawal mechanics to the caller.
    """

    def __init__(
        self,
        mexc_client: MexcClient,
        config: Optional[CapitalManagerConfig] = None,
    ) -> None:
        self._mexc = mexc_client
        self.config = config or CapitalManagerConfig()
        self.trade_count_since_bridge: int = 0

    # ── trade accounting ───────────────────────────────────────────

    def record_trade(self, profit_usd: float) -> None:
        """
        Record that a trade completed.

        For now we only track trade count; PnL attribution across sides is
        handled by the caller using their own balance snapshots.
        """
        if profit_usd is None:
            return
        self.trade_count_since_bridge += 1

    def get_effective_bridge_cost(self) -> float:
        """
        Return amortized bridge cost per trade in USD.
        """
        if self.trade_count_since_bridge <= 0:
            return 0.0
        return self.config.bridge_fixed_cost_usd / float(self.trade_count_since_bridge)

    # ── bridging decision logic ───────────────────────────────────

    def should_bridge(
        self,
        mex_balance_usd: float,
        chain_balance_usd: float,
    ) -> tuple[bool, str, Optional[str]]:
        """
        Decide whether to initiate a rebalance between MEXC and Arbitrum.

        Returns ``(should_bridge, reason, direction)`` where ``direction`` is
        either ``'arbitrum_to_mex'``, ``'mex_to_arbitrum'``, or ``None``.
        """
        min_trade = self.config.min_tradable_usd

        mex_empty = mex_balance_usd < min_trade
        chain_empty = chain_balance_usd < min_trade

        if not (mex_empty or chain_empty):
            return False, "Both sides have sufficient capital", None

        if mex_empty:
            accumulated = chain_balance_usd - self.config.starting_chain_usd
            direction = "arbitrum_to_mex"
        else:
            accumulated = mex_balance_usd - self.config.starting_cex_usd
            direction = "mex_to_arbitrum"

        if accumulated <= 0:
            return False, "No accumulated profit to bridge", None

        if accumulated < self.config.bridge_threshold_usd:
            return (
                False,
                f"Accumulated ${accumulated:.2f} < threshold "
                f"${self.config.bridge_threshold_usd:.2f}",
                None,
            )

        effective_fee = self.config.bridge_fixed_cost_usd / accumulated
        return True, f"Effective bridge fee {effective_fee:.2%}", direction

    # ── execution hook (CEX side only for now) ────────────────────

    def execute_cex_withdrawal(
        self,
        coin: str,
        amount: float,
        address: str,
        network: str = "Arbitrum One",
    ) -> str:
        """
        Trigger a MEXC withdrawal to Arbitrum.

        This is a thin wrapper around ``MexcClient.withdraw`` so callers can
        keep all bridge-related side effects in one place.
        """
        withdrawal_id = self._mexc.withdraw(
            coin=coin,
            amount=amount,
            address=address,
            network=network,
        )
        self.trade_count_since_bridge = 0
        return withdrawal_id
