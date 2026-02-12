"""Fee structure and economics for arbitrage trades."""

from dataclasses import dataclass


@dataclass
class FeeStructure:
    """
    All-in cost model for a CEX↔DEX arbitrage round-trip.

    Components:
      - cex_taker_bps  : CEX taker fee (Binance default ~10 bps)
      - dex_swap_bps   : DEX swap fee (Uniswap V3 ~30 bps on 0.3% pool)
      - gas_cost_usd   : Estimated gas for the on-chain leg
      - slippage_bps   : Expected slippage beyond mid-price
    """

    cex_taker_bps: float = 10.0
    dex_swap_bps: float = 30.0
    gas_cost_usd: float = 5.0
    slippage_bps: float = 5.0

    def total_fee_bps(self, trade_value_usd: float) -> float:
        """Total round-trip cost in basis points."""
        if trade_value_usd <= 0:
            return float("inf")
        gas_bps = (self.gas_cost_usd / trade_value_usd) * 10_000
        return self.cex_taker_bps + self.dex_swap_bps + self.slippage_bps + gas_bps

    def breakeven_spread_bps(self, trade_value_usd: float) -> float:
        """Minimum spread to cover all costs."""
        return self.total_fee_bps(trade_value_usd)

    def net_profit_usd(self, spread_bps: float, trade_value_usd: float) -> float:
        """Expected profit after all fees."""
        gross = (spread_bps / 10_000) * trade_value_usd
        fees = (self.total_fee_bps(trade_value_usd) / 10_000) * trade_value_usd
        return gross - fees
