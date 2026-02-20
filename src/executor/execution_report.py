"""
Unified execution report: link signal/opportunity with execution outcome.

Abstractions:
  - CEX leg: order_id, status, filled qty, price (wrapped from exchange-specific).
  - DEX leg: tx_hash or position_id, swap_executed (bool), filled; same abstraction.

When execution finishes we produce a full report: signal/opportunity structure,
per-leg status (CEX + DEX), overall success/fail, and send it (e.g. to Telegram).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from executor.engine import ExecutionContext, ExecutorState

# Telegram message limit; truncate reports longer than this
TELEGRAM_MAX_LEN = 4000


@dataclass
class LegSummary:
    """One leg (CEX order or DEX swap) in a unified format."""

    venue: str  # "cex" | "dex"
    order_id_or_tx: Optional[str] = None
    status: str = ""
    filled: Optional[float] = None
    price: Optional[float] = None
    # CEX: status from exchange (FILLED, NEW, ...)
    # DEX: swap_executed True = liquidity was used (swap happened)
    swap_executed: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_line(self) -> str:
        parts = [f"  {self.venue.upper()}: status={self.status}"]
        if self.order_id_or_tx:
            parts.append(
                f"id/tx={self.order_id_or_tx[:16]}..."
                if len(self.order_id_or_tx or "") > 16
                else f"id/tx={self.order_id_or_tx}"
            )
        if self.filled is not None:
            parts.append(f"filled={self.filled}")
        if self.price is not None:
            parts.append(f"price={self.price}")
        if self.swap_executed is not None:
            parts.append(f"swap_executed={self.swap_executed}")
        return " | ".join(parts)


def format_cex_dex_execution_report(ctx: ExecutionContext) -> str:
    """
    Full execution report for CEX/DEX executor: signal + both legs + outcome.
    """
    s = ctx.signal
    lines = [
        "━━ CEX/DEX EXECUTION REPORT ━━",
        f"Signal: {s.signal_id}",
        f"  pair={s.pair}  direction={s.direction.value}",
        f"  size={s.size}  cex_price={s.cex_price}  dex_price={s.dex_price}",
        f"  spread_bps={s.spread_bps}  expected_net_pnl=${s.expected_net_pnl:.4f}",
        "",
        "Leg 1 (CEX):",
    ]
    leg1 = LegSummary(
        venue="cex",
        order_id_or_tx=ctx.leg1_order_id,
        status=ctx.state.name if ctx.leg1_order_id else "—",
        filled=ctx.leg1_fill_size,
        price=ctx.leg1_fill_price,
    )
    lines.append(leg1.to_line())
    lines.append("")
    lines.append("Leg 2 (DEX):")
    leg2 = LegSummary(
        venue="dex",
        order_id_or_tx=ctx.leg2_tx_hash,
        status=(
            "FILLED"
            if ctx.leg2_tx_hash and ctx.state == ExecutorState.DONE
            else (ctx.state.name if ctx.leg2_tx_hash else "—")
        ),
        filled=ctx.leg2_fill_size,
        price=ctx.leg2_fill_price,
        swap_executed=bool(ctx.leg2_tx_hash and ctx.state == ExecutorState.DONE),
    )
    lines.append(leg2.to_line())
    lines.append("")
    lines.append(f"Outcome: {ctx.state.name}")
    if ctx.error:
        lines.append(f"Error: {ctx.error}")
    if ctx.actual_net_pnl is not None:
        lines.append(f"Actual net PnL: ${ctx.actual_net_pnl:.4f}")
    if ctx.duration_ms is not None:
        lines.append(f"Duration: {ctx.duration_ms:.0f} ms")
    ev_str = ", ".join(e.to_dict().get("to", "") for e in ctx.events[:5])
    lines.append("Events: " + ev_str)
    text = "\n".join(lines)
    if len(text) > TELEGRAM_MAX_LEN:
        text = text[: TELEGRAM_MAX_LEN - 3] + "..."
    return text


def format_double_limit_report(
    result: Dict[str, Any],
    opp: Any,
) -> str:
    """
    Full execution report for Double Limit: opportunity + CEX + DEX + outcome.

    Supports both legacy V3 range-order results and new ODOS swap results.
    opp = DoubleLimitOpportunity; result = return value of execute_double_limit().
    """
    status = result.get("status", "UNKNOWN")
    tok = getattr(opp, "token_symbol", opp)
    direction = getattr(opp, "direction", "—")
    mex_bid = getattr(opp, "mex_bid", 0)
    mex_ask = getattr(opp, "mex_ask", 0)
    odos_price = getattr(opp, "odos_price", 0)
    gross = getattr(opp, "gross_spread", 0) * 100
    net_p = getattr(opp, "net_profit_usd", 0)
    exec_flag = getattr(opp, "executable", False)
    lines = [
        "━━ DOUBLE LIMIT EXECUTION REPORT ━━",
        f"Token: {tok}  dir: {direction}",
        f"  mex_bid={mex_bid:.4f}  mex_ask={mex_ask:.4f}  odos_price={odos_price:.4f}",
        (
            f"  gross_spread={gross:.2f}%  net_profit_usd=${net_p:.4f}  "
            f"executable={exec_flag}"
        ),
        "",
        "CEX (MEXC):",
    ]
    mex = result.get("mex_order")
    if mex is not None:
        order_id = getattr(mex, "order_id", None) or mex.get("order_id")
        st = getattr(mex, "status", None) or mex.get("status", "—")
        filled = getattr(mex, "executed_qty", None) or mex.get("executed_qty")
        price = getattr(mex, "price", None) or mex.get("price")
        leg_cex = LegSummary(
            venue="cex",
            order_id_or_tx=str(order_id) if order_id else None,
            status=st,
            filled=float(filled) if filled is not None else None,
            price=float(price) if price is not None else None,
        )
        lines.append(leg_cex.to_line())
    else:
        lines.append("  (no order)")
    lines.append("")

    # DEX leg: new ODOS swap path or legacy V3 range order
    dex_tx_hash = result.get("dex_tx_hash")
    dex_ok = result.get("dex_success", False)

    if dex_tx_hash is not None or "dex_success" in result:
        lines.append("DEX (ODOS swap):")
        dex_swap_res = result.get("dex_swap_result")
        gas_used = getattr(dex_swap_res, "gas_used", None) if dex_swap_res else None
        leg_dex = LegSummary(
            venue="dex",
            order_id_or_tx=dex_tx_hash,
            status="FILLED" if dex_ok else "FAILED",
            swap_executed=dex_ok,
        )
        lines.append(leg_dex.to_line())
        if gas_used:
            lines.append(f"  gas_used={gas_used}")
    else:
        lines.append("DEX (V3 range):")
        v3 = result.get("v3_status") or {}
        v3_pos_id = result.get("v3_position_id")
        swap_done = bool(v3.get("is_executed"))
        leg_dex = LegSummary(
            venue="dex",
            order_id_or_tx=f"position#{v3_pos_id}" if v3_pos_id is not None else None,
            status=(
                "EXECUTED"
                if swap_done
                else ("PENDING" if v3_pos_id is not None else "—")
            ),
            swap_executed=swap_done if v3 else None,
            extra=dict(v3) if v3 else {},
        )
        lines.append(leg_dex.to_line())
        if v3:
            lines.append(
                f"  liquidity={v3.get('liquidity')}  in_range={v3.get('in_range')}"
            )

    lines.append("")
    lines.append(f"Outcome: {status}")
    if result.get("error"):
        lines.append(f"Error: {result['error']}")
    if status == "TIMEOUT" and result.get("unwind_attempted"):
        lines.append(
            f"Unwind (MEXC): attempted=True success={result.get('unwind_success')}"
        )
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    text = "\n".join(lines)
    if len(text) > TELEGRAM_MAX_LEN:
        text = text[: TELEGRAM_MAX_LEN - 3] + "..."
    return text
