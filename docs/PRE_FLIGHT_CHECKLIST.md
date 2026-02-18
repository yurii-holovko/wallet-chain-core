# Pre-Flight Checklist (1.6) — Double Limit

**Path:** MEXC + Arbitrum (Uniswap V3 + ODOS).

Complete this checklist **before any real trades**. Submit to instructor for co-signing.

---

## Code Readiness

| Item | Status | Notes |
|------|--------|--------|
| Bot connects to MEXC | ✅ / ☐ | `.env`: `MEXC_API_KEY`, `MEXC_API_SECRET`, `MEXC_BASE_URL`. Bot uses `MexcClient` for order book and limit orders. Verify: run `python scripts/arb_bot.py --mode double_limit` and confirm logs show MEXC bid/ask (no API errors). |
| Bot connects to Arbitrum | ✅ | `.env`: `ARBITRUM_RPC_HTTPS`. ODOS quotes + Uniswap V3 range orders. Verify: `python scripts/verify_all_costs.py` or run double_limit and see ODOS prices in logs. |
| Fee calculation uses real fees | ✅ | `DoubleLimitConfig`: gas, LP fee by tier (0.05% / 0.3% / 1%), ODOS fee, bridge amortization (`src/executor/double_limit_engine.py`). Min spread by tier: 0.7% (500), 1.0% (3000), 1.5% (10000). |
| Risk limits configured | ✅ | `TRADE_SIZE_USD` (e.g. $5), `MIN_SPREAD_PCT` / per-tier min spread, `MIN_PROFIT_USD`. Safety: `ABSOLUTE_*` in `src/safety.py` (max trade, daily loss, min capital, trades/hour). |
| Kill switch tested | ✅ / ☐ | Telegram `/kill` → creates file; bot stops new trades. `/resume` → clears. Double Limit checks `is_kill_switch_active()` every loop. Test: send `/kill`, confirm no new executions; `/resume` to continue. |
| Safety constants hardcoded (ABSOLUTE_*) | ✅ | `src/safety.py`: `ABSOLUTE_MAX_TRADE_USD`, `ABSOLUTE_MAX_DAILY_LOSS`, `ABSOLUTE_MIN_CAPITAL`, `ABSOLUTE_MAX_TRADES_PER_HOUR` — "DO NOT MODIFY". |
| Dry run completed (30+ min logs) | ☐ | Run `python scripts/arb_bot.py --mode double_limit` for ≥30 min (no `--execute`). Attach `logs/double_limit_YYYYMMDD.log`. |

---

## Security

| Item | Status | Notes |
|------|--------|--------|
| MEXC API key: Spot Trading only | ☐ | In MEXC API management, restrict key to Spot only (no Futures). |
| MEXC API key: IP whitelist set | ☐ | If available, whitelist your server/IP. |
| MEXC API key: NO withdrawal permission | ✅ / ☐ | `.env` has `MEXC_ENABLE_WITHDRAWAL=false`. Confirm in MEXC dashboard the key has **no withdrawal** permission. |
| `.env` in `.gitignore` | ✅ | `.gitignore` contains `.env`. |
| No secrets in git history | ☐ | Run `git log -p` or `detect-secrets scan`; ensure `.env` / keys were never committed. |

---

## Operational

| Item | Status | Notes |
|------|--------|--------|
| Logging writes to files | ✅ | `logs/double_limit_YYYYMMDD.log` (and `logs/bot_YYYYMMDD.log` when run via arb_bot). |
| Telegram alerts working | ✅ / ☐ | Logs (level `TELEGRAM_LOG_LEVEL`), EXECUTABLE opportunities (WARNING), execution reports. Test: `python scripts/test_telegram_output.py --live`. |
| Know how to read logs | ☐ | `logs/`; format: timestamp, level, message. EXECUTABLE = opportunity above min spread/profit. |
| Emergency flatten documented | ✅ | See **Emergency flatten** below. |
| MEXC app/web ready for manual intervention | ☐ | Have MEXC Spot open to cancel orders or flatten if needed. |

---

## Emergency flatten (Double Limit)

1. **Stop the bot**
   - Send **`/kill`** in Telegram, or stop the process (Ctrl+C).

2. **Cancel open orders on MEXC**
   - MEXC Spot → Open Orders → cancel any limit orders placed by the bot.

3. **Arbitrum (Uniswap V3)**
   - Open positions: Uniswap app → connect wallet (same as `ARBITRUM_WALLET_ADDRESS`) → Positions → withdraw/reduce liquidity if needed.
   - Bot can also close via `UniswapV3RangeOrderManager` on next run if you prefer.

4. **Withdrawals**
   - Only if needed; use MEXC/Arbitrum as usual (withdrawal from MEXC only if key has permission — recommended: keep it off).

---

## Sign-off

- **Student signature:** ________________  **Date:** ________
- **Instructor sign-off:** ______________  **Date:** ________
