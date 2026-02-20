# Pre-Flight Checklist (1.6) — MEXC + V3 Arb (formerly Double Limit)

**Path:** MEXC + Arbitrum (Uniswap V3 + ODOS).

Complete this checklist **before any real trades**. Submit to instructor for co-signing.

---

## Required starting balances (реальний арбітраж)

Якщо на початку на обох сторонах тільки стейблкоїни — що потрібно і як відбувається перший крок:

| Веню | Що потрібно | Чому |
|------|-------------|------|
| **MEXC** | **USDT** (мінімум ~TRADE_SIZE_USD на один трейд) | Бот купує базовий токен (ARB, GMX тощо) за USDT або продає базовий токен за USDT. Старт тільки з USDT на MEXC — **нормально**. |
| **Arbitrum** | **USDC** (не USDT) + **ETH** на газ | На Arbitrum бот торгує парами **USDC/токен** (Uniswap V3). Ноги: або віддаєш USDC і отримуєш токен, або віддаєш токен і отримуєш USDC. Тому на кошельку має бути **USDC** і трохи **ETH**. |

**Якщо на Arbitrum тільки USDT:**
Пул на Arbitrum у боті — USDC/токен. Щоб бот міг виставляти ордери на Arbitrum, потрібен **USDC**. Варіанти: обміняти USDT→USDC на Arbitrum (Uniswap, 1inch тощо) або перевести USDC з CEX/іншої мережі. Після появи USDC (+ ETH на газ) можна вмикати live.

**Перший можливий трейд (старт тільки з USDT на MEXC і USDC на Arbitrum):**

1. **mex_to_arb:** на MEXC бот **купує** базовий токен за USDT (лімітний ордер); на Arbitrum **продає** USDC за базовий токен (range order). Потрібно: USDT на MEXC, USDC на Arbitrum.
2. Після заповнення обох ніг у тебе з’являється базовий токен на обох сторонах.
3. **arb_to_mex:** можливий наступний крок — продаж базового токена на MEXC за USDT і на Arbitrum — продаж токена за USDC.

Підсумок: **MEXC — достатньо USDT; Arbitrum — обов’язково USDC + ETH.** Якщо на Arbitrum лише USDT, спочатку конвертуй у USDC.

**Тестовий ран (наприклад $40: по $20 на кожну сторону)**
Купи на Binance USDT, потім: MEXC — депозит ~$20 USDT; Arbitrum — вивід ~$20 USDC (+ трохи ETH на газ). У `.env`: `TRADE_SIZE_USD=5` (або `3` для більшої кількості менших трейдів). Double Limit не перевіряє мінімальний капітал — тест на $20+$20 можливий.

---

## Code Readiness

| Item | Status | Notes |
|------|--------|--------|
| Bot connects to MEXC | ✅ / ☐ | `.env`: `MEXC_API_KEY`, `MEXC_API_SECRET`, `MEXC_BASE_URL`. Bot uses `MexcClient` for order book and limit orders. Verify: run `python scripts/arb_bot.py --mode mexc_v3` (or legacy `double_limit`) and confirm logs show MEXC bid/ask (no API errors). |
| Bot connects to Arbitrum | ✅ | `.env`: `ARBITRUM_RPC_HTTPS`. ODOS quotes + Uniswap V3 range orders. Verify: `python scripts/verify_all_costs.py` or run `--mode mexc_v3` and see ODOS prices in logs. |
| Fee calculation uses real fees | ✅ | `DoubleLimitConfig`: gas, LP fee by tier (0.05% / 0.3% / 1%), ODOS fee, bridge amortization (`src/executor/double_limit_engine.py`). Min spread by tier: 0.7% (500), 1.0% (3000), 1.5% (10000). |
| Risk limits configured & justified | ✅ | `TRADE_SIZE_USD` (e.g. $5), `MIN_SPREAD_PCT` / per-tier min spread, `MIN_PROFIT_USD`. Safety: `ABSOLUTE_*` in `src/safety.py` (max trade $25, daily loss $20, min capital $50, 30 trades/hr). Written justification in `safety.py` comments (sized for ~$100 starting capital). |
| Kill switch tested | ✅ / ☐ | Telegram `/kill` → creates file; bot stops new trades. `/resume` → clears. Double Limit checks `is_kill_switch_active()` every loop. Test: send `/kill`, confirm no new executions; `/resume` to continue. |
| Safety constants hardcoded (ABSOLUTE_*) | ✅ | `src/safety.py`: `ABSOLUTE_MAX_TRADE_USD`, `ABSOLUTE_MAX_DAILY_LOSS`, `ABSOLUTE_MIN_CAPITAL`, `ABSOLUTE_MAX_TRADES_PER_HOUR` — "DO NOT MODIFY". |
| `safety_check()` wired in Double Limit | ✅ | Called before every execution in `demo_double_limit.py`: checks trade size, daily loss, capital, hourly trades. |
| Circuit breaker wired in Double Limit | ✅ | Global + per-pair `CircuitBreaker` from `recovery.py` in `demo_double_limit.py`. Trips on 3 failures / $50 drawdown. |
| Suspicious spread rejection (>500 bps) | ✅ | `MAX_SUSPICIOUS_SPREAD_PCT = 0.05` in `double_limit_engine.py` — rejects spreads > 5% as likely stale/bad data. |
| PRODUCTION flag logged | ✅ | `PRODUCTION=true/false` in `.env`; logged prominently at startup in `demo_double_limit.py`. |
| Heartbeat / liveness | ✅ | Prometheus `/health` endpoint via `MetricsServer` (port 9090). Heartbeat log every 5 min in main loop. |
| Start / stop alerts | ✅ | Telegram WARNING-level messages on bot start and stop with mode, PnL, trade count. |
| Dry run completed (30+ min logs) | ☐ | Run `python scripts/arb_bot.py --mode mexc_v3` (or `double_limit`) for ≥30 min (no `--execute`). Attach `logs/double_limit_YYYYMMDD.log`. |

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
