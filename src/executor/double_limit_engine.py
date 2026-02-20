from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import get_env
from exchange.mexc_client import MexcApiError, MexcClient, MexcOrderStatus
from pricing.odos_client import OdosClient, OdosQuote

logger = logging.getLogger(__name__)

V3_GAS_ESTIMATE = 150_000  # Typical V3 exactInputSingle gas
MAX_PRICE_IMPACT_PCT = 0.02  # Hard stop: skip if computed price impact > 2% (0.02 = 2%)
MAX_SUSPICIOUS_SPREAD_PCT = 0.05  # 5% (500 bps) — reject as likely stale/bad data


# ── Route health tracking ─────────────────────────────────────────


class RouteHealthTracker:
    """
    Track gas usage per token/route to auto-detect unreliable routes.

    If a route consistently uses high gas (avg > threshold), the tracker
    marks it unreliable so the engine can prefer alternatives.
    """

    def __init__(self, window: int = 10, unreliable_avg_gas: int = 600_000) -> None:
        self._gas_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._unreliable_threshold = unreliable_avg_gas

    def record(self, token: str, source: str, gas_used: int) -> None:
        key = f"{token}:{source}"
        self._gas_history[key].append(gas_used)

    def avg_gas(self, token: str, source: str) -> Optional[float]:
        key = f"{token}:{source}"
        h = self._gas_history[key]
        return sum(h) / len(h) if h else None

    def is_reliable(self, token: str, source: str) -> bool:
        avg = self.avg_gas(token, source)
        if avg is None:
            return True
        return avg < self._unreliable_threshold

    def estimated_gas(self, token: str, source: str, fallback: int) -> int:
        """
        Return an integer gas estimate based on historical usage for the
        given token/source pair, falling back to ``fallback`` when there
        is not enough data yet.
        """
        avg = self.avg_gas(token, source)
        if avg is None or avg <= 0:
            return int(fallback)
        return int(avg)


@dataclass
class DoubleLimitConfig:
    """
    Configuration for the Double Limit micro-arbitrage engine.
    """

    trade_size_usd: float = 5.0
    # Legacy flat thresholds (kept as sensible defaults / fallback).
    # NOTE: dynamic per-fee-tier thresholds are applied via ``min_spread_by_tier``.
    min_spread_pct: float = 0.004  # 0.4% minimum gross spread (fallback)
    min_profit_usd: float = 0.001  # do not bother for < $0.001
    max_slippage_pct: float = 0.5
    # Cost model — dynamic gas estimation from ODOS quote
    # ``gas_cost_usd`` is a static fallback used when ODOS gas_estimate is unavailable.
    gas_cost_usd: float = 0.03
    # Arbitrum gas price (L2 execution price, ~0.01-0.1 gwei typical)
    arb_gas_price_gwei: float = 0.1
    # ETH price in USD — used to convert gas units to USD.
    # Override via ETH_PRICE_USD env var or update at startup.
    eth_price_usd: float = 2600.0
    # ODOS "surplus" fee is small but non-zero (0.01% = 1 bp)
    odos_fee_pct: float = 0.0001
    # MEXC fee model: maker=0% (post-only), taker=0.1%
    mexc_maker_fee_pct: float = 0.0
    mexc_taker_fee_pct: float = 0.001
    # Bridge amortization: do not assume infinite trades; use at least N.
    min_trades_for_bridge_amortization: int = 5
    position_ttl_seconds: int = 600  # 10 minutes
    monitor_interval_seconds: float = 1.0
    usdc_decimals: int = 6
    simulation_mode: bool = True
    # When only one leg fills by deadline: reverse the filled leg to flatten position.
    enable_unwind_on_timeout: bool = True
    # Hard cap: reject ODOS routes that estimate more gas than this
    max_dex_gas_limit: int = 1_200_000
    # ODOS quote cache staleness (seconds). Reuse a cached quote if younger than this.
    quote_cache_ttl_seconds: float = 4.0

    # Maximum number of concurrent ODOS /quote requests when evaluating
    # all tokens in parallel.  Helps avoid hitting ODOS HTTP 429 limits
    # when many tokens are active.
    max_concurrent_odos_quotes: int = 4

    # Dynamic spread thresholds by Uniswap V3 fee tier.
    # Tuned for $5-10 micro-arb: tighter than before because we now model
    # MEXC fees explicitly and use bidirectional ODOS quotes.
    # 500   → 0.05% pool fee  → require 0.45% spread  (ARB, GNS)
    # 3000  → 0.30% pool fee  → require 0.65% spread  (MAGIC, PENDLE, LINK, UNI, GMX)
    # 10000 → 1.00% pool fee  → require 1.2% spread   (RDNT)
    min_spread_by_tier: Dict[int, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.min_spread_by_tier is None:
            self.min_spread_by_tier = {
                # 0.45% for 0.05% fee tier (ARB, GNS)
                500: 0.0045,
                # 0.65% for 0.3% fee tier (GMX, MAGIC, PENDLE, LINK, UNI)
                3_000: 0.0065,
                # 1.2% for 1% fee-tier pools (RDNT)
                10_000: 0.012,
            }

    def get_min_spread(self, fee_tier: int) -> float:
        """
        Return the minimum required gross spread for a given V3 fee tier.

        Falls back to ``min_spread_pct`` if tier is unknown.
        """
        return float(self.min_spread_by_tier.get(int(fee_tier), self.min_spread_pct))

    def estimate_gas_cost_usd(self, gas_estimate: int) -> float:
        """
        Convert ODOS gas estimate (gas units) to USD.

        Formula: gas_units * gas_price_gwei * 1e-9 * eth_price_usd
        Example: 200_000 * 0.1 * 1e-9 * 2600 = $0.052
        """
        if gas_estimate <= 0:
            return self.gas_cost_usd
        return gas_estimate * self.arb_gas_price_gwei * 1e-9 * self.eth_price_usd

    def mexc_fee_usd(self, trade_usd: float, post_only: bool = True) -> float:
        """MEXC fee for a trade. Post-only (maker) = 0%, market (taker) = 0.1%."""
        rate = self.mexc_maker_fee_pct if post_only else self.mexc_taker_fee_pct
        return rate * trade_usd

    @staticmethod
    def score_route(
        net_profit_usd: float,
        gas_units: int,
        source: str,
    ) -> float:
        """
        Gas-adjusted route score.

        Penalizes complex routes (high gas) and adds uncertainty penalty
        for ODOS (its gas estimates are less predictable than V3 direct).
        """
        gas_penalty = max(0, gas_units - 250_000) / 1_000_000 * 0.50
        uncertainty_penalty = 0.002 if source == "odos" else 0.0
        return net_profit_usd - gas_penalty - uncertainty_penalty


# ── ODOS quote cache ──────────────────────────────────────────


@dataclass
class _CachedQuote:
    quote: OdosQuote
    timestamp: float


class OdosQuoteCache:
    """
    Per-token, per-direction cache to avoid redundant ODOS HTTP calls
    within a short staleness window (default 4s).
    """

    def __init__(self, ttl: float = 4.0) -> None:
        self._ttl = ttl
        self._store: Dict[str, _CachedQuote] = {}

    def get(self, key: str) -> Optional[OdosQuote]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() - entry.timestamp > self._ttl:
            del self._store[key]
            return None
        return entry.quote

    def put(self, key: str, quote: OdosQuote) -> None:
        self._store[key] = _CachedQuote(quote=quote, timestamp=time.time())


@dataclass
class DoubleLimitOpportunity:
    """
    Evaluated opportunity between MEXC and Arbitrum for a single token.
    """

    token_symbol: str
    token_address: str
    mex_symbol: str
    direction: str
    mex_bid: float
    mex_ask: float
    odos_price: float
    gross_spread: float
    total_cost_usd: float
    net_profit_usd: float
    net_profit_pct: float
    executable: bool
    odos_gas_estimate: int = 0
    estimated_gas_cost_usd: float = 0.0
    use_v3_direct: bool = False
    price_impact: float = 0.0
    route_score: float = 0.0


class DoubleLimitArbitrageEngine:
    """
    High-level coordination for the Double Limit strategy.

    This component is intentionally decoupled from the existing ``Executor``
    state machine so that micro-arbitrage can evolve independently while
    still sharing lower-level primitives (MEXC client, chain client, ODOS).

    The on-chain DEX leg is executed via ``dex_swap_manager`` (DexSwapManager)
    which performs an immediate ODOS swap (quote → assemble → approve → send).
    The legacy ``range_manager`` (V3 range orders) is still accepted for
    backward compatibility but ``dex_swap_manager`` takes precedence.
    """

    def __init__(
        self,
        mexc_client: MexcClient,
        odos_client: OdosClient,
        token_mappings: Dict[str, Dict[str, Any]],
        config: Optional[DoubleLimitConfig] = None,
        range_manager: Any | None = None,
        dex_swap_manager: Any | None = None,
        capital_manager: Any | None = None,
        route_health: Optional[RouteHealthTracker] = None,
    ) -> None:
        self.mexc = mexc_client
        self.odos = odos_client
        self.tokens = token_mappings
        self.config = config or DoubleLimitConfig()
        self.range_manager = range_manager
        self.dex_swap_manager = dex_swap_manager
        self.capital_manager = capital_manager
        self.route_health = route_health or RouteHealthTracker()
        self._quote_cache = OdosQuoteCache(ttl=self.config.quote_cache_ttl_seconds)

        # Environment-derived addresses
        usdc = get_env("USDC_ADDRESS", required=True)
        user = get_env("ARBITRUM_WALLET_ADDRESS", required=True)
        assert usdc is not None
        assert user is not None
        self.usdc_address: str = usdc
        self.user_address: str = user

    # ── Batch evaluation (parallel) ────────────────────────────────

    async def evaluate_all(
        self,
        keys: Optional[List[str]] = None,
    ) -> List[Optional[DoubleLimitOpportunity]]:
        """
        Evaluate all tokens in parallel using concurrent ODOS quotes
        and MEXC order book fetches.

        Returns a list of opportunities (None for tokens with no opportunity).
        This replaces the sequential per-token loop in the demo script.
        """
        if keys is None:
            keys = list(self.tokens.keys())

        loop = asyncio.get_event_loop()
        odos_semaphore = asyncio.Semaphore(self.config.max_concurrent_odos_quotes)

        # Phase 1: fetch all MEXC order books concurrently (sync → thread pool)
        async def _fetch_book(mex_symbol: str) -> Optional[dict]:
            try:
                return await loop.run_in_executor(
                    None, lambda: self.mexc.get_order_book(mex_symbol, limit=5)
                )
            except MexcApiError as exc:
                logger.warning("MEXC order book failed for %s: %s", mex_symbol, exc)
                return None

        book_tasks = {}
        for key in keys:
            cfg = self.tokens.get(key)
            if (
                not cfg
                or not cfg.get("active", True)
                or not cfg.get("odos_supported", True)
            ):
                continue
            book_tasks[key] = _fetch_book(cfg["mex_symbol"])

        books: Dict[str, Optional[dict]] = {}
        if book_tasks:
            results = await asyncio.gather(*book_tasks.values(), return_exceptions=True)
            for key_name, result in zip(book_tasks.keys(), results):
                books[key_name] = result if isinstance(result, dict) else None

        # Phase 2: fetch ODOS quotes concurrently for both directions
        amount_usdc_raw = int(
            self.config.trade_size_usd * (10**self.config.usdc_decimals)
        )

        async def _fetch_odos(
            input_token: str,
            output_token: str,
            amount_in: int,
            cache_key: str,
        ) -> Optional[OdosQuote]:
            cached = self._quote_cache.get(cache_key)
            if cached is not None:
                return cached
            try:
                async with odos_semaphore:
                    # First attempt
                    try:
                        q = await loop.run_in_executor(
                            None,
                            lambda: self.odos.quote(
                                input_token=input_token,
                                output_token=output_token,
                                amount_in=amount_in,
                                user_address=self.user_address,
                                slippage_percent=self.config.max_slippage_pct,
                            ),
                        )
                    except Exception as exc:
                        msg = str(exc)
                        if (
                            "429" in msg
                            or "Too Many Requests" in msg
                            or "Rate limit exceeded" in msg
                        ):
                            logger.warning(
                                "ODOS rate limited (%s) – "
                                "backing off and retrying once",
                                cache_key,
                            )
                            await asyncio.sleep(0.5)
                            q = await loop.run_in_executor(
                                None,
                                lambda: self.odos.quote(
                                    input_token=input_token,
                                    output_token=output_token,
                                    amount_in=amount_in,
                                    user_address=self.user_address,
                                    slippage_percent=self.config.max_slippage_pct,
                                ),
                            )
                        else:
                            raise
                self._quote_cache.put(cache_key, q)
                return q
            except Exception as exc:
                logger.warning("ODOS quote failed (%s): %s", cache_key, exc)
                return None

        odos_coroutines: Dict[str, Tuple[Any, Any]] = {}
        for key in books:
            book = books[key]
            if book is None:
                continue
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            if not bids or not asks:
                continue

            cfg = self.tokens[key]
            token_address = cfg["address"]
            token_decimals = int(cfg.get("decimals", 18))

            buy_key = f"{key}:buy"
            buy_coro = _fetch_odos(
                self.usdc_address,
                token_address,
                amount_usdc_raw,
                buy_key,
            )

            mex_mid = (bids[0][0] + asks[0][0]) / 2.0
            est_tokens = self.config.trade_size_usd / mex_mid
            token_amount_raw = int(est_tokens * (10**token_decimals))
            sell_key = f"{key}:sell"
            sell_coro = _fetch_odos(
                token_address,
                self.usdc_address,
                token_amount_raw,
                sell_key,
            )

            odos_coroutines[key] = (buy_coro, sell_coro)

        flat_coros: List[Any] = []
        task_map: List[Tuple[str, str]] = []
        for key, (bc, sc) in odos_coroutines.items():
            flat_coros.append(bc)
            task_map.append((key, "buy"))
            flat_coros.append(sc)
            task_map.append((key, "sell"))

        odos_results: Dict[str, Dict[str, Optional[OdosQuote]]] = defaultdict(dict)
        if flat_coros:
            results = await asyncio.gather(*flat_coros, return_exceptions=True)
            for (key_name, direction), result in zip(task_map, results):
                if isinstance(result, BaseException):
                    odos_results[key_name][direction] = None
                else:
                    # type: ignore[assignment]
                    odos_results[key_name][direction] = result

        # Phase 3: evaluate each token synchronously (fast — just arithmetic)
        opportunities: List[Optional[DoubleLimitOpportunity]] = []
        for key in keys:
            book = books.get(key)
            quotes = odos_results.get(key)
            if book is None or quotes is None:
                opportunities.append(None)
                continue
            opp = self._evaluate_with_quotes(
                key, book, quotes.get("buy"), quotes.get("sell")
            )
            opportunities.append(opp)

        return opportunities

    # ── Single-token evaluation (with pre-fetched data) ────────────

    def _evaluate_with_quotes(
        self,
        key: str,
        book: dict,
        buy_quote: Optional[OdosQuote],
        sell_quote: Optional[OdosQuote],
    ) -> Optional[DoubleLimitOpportunity]:
        """
        Evaluate a single token with pre-fetched MEXC book and ODOS quotes.

        ``buy_quote``: USDC→token (for arb sell side / mex_to_arb direction)
        ``sell_quote``: token→USDC (for arb buy side / arb_to_mex direction)
        """
        cfg = self.tokens.get(key)
        if not cfg:
            return None

        token_symbol = key
        token_address = cfg["address"]
        token_decimals = int(cfg.get("decimals", 18))
        mex_symbol = cfg["mex_symbol"]

        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            return None

        mex_bid = bids[0][0]
        mex_ask = asks[0][0]

        # --- Direction A: mex_to_arb (buy MEXC cheap, sell on Arb via ODOS) ---
        # We buy tokens on MEXC at ask, sell on Arb: need sell_quote (token→USDC)
        # But we also need buy_quote to know the "arb buy price" for reference.
        spread_mex_cheaper = 0.0
        arb_price_buy = None
        if buy_quote and buy_quote.amount_out > 0:
            token_out_human = buy_quote.amount_out / float(10**token_decimals)
            if token_out_human > 0:
                arb_price_buy = self.config.trade_size_usd / token_out_human
                spread_mex_cheaper = (arb_price_buy - mex_ask) / mex_ask

        # --- Direction B: arb_to_mex (buy on Arb via ODOS, sell MEXC) ---
        # We buy tokens on Arb, sell on MEXC at bid: need sell_quote for actual USD out
        spread_arb_cheaper = 0.0
        arb_price_sell = None
        if sell_quote and sell_quote.amount_out > 0:
            usdc_out_human = sell_quote.amount_out / float(
                10**self.config.usdc_decimals
            )
            mex_mid = (mex_bid + mex_ask) / 2.0
            est_tokens = self.config.trade_size_usd / mex_mid
            if est_tokens > 0:
                arb_price_sell = self.config.trade_size_usd / est_tokens
                actual_sell_price = usdc_out_human / est_tokens
                spread_arb_cheaper = (mex_bid - actual_sell_price) / actual_sell_price

        # Pick better direction
        if spread_mex_cheaper <= 0 and spread_arb_cheaper <= 0:
            self._record_gas_from_quotes(token_symbol, buy_quote, sell_quote)
            return None

        if spread_mex_cheaper >= spread_arb_cheaper:
            direction = "mex_to_arb"
            best_spread = spread_mex_cheaper
            arbitrum_price = arb_price_buy or 0.0
            active_quote = buy_quote
        else:
            direction = "arb_to_mex"
            best_spread = spread_arb_cheaper
            arbitrum_price = arb_price_sell or 0.0
            active_quote = sell_quote

        if active_quote is None or arbitrum_price <= 0:
            return None

        # Cost model
        fee_tier = int(cfg.get("fee_tier", 3_000))
        lp_fee_pct = self._lp_fee_pct(fee_tier)
        lp_fee_usd = lp_fee_pct * self.config.trade_size_usd
        odos_fee_usd = float(self.config.odos_fee_pct) * self.config.trade_size_usd
        mexc_fee_usd = self.config.mexc_fee_usd(
            self.config.trade_size_usd, post_only=True
        )

        bridge_amortized = 0.0
        if self.capital_manager is not None:
            fixed = float(
                getattr(
                    getattr(self.capital_manager, "config", None),
                    "bridge_fixed_cost_usd",
                    0.0,
                )
                or 0.0
            )
            trades = int(
                getattr(self.capital_manager, "trade_count_since_bridge", 0) or 0
            )
            denom = max(trades, int(self.config.min_trades_for_bridge_amortization))
            bridge_amortized = fixed / float(denom) if fixed > 0 else 0.0

        # Price impact hard stop (computed vs MEXC mid)
        mex_mid_price = (mex_bid + mex_ask) / 2.0
        if buy_quote and buy_quote.amount_out > 0:
            expected_token_out = self.config.trade_size_usd / mex_mid_price
            actual_token_out = buy_quote.amount_out / float(10**token_decimals)
            if expected_token_out > 0 and actual_token_out < expected_token_out:
                computed_price_impact = (
                    expected_token_out - actual_token_out
                ) / expected_token_out
            else:
                computed_price_impact = 0.0
        else:
            computed_price_impact = 0.0

        price_impact = computed_price_impact
        if price_impact > MAX_PRICE_IMPACT_PCT:
            logger.warning(
                "%s SKIP: price impact %.2f%% > %.1f%% hard stop",
                token_symbol,
                price_impact * 100,
                MAX_PRICE_IMPACT_PCT * 100,
            )
            self._record_gas_from_quotes(token_symbol, buy_quote, sell_quote)
            return None

        gross_profit_usd = best_spread * self.config.trade_size_usd

        # Route comparison — V3 direct vs ODOS
        odos_gas_est = active_quote.gas_estimate
        v3_pool = cfg.get("v3_pool")
        has_v3_direct = bool(v3_pool and fee_tier)

        base_cost = lp_fee_usd + bridge_amortized + mexc_fee_usd

        # ODOS route
        odos_gas_cost = (
            self.config.estimate_gas_cost_usd(odos_gas_est)
            if odos_gas_est > 0
            else float(self.config.gas_cost_usd)
        )
        odos_total_cost = odos_gas_cost + base_cost + odos_fee_usd
        odos_net_profit = gross_profit_usd - odos_total_cost
        odos_score = self.config.score_route(odos_net_profit, odos_gas_est, "odos")

        odos_reliable = self.route_health.is_reliable(token_symbol, "odos")
        if not odos_reliable:
            avg_g = self.route_health.avg_gas(token_symbol, "odos")
            logger.info(
                "%s ODOS route unreliable (avg_gas=%.0f), penalizing",
                token_symbol,
                avg_g or 0,
            )
            odos_score -= 0.01

        # V3 direct route
        v3_net_profit = float("-inf")
        v3_gas_cost = 0.0
        v3_total_cost = 0.0
        v3_score = float("-inf")
        v3_reliable = self.route_health.is_reliable(token_symbol, "v3_direct")
        if has_v3_direct and v3_reliable:
            v3_gas_units = self.route_health.estimated_gas(
                token_symbol, "v3_direct", V3_GAS_ESTIMATE
            )
            v3_gas_cost = self.config.estimate_gas_cost_usd(v3_gas_units)
            v3_total_cost = v3_gas_cost + base_cost
            v3_net_profit = gross_profit_usd - v3_total_cost
            v3_score = self.config.score_route(v3_net_profit, v3_gas_units, "v3_direct")

        if has_v3_direct and v3_score > odos_score:
            use_v3_direct = True
            gas_cost_usd = v3_gas_cost
            total_cost = v3_total_cost
            net_profit_usd = v3_net_profit
            effective_gas_estimate = int(
                self.route_health.estimated_gas(
                    token_symbol, "v3_direct", V3_GAS_ESTIMATE
                )
            )
            chosen_score = v3_score
            route_label = "V3 direct"
        else:
            use_v3_direct = False
            gas_cost_usd = odos_gas_cost
            total_cost = odos_total_cost
            net_profit_usd = odos_net_profit
            effective_gas_estimate = odos_gas_est
            chosen_score = odos_score
            route_label = "ODOS"

        net_profit_pct = net_profit_usd / self.config.trade_size_usd
        min_spread_required = self.config.get_min_spread(fee_tier)

        if best_spread > MAX_SUSPICIOUS_SPREAD_PCT:
            logger.warning(
                "%s REJECTED: spread %.2f%% > %d bps — likely stale/bad data",
                token_symbol,
                best_spread * 100,
                int(MAX_SUSPICIOUS_SPREAD_PCT * 10_000),
            )
            return DoubleLimitOpportunity(
                token_symbol=token_symbol,
                token_address=token_address,
                mex_symbol=mex_symbol,
                direction=direction,
                mex_bid=mex_bid,
                mex_ask=mex_ask,
                odos_price=arbitrum_price,
                gross_spread=best_spread,
                total_cost_usd=total_cost,
                net_profit_usd=net_profit_usd,
                net_profit_pct=net_profit_pct,
                executable=False,
                odos_gas_estimate=effective_gas_estimate,
                estimated_gas_cost_usd=gas_cost_usd,
                use_v3_direct=use_v3_direct,
                price_impact=price_impact,
                route_score=chosen_score,
            )

        executable = (
            best_spread >= min_spread_required
            and net_profit_usd >= self.config.min_profit_usd
            and chosen_score > 0.0
        )

        if executable:
            logger.info(
                "%s VIABLE [%s|%s]: spread=%.3f%% gas=%d "
                "gas$=%.4f mex_fee$=%.4f net=$%.4f score=%.5f",
                token_symbol,
                route_label,
                direction,
                best_spread * 100,
                effective_gas_estimate,
                gas_cost_usd,
                mexc_fee_usd,
                net_profit_usd,
                chosen_score,
            )
        else:
            skip_reasons = []
            if best_spread < min_spread_required:
                skip_reasons.append(
                    f"spread {best_spread*100:.2f}% < {min_spread_required*100:.2f}%"
                )
            if net_profit_usd < self.config.min_profit_usd:
                skip_reasons.append(
                    f"net ${net_profit_usd:.4f} < ${self.config.min_profit_usd:.4f}"
                )
            if chosen_score <= 0.0:
                skip_reasons.append(f"score {chosen_score:.5f} <= 0")
            logger.debug(
                "%s SKIP [%s|%s]: %s (gas=%d score=%.5f)",
                token_symbol,
                route_label,
                direction,
                ", ".join(skip_reasons) if skip_reasons else "unknown",
                effective_gas_estimate,
                chosen_score,
            )

        if effective_gas_estimate > 0:
            route_key = "v3_direct" if use_v3_direct else "odos"
            self.route_health.record(token_symbol, route_key, effective_gas_estimate)

        return DoubleLimitOpportunity(
            token_symbol=token_symbol,
            token_address=token_address,
            mex_symbol=mex_symbol,
            direction=direction,
            mex_bid=mex_bid,
            mex_ask=mex_ask,
            odos_price=arbitrum_price,
            gross_spread=best_spread,
            total_cost_usd=total_cost,
            net_profit_usd=net_profit_usd,
            net_profit_pct=net_profit_pct,
            executable=executable,
            odos_gas_estimate=effective_gas_estimate,
            estimated_gas_cost_usd=gas_cost_usd,
            use_v3_direct=use_v3_direct,
            price_impact=price_impact,
            route_score=chosen_score,
        )

    def _record_gas_from_quotes(
        self,
        token_symbol: str,
        buy_quote: Optional[OdosQuote],
        sell_quote: Optional[OdosQuote],
    ) -> None:
        """Record gas estimates from quotes for health tracking."""
        for q in (buy_quote, sell_quote):
            if q and q.gas_estimate > 0:
                self.route_health.record(token_symbol, "odos", q.gas_estimate)

    # ── Legacy synchronous evaluate (kept for backward compatibility) ──

    def evaluate_opportunity(self, key: str) -> Optional[DoubleLimitOpportunity]:
        """
        Synchronous single-token evaluation. Prefer ``evaluate_all()`` for
        parallel batch evaluation.
        """
        cfg = self.tokens.get(key)
        if (
            not cfg
            or not cfg.get("active", True)
            or not cfg.get("odos_supported", True)
        ):
            return None

        token_address = cfg["address"]
        token_decimals = int(cfg.get("decimals", 18))
        mex_symbol = cfg["mex_symbol"]

        try:
            book = self.mexc.get_order_book(mex_symbol, limit=5)
        except MexcApiError as exc:
            logger.warning("MEXC order book failed for %s: %s", mex_symbol, exc)
            return None

        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            return None

        amount_usdc_raw = int(
            self.config.trade_size_usd * (10**self.config.usdc_decimals)
        )

        # Buy quote: USDC → token
        buy_quote: Optional[OdosQuote] = None
        cache_key_buy = f"{key}:buy"
        buy_quote = self._quote_cache.get(cache_key_buy)
        if buy_quote is None:
            try:
                buy_quote = self.odos.quote(
                    input_token=self.usdc_address,
                    output_token=token_address,
                    amount_in=amount_usdc_raw,
                    user_address=self.user_address,
                    slippage_percent=self.config.max_slippage_pct,
                )
                self._quote_cache.put(cache_key_buy, buy_quote)
            except Exception as exc:
                logger.warning("ODOS buy quote failed for %s: %s", key, exc)

        # Sell quote: token → USDC
        sell_quote: Optional[OdosQuote] = None
        cache_key_sell = f"{key}:sell"
        sell_quote = self._quote_cache.get(cache_key_sell)
        if sell_quote is None:
            mex_mid = (bids[0][0] + asks[0][0]) / 2.0
            est_tokens = self.config.trade_size_usd / mex_mid
            token_amount_raw = int(est_tokens * (10**token_decimals))
            try:
                sell_quote = self.odos.quote(
                    input_token=token_address,
                    output_token=self.usdc_address,
                    amount_in=token_amount_raw,
                    user_address=self.user_address,
                    slippage_percent=self.config.max_slippage_pct,
                )
                self._quote_cache.put(cache_key_sell, sell_quote)
            except Exception as exc:
                logger.warning("ODOS sell quote failed for %s: %s", key, exc)

        return self._evaluate_with_quotes(key, book, buy_quote, sell_quote)

    @staticmethod
    def _lp_fee_pct(fee_tier: int) -> float:
        """
        Convert Uniswap V3 fee tier (uint24) to LP fee percent.

        fee_tier values:
          100   = 0.01%
          500   = 0.05%
          3000  = 0.30%
          10000 = 1.00%
        """
        return {
            100: 0.0001,
            500: 0.0005,
            3_000: 0.003,
            10_000: 0.01,
        }.get(int(fee_tier), 0.003)

    # ── Execution ─────────────────────────────────────────────────

    async def execute_double_limit(self, opp: DoubleLimitOpportunity) -> Dict[str, Any]:
        """
        Place both legs (MEXC limit + DEX swap) and monitor them.

        The DEX leg is an immediate swap via ODOS (DexSwapManager) when
        ``dex_swap_manager`` is provided. Falls back to V3 range orders
        via ``range_manager`` if present.  When neither is provided we
        still place and monitor the MEXC leg (useful for dry-runs).
        """
        if self.config.simulation_mode:
            return self._simulate_execute_double_limit(opp)

        if not opp.executable:
            return {"status": "SKIPPED", "reason": "not executable", "opportunity": opp}

        if opp.direction == "mex_to_arb":
            mex_side = "BUY"
            mex_price = opp.mex_ask * 0.9995
        else:
            mex_side = "SELL"
            mex_price = opp.mex_bid * 1.0005

        trade_size_base = self.config.trade_size_usd / opp.odos_price

        # Pre-check balances on MEXC to avoid HTTP 400
        if mex_side == "BUY":
            try:
                usdt_balance = self.mexc.get_balance("USDT")
                required_usdt = trade_size_base * mex_price
                if usdt_balance < required_usdt * 0.999:
                    logger.warning(
                        "Skip %s mex_to_arb: MEXC USDT balance %.6f < required %.6f",
                        opp.token_symbol,
                        usdt_balance,
                        required_usdt,
                    )
                    return {
                        "status": "SKIPPED",
                        "reason": "insufficient_quote_on_mexc",
                        "opportunity": opp,
                    }
            except Exception as e:
                logger.warning("Could not check MEXC USDT balance: %s", e)

        if mex_side == "SELL":
            try:
                base_balance = self.mexc.get_balance(opp.token_symbol)
                if base_balance < trade_size_base * 0.999:
                    logger.warning(
                        "Skip %s arb_to_mex: MEXC %s balance %.6f < "
                        "required %.6f (Oversold)",
                        opp.token_symbol,
                        opp.token_symbol,
                        base_balance,
                        trade_size_base,
                    )
                    return {
                        "status": "SKIPPED",
                        "reason": "insufficient_base_on_mexc",
                        "opportunity": opp,
                    }
            except Exception as e:
                logger.warning(
                    "Could not check MEXC balance for %s: %s", opp.token_symbol, e
                )

        # Side A: post-only limit on MEXC
        try:
            mex_order = self.mexc.place_limit_order(
                symbol=opp.mex_symbol,
                side=mex_side,
                quantity=trade_size_base,
                price=mex_price,
                post_only=True,
            )
        except Exception as exc:
            logger.warning("Failed to place MEXC limit order: %s", exc)
            return {"status": "FAILED", "error": str(exc), "opportunity": opp}

        # Side B: DEX leg (prefer dex_swap_manager over legacy range_manager)
        dex_swap_result: Optional[Any] = None
        dex_tx_hash: Optional[str] = None
        dex_success: bool = False

        if self.dex_swap_manager is not None:
            token_cfg = self.tokens.get(opp.token_symbol, {})
            token_decimals = int(token_cfg.get("decimals", 18))
            # Use V3 direct only if opportunity evaluation determined it's better
            fee_tier: Optional[int] = (
                int(token_cfg["fee_tier"])
                if opp.use_v3_direct and token_cfg.get("fee_tier")
                else None
            )

            if opp.direction == "mex_to_arb":
                dex_input = opp.token_address
                dex_output = self.usdc_address
                dex_amount_in = int(trade_size_base * (10**token_decimals))
            else:
                dex_input = self.usdc_address
                dex_output = opp.token_address
                dex_amount_in = int(
                    self.config.trade_size_usd * (10**self.config.usdc_decimals)
                )

            try:
                dex_swap_result = self.dex_swap_manager.execute_swap(
                    input_token=dex_input,
                    output_token=dex_output,
                    amount_in=dex_amount_in,
                    slippage_percent=self.config.max_slippage_pct,
                    fee_tier=fee_tier,
                )
                dex_success = bool(getattr(dex_swap_result, "success", False))
                dex_tx_hash = getattr(dex_swap_result, "tx_hash", None)
                route_used = getattr(dex_swap_result, "route", "unknown")
                gas_used = getattr(dex_swap_result, "gas_used", 0)

                # Record gas usage for health tracker
                if gas_used > 0:
                    self.route_health.record(opp.token_symbol, route_used, gas_used)

                if dex_success:
                    logger.info(
                        "DEX swap succeeded [%s]: tx=%s amount_out=%d gas=%d",
                        route_used,
                        dex_tx_hash,
                        getattr(dex_swap_result, "amount_out", 0),
                        gas_used,
                    )
                else:
                    logger.warning(
                        "DEX swap failed [%s]: %s",
                        route_used,
                        getattr(dex_swap_result, "error", "unknown"),
                    )
            except Exception as exc:
                logger.warning("DEX swap raised exception: %s", exc)

        # Monitor MEXC leg + check DEX outcome
        result = await self._monitor_positions(
            mex_order=mex_order,
            dex_success=dex_success,
            dex_tx_hash=dex_tx_hash,
            dex_swap_result=dex_swap_result,
            opportunity=opp,
        )
        return result

    def _simulate_execute_double_limit(
        self, opp: DoubleLimitOpportunity
    ) -> Dict[str, Any]:
        """
        Simulate a full two-leg execution without touching MEXC or Arbitrum.

        Control via env:
          DOUBLE_LIMIT_SIM_SCENARIO=success|timeout|mex_reject|dex_failed
        Default: success
        """
        scenario = (
            (get_env("DOUBLE_LIMIT_SIM_SCENARIO", "success") or "success")
            .strip()
            .lower()
        )

        if not opp.executable:
            return {"status": "SKIPPED", "reason": "not executable", "opportunity": opp}

        if opp.direction == "mex_to_arb":
            mex_side = "BUY"
        else:
            mex_side = "SELL"

        trade_size_base = self.config.trade_size_usd / max(opp.odos_price, 1e-12)

        if scenario == "mex_reject":
            return {
                "status": "FAILED",
                "error": "SIM: MEXC rejected order",
                "opportunity": opp,
            }

        mex_status = "FILLED" if scenario != "timeout" else "NEW"
        mex_order = MexcOrderStatus(
            order_id=f"sim_{opp.token_symbol}_{int(time.time())}",
            symbol=opp.mex_symbol,
            side=mex_side,
            status=mex_status,
            price=float(opp.mex_ask if mex_side == "BUY" else opp.mex_bid),
            orig_qty=float(trade_size_base),
            executed_qty=float(trade_size_base if mex_status == "FILLED" else 0.0),
        )

        dex_success = scenario in {"success"}
        dex_tx_hash: Optional[str] = None
        if scenario == "dex_failed":
            dex_success = False
        if dex_success:
            dex_tx_hash = f"0xsim_dex_{opp.token_symbol}_{int(time.time())}"

        # Legacy V3 simulation fields for backward compat
        v3_position_id: Optional[int] = None
        v3_status: Dict[str, Any] = {}
        if self.range_manager is not None:
            v3_position_id = int(time.time()) % 1_000_000
            v3_executed = scenario in {"success"}
            if scenario in {"v3_not_executed", "dex_failed"}:
                v3_executed = False
            v3_status = {
                "pool": "sim",
                "fee_tier": int(
                    self.tokens.get(opp.token_symbol, {}).get("fee_tier", 3_000)
                ),
                "in_range": not v3_executed,
                "liquidity": 1,
                "is_executed": bool(v3_executed),
                "can_withdraw": bool(v3_executed),
            }

        if scenario == "timeout":
            return {
                "status": "TIMEOUT",
                "mex_order": mex_order,
                "dex_success": False,
                "dex_tx_hash": None,
                "v3_position_id": v3_position_id,
                "v3_status": v3_status,
                "opportunity": opp,
                "unwind_attempted": True,
                "unwind_success": True,
            }

        if scenario == "dex_failed":
            return {
                "status": "TIMEOUT",
                "mex_order": mex_order,
                "dex_success": False,
                "dex_tx_hash": None,
                "v3_position_id": v3_position_id,
                "v3_status": v3_status,
                "opportunity": opp,
                "unwind_attempted": True,
                "unwind_success": True,
            }

        if scenario == "v3_not_executed" and v3_position_id is not None:
            return {
                "status": "TIMEOUT",
                "mex_order": mex_order,
                "dex_success": False,
                "dex_tx_hash": None,
                "v3_position_id": v3_position_id,
                "v3_status": v3_status,
                "opportunity": opp,
            }

        return {
            "status": "SUCCESS",
            "mex_order": mex_order,
            "dex_success": dex_success,
            "dex_tx_hash": dex_tx_hash,
            "v3_status": v3_status,
            "v3_position_id": v3_position_id,
            "opportunity": opp,
        }

    async def _monitor_positions(
        self,
        mex_order: MexcOrderStatus,
        dex_success: bool,
        dex_tx_hash: Optional[str],
        dex_swap_result: Any,
        opportunity: DoubleLimitOpportunity,
    ) -> Dict[str, Any]:
        """
        Monitor the MEXC leg; the DEX leg is already resolved (immediate swap).

        - Both OK → SUCCESS.
        - MEXC fills but DEX failed → immediate unwind on MEXC.
        - DEX OK but MEXC never fills → cancel MEXC (we hold the position until
          we can unwind the DEX side via a reverse swap).
        - Neither fills → timeout, cancel MEXC.
        """
        start = time.time()

        while time.time() - start < self.config.position_ttl_seconds:
            try:
                mex_status = self.mexc.get_order_status(
                    symbol=mex_order.symbol, order_id=mex_order.order_id
                )
            except Exception as exc:
                logger.warning("MEXC status check failed: %s", exc)
                mex_status = mex_order

            both_filled = mex_status.is_filled and dex_success

            if both_filled:
                if self.capital_manager is not None:
                    try:
                        self.capital_manager.record_trade(opportunity.net_profit_usd)
                    except Exception:
                        logger.exception("Capital manager record_trade failed")

                return {
                    "status": "SUCCESS",
                    "mex_order": mex_status,
                    "dex_success": True,
                    "dex_tx_hash": dex_tx_hash,
                    "dex_swap_result": dex_swap_result,
                    "opportunity": opportunity,
                    "both_legs": True,
                    "unwind_attempted": False,
                    "unwind_success": None,
                }

            # MEXC filled but DEX failed → unwind immediately
            if mex_status.is_filled and not dex_success:
                logger.warning(
                    "MEXC filled but DEX swap failed for %s — unwinding MEXC leg",
                    opportunity.token_symbol,
                )
                unwind_result = self._unwind_mexc_filled(opportunity, mex_status)
                return {
                    "status": "TIMEOUT",
                    "mex_order": mex_status,
                    "dex_success": False,
                    "dex_tx_hash": dex_tx_hash,
                    "dex_swap_result": dex_swap_result,
                    "opportunity": opportunity,
                    "unwind_attempted": True,
                    "unwind_success": unwind_result.get("success", False),
                }

            # DEX succeeded but MEXC not filled yet → keep waiting
            await asyncio.sleep(self.config.monitor_interval_seconds)

        # Timeout: cancel remaining MEXC order
        try:
            if mex_order.is_active:
                self.mexc.cancel_order(
                    symbol=mex_order.symbol, order_id=mex_order.order_id
                )
        except Exception:
            logger.exception(
                "Failed to cancel expired MEXC order %s", mex_order.order_id
            )

        # Fetch final MEXC status
        try:
            mex_final = self.mexc.get_order_status(
                symbol=mex_order.symbol, order_id=mex_order.order_id
            )
        except Exception as exc:
            logger.warning("Could not fetch final MEXC order status: %s", exc)
            mex_final = mex_order

        unwind_attempted = False
        unwind_success: Optional[bool] = None
        if (
            self.config.enable_unwind_on_timeout
            and mex_final.is_filled
            and mex_final.executed_qty > 0
        ):
            unwind_attempted = True
            unwind_result = self._unwind_mexc_filled(opportunity, mex_final)
            unwind_success = unwind_result.get("success", False)
            if unwind_success:
                logger.info(
                    "Unwind (MEXC) succeeded for %s %s qty=%.6f",
                    opportunity.token_symbol,
                    opportunity.direction,
                    mex_final.executed_qty,
                )
            else:
                logger.error(
                    "Unwind (MEXC) failed for %s: %s",
                    opportunity.token_symbol,
                    unwind_result.get("error", "unknown"),
                )

        return {
            "status": "TIMEOUT",
            "mex_order": mex_final,
            "dex_success": dex_success,
            "dex_tx_hash": dex_tx_hash,
            "dex_swap_result": dex_swap_result,
            "opportunity": opportunity,
            "unwind_attempted": unwind_attempted,
            "unwind_success": unwind_success,
        }

    def _unwind_mexc_filled(
        self,
        opportunity: DoubleLimitOpportunity,
        mex_status: MexcOrderStatus,
    ) -> Dict[str, Any]:
        """
        Reverse the MEXC leg when it filled but the DEX leg did not.

        - mex_to_arb: we bought token on MEXC → market SELL to flatten.
        - arb_to_mex: we sold token on MEXC → market BUY to flatten.
        """
        if self.config.simulation_mode:
            logger.info(
                "Simulated unwind: would %s %.6f %s on MEXC (%s)",
                "SELL" if opportunity.direction == "mex_to_arb" else "BUY",
                mex_status.executed_qty,
                opportunity.token_symbol,
                opportunity.mex_symbol,
            )
            return {"success": True}

        qty = mex_status.executed_qty
        if qty <= 0:
            return {"success": False, "error": "executed_qty <= 0"}

        if opportunity.direction == "mex_to_arb":
            reverse_side = "SELL"
        else:
            reverse_side = "BUY"

        try:
            order = self.mexc.place_market_order(
                symbol=opportunity.mex_symbol,
                side=reverse_side,
                quantity=qty,
            )
            filled = order.is_filled or order.executed_qty >= qty * 0.999
            return {"success": filled, "order_id": order.order_id}
        except MexcApiError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.exception("Unwind MEXC market order failed")
            return {"success": False, "error": str(e)}
