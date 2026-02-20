"""
DexSwapManager: execute DEX swaps on Arbitrum.

Two execution paths (chosen automatically per swap):

  1. **Direct Uniswap V3** (~130-180K gas) — for tokens with a known
     USDC pool.  Calls ``exactInputSingle`` on the V3 SwapRouter.
  2. **ODOS aggregator** (200K-1M+ gas) — fallback when no direct pool
     is available, or the V3 swap reverts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from eth_abi import encode as abi_encode
from eth_utils import keccak

from chain import ChainClient
from chain.transaction_builder import TransactionBuilder
from core.base_types import Address, TokenAmount, TransactionRequest
from core.wallet_manager import WalletManager
from pricing.odos_client import OdosClient

logger = logging.getLogger(__name__)

ARBITRUM_CHAIN_ID = 42161
UNISWAP_V3_SWAP_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
V3_DIRECT_GAS_LIMIT = 200_000


class V3FailReason(Enum):
    """Why a V3 direct swap failed — determines whether ODOS fallback is safe."""

    APPROVE_FAILED = "approve_failed"  # token approve reverted -> ODOS OK
    TX_ERROR = "tx_error"  # network / RPC error -> ODOS OK
    REVERTED = "reverted"  # on-chain revert (low liq, wrong pool) -> ODOS OK
    PRICE_IMPACT = "price_impact"  # high gas on revert hints at price impact -> SKIP
    UNKNOWN = "unknown"

    @property
    def safe_for_fallback(self) -> bool:
        return self != V3FailReason.PRICE_IMPACT


@dataclass
class DexSwapResult:
    """Outcome of an on-chain DEX swap."""

    success: bool
    tx_hash: Optional[str] = None
    amount_in: int = 0
    amount_out: int = 0
    gas_used: int = 0
    error: Optional[str] = None
    route: str = "unknown"
    v3_fail_reason: Optional[V3FailReason] = None


class DexSwapManager:
    """
    Execute DEX swaps on Arbitrum.

    Preferred path: direct Uniswap V3 single-pool swap (~150K gas).
    Fallback: ODOS aggregator (multi-hop, higher gas).
    """

    def __init__(
        self,
        odos: OdosClient,
        chain_client: ChainClient,
        wallet: WalletManager,
        chain_id: int = ARBITRUM_CHAIN_ID,
        gas_priority: str = "medium",
        tx_timeout: int = 120,
        max_gas_limit: int = 1_000_000,
        v3_router: str = UNISWAP_V3_SWAP_ROUTER,
    ) -> None:
        self._odos = odos
        self._client = chain_client
        self._wallet = wallet
        self._chain_id = chain_id
        self._gas_priority = gas_priority
        self._tx_timeout = tx_timeout
        self._max_gas_limit = max_gas_limit
        self._v3_router = v3_router

    # ── Public API ────────────────────────────────────────────

    def execute_swap(
        self,
        input_token: str,
        output_token: str,
        amount_in: int,
        slippage_percent: float = 0.5,
        fee_tier: Optional[int] = None,
    ) -> DexSwapResult:
        """
        Execute a DEX swap.

        When ``fee_tier`` is provided, attempts a direct Uniswap V3
        single-pool swap first (~150K gas).  Falls back to ODOS if
        the V3 swap fails or ``fee_tier`` is ``None``.
        """
        if fee_tier is not None:
            v3_result = self._execute_v3_direct(
                input_token=input_token,
                output_token=output_token,
                amount_in=amount_in,
                fee_tier=fee_tier,
            )
            if v3_result.success:
                return v3_result

            reason = v3_result.v3_fail_reason or V3FailReason.UNKNOWN
            if not reason.safe_for_fallback:
                logger.warning(
                    "V3 failed with %s -- skipping ODOS fallback "
                    "(same root cause likely)",
                    reason.value,
                )
                return v3_result
            logger.info(
                "V3 direct swap failed (%s, reason=%s), falling back to ODOS",
                v3_result.error,
                reason.value,
            )

        return self._execute_odos(
            input_token=input_token,
            output_token=output_token,
            amount_in=amount_in,
            slippage_percent=slippage_percent,
        )

    # ── Direct Uniswap V3 swap ────────────────────────────────

    def _execute_v3_direct(
        self,
        input_token: str,
        output_token: str,
        amount_in: int,
        fee_tier: int,
    ) -> DexSwapResult:
        """
        Single-pool swap via Uniswap V3 SwapRouter.exactInputSingle.

        Typical gas: 130-180K (vs 500K-1.4M through ODOS).
        """
        user_address = self._wallet.address
        router = self._v3_router

        # 1. Approve V3 router for input token
        try:
            self._ensure_allowance(
                token=Address.from_string(input_token),
                owner=Address.from_string(user_address),
                spender=Address.from_string(router),
                min_amount=amount_in,
            )
        except Exception as exc:
            return DexSwapResult(
                success=False,
                error=f"v3 approve failed: {exc}",
                route="v3_direct",
                v3_fail_reason=V3FailReason.APPROVE_FAILED,
            )

        # 2. Build exactInputSingle calldata
        # Note: amountOutMinimum=1 is intentionally low because:
        #   - Spread already validated in evaluate_opportunity
        #   - If V3 reverts (insufficient liquidity, pool doesn't exist),
        #     we fallback to ODOS
        #   - Direct swap is simpler than range orders
        #     (no liquidity provision, just swap)
        deadline = int(time.time()) + 300
        amount_out_minimum = 1  # Minimal check; fallback to ODOS if V3 fails
        sqrt_price_limit_x96 = 0  # No price limit

        selector = keccak(
            text=(
                "exactInputSingle((address,address,uint24,address,"
                "uint256,uint256,uint256,uint160))"
            )
        )[:4]

        params = abi_encode(
            ["(address,address,uint24,address,uint256,uint256,uint256,uint160)"],
            [
                (
                    Address.from_string(input_token).checksum,
                    Address.from_string(output_token).checksum,
                    fee_tier,
                    Address.from_string(user_address).checksum,
                    deadline,
                    amount_in,
                    amount_out_minimum,
                    sqrt_price_limit_x96,
                )
            ],
        )
        calldata = selector + params

        # 3. Send transaction
        try:
            receipt = (
                TransactionBuilder(self._client, self._wallet)
                .to(Address.from_string(router))
                .value(TokenAmount(raw=0, decimals=18, symbol="ETH"))
                .data(calldata)
                .chain_id(self._chain_id)
                .gas_limit(V3_DIRECT_GAS_LIMIT)
                .with_gas_price(self._gas_priority)
                .send_and_wait(timeout=self._tx_timeout)
            )
        except Exception as exc:
            logger.warning("V3 direct swap tx failed (will fallback to ODOS): %s", exc)
            return DexSwapResult(
                success=False,
                error=f"v3 tx failed: {exc}",
                route="v3_direct",
                v3_fail_reason=V3FailReason.TX_ERROR,
            )

        if not receipt.status:
            # High gas usage on revert often signals price impact / liquidity issues
            # that ODOS would also hit (same underlying pools)
            if receipt.gas_used > V3_DIRECT_GAS_LIMIT * 0.9:
                fail_reason = V3FailReason.PRICE_IMPACT
            else:
                fail_reason = V3FailReason.REVERTED
            logger.warning(
                "V3 direct swap reverted (tx=%s, gas=%d, reason=%s).",
                receipt.tx_hash,
                receipt.gas_used,
                fail_reason.value,
            )
            return DexSwapResult(
                success=False,
                tx_hash=receipt.tx_hash,
                gas_used=receipt.gas_used,
                error="v3 swap reverted",
                route="v3_direct",
                v3_fail_reason=fail_reason,
            )

        logger.info(
            "V3 direct swap OK: tx=%s in=%d gas=%d fee_tier=%d",
            receipt.tx_hash,
            amount_in,
            receipt.gas_used,
            fee_tier,
        )
        return DexSwapResult(
            success=True,
            tx_hash=receipt.tx_hash,
            amount_in=amount_in,
            amount_out=0,
            gas_used=receipt.gas_used,
            route="v3_direct",
        )

    # ── ODOS aggregator swap ──────────────────────────────────

    def _execute_odos(
        self,
        input_token: str,
        output_token: str,
        amount_in: int,
        slippage_percent: float = 0.5,
    ) -> DexSwapResult:
        """Full ODOS swap: quote -> assemble -> approve -> send."""
        user_address = self._wallet.address

        # 1. Quote
        try:
            quote = self._odos.quote(
                input_token=input_token,
                output_token=output_token,
                amount_in=amount_in,
                user_address=user_address,
                slippage_percent=slippage_percent,
            )
        except Exception as exc:
            logger.warning("ODOS quote failed: %s", exc)
            return DexSwapResult(
                success=False,
                error=f"quote failed: {exc}",
                route="odos",
            )

        if not quote.path_id:
            return DexSwapResult(
                success=False,
                error="ODOS quote returned no pathId",
                route="odos",
            )

        # 2. Assemble
        try:
            assembled = self._odos.assemble(
                path_id=quote.path_id,
                user_address=user_address,
            )
        except Exception as exc:
            logger.warning("ODOS assemble failed: %s", exc)
            return DexSwapResult(
                success=False,
                error=f"assemble failed: {exc}",
                route="odos",
            )

        # 2b. Note: Gas cap check removed - we compare net profit instead
        # High gas routes may still be profitable if ODOS finds better prices

        # 3. Ensure ERC-20 allowance to the ODOS router
        native_zero = "0x0000000000000000000000000000000000000000"
        if input_token.lower() != native_zero:
            try:
                self._ensure_allowance(
                    token=Address.from_string(input_token),
                    owner=Address.from_string(user_address),
                    spender=Address.from_string(assembled.to),
                    min_amount=amount_in,
                )
            except Exception as exc:
                logger.warning("ERC-20 approve failed: %s", exc)
                return DexSwapResult(
                    success=False,
                    error=f"approve failed: {exc}",
                    route="odos",
                )

        # 4. Build and send
        try:
            calldata = bytes.fromhex(
                assembled.data[2:]
                if assembled.data.startswith("0x")
                else assembled.data
            )
            receipt = (
                TransactionBuilder(self._client, self._wallet)
                .to(Address.from_string(assembled.to))
                .value(TokenAmount(raw=assembled.value, decimals=18, symbol="ETH"))
                .data(calldata)
                .chain_id(self._chain_id)
                .gas_limit(int(assembled.gas * 1.2))
                .with_gas_price(self._gas_priority)
                .send_and_wait(timeout=self._tx_timeout)
            )
        except Exception as exc:
            logger.warning("ODOS swap tx failed: %s", exc)
            return DexSwapResult(
                success=False,
                error=f"tx failed: {exc}",
                route="odos",
            )

        if not receipt.status:
            return DexSwapResult(
                success=False,
                tx_hash=receipt.tx_hash,
                gas_used=receipt.gas_used,
                error="transaction reverted",
                route="odos",
            )

        logger.info(
            "ODOS swap OK: tx=%s in=%d out_expected=%d gas=%d",
            receipt.tx_hash,
            amount_in,
            quote.amount_out,
            receipt.gas_used,
        )
        return DexSwapResult(
            success=True,
            tx_hash=receipt.tx_hash,
            amount_in=amount_in,
            amount_out=quote.amount_out,
            gas_used=receipt.gas_used,
            route="odos",
        )

    # ── Internal helpers ───────────────────────────────────────

    def _ensure_allowance(
        self,
        token: Address,
        owner: Address,
        spender: Address,
        min_amount: int,
    ) -> None:
        """Approve spender if current allowance is below min_amount."""
        selector = keccak(text="allowance(address,address)")[:4]
        calldata = selector + abi_encode(
            ["address", "address"], [owner.checksum, spender.checksum]
        )
        call = TransactionRequest(
            to=token,
            value=TokenAmount(raw=0, decimals=18, symbol="ETH"),
            data=calldata,
            from_address=owner,
            chain_id=self._chain_id,
        )
        raw = self._client.call(call)
        current = int.from_bytes(raw, "big") if raw else 0
        if current >= min_amount:
            return

        max_uint256 = 2**256 - 1
        approve_sel = keccak(text="approve(address,uint256)")[:4]
        approve_data = approve_sel + abi_encode(
            ["address", "uint256"], [spender.checksum, max_uint256]
        )
        receipt = (
            TransactionBuilder(self._client, self._wallet)
            .to(token)
            .value(TokenAmount(raw=0, decimals=18, symbol="ETH"))
            .data(approve_data)
            .chain_id(self._chain_id)
            .with_gas_estimate()
            .with_gas_price(self._gas_priority)
            .send_and_wait(timeout=self._tx_timeout)
        )
        if not receipt.status:
            raise RuntimeError("ERC-20 approve transaction reverted")
        logger.info(
            "Approved %s for spender %s (tx=%s)",
            token.checksum,
            spender.checksum,
            receipt.tx_hash,
        )
