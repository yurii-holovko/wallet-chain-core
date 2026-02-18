from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_utils import keccak

from chain import ChainClient, TransactionBuilder
from config import get_env
from core.base_types import Address, TokenAmount, TransactionRequest
from core.wallet_manager import WalletManager
from pricing.uniswap_v3_math import TickRange, single_tick_range

logger = logging.getLogger(__name__)


@dataclass
class V3PoolConfig:
    fee_tier: int
    pool_address: Address | None = None


class UniswapV3RangeOrderManager:
    """
    Minimal Uniswap V3 range-order helper for Arbitrum.

    This manager mints single-tick range positions around the current pool
    price and exposes a small API used by the Double Limit engine:

      * place_limit_sell_order(USDC → token)
      * place_limit_buy_order(token → USDC)
      * check_position_status(token_id)
      * withdraw_executed_position(token_id)

    Notes:
      * All swaps are ERC20-based; no native ETH is handled here.
      * Pool addresses and fee tiers are provided via a per-token config map.
    """

    def __init__(
        self,
        client: ChainClient,
        wallet: WalletManager,
        usdc_address: str,
        pool_config: Dict[str, Dict[str, Any]],
        position_manager_address: Optional[str] = None,
        chain_id: int = 42161,
        gas_priority: str = "medium",
    ) -> None:
        self._client = client
        self._wallet = wallet
        self._chain_id = chain_id
        self._gas_priority = gas_priority

        self._usdc = Address.from_string(usdc_address)

        pm_env = position_manager_address or get_env(
            "UNISWAP_V3_POSITION_MANAGER",
            "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        )
        self._position_manager = Address.from_string(pm_env)

        factory_env = get_env(
            "UNISWAP_V3_FACTORY",
            "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        )
        self._factory = Address.from_string(factory_env)

        # Map non-USDC token address (lowercase) → V3PoolConfig
        self._pools: Dict[str, V3PoolConfig] = {}
        for token_addr, cfg in pool_config.items():
            try:
                fee_tier = int(cfg["fee_tier"])
                pool_value = cfg.get("pool")
                pool_addr = Address.from_string(pool_value) if pool_value else None
            except Exception:
                continue
            self._pools[token_addr.lower()] = V3PoolConfig(
                fee_tier=fee_tier,
                pool_address=pool_addr,
            )

    # ── public API ────────────────────────────────────────────────

    def place_limit_sell_order(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> Dict[str, Any]:
        """
        Place a single-tick range position representing a limit SELL order.

        token_in: address of asset deposited (e.g. USDC)
        token_out: address of asset to receive when price crosses the tick
        amount_in: raw token units of token_in
        """
        return self._mint_range_order(
            token_in=Address.from_string(token_in),
            token_out=Address.from_string(token_out),
            amount_in=amount_in,
            direction="above",
        )

    def place_limit_buy_order(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> Dict[str, Any]:
        """
        Place a single-tick range position representing a limit BUY order.

        Implemented as a range just below the current price.
        """
        return self._mint_range_order(
            token_in=Address.from_string(token_in),
            token_out=Address.from_string(token_out),
            amount_in=amount_in,
            direction="below",
        )

    def check_position_status(self, token_id: int) -> Dict[str, Any]:
        """
        Inspect position via NonfungiblePositionManager.positions(tokenId).
        """
        data = self._encode_call(
            "positions(uint256)",
            ["uint256"],
            [token_id],
        )
        raw = self._eth_call(self._position_manager, data)
        # See V3 NFPositionManager ABI for exact layout.
        decoded = abi_decode(
            [
                "uint96",  # nonce
                "address",  # operator
                "address",  # token0
                "address",  # token1
                "uint24",  # fee
                "int24",  # tickLower
                "int24",  # tickUpper
                "uint128",  # liquidity
                "uint256",  # feeGrowthInside0LastX128
                "uint256",  # feeGrowthInside1LastX128
                "uint128",  # tokensOwed0
                "uint128",  # tokensOwed1
            ],
            raw,
        )
        token0 = Address.from_string(decoded[2])
        token1 = Address.from_string(decoded[3])
        fee_tier = int(decoded[4])
        tick_lower = int(decoded[5])
        tick_upper = int(decoded[6])
        liquidity = int(decoded[7])
        tokens_owed0 = int(decoded[10])
        tokens_owed1 = int(decoded[11])

        pool = self._resolve_pool(token0, token1, fee_tier)
        current_tick = self._get_current_tick(pool) if pool else None
        in_range = current_tick is not None and tick_lower <= current_tick < tick_upper
        # For single-tick "range order" semantics, consider the position executed
        # once price moves out of the range (fully converted to one side).
        is_executed = bool(current_tick is not None and not in_range and liquidity > 0)

        return {
            "token0": token0.checksum,
            "token1": token1.checksum,
            "fee_tier": fee_tier,
            "pool": pool.checksum if pool else None,
            "current_tick": current_tick,
            "tick_lower": tick_lower,
            "tick_upper": tick_upper,
            "in_range": in_range,
            "liquidity": liquidity,
            "tokens_owed0": tokens_owed0,
            "tokens_owed1": tokens_owed1,
            "is_executed": is_executed,
            "can_withdraw": is_executed,
        }

    def withdraw_executed_position(self, token_id: int) -> str:
        """
        Collect all owed tokens for a position.

        Assumes the position has already been fully crossed (liquidity==0)
        and only fees / proceeds remain to be collected.
        """
        owner = Address.from_string(self._wallet.address)

        # Read liquidity from the position
        pos = self.check_position_status(token_id)
        liquidity = int(pos.get("liquidity") or 0)
        if liquidity <= 0:
            raise RuntimeError("Position has zero liquidity; nothing to withdraw")

        deadline = int(time.time()) + 600
        dec_params = (
            token_id,
            liquidity,
            0,  # amount0Min
            0,  # amount1Min
            deadline,
        )
        dec_calldata = self._encode_call(
            "decreaseLiquidity((uint256,uint128,uint256,uint256,uint256))",
            ["(uint256,uint128,uint256,uint256,uint256)"],
            [dec_params],
        )
        (
            TransactionBuilder(self._client, self._wallet)
            .to(self._position_manager)
            .value(TokenAmount(raw=0, decimals=18, symbol="ETH"))
            .data(dec_calldata)
            .chain_id(self._chain_id)
            .with_gas_estimate()
            .with_gas_price(self._gas_priority)
            .send_and_wait(timeout=300)
        )

        max_uint128 = 2**128 - 1
        collect_params = (
            token_id,
            owner.checksum,
            max_uint128,
            max_uint128,
        )
        collect_calldata = self._encode_call(
            "collect((uint256,address,uint128,uint128))",
            ["(uint256,address,uint128,uint128)"],
            [collect_params],
        )
        receipt = (
            TransactionBuilder(self._client, self._wallet)
            .to(self._position_manager)
            .value(TokenAmount(raw=0, decimals=18, symbol="ETH"))
            .data(collect_calldata)
            .chain_id(self._chain_id)
            .with_gas_estimate()
            .with_gas_price(self._gas_priority)
            .send_and_wait(timeout=300)
        )
        return receipt.tx_hash

    # ── internals ────────────────────────────────────────────────

    def _mint_range_order(
        self,
        token_in: Address,
        token_out: Address,
        amount_in: int,
        direction: str,
    ) -> Dict[str, Any]:
        # Identify the non-USDC token for selecting pool config.
        non_usdc = token_in if token_out == self._usdc else token_out
        token_key = non_usdc.checksum.lower()
        pool_cfg = self._pools.get(token_key)
        if pool_cfg is None:
            raise ValueError(f"No V3 pool config for token {non_usdc.checksum}")

        fee_tier = pool_cfg.fee_tier

        # Determine token0/token1 ordering as Uniswap V3 does (by address)
        if token_in.checksum.lower() < token_out.checksum.lower():
            token0 = token_in
            token1 = token_out
            amount0_desired = amount_in
            amount1_desired = 0
        else:
            token0 = token_out
            token1 = token_in
            amount0_desired = 0
            amount1_desired = amount_in

        # Ensure allowance for token_in towards the position manager
        owner = Address.from_string(self._wallet.address)
        self._ensure_allowance(
            token=token_in,
            owner=owner,
            spender=self._position_manager,
            min_amount=amount_in,
        )

        # Resolve pool address (prefer explicit config pool, else factory.getPool)
        pool_addr = pool_cfg.pool_address
        resolved_pool = self._resolve_pool(
            token0, token1, fee_tier, preferred_pool=pool_addr
        )
        if resolved_pool is None:
            raise ValueError("Could not resolve a Uniswap V3 pool for this pair/tier")

        # Current tick from pool slot0
        current_tick = self._get_current_tick(resolved_pool)
        tick_range: TickRange = single_tick_range(
            current_tick=current_tick,
            fee_tier=fee_tier,
            direction=direction,
        )

        deadline = int(time.time()) + 600  # 10 minutes
        mint_params = (
            token0.checksum,
            token1.checksum,
            fee_tier,
            tick_range.tick_lower,
            tick_range.tick_upper,
            amount0_desired,
            amount1_desired,
            0,  # amount0Min
            0,  # amount1Min
            owner.checksum,
            deadline,
        )

        mint_sig = (
            "mint((address,address,uint24,int24,int24,uint256,uint256,"
            "uint256,uint256,address,uint256))"
        )
        mint_type = (
            "(address,address,uint24,int24,int24,uint256,uint256,"
            "uint256,uint256,address,uint256)"
        )
        calldata = self._encode_call(mint_sig, [mint_type], [mint_params])

        receipt = (
            TransactionBuilder(self._client, self._wallet)
            .to(self._position_manager)
            .value(TokenAmount(raw=0, decimals=18, symbol="ETH"))
            .data(calldata)
            .chain_id(self._chain_id)
            .with_gas_estimate()
            .with_gas_price(self._gas_priority)
            .send_and_wait(timeout=300)
        )

        token_id = self._parse_token_id_from_receipt(receipt)
        return {
            "tx_hash": receipt.tx_hash,
            "token_id": token_id,
            "tick_lower": tick_range.tick_lower,
            "tick_upper": tick_range.tick_upper,
        }

    def _parse_token_id_from_receipt(self, receipt) -> Optional[int]:
        """
        Best-effort parse of tokenId from NFPositionManager events.

        We look for the standard ERC-721 Transfer event where the sender is
        the zero address (mint).
        """
        zero_address_topic = (
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        )
        transfer_sig = keccak(text="Transfer(address,address,uint256)").hex()
        transfer_topic0 = f"0x{transfer_sig}"

        for log in getattr(receipt, "logs", []):
            try:
                address = str(log.get("address") or "").lower()
                if address != self._position_manager.checksum.lower():
                    continue
                topics = [str(t) for t in (log.get("topics") or [])]
                if not topics or topics[0].lower() != transfer_topic0.lower():
                    continue
                if len(topics) < 3:
                    continue
                # topics[1] is from, topics[2] is to
                if topics[1].lower() != zero_address_topic.lower():
                    continue
                # tokenId is encoded in data as uint256
                data_hex = str(log.get("data") or "")
                if not data_hex:
                    continue
                token_id = int(data_hex, 16)
                return token_id
            except Exception:
                continue
        return None

    def _resolve_pool(
        self,
        token0: Address,
        token1: Address,
        fee_tier: int,
        preferred_pool: Address | None = None,
    ) -> Address | None:
        """
        Resolve the V3 pool for (token0, token1, fee).

        If a preferred_pool is supplied, it will be used only if it appears to
        be a V3 pool for the same token pair + fee. Otherwise, we fall back to
        querying the factory via getPool(...).
        """
        if preferred_pool is not None and self._pool_matches(
            preferred_pool, token0, token1, fee_tier
        ):
            return preferred_pool
        factory_pool = self._get_pool_from_factory(token0, token1, fee_tier)
        return factory_pool

    def _get_pool_from_factory(
        self, token0: Address, token1: Address, fee_tier: int
    ) -> Address | None:
        a = token0.checksum
        b = token1.checksum
        t0, t1 = (a, b) if a.lower() < b.lower() else (b, a)
        data = self._encode_call(
            "getPool(address,address,uint24)",
            ["address", "address", "uint24"],
            [t0, t1, fee_tier],
        )
        raw = self._eth_call(self._factory, data)
        (addr,) = abi_decode(["address"], raw)
        if (
            not addr
            or str(addr).lower() == "0x0000000000000000000000000000000000000000"
        ):
            return None
        return Address.from_string(addr)

    def _pool_matches(
        self, pool: Address, token0: Address, token1: Address, fee_tier: int
    ) -> bool:
        try:
            raw0 = self._eth_call(pool, keccak(text="token0()")[:4])
            (p0,) = abi_decode(["address"], raw0)
            raw1 = self._eth_call(pool, keccak(text="token1()")[:4])
            (p1,) = abi_decode(["address"], raw1)
            raw_fee = self._eth_call(pool, keccak(text="fee()")[:4])
            (pfee,) = abi_decode(["uint24"], raw_fee)
            pool_tokens = {Address.from_string(p0).lower, Address.from_string(p1).lower}
            wanted_tokens = {token0.lower, token1.lower}
            return pool_tokens == wanted_tokens and int(pfee) == int(fee_tier)
        except Exception:
            return False

    def _get_current_tick(self, pool: Address) -> int:
        """
        Read slot0() from a V3 pool and return the current tick.
        """
        selector = keccak(text="slot0()")[:4]
        tx = TransactionRequest(
            to=pool,
            value=TokenAmount(raw=0, decimals=18, symbol="ETH"),
            data=selector,
            chain_id=self._chain_id,
        )
        raw = self._client.call(tx)
        decoded = abi_decode(
            ["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"],
            raw,
        )
        return int(decoded[1])

    def _eth_call(self, to: Address, data: bytes) -> bytes:
        tx = TransactionRequest(
            to=to,
            value=TokenAmount(raw=0, decimals=18, symbol="ETH"),
            data=data,
            chain_id=self._chain_id,
        )
        return self._client.call(tx)

    def _ensure_allowance(
        self,
        token: Address,
        owner: Address,
        spender: Address,
        min_amount: int,
    ) -> None:
        selector = keccak(text="allowance(address,address)")[:4]
        calldata = selector + abi_encode(
            ["address", "address"], [owner.checksum, spender.checksum]
        )
        call = TransactionRequest(
            to=token,
            value=TokenAmount(raw=0, decimals=18, symbol="ETH"),
            data=calldata,
            chain_id=self._chain_id,
        )
        raw = self._client.call(call)
        current_allowance = int.from_bytes(raw, "big") if raw else 0
        if current_allowance >= min_amount:
            return

        max_uint256 = 2**256 - 1
        approve_data = self._encode_call(
            "approve(address,uint256)",
            ["address", "uint256"],
            [spender.checksum, max_uint256],
        )
        receipt = (
            TransactionBuilder(self._client, self._wallet)
            .to(token)
            .value(TokenAmount(raw=0, decimals=18, symbol="ETH"))
            .data(approve_data)
            .chain_id(self._chain_id)
            .with_gas_estimate()
            .with_gas_price(self._gas_priority)
            .send_and_wait(timeout=180)
        )
        if not receipt.status:
            raise RuntimeError("Token approve transaction failed")

    @staticmethod
    def _encode_call(signature: str, arg_types: list[str], args: list[Any]) -> bytes:
        selector = keccak(text=signature)[:4]
        return selector + abi_encode(arg_types, args)
