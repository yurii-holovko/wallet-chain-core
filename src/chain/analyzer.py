"""CLI for analyzing Ethereum transactions."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from eth_abi import decode
from eth_utils.crypto import keccak

import config
from core.base_types import Address, TokenAmount, TransactionRequest

from .client import ChainClient
from .errors import RPCError


def _selector_hash(signature: str) -> str:
    return f"0x{keccak(text=signature).hex()[:8]}"


TRANSFER_TOPIC = keccak(text="Transfer(address,address,uint256)").hex()
SWAP_V2_TOPIC = keccak(
    text="Swap(address,uint256,uint256,uint256,uint256,address)"
).hex()
SYNC_V2_TOPIC = keccak(text="Sync(uint112,uint112)").hex()
SWAP_V3_TOPIC = keccak(
    text="Swap(address,address,int256,int256,uint160,uint128,int24)"
).hex()

SELECTOR_MAP = {
    _selector_hash("transfer(address,uint256)"): (
        "transfer(address,uint256)",
        ["address", "uint256"],
    ),
    _selector_hash("approve(address,uint256)"): (
        "approve(address,uint256)",
        ["address", "uint256"],
    ),
    _selector_hash("transferFrom(address,address,uint256)"): (
        "transferFrom(address,address,uint256)",
        ["address", "address", "uint256"],
    ),
    _selector_hash(
        "swapExactTokensForTokens(uint256,uint256,address[],address,uint256)"
    ): (
        "swapExactTokensForTokens(uint256,uint256,address[],address,uint256)",
        ["uint256", "uint256", "address[]", "address", "uint256"],
    ),
    _selector_hash("swapExactETHForTokens(uint256,address[],address,uint256)"): (
        "swapExactETHForTokens(uint256,address[],address,uint256)",
        ["uint256", "address[]", "address", "uint256"],
    ),
    _selector_hash(
        "swapExactTokensForETH(uint256,uint256,address[],address,uint256)"
    ): (
        "swapExactTokensForETH(uint256,uint256,address[],address,uint256)",
        ["uint256", "uint256", "address[]", "address", "uint256"],
    ),
    _selector_hash(
        "addLiquidity(address,address,uint256,uint256,uint256,uint256,address,uint256)"
    ): (
        "addLiquidity(address,address,uint256,uint256,uint256,uint256,address,uint256)",
        [
            "address",
            "address",
            "uint256",
            "uint256",
            "uint256",
            "uint256",
            "address",
            "uint256",
        ],
    ),
    _selector_hash(
        "removeLiquidity(address,address,uint256,uint256,uint256,address,uint256)"
    ): (
        "removeLiquidity(address,address,uint256,uint256,uint256,address,uint256)",
        ["address", "address", "uint256", "uint256", "uint256", "address", "uint256"],
    ),
    _selector_hash("multicall(bytes[])"): ("multicall(bytes[])", ["bytes[]"]),
    _selector_hash("exactInput((bytes,address,uint256,uint256,uint256))"): (
        "exactInput((bytes,address,uint256,uint256,uint256))",
        ["(bytes,address,uint256,uint256,uint256)"],
    ),
    _selector_hash("exactOutput((bytes,address,uint256,uint256,uint256))"): (
        "exactOutput((bytes,address,uint256,uint256,uint256))",
        ["(bytes,address,uint256,uint256,uint256)"],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an Ethereum transaction")
    parser.add_argument("tx_hash", help="Transaction hash (0x...)")
    parser.add_argument("--rpc", help="RPC URL (or set RPC_URL env var)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    rpc_url = args.rpc or config.get_env("RPC_URL")
    if not rpc_url:
        raise SystemExit("RPC URL required via --rpc or RPC_URL env var")

    if not _is_valid_hash(args.tx_hash):
        raise SystemExit("Invalid transaction hash format")

    client = ChainClient([rpc_url])
    tx, receipt_data = _fetch_bundle(client, args.tx_hash)
    if receipt_data is None:
        if args.format == "json":
            print(json.dumps({"hash": args.tx_hash, "status": "PENDING"}, indent=2))
        else:
            print("Transaction Analysis")
            print("====================")
            print(f"Hash:           {args.tx_hash}")
            print("Status:         PENDING")
            print("Pending transaction. Receipt not available yet.")
        return

    receipt = client.get_receipt(args.tx_hash)
    if receipt is None:
        raise SystemExit("Transaction receipt not found")

    block = client.get_block(tx["blockNumber"])
    timestamp = _format_timestamp(block["timestamp"])
    status = "SUCCESS" if receipt.status else "FAILED"

    from_addr = Address.from_string(tx["from"]).checksum
    to_addr = tx.get("to")
    to_display = (
        Address.from_string(to_addr).checksum if to_addr else "Contract Creation"
    )

    value = _format_eth(_hex_to_int(tx["value"]))
    gas_limit = _hex_to_int(tx["gas"])
    gas_used = receipt.gas_used
    gas_pct = (gas_used / gas_limit) * 100 if gas_limit else 0

    base_fee = _hex_to_int(block.get("baseFeePerGas", "0x0"))
    effective_price = receipt.effective_gas_price
    priority_fee = max(effective_price - base_fee, 0)
    tx_fee = TokenAmount(raw=gas_used * effective_price, decimals=18, symbol="ETH")

    decoded = _decode_function(tx.get("input", "0x"))
    token_cache = _TokenCache(client)
    transfers = _extract_transfers(receipt.logs, token_cache)
    swaps = _extract_swaps(receipt.logs)
    syncs = _extract_syncs(receipt.logs)
    revert_reason = _get_revert_reason(client, tx) if not receipt.status else None

    if args.format == "json":
        output = {
            "hash": args.tx_hash,
            "block": _hex_to_int(tx["blockNumber"]),
            "timestamp": timestamp,
            "status": status,
            "from": from_addr,
            "to": to_display,
            "value": f"{value} ETH",
            "gas": {
                "limit": gas_limit,
                "used": gas_used,
                "used_pct": round(gas_pct, 2),
                "base_fee_gwei": _format_gwei(base_fee),
                "priority_fee_gwei": _format_gwei(priority_fee),
                "effective_price_gwei": _format_gwei(effective_price),
                "tx_fee_eth": str(tx_fee.human),
            },
            "function": (
                None
                if decoded is None
                else {
                    "selector": decoded.selector,
                    "name": decoded.name,
                    "args": decoded.args,
                }
            ),
            "transfers": transfers,
            "swaps": swaps,
            "syncs": syncs,
            "revert_reason": revert_reason,
        }
        print(json.dumps(output, indent=2))
        return

    print("Transaction Analysis")
    print("====================")
    print(f"Hash:           {args.tx_hash}")
    print(f"Block:          {_hex_to_int(tx['blockNumber'])}")
    print(f"Timestamp:      {timestamp}")
    print(f"Status:         {status}")
    print("")
    print(f"From:           {from_addr}")
    print(f"To:             {to_display}")
    print(f"Value:          {value} ETH")
    print("")
    print("Gas Analysis")
    print("------------")
    print(f"Gas Limit:      {gas_limit:,}")
    print(f"Gas Used:       {gas_used:,} ({gas_pct:.2f}%)")
    print(f"Base Fee:       {_format_gwei(base_fee)} gwei")
    print(f"Priority Fee:   {_format_gwei(priority_fee)} gwei")
    print(f"Effective Price: {_format_gwei(effective_price)} gwei")
    print(f"Transaction Fee: {tx_fee.human} ETH")
    print("")
    print("Function Called")
    print("---------------")
    selector = _selector(tx.get("input", "0x"))
    if selector:
        print(f"Selector:       {selector}")
        if decoded:
            print(f"Function:       {decoded.name}")
            if decoded.args:
                print("Arguments:")
                for label, value in decoded.args:
                    print(f"  - {label}: {value}")
        else:
            print("Function:       Unknown (ABI required)")
    else:
        print("Selector:       None")
        print("Function:       None")

    print("")
    print("Token Transfers")
    print("---------------")
    if not transfers:
        print("No ERC-20 Transfer events found.")
    else:
        for idx, transfer in enumerate(transfers, start=1):
            print(
                f"{idx}. {transfer['token']}: {transfer['from']} -> "
                f"{transfer['to']}  {transfer['value']}"
            )

    print("")
    print("Swap Events")
    print("-----------")
    if not swaps:
        print("No swap events found.")
    else:
        for idx, swap in enumerate(swaps, start=1):
            print(f"{idx}. {swap}")

    print("")
    print("Sync Events")
    print("-----------")
    if not syncs:
        print("No sync events found.")
    else:
        for idx, sync in enumerate(syncs, start=1):
            print(f"{idx}. {sync}")

    if revert_reason:
        print("")
        print("Revert Reason")
        print("-------------")
        print(revert_reason)


def _hex_to_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 16) if value.startswith("0x") else int(value)
    raise ValueError("Expected integer-like value")


def _format_timestamp(value: Any) -> str:
    ts = _hex_to_int(value)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_gwei(wei: int) -> str:
    return f"{Decimal(wei) / Decimal(10**9):.4f}".rstrip("0").rstrip(".")


def _format_eth(wei: int) -> str:
    return f"{Decimal(wei) / Decimal(10**18):.6f}".rstrip("0").rstrip(".")


def _selector(calldata: str) -> str | None:
    if not isinstance(calldata, str) or calldata == "0x":
        return None
    return calldata[:10] if calldata.startswith("0x") else f"0x{calldata[:8]}"


def _extract_transfers(logs: list[dict], token_cache: "_TokenCache") -> list[dict]:
    transfers = []
    for log in logs:
        topics = log.get("topics", [])
        if not topics or topics[0] != f"0x{TRANSFER_TOPIC}":
            continue
        if len(topics) < 3:
            continue
        from_addr = _topic_to_address(topics[1])
        to_addr = _topic_to_address(topics[2])
        value = _hex_to_int(log.get("data", "0x0"))
        token = log.get("address", "unknown")
        token_info = token_cache.get(token)
        if token_info:
            symbol, decimals = token_info
            amount = Decimal(value) / Decimal(10**decimals)
            display = f"{amount} {symbol}"
        else:
            display = str(value)
        transfers.append(
            {
                "token": token,
                "from": from_addr,
                "to": to_addr,
                "value": display,
            }
        )
    return transfers


def _extract_swaps(logs: list[dict]) -> list[str]:
    swaps = []
    for log in logs:
        topics = log.get("topics", [])
        if not topics:
            continue
        if topics[0] == f"0x{SWAP_V2_TOPIC}":
            data = _hex_to_bytes(log.get("data", "0x"))
            amounts = decode(["uint256", "uint256", "uint256", "uint256"], data)
            swaps.append(
                f"UniswapV2 Swap in={amounts[0]}/{amounts[1]} "
                f"out={amounts[2]}/{amounts[3]}"
            )
        elif topics[0] == f"0x{SWAP_V3_TOPIC}":
            data = _hex_to_bytes(log.get("data", "0x"))
            amounts = decode(["int256", "int256", "uint160", "uint128", "int24"], data)
            swaps.append(f"UniswapV3 Swap amount0={amounts[0]} amount1={amounts[1]}")
    return swaps


def _extract_syncs(logs: list[dict]) -> list[str]:
    syncs = []
    for log in logs:
        topics = log.get("topics", [])
        if not topics:
            continue
        if topics[0] == f"0x{SYNC_V2_TOPIC}":
            data = _hex_to_bytes(log.get("data", "0x"))
            reserves = decode(["uint112", "uint112"], data)
            syncs.append(f"Sync reserves={reserves[0]}/{reserves[1]}")
    return syncs


def _topic_to_address(topic: str) -> str:
    if not isinstance(topic, str):
        return "unknown"
    raw = topic[2:] if topic.startswith("0x") else topic
    addr = f"0x{raw[-40:]}"
    return Address.from_string(addr).checksum


def _is_valid_hash(value: str) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{64}", value or ""))


@dataclass(frozen=True)
class DecodedCall:
    selector: str
    name: str
    args: list[tuple[str, str]]


def _decode_function(calldata: str) -> DecodedCall | None:
    if not isinstance(calldata, str) or calldata in ("0x", ""):
        return None
    selector = _selector(calldata)
    if not selector:
        return None
    spec = SELECTOR_MAP.get(selector)
    if spec is None:
        return None
    name, types = spec
    data = (
        _hex_to_bytes(calldata[10:])
        if calldata.startswith("0x")
        else _hex_to_bytes(calldata[8:])
    )
    decoded = decode(types, data)
    labels = _label_args(name)
    formatted = []
    for label, value in zip(labels, decoded):
        formatted.append((label, _format_arg(value)))
    return DecodedCall(selector=selector, name=name, args=formatted)


def _label_args(signature: str) -> list[str]:
    if signature.startswith("transfer("):
        return ["to", "amount"]
    if signature.startswith("approve("):
        return ["spender", "amount"]
    if signature.startswith("transferFrom("):
        return ["from", "to", "amount"]
    if signature.startswith("swapExactTokensForTokens("):
        return ["amountIn", "amountOutMin", "path", "to", "deadline"]
    if signature.startswith("swapExactETHForTokens("):
        return ["amountOutMin", "path", "to", "deadline"]
    if signature.startswith("swapExactTokensForETH("):
        return ["amountIn", "amountOutMin", "path", "to", "deadline"]
    if signature.startswith("addLiquidity("):
        return [
            "tokenA",
            "tokenB",
            "amountADesired",
            "amountBDesired",
            "amountAMin",
            "amountBMin",
            "to",
            "deadline",
        ]
    if signature.startswith("removeLiquidity("):
        return [
            "tokenA",
            "tokenB",
            "liquidity",
            "amountAMin",
            "amountBMin",
            "to",
            "deadline",
        ]
    if signature.startswith("multicall("):
        return ["calls"]
    if signature.startswith("exactInput("):
        return ["params"]
    if signature.startswith("exactOutput("):
        return ["params"]
    return []


def _format_arg(value: Any) -> str:
    if isinstance(value, bytes):
        return f"0x{value.hex()}"
    if isinstance(value, (list, tuple)):
        return f"[{', '.join(_format_arg(v) for v in value)}]"
    if isinstance(value, str) and _is_valid_hash(value):
        return Address.from_string(value).checksum
    return str(value)


def _fetch_bundle(client: ChainClient, tx_hash: str) -> tuple[dict, Any]:
    tx, receipt = client._rpc_batch(
        [
            ("eth_getTransactionByHash", [tx_hash]),
            ("eth_getTransactionReceipt", [tx_hash]),
        ]
    )
    if tx is None:
        raise SystemExit("Transaction not found")
    return tx, receipt


def _get_revert_reason(client: ChainClient, tx: dict) -> str | None:
    try:
        client._rpc_call("eth_call", [tx, tx.get("blockNumber")])
    except RPCError as exc:
        data = getattr(exc, "data", None)
        reason = _decode_revert_reason(data)
        return reason or str(exc)
    except Exception:  # noqa: BLE001
        return None
    return None


def _decode_revert_reason(data: Any) -> str | None:
    if data is None:
        return None
    if isinstance(data, dict) and "data" in data:
        return _decode_revert_reason(data["data"])
    if isinstance(data, str) and data.startswith("0x08c379a0"):
        raw = _hex_to_bytes(data[10:])
        try:
            (reason,) = decode(["string"], raw)
            return str(reason)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(data, str):
        return data
    return None


class _TokenCache:
    def __init__(self, client: ChainClient):
        self._client = client
        self._cache: dict[str, tuple[str, int]] = {}

    def get(self, token: str) -> tuple[str, int] | None:
        if not isinstance(token, str) or not token.startswith("0x"):
            return None
        checksum = Address.from_string(token).checksum
        if checksum in self._cache:
            return self._cache[checksum]
        symbol = _call_token_string(self._client, checksum, "symbol()")
        decimals = _call_token_uint8(self._client, checksum, "decimals()")
        if symbol is None or decimals is None:
            return None
        self._cache[checksum] = (symbol, decimals)
        return self._cache[checksum]


def _call_token_string(client: ChainClient, token: str, signature: str) -> str | None:
    data = _selector_hash(signature)
    tx = TransactionRequest(
        to=Address.from_string(token),
        value=TokenAmount(raw=0, decimals=18),
        data=_hex_to_bytes(data),
    )
    try:
        raw = client.call(tx)
        if len(raw) == 32:
            return raw.rstrip(b"\x00").decode("utf-8", errors="ignore")
        (decoded,) = decode(["string"], raw)
        return str(decoded)
    except Exception:  # noqa: BLE001
        return None


def _call_token_uint8(client: ChainClient, token: str, signature: str) -> int | None:
    data = _selector_hash(signature)
    tx = TransactionRequest(
        to=Address.from_string(token),
        value=TokenAmount(raw=0, decimals=18),
        data=_hex_to_bytes(data),
    )
    try:
        raw = client.call(tx)
        (decoded,) = decode(["uint8"], raw)
        return int(decoded)
    except Exception:  # noqa: BLE001
        return None


def _hex_to_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        return b""
    normalized = value[2:] if value.startswith("0x") else value
    if normalized == "":
        return b""
    return bytes.fromhex(normalized)


if __name__ == "__main__":
    main()
