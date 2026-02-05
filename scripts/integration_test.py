"""Integration test for Sepolia: build, sign, send, confirm."""

from __future__ import annotations

import sys
from decimal import Decimal

from eth_account import Account

import config
from chain import ChainClient
from core.base_types import Address, TokenAmount, TransactionRequest
from core.wallet_manager import WalletManager

SEPOLIA_CHAIN_ID = 11155111


def main() -> None:
    rpc_url = config.get_env("SEPOLIA_RPC_URL", required=True)
    recipient = config.get_env("RECIPIENT_ADDRESS", required=True)
    amount = config.get_env("TRANSFER_AMOUNT", "0.001")

    wallet = WalletManager.from_env()
    client = ChainClient([rpc_url])

    balance = client.get_balance(Address.from_string(wallet.address))
    print(f"Wallet: {wallet.address}")
    print(f"Balance: {balance.human} ETH")
    print("")

    print("Building transaction...")
    to_addr = Address.from_string(recipient)
    value = TokenAmount.from_human(Decimal(amount), 18, "ETH")
    gas_price = client.get_gas_price()
    max_priority_fee = gas_price.priority_fee_medium
    max_fee = gas_price.get_max_fee("medium")
    nonce = client.get_nonce(Address.from_string(wallet.address))

    tx_request = TransactionRequest(
        to=to_addr,
        value=value,
        data=b"",
        nonce=nonce,
        gas_limit=0,
        max_fee_per_gas=max_fee,
        max_priority_fee=max_priority_fee,
        chain_id=SEPOLIA_CHAIN_ID,
    )
    estimated_gas = client.estimate_gas(tx_request)
    tx_request = TransactionRequest(
        to=to_addr,
        value=value,
        data=b"",
        nonce=nonce,
        gas_limit=estimated_gas,
        max_fee_per_gas=max_fee,
        max_priority_fee=max_priority_fee,
        chain_id=SEPOLIA_CHAIN_ID,
    )

    print(f"  To: {to_addr.checksum}")
    print(f"  Value: {value.human} ETH")
    print(f"  Estimated Gas: {estimated_gas}")
    print(f"  Max Fee: {_format_gwei(max_fee)} gwei")
    print(f"  Max Priority: {_format_gwei(max_priority_fee)} gwei")
    print("")

    print("Signing...")
    signed = wallet.sign_transaction(tx_request.to_dict())
    recovered = Account.recover_transaction(signed.raw_transaction)
    signature_valid = recovered.lower() == wallet.address.lower()
    print(
        "  Signature valid: \u2713" if signature_valid else "  Signature valid: \u2717"
    )
    print(
        "  Recovered address matches: \u2713"
        if signature_valid
        else "  Recovered address matches: \u2717"
    )
    if not signature_valid:
        raise SystemExit("Signature verification failed")
    print("")

    print("Sending...")
    tx_hash = client.send_transaction(signed.raw_transaction)
    print(f"  TX Hash: {tx_hash}")
    print("")

    print("Waiting for confirmation...")
    receipt = client.wait_for_receipt(tx_hash)
    status = "SUCCESS" if receipt.status else "FAILED"
    gas_used = receipt.gas_used
    gas_pct = (gas_used / estimated_gas) * 100 if estimated_gas else 0
    fee = receipt.tx_fee.human
    print(f"  Block: {receipt.block_number}")
    print(f"  Status: {status}")
    print(f"  Gas Used: {gas_used} ({gas_pct:.0f}%)")
    print(f"  Fee: {fee} ETH")
    print("")

    if receipt.status:
        print("Integration test PASSED")
    else:
        print("Integration test FAILED")
        sys.exit(1)


def _format_gwei(value: int) -> str:
    return f"{Decimal(value) / Decimal(10**9):.4f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    main()
