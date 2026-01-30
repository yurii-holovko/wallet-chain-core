import os
import time

from web3 import Web3

ROUTER = Web3.to_checksum_address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")
WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
USDC = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")


ABI = [
    {
        "name": "swapExactETHForTokens",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
    }
]


def main() -> None:
    rpc_url = os.environ.get("FORK_RPC_URL", "http://127.0.0.1:8545")
    sender = os.environ.get("ANVIL_SENDER")
    if not sender:
        raise SystemExit("ANVIL_SENDER is required (use one of Anvil's accounts)")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise SystemExit(f"Failed to connect to {rpc_url}")

    contract = w3.eth.contract(address=ROUTER, abi=ABI)
    deadline = int(time.time()) + 600
    tx = contract.functions.swapExactETHForTokens(
        1,  # amountOutMin
        [WETH, USDC],
        sender,
        deadline,
    ).build_transaction(
        {
            "from": sender,
            "value": w3.to_wei(0.01, "ether"),
            "nonce": w3.eth.get_transaction_count(sender),
        }
    )

    tx_hash = w3.eth.send_transaction(tx)
    print(f"sent: {tx_hash.hex()}")


if __name__ == "__main__":
    main()
