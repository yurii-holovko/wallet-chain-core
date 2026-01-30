from eth_abi import encode

from core.base_types import Address
from pricing.mempool_monitor import MempoolMonitor


def _calldata_swap_exact_tokens_for_tokens(
    amount_in: int,
    amount_out_min: int,
    path: list[str],
    to: str,
    deadline: int,
) -> str:
    selector = "0x38ed1739"
    encoded = encode(
        ["uint256", "uint256", "address[]", "address", "uint256"],
        [amount_in, amount_out_min, path, to, deadline],
    )
    return selector + encoded.hex()


def test_parse_transaction_non_swap():
    monitor = MempoolMonitor("ws://localhost:8545", lambda _: None)
    assert monitor.parse_transaction({"input": "0x"}) is None


def test_parse_transaction_malformed():
    monitor = MempoolMonitor("ws://localhost:8545", lambda _: None)
    assert monitor.parse_transaction({"input": 123}) is None
    assert monitor.parse_transaction({"input": "0x1234"}) is None


def test_parse_transaction_uniswap_v2_swap():
    monitor = MempoolMonitor("ws://localhost:8545", lambda _: None)
    path = [
        "0x0000000000000000000000000000000000000001",
        "0x0000000000000000000000000000000000000002",
    ]
    calldata = _calldata_swap_exact_tokens_for_tokens(
        amount_in=1000,
        amount_out_min=900,
        path=path,
        to="0x0000000000000000000000000000000000000003",
        deadline=1234567890,
    )
    tx = {
        "hash": "0xabc",
        "from": "0x0000000000000000000000000000000000000004",
        "to": "0x0000000000000000000000000000000000000005",
        "input": calldata,
        "gasPrice": "0x1",
    }
    parsed = monitor.parse_transaction(tx)
    assert parsed is not None
    assert parsed.dex == "UniswapV2"
    assert parsed.method == "swapExactTokensForTokens"
    assert parsed.amount_in == 1000
    assert parsed.min_amount_out == 900
    assert parsed.deadline == 1234567890
    assert parsed.token_in == Address.from_string(path[0])
    assert parsed.token_out == Address.from_string(path[-1])
    assert parsed.path == [Address.from_string(addr) for addr in path]
