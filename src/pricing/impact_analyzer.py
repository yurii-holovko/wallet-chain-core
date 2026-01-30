from __future__ import annotations

import argparse
import os
from decimal import Decimal

from chain.client import ChainClient
from core.base_types import Address

from .price_impact_analyzer import PriceImpactAnalyzer
from .uniswap_v2_pair import Token, UniswapV2Pair


def _parse_sizes(value: str, decimals: int) -> list[int]:
    sizes = []
    for chunk in value.split(","):
        text = chunk.strip().replace("_", "")
        if text == "":
            continue
        human = Decimal(text)
        sizes.append(int(human * (10**decimals)))
    return sizes


def _format_token_amount(raw: int, token: Token, decimals: int = 6) -> str:
    return _format_amount(Decimal(raw) / Decimal(10**token.decimals), decimals=decimals)


def _select_token(pair: UniswapV2Pair, token_in: str) -> Token:
    lowered = token_in.strip().lower()
    if lowered.startswith("0x"):
        addr = Address.from_string(lowered)
        if addr == pair.token0.address:
            return pair.token0
        if addr == pair.token1.address:
            return pair.token1
        raise SystemExit("token-in address not in pair")
    if lowered == pair.token0.symbol.lower():
        return pair.token0
    if lowered == pair.token1.symbol.lower():
        return pair.token1
    raise SystemExit("token-in symbol not in pair")


def _format_amount(value: Decimal, decimals: int = 4) -> str:
    return f"{value:,.{decimals}f}".rstrip("0").rstrip(".")


def _print_table(
    token_in: Token,
    token_out: Token,
    rows: list[dict],
    spot_price: Decimal,
    pair: UniswapV2Pair,
):
    header = f"Price Impact Analysis for {token_in.symbol} -> {token_out.symbol}"
    print(header)
    print(f"Pool: {pair.address.checksum}")

    reserve0 = Decimal(pair.reserve0) / Decimal(10**pair.token0.decimals)
    reserve1 = Decimal(pair.reserve1) / Decimal(10**pair.token1.decimals)
    print(
        f"Reserves: {_format_amount(reserve0)} {pair.token0.symbol} / "
        f"{_format_amount(reserve1)} {pair.token1.symbol}"
    )
    spot_line = (
        f"Spot Price: {_format_amount(spot_price, 2)} "
        f"{token_in.symbol}/{token_out.symbol}"
    )
    print(spot_line)
    print("")

    columns = [
        f"{token_in.symbol} In",
        f"{token_out.symbol} Out",
        "Exec Price",
        "Impact",
    ]
    widths = [13, 13, 13, 10]

    def sep(char: str) -> str:
        return "+" + "+".join(char * w for w in widths) + "+"

    def row(values: list[str]) -> str:
        padded = [values[i].rjust(widths[i]) for i in range(len(values))]
        return "|" + "|".join(padded) + "|"

    print(sep("-"))
    print(row(columns))
    print(sep("-"))
    for entry in rows:
        amount_in = Decimal(entry["amount_in"]) / Decimal(10**token_in.decimals)
        amount_out = Decimal(entry["amount_out"]) / Decimal(10**token_out.decimals)
        exec_price = entry["execution_price"]
        impact_pct = entry["price_impact_pct"]
        print(
            row(
                [
                    _format_amount(amount_in, 0),
                    _format_amount(amount_out, 6),
                    _format_amount(exec_price, 2),
                    f"{_format_amount(impact_pct, 2)}%",
                ]
            )
        )
    print(sep("-"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniswap V2 price impact analyzer")
    parser.add_argument("pair", help="Uniswap V2 pair address")
    parser.add_argument("--token-in", required=True, help="Token symbol or address")
    parser.add_argument(
        "--sizes",
        required=True,
        help="Comma-separated list of human amounts (e.g. 1000,10000,100000)",
    )
    parser.add_argument("--rpc", help="RPC URL (or set RPC_URL env var)")
    parser.add_argument(
        "--max-impact",
        default="1",
        help="Max impact percentage for max-size search (default 1)",
    )
    args = parser.parse_args()

    rpc_url = args.rpc or os.environ.get("RPC_URL")
    if not rpc_url:
        raise SystemExit("RPC URL required via --rpc or RPC_URL env var")

    client = ChainClient([rpc_url])
    pair_addr = Address.from_string(args.pair)
    pair = UniswapV2Pair.from_chain(pair_addr, client)

    token_in = _select_token(pair, args.token_in)
    token_out = pair.token1 if token_in.address == pair.token0.address else pair.token0

    sizes = _parse_sizes(args.sizes, token_in.decimals)
    analyzer = PriceImpactAnalyzer(pair)
    rows = analyzer.generate_impact_table(token_in, sizes)
    spot_price = analyzer._spot_price(token_in)

    _print_table(token_in, token_out, rows, spot_price, pair)

    max_impact = Decimal(args.max_impact)
    max_size = analyzer.find_max_size_for_impact(token_in, max_impact)
    max_size_human = Decimal(max_size) / Decimal(10**token_in.decimals)
    print("")
    print(
        f"Max trade for {max_impact}% impact: {_format_amount(max_size_human, 0)} "
        f"{token_in.symbol}"
    )


if __name__ == "__main__":
    main()
