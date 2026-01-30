import asyncio
import os

from pricing.mempool_monitor import MempoolMonitor

WS_URL_VALUE = os.environ.get("WS_URL")
if not WS_URL_VALUE:
    raise SystemExit("WS_URL env var is required (e.g. wss://... )")
WS_URL = WS_URL_VALUE


def on_swap(swap):
    print(f"{swap.dex} {swap.method} {swap.amount_in} -> min {swap.min_amount_out}")
    print(f"Slippage: {swap.slippage_tolerance:.2%}")


async def main():
    if "127.0.0.1" in WS_URL or "localhost" in WS_URL:
        monitor = MempoolMonitor(WS_URL, on_swap, rpc_url="http://127.0.0.1:8545")
    else:
        monitor = MempoolMonitor(
            WS_URL,
            on_swap,
            subscription_method="alchemy_pendingTransactions",
            subscription_params={"hashesOnly": False},
        )
    await monitor.start()


asyncio.run(main())
