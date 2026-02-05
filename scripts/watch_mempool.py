import asyncio

import config
from pricing.mempool_monitor import MempoolMonitor

WS_URL = config.get_env("WS_URL", required=True)


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
