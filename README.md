# Wallet Chain Core

## Quick Start

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
make setup
python src/main.py generate
```

PowerShell:
```powershell
$env:PRIVATE_KEY = "0xYOUR_PRIVATE_KEY"  # pragma: allowlist secret
make run
```

<details>
<summary><span style="font-size:1.25em"><strong>Week 1: Wallet + Chain</strong></span></summary>

Part 1 implements secure wallet management, canonical serialization, and base
types used across the project. A CLI wraps these features for quick testing.

## Getting Started

### Prerequisites
* **Python 3.14+**
* **Git**

### Installation
1. **Setup environment**:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

2. **Install tools**:
```bash
make setup
```
`make setup` also writes a `.pth` file so `src/` is on `PYTHONPATH` (enables
`python -m pricing.*`, `from chain...`, `from core...` without extra setup).

### IDE setup (important)
Mark `src/` as a **Sources Root** so imports like `from core...` and
`from chain...` resolve correctly.



## Development Tools

* **Formatter**: `black` & `isort`
* **Linter**: `flake8`
* **Tests**: `pytest`
* **Security**: `detect-secrets`

## Commands

Use the following `make` commands for a standardized workflow:

| Command | Description |
| --- | --- |
| `make run` | Executes the main application (`src/main.py`) |
| `make test` | Runs the full `pytest` suite |
| `make check` | Runs the full suite: formatting (black/isort), linting (flake8), and tests |
| `make setup` | Installs dependencies and sets up pre-commit hooks |

## Project Structure

```
src/
  core/
    wallet_manager.py   # key management + signing
    serializer.py       # canonical JSON + keccak hashing
    base_types.py       # Address, TokenAmount, Transaction types
  chain/
    __init__.py         # chain module exports
    client.py           # RPC client with retries and error handling
    transaction_builder.py  # fluent transaction builder
    analyzer.py         # tx analysis CLI
  pricing/
    uniswap_v2_pair.py  # Uniswap V2 pair math + on-chain helpers
    route.py            # routing + gas-aware selection
    fork_simulator.py   # forked swap simulation
    mempool_monitor.py  # pending tx watcher
    impact_analyzer.py  # price impact CLI
  main.py               # CLI entrypoint
docs/
  examples/             # sample JSON payloads for CLI
```

## Part 1: WalletManager CLI

The CLI is in `src/main.py` and is used by `make run`. Wallet logic lives in
`src/core/wallet_manager.py`. The `chain/` module is reserved for transaction
building, sending, and analysis.

### Environment setup

The default commands read the private key from `PRIVATE_KEY`:

PowerShell:
```powershell
$env:PRIVATE_KEY = "0xYOUR_PRIVATE_KEY"  # pragma: allowlist secret
```

macOS/Linux:
```bash
export PRIVATE_KEY="0xYOUR_PRIVATE_KEY"  # pragma: allowlist secret
```

### Run

Default command prints the address from `PRIVATE_KEY`:
```bash
make run
```

You can also call the CLI directly with subcommands:
```bash
python src/main.py address
python src/main.py generate
python src/main.py sign-message "hello"
```

### Canonical serialization

`CanonicalSerializer` produces deterministic JSON and its keccak hash.
Rules:
- Keys sorted (recursive)
- No whitespace
- Unicode preserved (UTF-8)
- Floats rejected (use strings/integers)

Use the CLI:
```bash
python src/main.py serialize --input docs/examples/payload.json
python src/main.py hash --input docs/examples/payload.json
python src/main.py verify-determinism --input docs/examples/payload.json --iterations 50
```

### Base types

`Address` validates and normalizes to checksum (case-insensitive equality).
`TokenAmount` stores raw integer (wei) and uses `Decimal` for human values.
`TransactionRequest` produces a web3-compatible dict.
`TransactionReceipt` parses a receipt and computes tx fee.

### Transaction input validation

`sign-transaction` validates inputs early with clear errors:
- Required fields: `nonce`, `gas`, `value`, `data`, `chainId`
- Fee: either `gasPrice` **or** both `maxFeePerGas` and `maxPriorityFeePerGas`
- `to` is checksummed (or `null` for contract creation)
- Numeric fields accept `int` or `0x...` hex string

### Typed data signing (EIP-712)

Provide JSON files for domain, types, and value:
```bash
python src/main.py sign-typed-data --domain docs/examples/domain.json --types docs/examples/types.json --value docs/examples/value.json
```

### Transaction signing

Provide a JSON file containing the transaction dict:
```bash
python src/main.py sign-transaction --tx docs/examples/tx.json
```

### Base types helpers

Build a transaction dict from a type-safe spec (uses `Address` + `TokenAmount`):
```bash
python src/main.py build-transaction --input docs/examples/tx_spec.json
```

Example `tx_spec.json`:
```json
{
  "to": "0x000000000000000000000000000000000000dead",
  "value_human": "1.5",
  "decimals": 18,
  "symbol": "ETH",
  "data": "0x",
  "nonce": 1,
  "gas_limit": 21000,
  "max_fee_per_gas": 1000000000,
  "max_priority_fee": 100000000,
  "chain_id": 1
}
```

Compute fee from a receipt (uses `TransactionReceipt` + `TokenAmount`):
```bash
python src/main.py receipt-fee --input docs/examples/receipt.json
```

### Encrypted keyfile import/export

Export from `PRIVATE_KEY`:
```bash
python src/main.py keyfile-export --path wallet.json --password "strong-password"
```

Import and print address:
```bash
python src/main.py keyfile-import --path wallet.json --password "strong-password"
```

### Examples directory

All sample payloads are under `docs/examples/`:
- `domain.json`, `types.json`, `value.json` (EIP-712)
- `tx.json` (transaction signing)
- `payload.json` (canonical serialization)
- `tx_spec.json` (base types transaction builder)
- `receipt.json` (fee calculation)

### Tests

Run all tests:
```bash
make test
```

Run formatting, linting, and tests:
```bash
make check
```

## Part 2: Chain Module

### RPC Client

`ChainClient` provides resilient JSON-RPC calls with retries, backoff, and
endpoint fallback. It classifies common errors (insufficient funds, nonce too
low, replacement underpriced) and exposes helpers for balance, nonce, gas price,
estimate gas, send transaction, and receipts.

Example:
```python
from chain import ChainClient
from core.base_types import Address

client = ChainClient(["https://rpc.example"])
balance = client.get_balance(Address.from_string("0x000000000000000000000000000000000000dead"))
nonce = client.get_nonce(Address.from_string("0x000000000000000000000000000000000000dead"))
```

### GasPrice

`GasPrice` wraps base/priority fees and computes `maxFeePerGas` with a buffer:
```python
gas = client.get_gas_price()
max_fee = gas.get_max_fee("high")
```

### Transaction Builder

Fluent builder that composes a `TransactionRequest`, estimates gas, sets fees,
signs, and sends.

```python
from chain import TransactionBuilder
from core.base_types import Address, TokenAmount

tx = (
    TransactionBuilder(client, wallet)
    .to(Address.from_string("0x000000000000000000000000000000000000dead"))
    .value(TokenAmount.from_human("0.1", 18, "ETH"))
    .data(b"")
    .with_gas_estimate()
    .with_gas_price("high")
    .build()
)
```

To send:
```python
tx_hash = (
    TransactionBuilder(client, wallet)
    .to(Address.from_string("0x000000000000000000000000000000000000dead"))
    .value(TokenAmount.from_human("0.1", 18, "ETH"))
    .data(b"")
    .with_gas_estimate()
    .with_gas_price("medium")
    .send()
)
```

### Transaction Analyzer CLI (Optional)

Analyze any transaction:
```bash
python -m chain.analyzer <tx_hash> --rpc <URL>
python -m chain.analyzer <tx_hash> --rpc <URL> --format json
```

Features:
- Function decoding for common ERC-20/Uniswap calls
- Transfer/Swap/Sync event parsing
- Pending tx handling, invalid hash checks
- Revert reason (when available)
- Token metadata caching

## Part 3: Integration Test

Run the Sepolia integration test end-to-end:
```bash
SEPOLIA_RPC_URL=... RECIPIENT_ADDRESS=0x... PRIVATE_KEY=0x... python scripts/integration_test.py
```

Expected flow:
- Loads wallet from `PRIVATE_KEY`
- Connects to Sepolia
- Checks balance
- Builds an ETH transfer
- Estimates gas and sets fees
- Signs and verifies signature locally
- Sends transaction and waits for receipt
- Prints receipt analysis and PASS/FAIL

## Acceptance Checklist

Required:
- [x] `WalletManager` loads key from env, signs messages and transactions
- [x] Private key never appears in logs/repr/str (tested)
- [x] `CanonicalSerializer` deterministic over 1000 iterations
- [x] `Address` and `TokenAmount` with full validation
- [x] `ChainClient` connects, gets balance, estimates gas, sends transactions
- [x] `TransactionBuilder` creates and signs valid transactions
- [x] Transaction Analyzer CLI works on mainnet txs
- [x] Integration test passes on Sepolia (requires RPC + funded key)
- [x] Minimum 15 unit tests covering edge cases (37 tests)
- [x] README with setup and usage examples

Test coverage:
- [x] Wallet: key security, signing, verification
- [x] Serializer: all listed edge cases
- [x] Types: validation, arithmetic, equality
- [x] Client: retry logic, error classification
- [x] Builder: validation, gas estimation
- [x] Analyzer: parsing, decoding, edge cases

</details>
<details>
<summary><span style="font-size:1.25em"><strong>Week 2: Pricing</strong></span></summary>

### Overview
Pricing module for AMM math, routing, simulation, and mempool monitoring.

### Architecture
```
MempoolMonitor ──▶ ParsedSwap ──▶ PricingEngine ──┐
                                                   ├─▶ RouteFinder ──▶ Route
                                                   ├─▶ UniswapV2Pair (math)
                                                   └─▶ ForkSimulator (forked sim)
```

### Modules
- `src/pricing/uniswap_v2_pair.py`: Uniswap V2 pair math and helpers
- `src/pricing/route.py`: route modeling and route finding
- `src/pricing/fork_simulator.py`: swap/route simulation on forked RPC
- `src/pricing/mempool_monitor.py`: mempool swap parsing/monitoring
- `src/pricing/pricing_engine.py`: orchestration entry point
- `src/pricing/impact_analyzer.py`: price impact CLI
- `src/pricing/price_impact_analyzer.py`: price impact math utilities

### Setup
Start a local fork (requires Foundry/Anvil):
```bash
export ETH_RPC_URL="https://mainnet.example"
bash scripts/start_fork.sh
```
Install Foundry (Git Bash):
```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

Set RPC/WS endpoints (or copy from `env.example`):
```bash
export RPC_URL="https://mainnet.example"
export WS_URL="wss://mainnet.example"
```
Environment loading is centralized in `config.py`: all entrypoints call `config.get_env()` which loads `.env` once via `python-dotenv`. You can still export env vars, but `.env` works automatically.
For the mempool watcher script:
```bash
export WS_URL="wss://mainnet.example"
```
PowerShell convenience loader:
```powershell
.\scripts\load_env.ps1 -EnvFile .env
```

### Configuration
All entrypoints load environment variables via `config.get_env()` which lazily loads `.env` once. Prefer `config.get_env("NAME", required=True)` instead of `os.environ.get()` to keep configuration consistent and avoid leaking sensitive values in errors.

### Examples
```bash
python -m pricing.impact_analyzer 0xPairAddress --token-in USDC --sizes 1000,10000 --rpc $RPC_URL
```

```python
async def main():
    def on_swap(swap: ParsedSwap):
        print(f\"Detected swap: {swap.dex} {swap.method}\")
        print(f\"  {swap.amount_in} → min {swap.min_amount_out}\")
        print(f\"  Slippage tolerance: {swap.slippage_tolerance:.2%}\")

    monitor = MempoolMonitor(WS_URL, on_swap)
    await monitor.start()
```

```python
client = ChainClient([RPC_URL])
engine = PricingEngine(client, "http://127.0.0.1:8545", WS_URL)
engine.load_pools([Address.from_string("0xPairAddress")])
quote = engine.get_quote(token_in, token_out, amount_in, gas_price_gwei=10, sender=sender)
print(quote.expected_output, quote.simulated_output)
```

### Demo Runbook (Week 2)
1) Load env vars:
```powershell
.\scripts\load_env.ps1 -EnvFile .env
```

2) Start the fork (Git Bash):
```bash
export ETH_RPC_URL="https://mainnet.example"
bash scripts/start_fork.sh
```

3) Start mempool watcher (PowerShell):
```powershell
$env:WS_URL = "ws://127.0.0.1:8545"
python scripts/watch_mempool.py
```

4) Send a test swap on the fork (PowerShell):
```powershell
$env:ANVIL_SENDER = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
python scripts/send_test_swap.py
```

5) Show price impact table (mainnet pool):
```powershell
python -m pricing.impact_analyzer `
  0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc `
  --token-in USDC `
  --sizes 1000,10000,100000 `
  --rpc $env:RPC_URL
```

6) Show best route with gas flip:
```powershell
python scripts/demo_route.py
```

7) Show Solidity-matching math (tests):
```powershell
pytest tests/test_uniswap_v2_pair.py
```

### Tests
```bash
pytest tests/test_uniswap_v2_pair.py
pytest tests/test_route.py
pytest tests/test_price_impact_analyzer.py
pytest tests/test_mempool_monitor.py
pytest tests/test_mainnet_uniswap_v2_swap.py
```

### Checklist (Week 2)
- AMM math: integer-only, matches Solidity outputs
- Routing: multi-hop discovery with gas-aware selection
- Monitoring: pending tx watcher + swap decoding
- Simulation: forked swap execution + receipt parsing
- CLI: price impact analyzer output
- Docs: architecture + examples


</details>
<details>
<summary><span style="font-size:1.25em"><strong>Week 3: Exchange + Inventory</strong></span></summary>

### Overview
Week 3 integrates a CEX via `ccxt`, adds order book analytics, portfolio
tracking across venues, and PnL + rebalance planning for arbitrage workflows.

### Architecture
```
PricingEngine (Week 2) ──▶ ArbChecker ──▶ ExchangeClient (ccxt)
                                       ├─▶ OrderBookAnalyzer
                                       ├─▶ InventoryTracker
                                       └─▶ PnLEngine / RebalancePlanner
```

### Modules
- `src/exchange/client.py`: ccxt wrapper (order books, balances, orders, fees)
- `src/exchange/orderbook.py`: order book analytics and slippage modeling
- `src/inventory/tracker.py`: multi-venue inventory tracking
- `src/inventory/pnl.py`: arb trade PnL tracking and summaries
- `src/inventory/rebalancer.py`: rebalance planning (no execution)
- `src/integration/arb_checker.py`: end-to-end arb opportunity checks

### Architecture
```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│ Pricing      │────▶│ ArbChecker      │────▶│ InventoryTracker     │
│ (Uniswap V2) │     │ (decision core) │     │ + RebalancePlanner   │
└──────┬───────┘     └───────┬─────────┘     └──────────┬───────────┘
       │                     │                            │
       │                     │                            │
┌──────▼────────┐     ┌──────▼──────────┐        ┌────────▼─────────┐
│ ChainClient   │     │ ExchangeClient  │        │ PnLEngine         │
│ (RPC/tx)      │     │ (Binance)       │        │ (trades, charts)  │
└───────────────┘     └──────┬──────────┘        └──────────────────┘
                             │
                             │
                      ┌──────▼─────────┐
                      │ OrderBook      │
                      │ Analyzer + WS  │
                      └────────────────┘
```

### Setup
Set Binance testnet credentials (see `env.example`):
```powershell
$env:BINANCE_TESTNET_API_KEY = "your_key_here"
$env:BINANCE_TESTNET_SECRET = "your_secret_here"
```

`config.BINANCE_CONFIG` reads env vars via `config.get_env()` and enables the
Binance sandbox + rate limiting by default.

### CLIs (Quick Reference)
Live order book + analysis:
```bash
python -m src.exchange.orderbook ETH/USDT --depth 50 --depth-bps 15 --imbalance-levels 10 --walk-sizes 1,5,10
```

Live order book (WebSocket):
```bash
python -m src.exchange.orderbook_ws ETH/USDT --ws-mode stream
python -m src.exchange.orderbook_ws ETH/USDT --ws-mode api
```

Notes:
- For mainnet streams, the REST snapshot runs without testnet keys; if you want
  authenticated mainnet REST calls, set `BINANCE_API_KEY`/`BINANCE_SECRET`.
- For testnet streams, pass `--testnet` and set
  `BINANCE_TESTNET_API_KEY`/`BINANCE_TESTNET_SECRET`.
- Enable debug logs with `ORDERBOOK_WS_DEBUG=1`.

LIMIT IOC (testnet):
```bash
python scripts/demo_ioc.py --symbol ETH/USDT --amount 0.01 --price-multiplier 0.999
```

Portfolio snapshot (Binance + wallet):
```bash
python scripts/demo_snapshot.py --wallet-balances docs/examples/wallet_balances.json
```

Arb checker (real prices):
```bash
python -m src.integration.arb_checker ETH/USDT --size 1 --gas-usd 5 --balances docs/examples/wallet_balances.json
```

Arb checker (marginal-size optimization):
```bash
python -m src.integration.arb_checker ETH/USDT --size 2 --step 0.1 --gas-usd 5 --balances docs/examples/wallet_balances.json
```
Notes:
- Uses marginal slice pricing on DEX/CEX order books to find the optimal size.
- Stops when the next slice becomes unprofitable; gas is amortized across slices.
- DEX fee + price impact are already included in AMM execution prices.

PnL report (5 trades):
```bash
python -m src.inventory.pnl --summary --trades docs/examples/pnl_trades.json
```

Check inventory skew:
```bash
python -m src.inventory.rebalancer --check
```

Generate rebalance plan:
```bash
python -m src.inventory.rebalancer --plan ETH
```

Stretch goals

Real-time inventory dashboard (TUI):
```bash
python -m src.inventory.dashboard --interval 5 --wallet-balances docs/examples/wallet_balances.json --wallet-only --nonzero-only
```

Historical PnL chart export:
```bash
python -m src.inventory.pnl --export-chart pnl.png --trades docs/examples/pnl_trades.json
```

Arb check with CSV logging:
```bash
python -m src.integration.arb_checker ETH/USDT --size 1 --log-csv arb.csv --log-min-bps 0
```

### Definition of Done
## Required (all must pass)
- [x] `ExchangeClient` connects to Binance testnet and fetches order books
- [x] `ExchangeClient` places and cancels LIMIT IOC orders
- [x] Rate limiter prevents API ban
- [x] `OrderBookAnalyzer.walk_the_book` simulates fills correctly
- [x] `OrderBookAnalyzer` CLI shows depth, spread, imbalance
- [x] `InventoryTracker` aggregates balances across venues
- [x] `InventoryTracker.can_execute` validates both legs before trade
- [x] `InventoryTracker.skew` detects imbalanced positions
- [x] `RebalancePlanner` generates valid transfer plans
- [x] `RebalancePlanner` respects min operating balances
- [x] `PnLEngine` tracks per-trade and aggregate PnL correctly
- [x] `ArbChecker` integrates pricing (Week 2) + exchange + inventory
- [x] Minimum 25 tests covering edge cases (current: 70+ unit tests)
- [x] README with architecture diagram showing module interactions

## Tests Must Cover
- [x] Order book parsing with real testnet data (live integration test, skipped without keys)
- [x] Walk-the-book with various sizes (small, large, insufficient)
- [x] Inventory update after trades (buy/sell/fee deductions)
- [x] Skew calculation with various distributions
- [x] Rebalance plan generation with fee accounting
- [x] PnL calculation with real fee structures
- [x] Integration: arb check passes when profitable, rejects when not

## Stretch Goals
- [x] WebSocket-based order book with incremental updates (+10)
- [ ] Multi-exchange support (add Bybit testnet via ccxt) (+10)
- [x] Real-time inventory dashboard (terminal UI) (+5)
- [x] Historical PnL chart export (matplotlib/plotly) (+5)
- [x] Arb opportunity logger with CSV export (+5)

to be continued...
</details>
<details>
<summary><span style="font-size:1.25em"><strong>Week 4: Strategy + Execution + Recovery</strong></span></summary>

### Overview
Week 4 turns the system into a production-style arbitrage loop:
- strategy generates opportunities from CEX/DEX prices,
- scorer ranks them by quality,
- executor runs two-leg state-machine execution with retries/unwind,
- recovery layer protects against duplicates, stale signals, and cascading failures.

This week introduces the decision-and-control plane for reliable operation.

### Architecture
```
SignalGenerator ──▶ SignalScorer ──▶ SignalPriorityQueue ──▶ Executor
      │                    │                   │                 │
      │                    │                   │                 ├─▶ RecoveryManager
      │                    │                   │                 │    ├─ CircuitBreaker
      │                    │                   │                 │    ├─ ReplayProtection
      │                    │                   │                 │    └─ FailureClassifier
      │                    │                   │                 │
      └─ uses CEX+DEX prices ───────────────────────────────────┘
```

### Key Modules
- `src/strategy/signal.py`
  - `Signal` dataclass + `Direction` enum.
  - Built-in validity checks (`expiry`, inventory/limits, expected net PnL, score).
  - Unique `signal_id` and helper methods (`create`, `age_seconds`).

- `src/strategy/fees.py`
  - Fee model (`FeeStructure`) for CEX fee + DEX fee + slippage + gas.
  - Computes `total_fee_bps`, breakeven spread, and expected net profit.

- `src/strategy/generator.py`
  - Fetches CEX order book and DEX quote (real pricer when configured, otherwise fallback simulation).
  - Computes spread in both directions:
    - BUY CEX / SELL DEX
    - BUY DEX / SELL CEX
  - Applies profitability gates (`min_spread_bps`, `min_profit_usd`), cooldown, inventory checks, and position limits.
  - Emits a rich `Signal` with metadata (depth, breakeven bps, raw prices).

- `src/strategy/scorer.py`
  - Multi-factor scoring (0-100):
    - spread (optionally net-of-fees),
    - liquidity (top-of-book depth),
    - inventory impact (rebalancing vs worsening skew),
    - pair-specific history (EMA),
    - urgency/freshness.
  - Produces transparent score breakdown in `signal.meta["score_breakdown"]`.
  - Supports score decay over time for queue prioritization.

- `src/executor/engine.py`
  - Strict state machine with guarded transitions:
    - `IDLE -> VALIDATING -> LEG1_* -> LEG2_* -> DONE/FAILED/UNWINDING`
  - Supports DEX-first (Flashbots style) and CEX-first sequencing.
  - Retry with exponential backoff and per-leg timeout.
  - Unwind flow if first leg fills and second leg fails.
  - Tracks detailed execution metrics (latency, retries, slippage, fill ratio) and event audit trail.
  - Computes realized PnL from actual fills and fee assumptions.

- `src/executor/recovery.py`
  - `FailureClassifier`: categorizes errors (TRANSIENT, PERMANENT, RATE_LIMIT, NETWORK, UNKNOWN).
  - `CircuitBreaker`: global + per-pair breaker with failure-window and drawdown trip logic.
  - `ReplayProtection`: dedup, staleness checks, nonce monotonicity, bounded memory (LRU), audit log.
  - `RecoveryManager`: single pre-flight and post-outcome interface used by executor.

- `scripts/arb_bot.py`
  - Orchestrates end-to-end runtime loop:
    - signal generation,
    - scoring + queueing,
    - execution,
    - balance sync,
    - metrics and webhook integration,
    - recovery snapshots / breaker visibility.

### Runtime Flow
1. Load balances and market data.
2. Generate candidate signals per pair.
3. Score and filter by minimum score.
4. Push to priority queue and drain highest-priority first.
5. Execute via state machine (with retries and guarded transitions).
6. On failure, trigger unwind if needed and record outcome in recovery.
7. Update historical scorer outcomes and metrics.

### Configuration Highlights
- Strategy thresholds: `min_spread_bps`, `min_profit_usd`, `max_position_usd`, `signal_ttl_seconds`, `cooldown_seconds`.
- Scorer threshold: `min_score` (typically 50-60).
- Executor controls: `use_flashbots`, leg timeouts, retry counts, `simulation_mode`.
- Recovery controls:
  - breaker `failure_threshold`, `window_seconds`, `max_drawdown_usd`, `cooldown_seconds`,
  - replay `ttl_seconds`, `max_age_seconds`, `nonce_check`.

### Run
Both modes live in a single script — `scripts/arb_bot.py`:

```bash
# Simulation — fake DEX prices, frequent trades, full pipeline
python scripts/arb_bot.py --mode simulation

# Paper — REAL CEX + DEX on-chain prices, simulated execution, PnL dashboard
python scripts/arb_bot.py --mode paper
```

Default (no flag) is `simulation`.

### Tests (Week 4)
Main suites:
- `tests/test_signal.py`
- `tests/test_scorer.py`
- `tests/test_executor.py`
- `tests/test_recovery.py`

Run just Week 4 tests:
```bash
pytest tests/test_signal.py tests/test_scorer.py tests/test_executor.py tests/test_recovery.py
```

Run all:
```bash
make test
```

### Definition of Done (Week 4)
- [x] Signals include economics + validity gates
- [x] Generator validates spread/profit/inventory/limits
- [x] Scorer ranks opportunities with transparent breakdown
- [x] Executor enforces strict transition-safe state machine
- [x] Retry/backoff and unwind logic implemented
- [x] Recovery layer protects against replay/stale/duplicate signals
- [x] Circuit breaker trips on failures and drawdown
- [x] Comprehensive unit tests for signal/scorer/executor/recovery

### Stretch Goals
- [x] Webhook alerts on circuit breaker trip
- [x] Real DEX execution (not simulation)
- [x] Multiple signals queued by priority
- [x] Prometheus metrics export

---

### Stretch Goal 1 — Webhook Alerts on Circuit Breaker Trip ✅

**Module:** `src/executor/alerts.py`

**What's implemented:**
- `WebhookAlerter` with background delivery thread — fire-and-forget, never blocks the hot path.
- Alert types: `CIRCUIT_BREAKER_TRIP`, `CIRCUIT_BREAKER_HALF_OPEN`, `CIRCUIT_BREAKER_RESET`, `EXECUTION_FAILURE`, `DRAWDOWN`.
- `WebhookConfig.from_env()` reads `WEBHOOK_URLS`, `WEBHOOK_TIMEOUT`, `WEBHOOK_MAX_RETRIES`, `WEBHOOK_COOLDOWN` from `.env`.
- Per-alert-type + per-pair cooldown to avoid alert storms.
- Exponential back-off retry on failed deliveries.
- Bounded queue (500 max) — drops oldest if full.
- `RecoveryManager.record_outcome()` automatically fires `on_circuit_breaker_trip()` when breaker transitions from CLOSED → OPEN.
- `ArbBot._tick()` also fires `on_execution_failure()` on every failed execution.

**Integration in `scripts/arb_bot.py`:**
```python
self.alerter = WebhookAlerter(WebhookConfig.from_env())
self.alerter.start()
self.executor.recovery.alerter = self.alerter  # auto-fires on CB trip
```

**How to test:**
```bash
pytest tests/test_alerts.py -v
```
Tests cover: payload structure, env config parsing, disabled alerter, cooldown throttling, real HTTP delivery to a local test server, retry on 500, all-retries-fail tracking, convenience senders (`on_circuit_breaker_trip`, `on_circuit_breaker_half_open`, `on_circuit_breaker_reset`, `on_execution_failure`, `on_drawdown`), and stats structure.

**Manual smoke test:**
1. Set `WEBHOOK_URLS=https://your-endpoint.com/hook` in `.env`.
2. Run the bot and force repeated failures (e.g. misconfigure DEX config so every execution fails).
3. After `failure_threshold` failures within the window, the circuit breaker trips → webhook fires with a JSON payload containing `type`, `level`, `pair`, `message`, `details`, `timestamp`.

---

### Stretch Goal 2 — Real DEX Execution (Not Simulation) ✅

**Module:** `src/executor/engine.py` → `_execute_real_dex_leg()`

**What's implemented:**
- When `ExecutorConfig.simulation_mode = False`, the executor calls `_execute_real_dex_leg()` instead of returning a simulated fill.
- Builds and signs real Uniswap V2 swap transactions via `TransactionBuilder`:
  - `BUY_CEX_SELL_DEX` → `swapExactETHForTokens` (sell ETH for quote token on DEX).
  - `BUY_DEX_SELL_CEX` → `swapExactTokensForETH` (sell quote token for ETH on DEX).
- Configurable slippage (`dex_slippage_bps`), deadline (`dex_deadline_seconds`), gas priority (`dex_gas_priority`), and chain ID (`dex_chain_id`).
- Uses `ChainClient` for RPC and `WalletManager` for key management — initialized lazily via `_ensure_dex_ready()`.
- Returns real tx hash from on-chain receipt.
- Simulation path returns `0xsim_...` hashes; real path returns actual `0x...` hashes.

**Branching logic:**
```python
async def _execute_dex_leg(self, signal, size):
    if self.config.simulation_mode:
        return {"success": True, "price": ..., "tx_hash": "0xsim_..."}
    return await asyncio.to_thread(self._execute_real_dex_leg, signal, size)
```

**How to test:**
```bash
pytest tests/test_executor.py -v
```
The executor test suite covers the full state machine (IDLE → VALIDATING → LEG1 → LEG2 → DONE/FAILED/UNWINDING) in simulation mode. Real DEX execution is integration-level and requires a testnet setup.

**Manual smoke test (Sepolia testnet):**
1. Set in `.env`:
   ```
   SEPOLIA_RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY
   PRIVATE_KEY=0x...
   DEX_ROUTER_ADDRESS=0x...   # Uniswap V2 Router on Sepolia
   DEX_WETH_ADDRESS=0x...
   DEX_QUOTE_TOKEN_ADDRESS=0x...
   ```
2. Run `scripts/arb_bot.py` with `simulation=False` in config.
3. Confirm execution context has a real tx hash format (`0x...64hex`) instead of `0xsim_...`.
4. Verify transaction on Sepolia Etherscan.

---

### Stretch Goal 3 — Multiple Signals Queued by Priority ✅

**Module:** `src/strategy/priority_queue.py`

**What's implemented:**
- `SignalPriorityQueue` backed by a min-heap (negated scores for max-priority-first ordering).
- Configurable via `PriorityQueueConfig`:
  - `max_depth` (default 50) — evicts lowest-scoring signals when over capacity.
  - `max_per_pair` (default 1) — prevents executing 2 signals for the same pair in one tick.
  - `score_decay` — re-applies decay function before yielding (stale signals lose priority).
  - `min_score` — drops signals that fall below threshold after decay.
- Deduplication by `signal_id` — rejects duplicate pushes.
- `drain()` yields signals in descending score order with expiry checks.
- Stats tracking: `total_pushed`, `total_dropped`, `total_yielded`, `queued`.

**Integration in `scripts/arb_bot.py`:**
```python
# Phase 1: Collect signals into priority queue
self.priority_queue.clear()
for pair in self.pairs:
    signal = self.generator.generate(pair, self.trade_size)
    signal.score = self.scorer.score(signal, skews)
    self.priority_queue.push(signal)

# Phase 2: Execute signals in priority order
for signal in self.priority_queue.drain():
    ctx = await self.executor.execute(signal)
```

**How to test:**
```bash
pytest tests/test_priority_queue.py -v
```
Tests cover: push/drain ordering, deduplication, max-depth eviction with stats, per-pair concurrency limits, signal expiry, score decay application, decay-below-min-score drop, peek without removal, clear, and stats accumulation.

**Manual smoke test:**
1. Configure multiple pairs (e.g. `ETH/USDT`, `BTC/USDT`, `ARB/USDT`) in bot config.
2. Lower `min_score` to ~30 so multiple signals are accepted.
3. Run bot and observe logs — higher-score signals are executed first within each tick.
4. Confirm `arb_queue_depth` gauge in `/metrics` reflects queue size before draining.

---

### Stretch Goal 4 — Prometheus Metrics Export ✅

**Module:** `src/executor/metrics.py`

**What's implemented:**
- Custom Prometheus-compatible metrics (no `prometheus_client` dependency):
  - **Counter**: `arb_signals_total{pair,direction}`, `arb_executions_total{pair,state}`, `arb_unwinds_total{pair,success}`, `arb_circuit_breaker_trips_total{pair}`, `arb_webhook_sent_total`.
  - **Gauge**: `arb_spread_bps{pair}`, `arb_score{pair}`, `arb_pnl_total_usd`, `arb_inventory_skew_pct{pair,venue}`, `arb_circuit_breaker_state{pair}` (0=closed, 1=open, 2=half_open), `arb_queue_depth`.
  - **Histogram**: `arb_execution_latency_ms{pair,leg}` with configurable buckets.
- `MetricsRegistry` collects all metrics and emits Prometheus text exposition format.
- `MetricsServer` runs a background HTTP server (default port 9090):
  - `GET /metrics` — full Prometheus scrape endpoint.
  - `GET /health` — health check (`{"status":"ok"}`).
- Thread-safe counters and gauges (tested with concurrent writes).

**Integration in `scripts/arb_bot.py`:**
```python
self.metrics = MetricsRegistry()
self.metrics_server = MetricsServer(self.metrics, port=metrics_port)
self.metrics_server.start()

# In _tick():
self.metrics.signals_total.inc(pair=pair, direction=signal.direction.name)
self.metrics.spread_bps.set(signal.spread_bps, pair=pair)
self.metrics.pnl_total.inc(ctx.actual_net_pnl or 0)
self.metrics.queue_depth.set(self.priority_queue.size)
self.metrics.cb_state.set(cb_val, pair=pair)
```

**How to test:**
```bash
pytest tests/test_metrics.py -v
```
Tests cover: counter inc (with/without labels), gauge set/inc/overwrite, histogram observe with buckets and labels, HELP/TYPE header lines, registry `collect_all()` output, real HTTP `/metrics` endpoint, `/health` endpoint, 404 on unknown paths, server start/stop lifecycle, and concurrent thread-safety for counters and gauges.

**Manual smoke test:**
1. Start the bot (default `METRICS_PORT=9090`).
2. Open `http://localhost:9090/metrics` in a browser or `curl`.
3. Verify Prometheus text format with key series:
   ```
   # HELP arb_signals_total Total arbitrage signals generated
   # TYPE arb_signals_total counter
   arb_signals_total{pair="ETH/USDT",direction="BUY_CEX_SELL_DEX"} 5.0

   # HELP arb_pnl_total_usd Cumulative PnL in USD
   # TYPE arb_pnl_total_usd gauge
   arb_pnl_total_usd 12.5
   ```
4. Add to `prometheus.yml`:
   ```yaml
   scrape_configs:
     - job_name: 'arb-bot'
       static_configs:
         - targets: ['localhost:9090']
   ```

---

### Run All Stretch Goal Tests
```bash
pytest tests/test_alerts.py tests/test_priority_queue.py tests/test_metrics.py tests/test_executor.py -v
```
</details>
<details>
<summary><span style="font-size:1.25em"><strong>Week 5: Dry Run &amp; Micro-Arb Bot</strong></span></summary>

### Overview

Week 5 is about running the **full arbitrage stack in dry-run mode** with:

- a focused **Arbitrum ⇄ MEXC micro-arbitrage loop** (`DoubleLimitArbitrageEngine`),
- strict **safety limits** and **kill switch**,
- **structured logging**, **webhook alerts**, and an optional **Telegram bot**,
- and a repeatable **pre‑flight token verification** flow.

The system supports three execution modes: **observation** (default, no orders), **simulated execution** (full flow with fake fills), and **production/live execution** (real orders on MEXC and Arbitrum). Week 5 focuses on dry-run validation, but the codebase fully supports live trading when `--execute` is enabled.

---

### Execution Modes

The bot operates in three distinct modes, controlled by flags and environment variables:

#### 1. **Observation Mode** (Default)
- **What it does**: Evaluates opportunities, logs spreads and net PnL, but **never places orders**.
- **How to run**: `python scripts/arb_bot.py --mode mexc_v3` (no flags)
- **Use case**: Validate economics, test token universe, verify cost models.
- **Safety**: Zero risk — no orders sent to MEXC or Arbitrum.

#### 2. **Simulated Execution Mode**
- **What it does**: Exercises the full two-leg execution flow with **simulated fills/timeouts**. No real orders.
- **How to run**: `python scripts/arb_bot.py --mode mexc_v3 --simulate-execution`
- **Control**: Set `DOUBLE_LIMIT_SIM_SCENARIO=success|timeout|mex_reject|dex_failed` to test different outcomes.
- **Use case**: Test state machine, unwind logic, balance verification, execution reports.
- **Safety**: Zero risk — simulated fills only.

#### 3. **Production / Live Execution Mode** ⚠️
- **What it does**: Places **real orders** on MEXC (limit orders) and Arbitrum (ODOS or V3 direct swaps, chosen by route selection). Monitors fills, handles timeouts, unwinds if needed.
- **How to run**:
  ```bash
  python scripts/arb_bot.py --mode mexc_v3 --execute --trade-size 5.0
  ```
  Or set `ENABLE_LIVE_EXECUTION=1` in `.env`.
- **Production flag**: Set `PRODUCTION=true` in `.env` to enable production logging and stricter checks.
- **Use case**: Live arbitrage trading with real capital.
- **Safety**:
  - ⚠️ **REAL MONEY AT RISK** — orders execute on-chain and on MEXC.
  - Safety limits enforced (`ABSOLUTE_MAX_TRADE_USD=$25`, `ABSOLUTE_MAX_DAILY_LOSS=$20`, etc.).
  - Kill switch available (`/kill` via Telegram or `arb_bot_kill` file).
  - Circuit breaker trips on failures/drawdowns.
  - Pre-flight checks: balances, approvals, cost verification.

**Mode Detection**: The bot logs its mode at startup:
```
============================================================
  MODE: *** PRODUCTION *** / LIVE EXECUTION
============================================================
```
or
```
============================================================
  MODE: TESTNET / OBSERVATION
============================================================
```

---

### High-Level Architecture (Implemented)

#### Layer 1 — Infrastructure

Conceptually:

```
┌─────────────────────────────────────────┐
│  Arbitrum RPC (Alchemy or similar)     │
│  • Used by ChainClient + Uniswap V3    │
│  • Low-latency HTTPS RPC               │
└─────────────────┬──────────────────────┘
                  │
┌─────────────────▼──────────────────────┐
│  Wallet + Key Management               │
│  • WalletManager (signing, address)    │
│  • Capital limited to test size        │
└────────────────────────────────────────┘
```

In code:

- `ChainClient` (`src/chain/client.py`) uses `ARBITRUM_RPC_HTTPS`/`ARBITRUM_RPC_WSS` from `.env` for both ODOS swaps and V3 direct swaps.
- `WalletManager` (`src/core/wallet_manager.py`) holds the Arbitrum key for signing transactions (ODOS swaps or V3 direct swaps).
- For micro-arb dry run, **no real on-chain trades are sent**; the engine uses ODOS quotes, V3 route evaluation, and MEXC order books only.

#### Layer 2 — Monitoring & Detection

Instead of external price feeds, Week 5 uses **direct venue data** with **parallel DEX route monitoring**:

```
┌─────────────────────────────────────────┐
│  MEXC REST API (spot)                  │
│  • get_order_book() top-of-book L2     │
│  • 0% maker-fee limit orders           │
└─────────────────┬──────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌───────▼──────────────┐
│  ODOS          │  │  Uniswap V3 Direct   │
│  Aggregator    │  │  (parallel monitor)  │
│  • USDC↔token  │  │  • V3 pool address   │
│  • Gas est.    │  │  • Fee tier (0.05%/  │
│  • Route info  │  │    0.3%/1%)          │
│                │  │  • Historical gas    │
└────────────────┘  └──────────────────────┘
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │ RouteHealthTracker│
        │ • Gas history     │
        │ • Reliability     │
        │ • Route selection │
        └───────────────────┘
```

In code:

- `scripts/demo_double_limit.py` (MEXC + ODOS/V3 arb):
  - pulls MEXC order books via `MexcClient.get_order_book()` (`src/exchange/mexc_client.py`),
  - **fetches ODOS quotes in parallel** for both directions (buy/sell) via `OdosClient.quote()` (`src/pricing/odos_client.py`),
  - **evaluates V3 direct routes** in parallel (when `v3_pool` configured in token config),
  - `RouteHealthTracker` compares routes by:
    - Net profit (gross spread - fees - gas - bridge amortization),
    - Gas cost (ODOS estimate vs V3 historical average),
    - Route reliability (unreliable routes penalized),
  - **selects best route** (ODOS vs V3 direct) per token based on score,
  - loops every few seconds and logs spreads + **net** PnL + chosen route for a curated token universe from `config_tokens_arb_mex.py`.

#### Layer 3 — Execution Engine (MEXC + ODOS/V3 Arb)

Core micro-arb logic lives in `src/executor/double_limit_engine.py`:

```
┌─────────────────────────────────────────┐
│  DoubleLimitArbitrageEngine            │
│  • Evaluates CEX vs DEX prices          │
│  • Parallel route monitoring:           │
│    - ODOS aggregator quotes            │
│    - V3 direct pool evaluation        │
│  • Models LP fee + gas + bridge costs  │
│  • Chooses direction (mex→arb/arb→mex) │
│  • RouteHealthTracker: selects best     │
│    route (ODOS vs V3 direct) per token │
└─────────────────┬──────────────────────┘
                  │
        (Week 5: evaluation only by default)
```

- `evaluate_opportunity()`:
  - reads top-of-book from MEXC,
  - **fetches ODOS quotes in parallel** for both directions (USDC ↔ token),
  - **evaluates V3 direct route** in parallel (when `v3_pool` configured):
    - Uses V3 pool address and fee tier from token config,
    - Estimates gas from `RouteHealthTracker` historical data,
    - Computes net profit assuming direct V3 swap,
  - **compares routes**:
    - ODOS: uses quote gas estimate, includes ODOS fee (0.01%),
    - V3 direct: uses historical gas average, no aggregator fee,
    - `RouteHealthTracker` penalizes unreliable routes (avg gas > threshold),
    - **selects route with higher score** (net profit - gas - fees),
  - computes:
    - gross spread,
    - LP fee by tier (0.05% / 0.3% / 1%) + gas + amortized bridge cost via `CapitalManager`,
    - **net_profit_usd** and **net_profit_pct**,
    - whether the opportunity is `executable` under per-tier thresholds (`min_spread_by_tier`: 0.45% / 0.65% / 1.2%),
    - **chosen route** (`use_v3_direct` flag) stored in `DoubleLimitOpportunity`.
- `RouteHealthTracker` records gas usage per token/route after execution; unreliable routes (avg gas > threshold) are penalized in future evaluations.
- In Week‑5 **dry run** (default), `demo_double_limit` only calls `evaluate_opportunity` and logs:
  - `spread=…% net=$… (%) route=ODOS/V3 EXECUTABLE/SKIP` — **no orders are sent**.

The full `execute_double_limit()` path exists: **post-only MEXC limit** + **DEX swap** via chosen route:
- **ODOS**: `DexSwapManager` (quote → assemble → approve → send),
- **V3 direct**: Uniswap V3 `exactInputSingle` swap (when `use_v3_direct=True`).

#### Layer 4 — Capital Management

Capital and bridging policy are encapsulated in `src/core/capital_manager.py`:

```
┌─────────────────────────────────────────┐
│  CapitalManager                         │
│  • Tracks trade_count_since_bridge      │
│  • Amortizes bridge cost over trades    │
│  • Decides when to bridge ≥ $20 profit  │
└─────────────────────────────────────────┘
```

- `CapitalManagerConfig` sets:
  - `starting_cex_usd`, `starting_chain_usd`,
  - `bridge_threshold_usd` (e.g. $20),
  - `bridge_fixed_cost_usd` (e.g. $0.05 — actual MEXC withdrawal fee for USDT on Arbitrum One).
- The MEXC+V3 engine pulls an **amortized bridge cost per trade** from `get_effective_bridge_cost()`, so micro trades are only considered if they can dilute fixed costs.
- **Cost verification**: Run `python scripts/verify_all_costs.py` to verify gas costs, bridge fees, and LP fees match actual market conditions.

---

### Safety, Dry Run & Kill Switch

#### Absolute Safety Constants

`src/safety.py` defines **hard-coded, non-configurable** limits:

```python
ABSOLUTE_MAX_TRADE_USD = 25.0
ABSOLUTE_MAX_DAILY_LOSS = 20.0
ABSOLUTE_MIN_CAPITAL = 50.0
ABSOLUTE_MAX_TRADES_PER_HOUR = 30
```

and a final gate:

```python
def safety_check(trade_usd, daily_loss, total_capital, trades_this_hour) -> tuple[bool, str]:
    ...
```

`scripts/arb_bot.py` calls `safety_check(...)` **after all other checks** (spread, inventory, circuit breaker, etc.). If any limit is breached, the trade is blocked and an alert is sent.

#### Dry-Run Mode

`ArbBot` supports a strict dry-run flag:

```python
self.dry_run = config.get("dry_run", True)
...
if self.dry_run:
    logging.info(
        "DRY RUN | Would trade: %s %s size=%.4f spread=%.1fbps expected_pnl=$%.2f",
        pair,
        signal.direction.value,
        signal.size,
        signal.spread_bps,
        signal.expected_net_pnl,
    )
    return  # no Executor.execute()
```

For Week 5, you typically:

- use `--mode mexc_v3` (observation-only MEXC + ODOS/V3 arb; `double_limit` is a deprecated alias),
- or `--mode simulation`/`paper` with `dry_run=True` to exercise the full state machine without real orders.

#### Kill Switch (File + Telegram)

Kill switch is a **shared file** in your OS temp directory:

```python
from pathlib import Path
import tempfile

KILL_SWITCH_FILE = str(Path(tempfile.gettempdir()) / "arb_bot_kill")
```

All main loops (`ArbBot.run`, `demo_double_limit.main`) do:

```python
if is_kill_switch_active():
    # pause: no new trades, but process + Telegram bot stay alive
```

The **Telegram bot** (`src/telegram_bot.py`) gives you remote control:

- `/kill` → creates `arb_bot_kill` → bots pause.
- `/resume` → deletes `arb_bot_kill` → bots resume.
- `/status` → replies with current kill-switch status.

Environment variables:

```env
TELEGRAM_BOT_TOKEN=123456:ABC...   # from BotFather
TELEGRAM_CHAT_ID=617012126        # your chat ID
TELEGRAM_POLL_SEC=1.0             # optional, default 2.0
```

The entrypoint (`scripts/arb_bot.py`) starts a shared `TelegramBot` for all modes and stops it on shutdown.

---

### Monitoring & Alerts

**Structured logging** (files + stdout):

- `scripts/arb_bot.py` → `logs/bot_YYYYMMDD.log`
- `scripts/demo_double_limit.py` → `logs/double_limit_YYYYMMDD.log`

In **production mode** (`PRODUCTION=true` + `--execute`), logs include:
- Real order IDs from MEXC
- On-chain transaction hashes from ODOS swaps
- Actual fill prices and amounts
- Unwind transactions (if one leg fills and the other times out)
- Balance verification after trades

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s |%(levelname)s |%(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
```

**Webhook alerts** (`src/executor/alerts.py`) are already integrated:

- Circuit breaker trips / resets.
- Execution failures + unwind status.
- Drawdown alerts.

`ArbBot` also sends custom alerts for:

- Bot start (`mode`, `dry_run`, `PRODUCTION` flag),
- Kill switch activated / cleared,
- Trade completed with realized net PnL (in production mode: includes real order IDs and tx hashes),
- Absolute safety limit violations (e.g. daily loss),
- Production mode: execution reports with MEXC order status and Arbitrum tx receipts.

To wire alerts to Slack/Telegram/etc., point `WEBHOOK_URLS` to your bridge endpoint in `.env`.

---

### How to Run Week 5 Dry Run

#### 1. Configure `.env`

Minimum required for MEXC + ODOS/V3 dry run:

```env
# Arbitrum + ODOS + MEXC
ALCHEMY_API_KEY=...
ARBITRUM_RPC_HTTPS=https://arb-mainnet.g.alchemy.com/v2/...
ARBITRUM_WALLET_ADDRESS=0x...
USDC_ADDRESS=0xaf88d065e77c8cC2239327C5EDb3A432268e5831

MEXC_API_KEY=...
MEXC_API_SECRET=...
MEXC_BASE_URL=https://api.mexc.com

# ODOS has no API key, just base URL in OdosClient

# MEXC + ODOS arb parameters
TRADE_SIZE_USD=5.0
MIN_SPREAD_PCT=0.004
MIN_PROFIT_USD=0.001
MAX_SLIPPAGE_PCT=0.5

# Optional: Telegram control
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=617012126
```

#### 2. Verify costs and token universe

**Cost verification** (recommended first):

```bash
python scripts/verify_all_costs.py
```

This verifies:
- Gas costs on Arbitrum (compares configured vs. actual),
- Bridge fees from MEXC (fetches actual withdrawal fees),
- LP fees by token (from configuration),
- Total cost estimation per trade.

**Token verification**:

```bash
python scripts/verify_tokens.py
```

This prints a **TOKEN VERIFICATION REPORT** and a JSON **CONFIG PATCH FOR FAILED TOKENS**. Week 5 already applied the patch for known failures (AAVE, CVX, FXS, YFI, LDO, etc.).

**Note**: Ensure your MEXC API key has:
- Correct `MEXC_API_KEY` and `MEXC_API_SECRET` in `.env` (whitespace is automatically stripped),
- "Withdrawal" permission enabled (not just "View deposit/withdrawal details"),
- IP address matches bound IP (if IP binding is enabled).

#### 3. Start MEXC + ODOS/V3 dry run

Use the unified entrypoint:

```bash
# Default trade size ($5.0 from TRADE_SIZE_USD env var or config)
python scripts/arb_bot.py --mode mexc_v3

# Custom trade size ($10.0)
python scripts/arb_bot.py --mode mexc_v3 --trade-size 10.0

# Limit to specific tokens
python scripts/arb_bot.py --mode mexc_v3 --tokens ARB,GMX,LINK
```

**CLI options**:
- `--trade-size 5.0` — $5 USD per trade (default; good for micro-arb testing)
- `--tokens ARB,GMX,LINK` — comma-separated symbols to track (default: all active tokens)
- `--max-trades N` — stop after N successful trades (e.g. 1 for a single transaction)

What this does:

- Starts structured logging and metrics.
- Starts the Telegram bot (if configured).
- Delegates to `scripts/demo_double_limit.py` (MEXC + ODOS/V3 arb):
  - Calls `DoubleLimitArbitrageEngine.evaluate_opportunity()` for each token.
  - Logs:
    - MEXC bid/ask,
    - ODOS price,
    - gross spread,
    - **net** profit after fees + gas + bridge amortization,
    - whether the opportunity is `EXECUTABLE` under your thresholds.
- **No real orders** are sent in this mode.

Let this run for **≥30 minutes** and keep `logs/double_limit_YYYYMMDD.log` as your Week‑5 dry‑run artifact.

**Simulate execution** (two legs, no real orders):

```bash
python scripts/arb_bot.py --mode mexc_v3 --simulate-execution --trade-size 5.0
```

This exercises the full `execute_double_limit()` flow with simulated fills/timeouts (controlled by `DOUBLE_LIMIT_SIM_SCENARIO=success|timeout|mex_reject|dex_failed`).

#### 3b. Production Mode — Real Trades ⚠️

**⚠️ WARNING: This mode places REAL orders and uses REAL capital. Only enable after completing dry-run validation and pre-flight checks.**

To place **real orders** (MEXC limit + DEX swap on Arbitrum, route chosen automatically: ODOS or V3 direct) in production mode:

1. **Environment & Production Flag**
   - Ensure `.env` has: `MEXC_API_KEY`, `MEXC_API_SECRET`, `ARBITRUM_RPC_HTTPS`, `PRIVATE_KEY`, `USDC_ADDRESS`, `ARBITRUM_WALLET_ADDRESS`.
   - Set `PRODUCTION=true` in `.env` for production logging and stricter checks.
   - Use `--execute` when running the bot (or `ENABLE_LIVE_EXECUTION=1` in `.env`).
   - **Verify**: Bot startup logs should show `MODE: *** PRODUCTION *** / LIVE EXECUTION`.

2. **Balances**
   - **MEXC**: Enough USDT and the token you trade (e.g. ARB) so a post-only limit order can be placed for `TRADE_SIZE_USD`.
   - **Arbitrum**: Enough USDC in `ARBITRUM_WALLET_ADDRESS` for the DEX leg (~`TRADE_SIZE_USD`), plus some ETH for gas.

3. **Approvals**
   - The first ODOS swap will request token approvals (USDC and the quote token) for the swap router. Ensure the wallet can sign those transactions.

4. **Cost verification (recommended)**
   ```bash
   python scripts/verify_all_costs.py
   python scripts/verify_tokens.py
   ```

5. **Run with live execution**
   ```bash
   # CLI flag (preferred)
   python scripts/arb_bot.py --mode mexc_v3 --execute --trade-size 5.0

   # Or env var
   set ENABLE_LIVE_EXECUTION=1
   python scripts/arb_bot.py --mode mexc_v3 --trade-size 5.0
   ```
   When an **EXECUTABLE** opportunity appears, the bot will:
   - Place **real MEXC limit order** (post-only, maker fee 0%).
   - Execute **real DEX swap** on Arbitrum using the **chosen route**:
     - **ODOS**: Aggregator swap via `DexSwapManager` (quote → assemble → approve → send),
     - **V3 direct**: Direct Uniswap V3 `exactInputSingle` swap (when route selection favors V3).
   - Monitor both legs until fills or timeout (~10 min).
   - **Unwind** if one leg fills and the other times out (reverse the filled leg to flatten position).
   - Log execution details (route used, order IDs, tx hashes) to `logs/double_limit_YYYYMMDD.log`.

6. **Safety & Monitoring**
   - ⚠️ **Start small**: Use `TRADE_SIZE_USD=5.0` for initial production runs.
   - **Kill switch**: `arb_bot_kill` file in temp dir (or `/kill` via Telegram) pauses new trades immediately.
   - **Circuit breaker**: Trips after 3 failures or $50 drawdown; logs alert.
   - **Absolute limits**: Hard-coded in `src/safety.py` — max trade $25, max daily loss $20, min capital $50.
   - **Monitor logs**: Watch `logs/double_limit_YYYYMMDD.log` for execution status, fills, unwinds.
   - **Telegram alerts**: Execution reports, circuit breaker trips, safety violations sent automatically.
   - **Metrics**: Prometheus `/metrics` endpoint (port 9090) tracks PnL, execution counts, queue depth.

#### 4. Optional: CEX/DEX arb bot dry run

To exercise the Week‑4 CEX/DEX engine with all safety gates:

```bash
python scripts/arb_bot.py --mode simulation    # or --mode paper
```

With `dry_run=True` (default), you’ll see:

```text
DRY RUN | Would trade: ETH/USDT buy_cex_sell_dex size=... spread=...bps expected_pnl=$...
```

covering:

- signal generation,
- scoring + priority queue,
- executor state machine and recovery (simulated),
- safety_check(),
- circuit breaker + webhook/Telegram alerts.

---

### Conceptual vs. Internship Baseline

Compared to a “basic” internship-style arb loop (single Uniswap V2 swap + MEXC/Binance market order with taker fees and public RPC), the Week‑5 framework pushes toward a **professional micro‑arbitrage architecture**:

- **Zero taker fees** on the CEX path (post‑only MEXC limit orders when real execution is enabled).
- **Parallel DEX route monitoring**: Evaluates **ODOS aggregator** and **Uniswap V3 direct** routes simultaneously; selects best route per token based on net profit, gas cost, and reliability (`RouteHealthTracker`).
- **Amortized fixed costs** via `CapitalManager` and bridge thresholding (bridge fee: $0.05 USDT per withdrawal, amortized over 5+ trades).
- **Strict safety limits** and kill switch, integrated at the execution boundary.
- **Observability**: structured logs, metrics, webhooks, and Telegram control.
- **Cost verification**: Automated scripts verify gas costs, bridge fees, and LP fees match actual market conditions.

**Recent updates**:
- **Parallel route monitoring**: System evaluates **ODOS aggregator** and **Uniswap V3 direct** routes simultaneously; `RouteHealthTracker` selects the best route per token based on net profit, gas cost, and reliability.
- DEX execution uses chosen route: **ODOS swap** via `DexSwapManager` (quote → assemble → approve → send) or **V3 direct** via `exactInputSingle` when `use_v3_direct=True`.
- `RouteHealthTracker` records gas per token/route after execution; unreliable routes (avg gas > threshold) are penalized in future evaluations.
- Per-tier min spread: 0.45% (500), 0.65% (3000), 1.2% (10000) for $5–10 micro-arb.
- `--simulate-execution` flag exercises full two-leg flow without real orders (`DOUBLE_LIMIT_SIM_SCENARIO`).
- `--tokens` and `--max-trades` CLI options for focused runs.
- Fixed MEXC API signature generation; bridge cost $0.05 (MEXC withdrawal fee); network matching for "Arbitrum One(ARB)".

The codebase supports **full production trading** with `--execute` flag: real MEXC limit orders and DEX swaps (ODOS or V3 direct, chosen by route selection) execute on-chain. The observation and simulated execution modes allow you to validate economics and safety before enabling live trading.

</details>

---

<details>
<summary><span style="font-size:1.25em"><strong>Conclusion</strong></span></summary>

After five weeks of development, this project has evolved from basic wallet management to a production-ready micro-arbitrage trading system. The journey covered secure key management, on-chain transaction building, AMM pricing and routing, CEX integration, strategy execution, and finally a focused Arbitrum ⇄ MEXC micro-arbitrage engine with parallel route monitoring.

### Documentation

For detailed analysis and reflections on the project:

- **Final Report (Part 6)**: See `docs/final_report_part6.docx` for comprehensive technical documentation, architecture decisions, and production deployment considerations.
- **Trading Journal (Days 1-3)**: See `docs/trading_journal_days1-3.docx` for real-world trading observations, PnL analysis, and operational learnings from live production runs.

### What I Learned

Through building this system, several key insights emerged:

- **Safety First**: Hard-coded absolute limits (`ABSOLUTE_MAX_TRADE_USD`, `ABSOLUTE_MAX_DAILY_LOSS`) and kill switches are essential for production trading. No amount of testing replaces having emergency stops.

- **Route Optimization Matters**: Parallel monitoring of ODOS aggregator and V3 direct routes, combined with `RouteHealthTracker`, can significantly improve execution quality. Gas costs vary dramatically, and historical tracking helps avoid unreliable routes.

- **Dry-Run Validation is Critical**: The observation and simulated execution modes caught numerous edge cases before real capital was at risk. Cost verification scripts (`verify_all_costs.py`, `verify_tokens.py`) are invaluable for catching configuration errors.

- **Micro-Arbitrage Economics**: For small trade sizes ($5–10), fixed costs (bridge fees, gas) must be amortized carefully. The `CapitalManager` bridge thresholding model prevents unprofitable trades that don't cover fixed costs.

- **Production Observability**: Structured logging, Prometheus metrics, Telegram alerts, and execution reports are not optional for production trading. When real money is at stake, visibility into every execution is essential.

- **Incremental Development**: Building week-by-week (wallet → chain → pricing → exchange → execution → production) allowed each layer to be tested independently before integration. This modular approach made debugging much easier.

The codebase now supports full production trading with real orders, but the safety mechanisms, dry-run modes, and comprehensive logging ensure that every execution is monitored, every failure is analyzed, and every improvement is data-driven.

</details>
