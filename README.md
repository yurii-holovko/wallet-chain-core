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
