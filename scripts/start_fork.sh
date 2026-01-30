#!/bin/bash
# scripts/start_fork.sh

# Requires: anvil (from foundry)
# Install: curl -L https://foundry.paradigm.xyz | bash && foundryup

anvil \
    --fork-url "$ETH_RPC_URL" \
    --port 8545 \
    --accounts 10 \
    --balance 10000
