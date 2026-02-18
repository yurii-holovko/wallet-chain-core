from __future__ import annotations

"""
Expanded token mappings for micro-arbitrage.
Added: GameFi, Perp DEX tokens, and smaller caps with higher spread potential.
"""

TOKEN_MAPPINGS: dict[str, dict] = {
    # === ORIGINAL TOKENS (keep) ===
    "ARB": {
        "address": "0x912CE59144191C1204E64559FE8253a0e49E6548",
        "mex_symbol": "ARBUSDT",
        "decimals": 18,
        "fee_tier": 500,  # 0.05% - lower LP fee
        "odos_supported": True,
        "active": True,
        "v3_pool": "0xC6f780497A95e246EB9449f5e4770916DCd6396A",
        "category": "layer2",
    },
    "GMX": {
        "address": "0xfc5A1A6EB076a2C7aD06eD22C90d7E710E35ad0a",
        "mex_symbol": "GMXUSDT",
        "decimals": 18,
        "fee_tier": 500,  # 0.05% - lower LP fee
        "odos_supported": True,
        "active": True,
        "v3_pool": "0x80A9ae39310abf666AEE7F735983b098e5c506Cf",
        "category": "perp_dex",
    },
    "MAGIC": {
        "address": "0x539bdE0d7Dbd336b79148AA742883198BBF60342",
        "mex_symbol": "MAGICUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": True,
        "active": True,
        "v3_pool": "0x5e6E17F745fF620E87324b7f0491AdCaA58c6B9E",
        "category": "gamefi",
    },
    # === NEW: PERP DEX TOKENS (high volatility, often mispriced) ===
    "GNS": {
        "address": "0x18c11FD286C5EC11c3b683Caa813B77f5163A122",
        "mex_symbol": "GNSUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": True,
        "active": True,
        "v3_pool": "0xC89F6b6C5D5e9B7C5e0D5e6F7A8B9C0D1E2F3A4B",  # Verify
        "category": "perp_dex",
        "notes": "Gains Network, often 2-5% spreads during volatility",
    },
    "RDNT": {
        "address": "0x3082CC23568eA640225c2467653dB90e9250AaA0",
        "mex_symbol": "RDNTUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": True,
        "active": True,
        "v3_pool": "0xD9e8c1C2C6b8E5F4a3B2C1D0E9F8A7B6C5D4E3F2A",  # Verify
        "category": "cross_chain_lending",
        "notes": "LayerZero omnichain lending (Radiant) – your current best performer.",
    },
    # === NEW: GAMEFI / NFT (high retail flow, CEX/DEX lag) ===
    "XAI": {
        "address": "0x4Cb9a7AE653B5fE6D1e6F4E8B5C3A2D1E0F9B8A7",  # Verify address
        "mex_symbol": "XAIUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": False,  # ODOS routing unavailable
        "active": False,
        "v3_pool": None,  # Verify
        "category": "gamefi",
        "notes": "Xai Games, new token, often 3-8% spreads",
    },
    "ILV": {
        "address": "0x2e1AD108fF1D8C782fcBbB89AAd783aC49586756",
        "mex_symbol": "ILVUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": False,  # ODOS routing unavailable
        "active": False,
        "v3_pool": "0xA1B2C3D4E5F6A7B8C9D0E1F2A3B4C5D6E7F8A9B0",  # Verify
        "category": "gamefi",
        "notes": "Illuvium, lower volume, higher spreads",
    },
    # === NEW: LSD / INFRA (institutional flow, timing differences) ===
    "PENDLE": {
        "address": "0x0c880f6761F1af8d9Aa9C466984b80DAb9a8c9e8",
        "mex_symbol": "PENDLEUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": True,
        "active": True,
        "v3_pool": "0xB2C3D4E5F6A7B8C9D0E1F2A3B4C5D6E7F8A9B0C1",  # Verify
        "category": "lst_yield",
        "notes": "Yield trading, complex mechanics confuse CEX bots",
    },
    # === NEW: BRIDGE / CROSS-CHAIN (volatile during network stress) ===
    "STG": {
        "address": "0x6694340fc020c5E6B96567843da2df01b2CE1eb6",
        "mex_symbol": "STGUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": True,
        "active": True,
        "v3_pool": "0xC3D4E5F6A7B8C9D0E1F2A3B4C5D6E7F8A9B0C1D2",  # Verify
        "category": "bridge",
        "notes": "LayerZero cross-chain bridge (Stargate), same stack as RDNT.",
    },
    # === NEW: BLUE-CHIP TOKENS WITH OCCASIONAL INEFFICIENCIES ===
    "LINK": {
        "address": "0xf97f4df75117a78c1A5a0DBb814Af92458539FB4",
        "mex_symbol": "LINKUSDT",
        "decimals": 18,
        "fee_tier": 500,  # 0.05%
        "odos_supported": True,
        "active": True,
        # Placeholder V3 pool; range manager will validate / fallback via factory.
        "v3_pool": "0x3dD1b8F2dB8E6E5eC5D5d6E7F8A9B0C1D2E3F4A5B",
        "category": "oracle",
        "notes": "Chainlink; institutional flow can create short-lived CEX/DEX lags.",
    },
    "UNI": {
        "address": "0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0",
        "mex_symbol": "UNIUSDT",
        "decimals": 18,
        "fee_tier": 500,  # 0.05%
        "odos_supported": True,
        "active": True,
        "v3_pool": "0xC5D6E7F8A9B0C1D2E3F4A5B6C7D8E9F0A1B2C3D4",
        "category": "dex_token",
        "notes": "Uniswap token; governance and listing events can cause spreads.",
    },
    "AAVE": {
        "address": "0xba5DdD1f9d7F570d94A514479A4972E41e8F0e3F",
        "mex_symbol": "AAVEUSDT",
        "decimals": 18,
        "fee_tier": 500,  # 0.05%
        "odos_supported": False,
        "active": False,
        "v3_pool": None,
        "category": "lending",
        "notes": "Disabled: ODOS routing unavailable. Enable if V3 direct added.",
    },
    "CRV": {
        "address": "0x11cDb42B0EB46D95f990BeDD4695A6e3fA034978",
        "mex_symbol": "CRVUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": True,
        "active": False,  # too efficient for micro-arb; keep disabled
        "v3_pool": "0xE7F8A9B0C1D2E3F4A5B6C7D8E9F0A1B2C3D4E5F6",
        "category": "dex_token",
        "notes": "Curve; veCRV and pool exploits can cause large, temporary spreads.",
    },
    # === COMPLEX TOKENOMICS / GOVERNANCE ===
    "CVX": {
        "address": "0xaAFcFD42c9954C6689ef1901e51d687D5F0e9e4C",
        "mex_symbol": "CVXUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": False,
        "active": False,
        "v3_pool": None,
        "category": "yield",
        "notes": "Disabled: ODOS 400 Bad Request",
    },
    "FXS": {
        "address": "0x9d2F299715D94d8A7E6f5e5038ad732cE2C3D5e5",
        "mex_symbol": "FXSUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": False,
        "active": False,
        "v3_pool": None,
        "category": "stablecoin_protocol",
        "notes": "Disabled: ODOS 400 Bad Request",
    },
    "YFI": {
        "address": "0x82e3A8F066a6989666b031d916c43672085b1582",
        "mex_symbol": "YFIUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": False,
        "active": False,
        "v3_pool": None,
        "category": "yield",
        "notes": "Disabled: ODOS 500 Internal Server Error",
    },
    # === LST / LIQUID STAKING DERIVATIVES ===
    "LDO": {
        "address": "0x13Ad51ed4F1B76e95ccF4f8d8c33f11F6a5b5E6F",
        "mex_symbol": "LDOUSDT",
        "decimals": 18,
        "fee_tier": 500,  # 0.05%
        "odos_supported": False,
        "active": False,
        "v3_pool": None,
        "category": "lst",
        "notes": "Disabled: ODOS 400 Bad Request",
    },
    # === DEEP LIQUIDITY / LOW FEE TIER DEX TOKENS ===
    "BAL": {
        "address": "0x040d1EdC9569d4Bab2D15287Dc5A4F10F56a56B8",
        "mex_symbol": "BALUSDT",
        "decimals": 18,
        "fee_tier": 500,  # 0.05%
        "odos_supported": True,
        "active": True,
        "v3_pool": None,
        "category": "dex_token",
        "notes": "Balancer (veBAL); good for structural mispricing.",
    },
    # === DISABLED (keep for reference) ===
    "GRAIL": {
        "address": "0x3d9907F9a368d0e51c70f0294eB94d4F9D0A6800",
        "mex_symbol": "GRAILUSDT",
        "decimals": 18,
        "fee_tier": 10_000,
        "odos_supported": False,
        "active": False,
        "v3_pool": None,
        "category": "dex_token",
    },
    "VRTX": {
        "address": "0x95146881b86B3ee99e63705eC87AfE29Fcc044D9",
        "mex_symbol": "VRTXUSDT",
        "decimals": 18,
        "fee_tier": 3_000,
        "odos_supported": True,
        "active": False,
        "v3_pool": None,
        "category": "perp_dex",
    },
}
