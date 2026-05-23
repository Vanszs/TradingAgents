"""Maps ticker symbols to CoinGecko coin IDs.

Fast path: hardcoded top-50 map.
Slow path: CoinGecko /coins/list endpoint (cached).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Hardcoded top-50 map (fast path, no API call needed)
_TICKER_TO_ID: dict[str, str] = {
    "BTC-USD": "bitcoin", "BTC-USDT": "bitcoin", "BTC-USDC": "bitcoin",
    "ETH-USD": "ethereum", "ETH-USDT": "ethereum", "ETH-USDC": "ethereum",
    "BNB-USD": "binancecoin", "BNB-USDT": "binancecoin",
    "SOL-USD": "solana", "SOL-USDT": "solana",
    "XRP-USD": "ripple", "XRP-USDT": "ripple",
    "ADA-USD": "cardano", "ADA-USDT": "cardano",
    "AVAX-USD": "avalanche-2", "AVAX-USDT": "avalanche-2",
    "DOGE-USD": "dogecoin", "DOGE-USDT": "dogecoin",
    "DOT-USD": "polkadot", "DOT-USDT": "polkadot",
    "MATIC-USD": "matic-network", "MATIC-USDT": "matic-network",
    "LINK-USD": "chainlink", "LINK-USDT": "chainlink",
    "LTC-USD": "litecoin", "LTC-USDT": "litecoin",
    "UNI-USD": "uniswap", "UNI-USDT": "uniswap",
    "ATOM-USD": "cosmos", "ATOM-USDT": "cosmos",
    "XLM-USD": "stellar", "XLM-USDT": "stellar",
    "ALGO-USD": "algorand", "ALGO-USDT": "algorand",
    "FIL-USD": "filecoin", "FIL-USDT": "filecoin",
    "NEAR-USD": "near", "NEAR-USDT": "near",
    "ARB-USD": "arbitrum", "ARB-USDT": "arbitrum",
    "OP-USD": "optimism", "OP-USDT": "optimism",
    "SUI-USD": "sui", "SUI-USDT": "sui",
    "APT-USD": "aptos", "APT-USDT": "aptos",
    "INJ-USD": "injective-protocol", "INJ-USDT": "injective-protocol",
    "TRX-USD": "tron", "TRX-USDT": "tron",
    "TON-USD": "the-open-network", "TON-USDT": "the-open-network",
    "SHIB-USD": "shiba-inu", "SHIB-USDT": "shiba-inu",
    "PEPE-USD": "pepe", "PEPE-USDT": "pepe",
    # Meme coins
    "CHILLGUY-USD": "just-a-chill-guy", "CHILLGUY-USDT": "just-a-chill-guy",
    "BONK-USD": "bonk", "BONK-USDT": "bonk",
    "WIF-USD": "dogwifcoin", "WIF-USDT": "dogwifcoin",
    "FLOKI-USD": "floki", "FLOKI-USDT": "floki",
    "MEME-USD": "memecoin-2", "MEME-USDT": "memecoin-2",
    "POPCAT-USD": "popcat", "POPCAT-USDT": "popcat",
    "MOG-USD": "mog-coin", "MOG-USDT": "mog-coin",
    "BRETT-USD": "based-brett", "BRETT-USDT": "based-brett",
    "TURBO-USD": "turbo", "TURBO-USDT": "turbo",
    "NEIRO-USD": "neiro-on-eth", "NEIRO-USDT": "neiro-on-eth",
}

_list_cache: dict[str, str] = {}
_list_cache_ts: float = 0.0
_LIST_CACHE_TTL = 3600  # 1 hour


def ticker_to_coingecko_id(ticker: str) -> Optional[str]:
    """Return CoinGecko coin ID for a ticker symbol, or None if not found."""
    if not ticker:
        return None
    normalized = ticker.strip().upper()
    # Fast path
    if normalized in _TICKER_TO_ID:
        return _TICKER_TO_ID[normalized]
    # Try base symbol (strip quote currency)
    base = normalized.split("-")[0]
    for key, cg_id in _TICKER_TO_ID.items():
        if key.startswith(base + "-"):
            return cg_id
    # Slow path: fetch CoinGecko coin list
    return _lookup_from_list(base)


def _lookup_from_list(symbol: str) -> Optional[str]:
    global _list_cache, _list_cache_ts
    now = time.time()
    if not _list_cache or (now - _list_cache_ts) > _LIST_CACHE_TTL:
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/list",
                timeout=10,
                headers={"accept": "application/json"},
            )
            resp.raise_for_status()
            coins = resp.json()
            _list_cache = {}
            for c in coins:
                sym = c["symbol"].upper()
                # Keep first occurrence (CoinGecko returns by market cap rank)
                if sym not in _list_cache:
                    _list_cache[sym] = c["id"]
            _list_cache_ts = now
        except Exception as exc:
            logger.warning("CoinGecko /coins/list failed: %s", exc)
            return None
    return _list_cache.get(symbol.upper())
