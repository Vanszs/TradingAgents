"""DeFiLlama API client for TVL and protocol data.

Protocol slugs are auto-resolved from DeFiLlama's /protocols endpoint,
so any DeFi protocol is supported without hardcoding.
Free API, no key required.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from .crypto_id_map import ticker_to_coingecko_id

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 600          # 10 minutes for protocol data
_PROTOCOLS_TTL = 3600     # 1 hour for full protocols list

# Fast-path: known slugs
_TICKER_TO_SLUG: dict[str, str] = {
    "UNI-USD": "uniswap", "AAVE-USD": "aave", "MKR-USD": "makerdao",
    "CRV-USD": "curve-dex", "COMP-USD": "compound-finance",
    "SNX-USD": "synthetix", "YFI-USD": "yearn-finance",
    "SUSHI-USD": "sushiswap", "BAL-USD": "balancer",
    "LDO-USD": "lido", "RPL-USD": "rocket-pool",
    "GMX-USD": "gmx", "DYDX-USD": "dydx", "INJ-USD": "injective",
    "ARB-USD": "arbitrum-one", "OP-USD": "optimism-bridge",
    "PENDLE-USD": "pendle", "JOE-USD": "trader-joe",
    "CAKE-USD": "pancakeswap", "1INCH-USD": "1inch-network",
}

# Native L1 / non-DeFi tokens — TVL not applicable
_NON_DEFI_BASE = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT",
                  "LTC", "AVAX", "ATOM", "XLM", "TRX", "TON", "SHIB", "PEPE",
                  "NEAR", "APT", "SUI", "FIL", "ALGO", "MATIC", "LINK"}

# Lazy-loaded protocols list cache
_protocols_cache: list[dict] = []
_protocols_cache_ts: float = 0.0


def _load_protocols() -> list[dict]:
    """Lazy-load and cache DeFiLlama full protocols list."""
    global _protocols_cache, _protocols_cache_ts
    now = time.time()
    if _protocols_cache and (now - _protocols_cache_ts) < _PROTOCOLS_TTL:
        return _protocols_cache
    try:
        resp = requests.get("https://api.llama.fi/protocols", timeout=20)
        resp.raise_for_status()
        _protocols_cache = resp.json()
        _protocols_cache_ts = now
        logger.debug("DeFiLlama: loaded %d protocols", len(_protocols_cache))
    except Exception as exc:
        logger.warning("DeFiLlama /protocols fetch failed: %s", exc)
    return _protocols_cache


def _resolve_defillama_slug(ticker: str) -> Optional[str]:
    """Return DeFiLlama slug for a ticker.

    Fast path: hardcoded map.
    Slow path: search DeFiLlama /protocols by symbol and name.
    Result is cached in _TICKER_TO_SLUG for future calls.
    """
    normalized = ticker.strip().upper()

    # Fast path
    if normalized in _TICKER_TO_SLUG:
        return _TICKER_TO_SLUG[normalized]

    base = normalized.split("-")[0]

    # Non-DeFi check
    if base in _NON_DEFI_BASE:
        return None

    # Get coin name from CoinGecko for better matching
    coin_name: Optional[str] = None
    coin_id = ticker_to_coingecko_id(ticker)
    if coin_id:
        try:
            from .coingecko import _get
            data = _get(f"/coins/{coin_id}", params={
                "localization": "false", "tickers": "false",
                "market_data": "false", "community_data": "false",
                "developer_data": "false",
            })
            if data:
                raw_name = data.get("name", "")
                # Strip parenthetical chain suffixes like "® (ETHEREUM)" or "(BSC)"
                import re as _re
                coin_name = _re.sub(r'\s*[\(\[].*?[\)\]]', '', raw_name).strip().lower()
        except Exception:
            pass

    # Search protocols list
    protocols = _load_protocols()
    if not protocols:
        return None

    base_lower = base.lower()
    name_lower = coin_name or base_lower

    # Pass 1: exact symbol match
    for p in protocols:
        if p.get("symbol", "").upper() == base:
            slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
            _TICKER_TO_SLUG[normalized] = slug
            return slug

    # Pass 2: coin name contains protocol name (protocol name must be >= 5 chars to avoid false positives)
    for p in protocols:
        p_name = p.get("name", "").lower()
        if len(p_name) >= 5 and p_name in name_lower:
            slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
            _TICKER_TO_SLUG[normalized] = slug
            return slug

    # Pass 3: protocol name contains coin base symbol (base must be >= 4 chars, word boundary)
    if len(base) >= 4:
        for p in protocols:
            p_name = p.get("name", "").lower()
            p_sym = p.get("symbol", "").upper()
            # Only match on base symbol, not on coin name words (avoids "ethereum" false positive)
            if p_sym == base or p_name == base_lower:
                slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
                _TICKER_TO_SLUG[normalized] = slug
                return slug

    return None


def _fetch_protocol(slug: str) -> Optional[dict]:
    url = f"https://api.llama.fi/protocol/{slug}"
    now = time.time()
    if url in _CACHE:
        ts, data = _CACHE[url]
        if now - ts < _CACHE_TTL:
            return data
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        _CACHE[url] = (now, data)
        return data
    except Exception as exc:
        logger.warning("DeFiLlama request failed for %s: %s", slug, exc)
        return None


def _fmt_usd(n: float | None) -> str:
    if n is None:
        return "N/A"
    if abs(n) >= 1e9:
        return f"${n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:.2f}M"
    return f"${n:,.0f}"


def _pct_change(current: float, previous: float) -> str:
    if not previous:
        return "N/A"
    return f"{(current - previous) / previous * 100:+.1f}%"


def get_tvl(ticker: str) -> str:
    """Return formatted TVL/protocol data string for a ticker."""
    base = ticker.strip().upper().split("-")[0]

    if base in _NON_DEFI_BASE:
        return (
            f"## TVL Data: {ticker}\n\n"
            f"TVL not applicable for {base}. "
            "It is a Layer-1/currency token, not a DeFi protocol. "
            "TVL metrics apply to protocols that hold user deposits "
            "(DEXs, lending, liquid staking, yield aggregators, etc.)."
        )

    slug = _resolve_defillama_slug(ticker)
    if not slug:
        return (
            f"## TVL Data: {ticker}\n\n"
            f"No DeFiLlama protocol found for {ticker}. "
            "This token may not be a DeFi protocol or is not yet indexed by DeFiLlama."
        )

    data = _fetch_protocol(slug)
    if not data:
        return f"## TVL Data: {ticker}\n\nDeFiLlama API error for protocol '{slug}'."

    current_tvl_by_chain = data.get("currentChainTvls", {})
    total_tvl = sum(
        v for k, v in current_tvl_by_chain.items()
        if not k.endswith("-borrowed") and not k.endswith("-staking")
        and isinstance(v, (int, float))
    )

    tvl_history = data.get("tvl", [])
    tvl_7d_ago = tvl_history[-8]["totalLiquidityUSD"] if len(tvl_history) > 8 else None
    tvl_30d_ago = tvl_history[-31]["totalLiquidityUSD"] if len(tvl_history) > 31 else None

    category = data.get("category", "N/A")
    chains = data.get("chains", [])
    chains_str = ", ".join(chains[:10]) + (f" (+{len(chains)-10} more)" if len(chains) > 10 else "")

    lines = [
        f"## TVL Data: {ticker} (DeFiLlama: {slug})",
        "",
        f"**Current TVL**: {_fmt_usd(total_tvl)}",
        f"- 7d TVL Change: {_pct_change(total_tvl, tvl_7d_ago) if tvl_7d_ago else 'N/A'}",
        f"- 30d TVL Change: {_pct_change(total_tvl, tvl_30d_ago) if tvl_30d_ago else 'N/A'}",
        f"- Category: {category}",
        f"- Chains: {chains_str}" if chains else "- Chains: N/A",
    ]

    sorted_chains = sorted(
        [(k, v) for k, v in current_tvl_by_chain.items()
         if not k.endswith("-borrowed") and not k.endswith("-staking")
         and isinstance(v, (int, float))],
        key=lambda x: x[1], reverse=True
    )[:5]
    if sorted_chains:
        lines += ["", "**TVL by Chain (top 5)**:"]
        for chain, tvl in sorted_chains:
            lines.append(f"- {chain}: {_fmt_usd(tvl)}")

    return "\n".join(lines)
