"""DeFiLlama API client for TVL and protocol data.

Free API, no key required. Docs: https://defillama.com/docs/api
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from .crypto_id_map import ticker_to_coingecko_id

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 600  # 10 minutes

_TICKER_TO_SLUG: dict[str, str] = {
    "UNI-USD": "uniswap", "AAVE-USD": "aave", "MKR-USD": "makerdao",
    "CRV-USD": "curve-dex", "COMP-USD": "compound-finance",
    "SNX-USD": "synthetix", "YFI-USD": "yearn-finance",
    "SUSHI-USD": "sushiswap", "BAL-USD": "balancer",
    "LDO-USD": "lido", "RPL-USD": "rocket-pool",
    "GMX-USD": "gmx", "DYDX-USD": "dydx", "INJ-USD": "injective",
    "ARB-USD": "arbitrum-one", "OP-USD": "optimism-bridge",
}

# Tokens where TVL is not a meaningful metric
_NON_DEFI = {"BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
             "DOGE-USD", "DOT-USD", "LTC-USD", "AVAX-USD", "ATOM-USD", "XLM-USD",
             "TRX-USD", "TON-USD", "SHIB-USD", "PEPE-USD", "NEAR-USD", "APT-USD",
             "SUI-USD", "FIL-USD", "ALGO-USD", "MATIC-USD", "LINK-USD"}


def _ticker_to_defillama_slug(ticker: str) -> Optional[str]:
    normalized = ticker.strip().upper()
    if normalized in _TICKER_TO_SLUG:
        return _TICKER_TO_SLUG[normalized]
    # Try without quote currency
    base = normalized.split("-")[0]
    for key, slug in _TICKER_TO_SLUG.items():
        if key.startswith(base + "-"):
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
    """Main entry point: return formatted TVL/protocol data string."""
    normalized = ticker.strip().upper()
    base = normalized.split("-")[0]

    # Check if non-DeFi
    if normalized in _NON_DEFI or any(normalized.startswith(b + "-") for b in
                                       [t.split("-")[0] for t in _NON_DEFI]):
        return (f"## TVL Data: {ticker}\n\n"
                f"TVL is not applicable for {base}. It is a Layer-1/currency token, "
                f"not a DeFi protocol. TVL metrics apply to protocols that hold user deposits "
                f"(DEXs, lending platforms, liquid staking, etc.).")

    slug = _ticker_to_defillama_slug(ticker)
    if not slug:
        return (f"## TVL Data: {ticker}\n\n"
                f"No DeFiLlama mapping found for {ticker}. "
                f"This token may not be a DeFi protocol, or is not yet mapped.")

    data = _fetch_protocol(slug)
    if not data:
        return f"## TVL Data: {ticker}\n\nDeFiLlama API error for protocol '{slug}'."

    current_tvl = data.get("currentChainTvls", {})
    total_tvl = sum(v for k, v in current_tvl.items()
                    if not k.endswith("-borrowed") and not k.endswith("-staking")
                    and isinstance(v, (int, float)))

    # TVL history for change calculation
    tvl_history = data.get("tvl", [])
    tvl_7d_ago = tvl_history[-8]["totalLiquidityUSD"] if len(tvl_history) > 8 else None
    tvl_30d_ago = tvl_history[-31]["totalLiquidityUSD"] if len(tvl_history) > 31 else None

    category = data.get("category", "N/A")
    chains = data.get("chains", [])
    chains_str = ", ".join(chains[:10]) + (f" (+{len(chains)-10} more)" if len(chains) > 10 else "")

    lines = [
        f"## TVL Data: {ticker} ({slug})",
        "",
        f"**Current TVL**: {_fmt_usd(total_tvl)}",
        f"- 7d TVL Change: {_pct_change(total_tvl, tvl_7d_ago) if tvl_7d_ago else 'N/A'}",
        f"- 30d TVL Change: {_pct_change(total_tvl, tvl_30d_ago) if tvl_30d_ago else 'N/A'}",
        f"- Category: {category}",
        f"- Chains: {chains_str}" if chains else "- Chains: N/A",
    ]

    # Chain breakdown (top 5)
    if current_tvl:
        sorted_chains = sorted(
            [(k, v) for k, v in current_tvl.items()
             if not k.endswith("-borrowed") and not k.endswith("-staking")
             and isinstance(v, (int, float))],
            key=lambda x: x[1], reverse=True
        )[:5]
        if sorted_chains:
            lines += ["", "**TVL by Chain (top 5)**:"]
            for chain, tvl in sorted_chains:
                lines.append(f"- {chain}: {_fmt_usd(tvl)}")

    return "\n".join(lines)
