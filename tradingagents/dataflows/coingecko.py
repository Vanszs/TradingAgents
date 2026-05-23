"""CoinGecko API client for crypto fundamental data.

Supports both keyless (30 req/min) and Demo API key (500 req/min) modes.
All functions return formatted strings ready for LLM prompt injection.
Gracefully degrades on rate limits or network errors.
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any, Optional

import requests

from .crypto_id_map import ticker_to_coingecko_id

logger = logging.getLogger(__name__)

_BASE = "https://api.coingecko.com/api/v3"
_PRO_BASE = "https://pro-api.coingecko.com/api/v3"
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 300  # 5 minutes


def _get(path: str, params: dict | None = None) -> Optional[dict]:
    """Make a GET request to CoinGecko, with caching and error handling."""
    api_key = os.environ.get("COINGECKO_API_KEY", "")
    # Demo keys (CG-...) use api.coingecko.com; Pro keys use pro-api.coingecko.com
    is_pro_key = api_key and not api_key.startswith("CG-")
    base = _PRO_BASE if is_pro_key else _BASE
    url = f"{base}{path}"
    cache_key = f"{url}:{params}"
    now = time.time()
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return data
    headers = {"accept": "application/json"}
    if api_key:
        hdr_key = "x-cg-pro-api-key" if is_pro_key else "x-cg-demo-api-key"
        headers[hdr_key] = api_key
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 429:
            logger.warning("CoinGecko rate limit hit for %s", path)
            return None
        resp.raise_for_status()
        data = resp.json()
        _CACHE[cache_key] = (now, data)
        return data
    except Exception as exc:
        logger.warning("CoinGecko request failed for %s: %s", path, exc)
        return None


def get_tokenomics(ticker: str) -> str:
    """Return tokenomics data for a crypto ticker as a formatted string."""
    coin_id = ticker_to_coingecko_id(ticker)
    if not coin_id:
        return f"Tokenomics data unavailable: could not resolve '{ticker}' to a CoinGecko ID."

    data = _get(f"/coins/{coin_id}", params={
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "true",
        "developer_data": "true",
    })
    if not data:
        return f"Tokenomics data unavailable for {ticker} (API error or rate limit)."

    md = data.get("market_data", {})
    mcap = md.get("market_cap", {}).get("usd")
    fdv = md.get("fully_diluted_valuation", {}).get("usd")
    circ = md.get("circulating_supply")
    total = md.get("total_supply")
    max_s = md.get("max_supply")
    price = md.get("current_price", {}).get("usd")
    vol_24h = md.get("total_volume", {}).get("usd")
    price_chg_24h = md.get("price_change_percentage_24h")
    price_chg_7d = md.get("price_change_percentage_7d")
    price_chg_30d = md.get("price_change_percentage_30d")
    ath = md.get("ath", {}).get("usd")
    ath_chg = md.get("ath_change_percentage", {}).get("usd")

    def fmt_num(n, prefix="$", suffix=""):
        if n is None:
            return "N/A"
        sign = "-" if n < 0 else ""
        n = abs(n)
        if n >= 1e12:
            return f"{sign}{prefix}{n/1e12:.2f}T{suffix}"
        if n >= 1e9:
            return f"{sign}{prefix}{n/1e9:.2f}B{suffix}"
        if n >= 1e6:
            return f"{sign}{prefix}{n/1e6:.2f}M{suffix}"
        return f"{sign}{prefix}{n:,.2f}{suffix}"

    supply_ratio = f"{circ/max_s*100:.1f}%" if circ and max_s else "N/A (unlimited supply)"
    is_deflationary = "Yes (max supply capped)" if max_s else "No (unlimited supply)"

    lines = [
        f"## Tokenomics: {data.get('name', ticker)} ({data.get('symbol', '').upper()})",
        "",
        f"**Current Price**: {fmt_num(price)}",
        f"**Market Cap**: {fmt_num(mcap)} (Rank #{data.get('market_cap_rank', 'N/A')})",
        f"**Fully Diluted Valuation**: {fmt_num(fdv)}",
        f"**24h Volume**: {fmt_num(vol_24h)}",
        "",
        "### Supply Dynamics",
        f"- Circulating Supply: {fmt_num(circ, prefix='', suffix=' tokens') if circ else 'N/A'}",
        f"- Total Supply: {fmt_num(total, prefix='', suffix=' tokens') if total else 'N/A'}",
        f"- Max Supply: {fmt_num(max_s, prefix='', suffix=' tokens') if max_s else 'Unlimited'}",
        f"- Supply Ratio (circ/max): {supply_ratio}",
        f"- Deflationary: {is_deflationary}",
        "",
        "### Price Performance",
        f"- 24h Change: {price_chg_24h:+.2f}%" if price_chg_24h is not None else "- 24h Change: N/A",
        f"- 7d Change: {price_chg_7d:+.2f}%" if price_chg_7d is not None else "- 7d Change: N/A",
        f"- 30d Change: {price_chg_30d:+.2f}%" if price_chg_30d is not None else "- 30d Change: N/A",
        f"- ATH: {fmt_num(ath)} ({ath_chg:+.1f}% from ATH)" if ath and ath_chg is not None else f"- ATH: {fmt_num(ath)}",
    ]

    # Community data
    comm = data.get("community_data", {})
    if comm:
        lines += [
            "",
            "### Community",
            f"- Twitter Followers: {comm.get('twitter_followers', 'N/A'):,}" if isinstance(comm.get('twitter_followers'), int) else "- Twitter Followers: N/A",
            f"- Reddit Subscribers: {comm.get('reddit_subscribers', 'N/A'):,}" if isinstance(comm.get('reddit_subscribers'), int) else "- Reddit Subscribers: N/A",
            f"- Telegram Users: {comm.get('telegram_channel_user_count', 'N/A'):,}" if isinstance(comm.get('telegram_channel_user_count'), int) else "- Telegram Users: N/A",
        ]

    # Description
    desc = data.get("description", {}).get("en", "")
    if desc:
        lines += ["", "### Project Description", desc[:500] + ("..." if len(desc) > 500 else "")]

    return "\n".join(lines)


def get_market_data(ticker: str) -> str:
    """Return current market data for a crypto ticker."""
    coin_id = ticker_to_coingecko_id(ticker)
    if not coin_id:
        return f"Market data unavailable: could not resolve '{ticker}'."
    data = _get(f"/coins/{coin_id}/market_chart", params={"vs_currency": "usd", "days": "30"})
    if not data:
        return f"Market data unavailable for {ticker}."
    prices = data.get("prices", [])
    if len(prices) < 2:
        return f"Insufficient market data for {ticker}."
    start_price = prices[0][1]
    end_price = prices[-1][1]
    pct_30d = (end_price - start_price) / start_price * 100
    return f"30-day price trend for {ticker}: {pct_30d:+.1f}% (from ${start_price:,.2f} to ${end_price:,.2f})"
