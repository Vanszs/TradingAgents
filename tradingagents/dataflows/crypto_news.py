"""Crypto news fetcher via RSS feeds (CoinDesk, CoinTelegraph, Decrypt, TheBlock).

No API key required. CryptoPanic discontinued their free API on April 1, 2026.
This module replaces it with multi-source RSS aggregation.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 600  # 10 minutes

_FEEDS = [
    ("CoinDesk",      "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt",       "https://decrypt.co/feed"),
    ("TheBlock",      "https://www.theblock.co/rss.xml"),
]

# Ticker → search keywords mapping
_KEYWORD_MAP: dict[str, list[str]] = {
    "BTC":      ["bitcoin", "btc"],
    "ETH":      ["ethereum", "eth"],
    "SOL":      ["solana", "sol"],
    "BNB":      ["bnb", "binance coin"],
    "XRP":      ["xrp", "ripple"],
    "ADA":      ["cardano", "ada"],
    "AVAX":     ["avalanche", "avax"],
    "DOGE":     ["dogecoin", "doge"],
    "DOT":      ["polkadot", "dot"],
    "MATIC":    ["polygon", "matic"],
    "LINK":     ["chainlink", "link"],
    "UNI":      ["uniswap", "uni"],
    "AAVE":     ["aave"],
    "LTC":      ["litecoin", "ltc"],
    "ATOM":     ["cosmos", "atom"],
    "NEAR":     ["near protocol", "near"],
    "ARB":      ["arbitrum", "arb"],
    "OP":       ["optimism"],
    "INJ":      ["injective", "inj"],
    "SUI":      ["sui"],
    "APT":      ["aptos", "apt"],
    "TON":      ["toncoin", "ton"],
    "TRX":      ["tron", "trx"],
    "PEPE":     ["pepe"],
    "SHIB":     ["shiba inu", "shib"],
    "BONK":     ["bonk"],
    "WIF":      ["dogwifhat", "wif"],
    "FLOKI":    ["floki"],
}


def _fetch_feed(url: str) -> list[dict]:
    """Fetch and parse a single RSS feed. Returns list of article dicts."""
    try:
        resp = requests.get(
            url, timeout=10,
            headers={"User-Agent": "TradingAgents/1.0 RSS Reader"},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            date_el = item.find("pubDate")
            link_el = item.find("link")
            title = title_el.text or "" if title_el is not None else ""
            desc = (desc_el.text or "")[:300] if desc_el is not None else ""
            # Strip HTML tags from description
            desc = re.sub(r"<[^>]+>", "", desc).strip()
            items.append({
                "title": title.strip(),
                "desc": desc,
                "date": (date_el.text or "")[:16] if date_el is not None else "",
                "link": link_el.text or "" if link_el is not None else "",
            })
        return items
    except Exception as exc:
        logger.debug("RSS feed failed for %s: %s", url, exc)
        return []


def _get_keywords(ticker: str) -> list[str]:
    """Return search keywords for a ticker."""
    base = ticker.split("-")[0].upper()
    if base in _KEYWORD_MAP:
        return _KEYWORD_MAP[base]
    # Fallback: use base symbol and lowercase
    return [base.lower()]


def get_crypto_news(ticker: str, limit: int = 10) -> str:
    """Return recent news headlines for a crypto ticker from RSS feeds.

    No API key required. Aggregates CoinDesk, CoinTelegraph, Decrypt, TheBlock.
    """
    cache_key = f"{ticker}:{limit}"
    now = time.time()
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return data

    keywords = _get_keywords(ticker)

    # Fetch all feeds in parallel-ish (sequential but fast)
    all_articles: list[tuple[str, dict]] = []  # (source_name, article)
    for source_name, url in _FEEDS:
        for article in _fetch_feed(url):
            text = (article["title"] + " " + article["desc"]).lower()
            if any(kw in text for kw in keywords):
                all_articles.append((source_name, article))

    if not all_articles:
        result = (
            f"## Crypto News: {ticker}\n\n"
            f"No recent news found for {ticker} across CoinDesk, CoinTelegraph, "
            f"Decrypt, and TheBlock. This may indicate low media coverage "
            f"(common for smaller/meme coins)."
        )
        _CACHE[cache_key] = (now, result)
        return result

    # Deduplicate by title similarity, take top `limit`
    seen_titles: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for src, art in all_articles:
        title_key = art["title"][:50].lower()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique.append((src, art))
        if len(unique) >= limit:
            break

    lines = [
        f"## Crypto News: {ticker} ({len(unique)} articles)",
        "*Sources: CoinDesk, CoinTelegraph, Decrypt, TheBlock — no API key required*",
        "",
    ]
    for src, art in unique:
        date = art["date"] or "recent"
        lines.append(f"**[{src}]** {art['title']}")
        if art["desc"]:
            lines.append(f"  > {art['desc'][:150]}")
        lines.append(f"  *{date}*")
        lines.append("")

    result = "\n".join(lines)
    _CACHE[cache_key] = (now, result)
    return result
