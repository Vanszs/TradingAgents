"""Exa.ai Time-Travel Search Client with Disk Caching and Zero Lookahead Leakage."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from tradingagents.dataflows.config import get_config, is_point_in_time_mode
from tradingagents.dataflows.searxng import search as searxng_search

logger = logging.getLogger(__name__)

EXA_SEARCH_URL = "https://api.exa.ai/search"


def _compute_cache_key(
    query: str,
    end_published_date: Optional[str],
    num_results: int,
    category: str = "general",
) -> str:
    """Generate deterministic SHA256 hex digest for query parameters."""
    normalized = f"{query.strip().lower()}|{end_published_date or 'NONE'}|{num_results}|{category}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _resolve_end_published_date(trade_date: Optional[str]) -> Optional[str]:
    """Convert trade_date (YYYY-MM-DD) to ISO8601 UTC end-of-day boundary."""
    if not trade_date:
        return None
    clean_date = str(trade_date).strip().split("T")[0].split(" ")[0]
    return f"{clean_date}T23:59:59.000Z"


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an Exa timestamp as timezone-aware UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _filter_historical_results(
    results: List[Dict[str, Any]], end_published_date: str
) -> List[Dict[str, Any]]:
    """Keep only dated results published and updated by the historical cutoff."""
    cutoff = _parse_timestamp(end_published_date)
    if cutoff is None:
        return []

    filtered = []
    for result in results:
        published = _parse_timestamp(result.get("publishedDate"))
        updated = _parse_timestamp(result.get("updatedDate"))
        if published is None or published > cutoff or (updated and updated > cutoff):
            continue
        filtered.append(result)
    return filtered


def _historical_unavailable(query: str) -> str:
    return (
        f"## Web Search (Exa Time-Travel): {query}\n\n"
        "Historical search unavailable; no point-in-time result was used."
    )


def _format_exa_response(query: str, results: List[Dict[str, Any]]) -> str:
    """Format Exa search results with [YYYY-MM-DD] publication badges."""
    if not results:
        return f"## Web Search (Exa Time-Travel): {query}\n\nNo historical results found."

    lines = [f"## Web Search Results (Exa Time-Travel): {query}", ""]
    for i, r in enumerate(results, 1):
        title = r.get("title") or "No title"
        url = r.get("url", "")
        raw_pub = r.get("publishedDate") or ""
        pub_date = raw_pub.split("T")[0] if raw_pub else "Unknown Date"

        highlights = r.get("highlights", [])
        snippet = " ".join(highlights)[:300] if highlights else r.get("text", "")[:200]

        lines.append(f"{i}. [{pub_date}] **{title}**")
        if snippet:
            lines.append(f"   Excerpt: {snippet}")
        if url:
            lines.append(f"   URL: {url}")
        lines.append("")

    return "\n".join(lines).strip()


class ExaTimeTravelSearch:
    """Zero-lookahead historical search engine with persistent disk caching."""

    def __init__(self, cache_dir: Optional[Path] = None):
        cfg = get_config()
        base_cache = Path(cfg.get("data_cache_dir", ".cache")) / "exa_search"
        self.cache_dir = cache_dir or base_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed reading Exa cache %s: %s", cache_file, e)
        return None

    def _write_cache(self, key: str, payload: Dict[str, Any]) -> None:
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed writing Exa cache %s: %s", cache_file, e)

    def search(
        self,
        query: str,
        trade_date: Optional[str] = None,
        num_results: int = 5,
        category: str = "general",
    ) -> str:
        """Execute historical search clamped to trade_date.

        // ponytail: JSON disk cache per query-hash, upgrade to SQLite when file-count > 50k.
        """
        if category not in {"news", "general"}:
            raise ValueError("category must be 'news' or 'general'")
        api_key = os.environ.get("EXA_API_KEY")
        if is_point_in_time_mode() and not trade_date:
            logger.warning("PIT search requires trade_date; historical search unavailable.")
            return _historical_unavailable(query)

        end_published_date = _resolve_end_published_date(trade_date)
        cache_key = _compute_cache_key(query, end_published_date, num_results, category)

        # 1. Disk Cache Hit (Fast & Deterministic)
        cached = self._read_cache(cache_key)
        if cached is not None:
            results = cached.get("results", [])
            if end_published_date:
                results = _filter_historical_results(results, end_published_date)
            return _format_exa_response(query, results)

        # Historical mode must never fall back to an unbounded live index.
        if not api_key:
            logger.warning("EXA_API_KEY not set; historical search unavailable.")
            return _historical_unavailable(query) if end_published_date else searxng_search(
                query, num_results=num_results
            )

        # 3. Exa API Request
        payload: Dict[str, Any] = {
            "query": query,
            "type": "auto",
            "numResults": num_results,
            "contents": {"highlights": True},
        }
        if category == "news":
            payload["category"] = "news"
        if end_published_date:
            payload["endPublishedDate"] = end_published_date

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "tradingagents",
        }

        try:
            resp = requests.post(EXA_SEARCH_URL, json=payload, headers=headers, timeout=12)

            if resp.status_code == 429:
                logger.warning("Exa API rate-limited; search unavailable for this request.")
                return _historical_unavailable(query) if end_published_date else searxng_search(
                    query, num_results=num_results
                )

            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if end_published_date:
                results = _filter_historical_results(results, end_published_date)

            self._write_cache(
                cache_key,
                {"results": results, "query": query, "endPublishedDate": end_published_date},
            )
            return _format_exa_response(query, results)

        except Exception as e:
            logger.error("Exa search request failed: %s", e)
            return _historical_unavailable(query) if end_published_date else searxng_search(
                query, num_results=num_results
            )
