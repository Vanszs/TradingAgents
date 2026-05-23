"""SearXNG web search client.

SearXNG is a free, self-hosted meta search engine.
Default URL: http://localhost:8080 (configurable via SEARXNG_URL env var).
No API key required.
Gracefully returns 'SearXNG not configured' if server not reachable.
"""
from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 300  # 5 minutes


def _get_base_url() -> str:
    return os.environ.get("SEARXNG_URL", "http://localhost:8080").rstrip("/")


def search(query: str, num_results: int = 8, category: str = "news") -> str:
    """Search SearXNG and return formatted results string."""
    base_url = _get_base_url()
    cache_key = f"{base_url}:{query}:{num_results}"
    now = time.time()
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return data

    try:
        resp = requests.get(
            f"{base_url}/search",
            params={"q": query, "format": "json", "language": "en", "categories": category},
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            return f"## Web Search: {query}\n\nSearXNG returned non-JSON response (possible HTML error page)."
    except requests.exceptions.ConnectionError:
        return (
            f"## Web Search: {query}\n\n"
            "SearXNG not reachable. To enable web search, run SearXNG locally:\n"
            "  docker run -d -p 8080:8080 searxng/searxng\n"
            "Then set SEARXNG_URL=http://localhost:8080 in .env"
        )
    except Exception as exc:
        logger.warning("SearXNG search failed for %r: %s", query, exc)
        return f"## Web Search: {query}\n\nSearch unavailable: {exc}"

    results = data.get("results", [])[:num_results]
    if not results:
        return f"## Web Search: {query}\n\nNo results found."

    lines = [f"## Web Search Results: {query}", ""]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")[:200]
        lines.append(f"{i}. **{title}**")
        if content:
            lines.append(f"   {content}")
        lines.append(f"   {url}")
        lines.append("")

    result_str = "\n".join(lines)
    _CACHE[cache_key] = (now, result_str)
    return result_str
