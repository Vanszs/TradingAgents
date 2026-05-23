"""GitHub API client for crypto project developer activity.

Uses CoinGecko's links.repos_url.github to find the repo, then fetches
stats from the GitHub API. Supports GITHUB_TOKEN for higher rate limits.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

from .crypto_id_map import ticker_to_coingecko_id

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_COMMITS = 3600  # 1 hour
_CACHE_TTL_STATS = 21600  # 6 hours


def _gh_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _cached_get(url: str, ttl: int) -> Optional[dict | list]:
    now = time.time()
    if url in _CACHE:
        ts, data = _CACHE[url]
        if now - ts < ttl:
            return data
    try:
        resp = requests.get(url, headers=_gh_headers(), timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        _CACHE[url] = (now, data)
        return data
    except Exception as exc:
        logger.warning("GitHub request failed for %s: %s", url, exc)
        return None


def _fetch_repo_url(coin_id: str) -> Optional[str]:
    """Get repo URL from CoinGecko links."""
    from .coingecko import _get
    data = _get(f"/coins/{coin_id}", params={
        "localization": "false", "tickers": "false",
        "market_data": "false", "community_data": "false",
        "developer_data": "false",
    })
    if not data:
        return None
    github_urls = data.get("links", {}).get("repos_url", {}).get("github", [])
    for url in github_urls:
        if url and "github.com" in url:
            return url.rstrip("/")
    return None


def _parse_owner_repo(repo_url: str) -> tuple[str, str] | None:
    """Extract owner/repo from a GitHub URL."""
    parts = repo_url.rstrip("/").split("github.com/")
    if len(parts) < 2:
        return None
    segments = parts[1].strip("/").split("/")
    if len(segments) < 2:
        return None
    return segments[0], segments[1]


def _fetch_github_stats(repo_url: str) -> Optional[dict]:
    """Fetch repo stats from GitHub API."""
    parsed = _parse_owner_repo(repo_url)
    if not parsed:
        return None
    owner, repo = parsed
    url = f"https://api.github.com/repos/{owner}/{repo}"
    return _cached_get(url, _CACHE_TTL_STATS)


def _fetch_commit_count_4w(owner: str, repo: str) -> Optional[int]:
    """Count commits in the last 28 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=28)).isoformat()
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?since={since}&per_page=1"
    now = time.time()
    cache_key = f"commits_4w:{owner}/{repo}"
    if cache_key in _CACHE:
        ts, count = _CACHE[cache_key]
        if now - ts < _CACHE_TTL_COMMITS:
            return count
    try:
        resp = requests.get(url, headers=_gh_headers(), timeout=15)
        if resp.status_code != 200:
            return None
        # GitHub returns total count in Link header for pagination
        link = resp.headers.get("Link", "")
        if 'rel="last"' in link:
            # Parse last page number
            import re
            match = re.search(r'page=(\d+)>; rel="last"', link)
            count = int(match.group(1)) if match else len(resp.json())
        else:
            count = len(resp.json())
        _CACHE[cache_key] = (now, count)
        return count
    except Exception as exc:
        logger.warning("GitHub commits request failed: %s", exc)
        return None


def _relative_time(iso_date: str) -> str:
    """Convert ISO date to relative time string."""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        if delta.days > 30:
            return f"{delta.days // 30} months ago"
        if delta.days > 0:
            return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago" if hours else "just now"
    except Exception:
        return iso_date


def get_dev_activity(ticker: str) -> str:
    """Main entry point: return formatted developer activity string."""
    coin_id = ticker_to_coingecko_id(ticker)
    if not coin_id:
        return f"Developer activity unavailable: could not resolve '{ticker}' to a CoinGecko ID."

    repo_url = _fetch_repo_url(coin_id)
    if not repo_url:
        return f"Developer activity unavailable for {ticker}: no GitHub repository found on CoinGecko."

    parsed = _parse_owner_repo(repo_url)
    if not parsed:
        return f"Developer activity unavailable for {ticker}: could not parse repo URL '{repo_url}'."

    owner, repo = parsed
    stats = _fetch_github_stats(repo_url)
    if not stats:
        return f"Developer activity unavailable for {ticker}: GitHub API error for {repo_url}."

    commits_4w = _fetch_commit_count_4w(owner, repo)
    stars = stats.get("stargazers_count")
    forks = stats.get("forks_count")
    open_issues = stats.get("open_issues_count")
    pushed_at = stats.get("pushed_at", "")
    subscribers = stats.get("subscribers_count")

    # Activity grade
    if commits_4w is None:
        grade = "Unknown"
        assessment = "Could not determine commit frequency."
    elif commits_4w >= 50:
        grade = "Very Active"
        assessment = "High development velocity indicates active maintenance and feature development."
    elif commits_4w >= 20:
        grade = "Active"
        assessment = "Healthy development pace with regular contributions."
    elif commits_4w >= 5:
        grade = "Moderate"
        assessment = "Some development activity; project is maintained but not rapidly evolving."
    else:
        grade = "Low Activity"
        assessment = "Minimal recent development; may indicate maintenance-only mode or declining interest."

    lines = [
        f"## Developer Activity: {ticker} ({coin_id})",
        "",
        f"**Activity Grade**: {grade}",
        f"**Repository**: {repo_url}",
        f"- Stars: {stars:,}" if isinstance(stars, int) else "- Stars: N/A",
        f"- Forks: {forks:,}" if isinstance(forks, int) else "- Forks: N/A",
        f"- Open Issues: {open_issues:,}" if isinstance(open_issues, int) else "- Open Issues: N/A",
        f"- Commits (last 4 weeks): {commits_4w}" if commits_4w is not None else "- Commits (last 4 weeks): N/A",
        f"- Watchers: {subscribers:,}" if isinstance(subscribers, int) else "- Watchers: N/A",
        f"- Last Commit: {_relative_time(pushed_at)}" if pushed_at else "- Last Commit: N/A",
        "",
        f"**Assessment**: {assessment}",
    ]
    return "\n".join(lines)
