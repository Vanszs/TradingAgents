"""Alternative.me Fear & Greed Index fetcher.

No API key required. Returns current crypto market sentiment on 0-100 scale.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CACHE: Optional[tuple[float, str]] = None
_CACHE_TTL = 3600  # 1 hour (index updates daily)
_HISTORICAL_CACHE: Optional[list[dict]] = None


def _get_interpretation(value: int) -> str:
    if value < 25:
        return "Extreme Fear — potential contrarian buy signal; market may be oversold."
    if value < 45:
        return "Fear — cautious sentiment; watch for reversal signals."
    if value < 55:
        return "Neutral — balanced market sentiment."
    if value < 75:
        return "Greed — positive momentum; watch for overextension."
    return "Extreme Greed — market may be overheated; caution advised."


def get_fear_greed_index(trade_date: Optional[str] = None) -> str:
    """Return the Crypto Fear & Greed Index as a formatted string.

    When trade_date is provided (YYYY-MM-DD), historical data is used.
    When trade_date is None, live data (limit=7) is used.
    """
    global _CACHE, _HISTORICAL_CACHE

    if trade_date:
        try:
            if _HISTORICAL_CACHE is None:
                resp = requests.get(
                    "https://api.alternative.me/fng/?limit=0",
                    timeout=15,
                    headers={"accept": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                _HISTORICAL_CACHE = data.get("data", [])

            # Filter entries where date <= trade_date
            # timestamp in alternative.me is a unix timestamp string
            valid_entries = []
            for entry in _HISTORICAL_CACHE:
                ts = entry.get("timestamp")
                if ts is not None:
                    try:
                        entry_date = time.strftime("%Y-%m-%d", time.gmtime(int(ts)))
                        if entry_date <= trade_date:
                            valid_entries.append((entry_date, entry))
                    except (ValueError, TypeError):
                        continue

            # Entries are usually ordered newest first, but sort to be certain: descending by date
            valid_entries.sort(key=lambda x: x[0], reverse=True)

            if not valid_entries:
                return f"Fear & Greed Index: data unavailable for date <= {trade_date}."

            current_entry = valid_entries[0][1]
            value = int(current_entry.get("value", 0))
            classification = current_entry.get("value_classification", "Unknown")

            trend_lines = []
            for _, e in valid_entries[:7]:
                v = e.get("value", "?")
                c = e.get("value_classification", "?")
                trend_lines.append(f"  - {c} ({v})")

            return (
                f"## Market Fear & Greed Index (As of {trade_date})\n\n"
                f"**Current**: {value}/100 — {classification}\n"
                f"**Interpretation**: {_get_interpretation(value)}\n\n"
                f"**7-Day Trend** (most recent first):\n"
                + "\n".join(trend_lines)
            )
        except Exception as exc:
            logger.warning("Historical Fear & Greed Index fetch failed: %s", exc)
            return "Fear & Greed Index: data unavailable (network error)."

    now = time.time()
    if _CACHE and (now - _CACHE[0]) < _CACHE_TTL:
        return _CACHE[1]

    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=7",
            timeout=10,
            headers={"accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("data", [])
        if not entries:
            return "Fear & Greed Index: data unavailable."

        current = entries[0]
        value = int(current.get("value", 0))
        classification = current.get("value_classification", "Unknown")

        # Build 7-day trend
        trend_lines = []
        for e in entries[:7]:
            v = e.get("value", "?")
            c = e.get("value_classification", "?")
            trend_lines.append(f"  - {c} ({v})")

        result = (
            f"## Crypto Fear & Greed Index\n\n"
            f"**Current**: {value}/100 — {classification}\n"
            f"**Interpretation**: {_get_interpretation(value)}\n\n"
            f"**7-Day Trend** (most recent first):\n"
            + "\n".join(trend_lines)
        )
        _CACHE = (now, result)
        return result
    except Exception as exc:
        logger.warning("Fear & Greed Index fetch failed: %s", exc)
        return "Fear & Greed Index: data unavailable (network error)."
