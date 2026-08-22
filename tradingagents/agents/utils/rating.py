"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.
"""

from __future__ import annotations

import re
from typing import Tuple

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: Tuple[str, ...] = ("Buy", "Overweight", "Hold", "WNS", "Underweight", "Sell")
_RATING_SET = {r.lower() for r in RATINGS_5_TIER} | {"wait and see", "wait & see", "wns"}

_RATING_LABEL_RE = re.compile(
    r"(?i)^\s*(?:[-*#\d.]+\s*)?(?:\*{1,2})?(?:portfolio\s+)?(?:recommendation|rating)(?:\*{1,2})?\s*[:\-]\s*\**\b(Buy|Overweight|Hold|WNS|Wait\s+and\s+See|Wait\s*&\s*See|Underweight|Sell)\b"
)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Extract a rating from prose text using line-anchored matching."""
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m:
            raw = m.group(1).lower().strip()
            if raw in ("wns", "wait and see", "wait & see"):
                return "WNS"
            if raw in _RATING_SET:
                return m.group(1).capitalize()

    return default
