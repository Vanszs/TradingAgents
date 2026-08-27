"""Canonical BUY/WNS parser with legacy text normalization at the boundary."""

from __future__ import annotations

import re
from typing import Tuple

# Legacy labels remain parseable, but callers receive only BUY or WNS.
RATINGS_5_TIER: Tuple[str, ...] = ("Buy", "Overweight", "Hold", "WNS", "Underweight", "Sell")
_RATING_SET = {r.lower() for r in RATINGS_5_TIER} | {"wait and see", "wait & see", "wns"}

_RATING_LABEL_RE = re.compile(
    r"(?i)^\s*(?:[-*#\d.]+\s*)?(?:\*{1,2})?(?:portfolio\s+)?(?:recommendation|rating)(?:\*{1,2})?\s*[:\-]\s*\**\b(Buy|Overweight|Hold|WNS|Wait\s+and\s+See|Wait\s*&\s*See|Underweight|Sell)\b"
)


def parse_rating(text: str, default: str = "WNS") -> str:
    """Extract a legacy label, then normalize it to canonical BUY/WNS."""
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m:
            raw = m.group(1).lower().strip()
            if raw in {"buy", "overweight"}:
                return "BUY"
            if raw in _RATING_SET:
                return "WNS"

    return "BUY" if default.strip().lower() in {"buy", "overweight"} else "WNS"
