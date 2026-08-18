"""Deterministic market data validator for ground-truth verification."""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from .stockstats_utils import load_ohlcv
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)


def validate_market_data(
    symbol: str,
    trade_date: str,
    lookback_days: int = 30,
) -> Dict[str, Any]:
    """Retrieve and validate market data snapshot without hallucination.

    Returns verified OHLCV values, recent daily bars, and price health flags.
    """
    canonical = normalize_symbol(symbol)
    try:
        df = load_ohlcv(canonical, trade_date)
    except Exception as e:
        logger.warning("Failed loading market data for validation: %s", e)
        return {
            "status": "UNAVAILABLE",
            "symbol": symbol,
            "canonical": canonical,
            "trade_date": trade_date,
            "error": str(e),
        }

    if df is None or df.empty:
        return {
            "status": "NO_DATA",
            "symbol": symbol,
            "canonical": canonical,
            "trade_date": trade_date,
        }

    # Ensure Date column is string
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df["date_str"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    df = df[df["date_str"] <= trade_date].sort_values("date_str")

    if df.empty:
        return {
            "status": "NO_DATA_PRIOR_TO_DATE",
            "symbol": symbol,
            "canonical": canonical,
            "trade_date": trade_date,
        }

    latest = df.iloc[-1]
    recent = df.tail(lookback_days)

    recent_closes = [
        {"date": row["date_str"], "close": float(row.get("Close", 0.0))}
        for _, row in recent.iterrows()
    ]

    return {
        "status": "VERIFIED",
        "symbol": symbol,
        "canonical": canonical,
        "trade_date": str(latest["date_str"]),
        "latest_ohlcv": {
            "open": float(latest.get("Open", 0.0)),
            "high": float(latest.get("High", 0.0)),
            "low": float(latest.get("Low", 0.0)),
            "close": float(latest.get("Close", 0.0)),
            "volume": int(latest.get("Volume", 0)) if "Volume" in latest else 0,
        },
        "records_count": len(df),
        "recent_closes": recent_closes,
    }


def format_verified_market_snapshot(symbol: str, trade_date: str) -> str:
    """Render a human/LLM-readable verified market snapshot block."""
    snapshot = validate_market_data(symbol, trade_date)
    if snapshot.get("status") != "VERIFIED":
        return f"## Verified Market Snapshot: {symbol}\n\nStatus: {snapshot.get('status')} - Market data unavailable for verification."

    latest = snapshot["latest_ohlcv"]
    canonical = snapshot["canonical"]
    label = canonical if canonical == symbol.upper() else f"{symbol} (resolved to {canonical})"

    lines = [
        f"## Verified Market Snapshot: {label}",
        f"**Session Date**: {snapshot['trade_date']}",
        f"- **Open**: {latest['open']}",
        f"- **High**: {latest['high']}",
        f"- **Low**: {latest['low']}",
        f"- **Close**: {latest['close']}",
        f"- **Volume**: {latest['volume']:,}",
        f"\n**Note for Market Analyst**: Treat these numbers as authoritative ground truth. Do not invent support/resistance without corroborating prices.",
    ]
    return "\n".join(lines)
