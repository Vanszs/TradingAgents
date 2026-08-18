"""Quantitative market price structural levels and dynamic volatility barriers."""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from .stockstats_utils import load_ohlcv

logger = logging.getLogger(__name__)


def compute_structural_levels(df: pd.DataFrame, trade_date: str) -> Dict[str, Any]:
    """Calculate 52-week High/Low, multi-month swing points, and Fibonacci retracements."""
    if df is None or df.empty:
        return {}

    work_df = df.copy()
    if isinstance(work_df.index, pd.DatetimeIndex) or (work_df.index.name and str(work_df.index.name).lower() in ("date", "datetime", "timestamp")):
        work_df = work_df.reset_index()

    # Case-insensitive column resolution
    col_map = {c: str(c).strip().title() for c in work_df.columns}
    work_df = work_df.rename(columns=col_map)

    date_candidates = [c for c in work_df.columns if str(c).lower() in ("date", "datetime", "timestamp", "trade_date")]
    date_col = date_candidates[0] if date_candidates else work_df.columns[0]

    work_df["date_str"] = pd.to_datetime(work_df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    history = work_df[work_df["date_str"] <= str(trade_date)].sort_values("date_str").copy()
    if history.empty:
        return {}

    # 1D OHLC Numeric check
    for col in ["Open", "High", "Low", "Close"]:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")

    history = history.dropna(subset=["Close"])
    if history.empty:
        return {}

    latest = history.iloc[-1]
    last_close = float(latest["Close"])

    # 52-Week (~252 trading days) High / Low
    w52 = history.tail(252)
    h52 = float(w52["High"].max()) if "High" in w52 else float(w52["Close"].max())
    l52 = float(w52["Low"].min()) if "Low" in w52 else float(w52["Close"].min())

    # Multi-month Swing Lows & Highs (60D and 20D)
    w60 = history.tail(60)
    l60 = float(w60["Low"].min()) if "Low" in w60 else float(w60["Close"].min())
    h60 = float(w60["High"].max()) if "High" in w60 else float(w60["Close"].max())
    w20 = history.tail(20)
    l20 = float(w20["Low"].min()) if "Low" in w20 else float(w20["Close"].min())
    h20 = float(w20["High"].max()) if "High" in w20 else float(w20["Close"].max())

    # Fibonacci calculation between 60D High and Low
    fib_range = h60 - l60
    fib_50 = round(h60 - (0.50 * fib_range), 2) if fib_range > 0 else last_close
    fib_618 = round(h60 - (0.618 * fib_range), 2) if fib_range > 0 else last_close

    return {
        "trade_date": str(latest["date_str"]),
        "last_close": round(last_close, 2),
        "52_week_high": round(h52, 2),
        "52_week_low": round(l52, 2),
        "60d_swing_low": round(l60, 2),
        "60d_swing_high": round(h60, 2),
        "20d_swing_low": round(l20, 2),
        "20d_swing_high": round(h20, 2),
        "fib_50_level": fib_50,
        "fib_618_level": fib_618,
    }


def get_market_structural_summary(symbol: str, trade_date: str) -> str:
    """Format quantitative structural levels into clean prompt text."""
    try:
        df = load_ohlcv(symbol, trade_date)
        levels = compute_structural_levels(df, trade_date)
    except Exception as exc:
        logger.warning("Could not calculate structural levels for %s: %s", symbol, exc)
        return ""

    if not levels:
        return ""

    return (
        f"Quantitative Structural Price Levels (1D Timeframe, as of {levels['trade_date']}):\n"
        f"- Last Close: {levels['last_close']}\n"
        f"- 52-Week Range: Low = {levels['52_week_low']} | High = {levels['52_week_high']}\n"
        f"- 60D Swing Range: Low = {levels['60d_swing_low']} | High = {levels['60d_swing_high']}\n"
        f"- 20D Swing Range: Low = {levels['20d_swing_low']} | High = {levels['20d_swing_high']}\n"
        f"- Key Retracements: Fib 50% = {levels['fib_50_level']} | Fib 61.8% = {levels['fib_618_level']}\n"
        f"Rule: Price levels in trading proposals should align with these calculated structural benchmarks."
    )
