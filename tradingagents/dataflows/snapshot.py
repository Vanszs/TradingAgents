"""Snapshot vendor — reads pre-cut data from config for backtesting.

All functions read from ``config["snapshot_data"]`` which is injected by
``TradingAgentsRunner._safe_agent_runtime_config()`` during backtesting.

Note: Most parameters (ticker, start_date, end_date, freq, etc.) exist to
match the interface signatures expected by the agent tools. The snapshot
data is already pre-filtered by the SnapshotProvider, so these parameters
are not used for filtering in most functions.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def _get_snapshot_data() -> dict:
    """Retrieve the snapshot data block from the global config."""
    from .config import get_config
    return get_config().get("snapshot_data", {})


def _format_news_items(items: list[dict], limit: int = 50) -> str:
    """Format news items into a readable string for LLM consumption."""
    if not items:
        return "No news data available."
    lines = []
    for item in items[:limit]:
        pub = item.get("published_at", item.get("date", ""))
        title = item.get("title", "")
        summary = item.get("summary", item.get("description", ""))
        source = item.get("source", "")
        line = f"[{pub}] {source}: {title}" if source else f"[{pub}] {title}"
        lines.append(line)
        if summary:
            lines.append(f"  {summary}")
    return "\n\n".join(lines)


def _fundamental_available_date(item: dict) -> str:
    """Return the normalized availability date, or an empty value if absent."""
    return str(item.get("available_date") or item.get("date") or "").split("T")[0].split(" ")[0]


def _runtime_trade_date() -> Optional[str]:
    from .config import get_config
    return get_config().get("trade_date") or get_config().get("curr_date")


def _date_only(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _visible_event(value: object, cutoff_date: str, cutoff_time: str = "16:30:00") -> bool:
    """Keep events visible through the configured post-market review cutoff."""
    try:
        event = pd.Timestamp(value)
        cutoff = pd.Timestamp(f"{cutoff_date} {cutoff_time}")
        if event.tzinfo is not None:
            event = event.tz_convert(None)
        return event <= cutoff
    except (TypeError, ValueError):
        return False


def _effective_cutoff(value: object = None) -> Optional[str]:
    """Never return injected snapshot data after the active trade date."""
    requested = _date_only(value) if value else None
    runtime = _runtime_trade_date()
    active = _date_only(runtime) if runtime else None
    if requested and active:
        return min(requested, active)
    return requested or active


def _ohlcv_frame(value) -> pd.DataFrame:
    """Normalize snapshot OHLCV records and frames at the vendor boundary."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _format_fundamental_items(items: list[dict]) -> str:
    """Format fundamental items into a readable string."""
    if not items:
        return "No fundamental data available."
    lines = []
    for item in items:
        date = item.get("available_date", item.get("date", ""))
        metric = item.get("metric", item.get("name", item.get("field", "")))
        value = item.get("value", item.get("amount", ""))
        period = item.get("period", item.get("fiscal_period", ""))
        parts = [f"[{date}]"]
        if period:
            parts.append(f"({period})")
        parts.append(f"{metric}: {value}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Core stock APIs
# --------------------------------------------------------------------------

def snapshot_get_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Return OHLCV CSV from snapshot data."""
    data = _get_snapshot_data()
    ohlcv = data.get("ohlcv")
    if ohlcv is None:
        return f"No OHLCV data available in snapshot for {symbol}."

    df = _ohlcv_frame(ohlcv)
    if df.empty or "date" not in df.columns:
        return f"No OHLCV data available in snapshot for {symbol}."
    df["date"] = df["date"].map(_date_only)
    cutoff = _effective_cutoff(end_date)
    if cutoff is None:
        return f"No OHLCV data available in snapshot for {symbol}."
    start = _date_only(start_date) if start_date else "0001-01-01"
    filtered = df[(df["date"] >= start) & (df["date"] <= cutoff)]

    if filtered.empty:
        return f"No OHLCV data for {symbol} between {start_date} and {end_date}."

    header = (
        f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        f"# Total records: {len(filtered)}\n"
        f"# Source: snapshot (backtest)\n\n"
    )
    return header + filtered.to_csv(index=False)


# --------------------------------------------------------------------------
# Technical indicators
# --------------------------------------------------------------------------

def snapshot_get_indicators(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
) -> str:
    """Compute technical indicators from snapshot OHLCV using stockstats."""
    data = _get_snapshot_data()
    ohlcv = data.get("ohlcv")
    if ohlcv is None:
        return "No OHLCV data available for indicator computation."

    from datetime import datetime, timedelta

    from stockstats import wrap

    raw_ind = indicator.strip().lower()
    alias_map = {
        "rsi_14": "rsi",
        "atr_14": "atr_14",
        "atr_20": "atr_20",
        "atr": "atr_14",
        "boll_ub": "boll_ub",
        "boll_lb": "boll_lb",
    }
    indicator = alias_map.get(raw_ind, raw_ind)

    best_ind_params = {
        "close_50_sma": "50 SMA: A medium-term trend indicator.",
        "close_200_sma": "200 SMA: A long-term trend benchmark.",
        "close_10_ema": "10 EMA: A responsive short-term average.",
        "macd": "MACD: Computes momentum via differences of EMAs.",
        "macds": "MACD Signal: An EMA smoothing of the MACD line.",
        "macdh": "MACD Histogram: Shows the gap between MACD and signal.",
        "rsi": "RSI: Measures momentum to flag overbought/oversold.",
        "boll": "Bollinger Middle: A 20 SMA basis for Bollinger Bands.",
        "boll_ub": "Bollinger Upper Band: 2 std dev above middle.",
        "boll_lb": "Bollinger Lower Band: 2 std dev below middle.",
        "atr": "ATR: Averages true range to measure volatility.",
        "atr_14": "ATR 14: 14-day Average True Range volatility measure.",
        "atr_20": "ATR 20: 20-day Average True Range volatility measure.",
        "vwma": "VWMA: A moving average weighted by volume.",
        "mfi": "MFI: Money Flow Index momentum indicator.",
    }

    if indicator not in best_ind_params:
        return (
            f"Indicator '{indicator}' not supported. "
            f"Choose from: {list(best_ind_params.keys())}"
        )

    # Prepare DataFrame for stockstats
    df = _ohlcv_frame(ohlcv)
    if df.empty or "date" not in df.columns:
        return "No OHLCV data available for indicator computation."
    df = df.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter to curr_date
    cutoff = _effective_cutoff(curr_date)
    curr_date_dt = pd.to_datetime(cutoff)
    df = df[df["Date"] <= curr_date_dt]
    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty:
        return f"No OHLCV data available before {curr_date}."

    try:
        if indicator in ("atr", "atr_14"):
            from .stockstats_utils import compute_atr
            atr_s = compute_atr(df, period=14)
            wrapped = df.copy()
            wrapped["Date"] = pd.to_datetime(wrapped["Date"]).dt.strftime("%Y-%m-%d")
            wrapped[indicator] = atr_s
        elif indicator == "atr_20":
            from .stockstats_utils import compute_atr
            atr_s = compute_atr(df, period=20)
            wrapped = df.copy()
            wrapped["Date"] = pd.to_datetime(wrapped["Date"]).dt.strftime("%Y-%m-%d")
            wrapped[indicator] = atr_s
        else:
            wrapped = wrap(df)
            wrapped["Date"] = wrapped["Date"].dt.strftime("%Y-%m-%d")
            # Trigger indicator calculation
            _ = wrapped[indicator]

        # Build lookback range
        before = curr_date_dt - timedelta(days=look_back_days)
        result_lines = []
        for _, row in wrapped.iterrows():
            date_str = row["Date"]
            if date_str >= before.strftime("%Y-%m-%d") and date_str <= curr_date:
                val = row[indicator]
                val_str = "N/A" if pd.isna(val) else str(val)
                result_lines.append(f"{date_str}: {val_str}")

        ind_string = "\n".join(result_lines)
    except Exception as e:
        return f"Error computing {indicator}: {e}"

    return (
        f"## {indicator} values from "
        f"{(curr_date_dt - timedelta(days=look_back_days)).strftime('%Y-%m-%d')} "
        f"to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "")
    )


# --------------------------------------------------------------------------
# News data
# --------------------------------------------------------------------------

def snapshot_get_news(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Return formatted news from snapshot."""
    data = _get_snapshot_data()
    news = data.get("news", [])
    cutoff = _effective_cutoff(end_date)
    start_cutoff = _date_only(start_date) if start_date else "0001-01-01"
    if cutoff:
        news = [
            item for item in news
            if (raw := str(item.get("published_at") or item.get("date") or ""))
            and start_cutoff <= _date_only(raw) <= cutoff
            and _visible_event(raw, cutoff)
        ]
    if not news:
        return f"No news data available for {ticker}."
    return _format_news_items(news)


def snapshot_get_global_news(
    curr_date: str,
    look_back_days: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    """Return formatted global/macro news from snapshot."""
    data = _get_snapshot_data()
    news = data.get("news", [])
    cutoff = _effective_cutoff(curr_date)
    if cutoff:
        news = [
            item for item in news
            if (raw := str(item.get("published_at") or item.get("date") or ""))
            and _date_only(raw) <= cutoff
            and _visible_event(raw, cutoff)
        ]
    if not news:
        return "No global news data available."
    return _format_news_items(news, limit=limit or 50)


def snapshot_get_insider_transactions(ticker: str) -> str:
    """Return insider transactions from snapshot (if available)."""
    from .config import get_config
    curr_date = get_config().get("trade_date") or get_config().get("curr_date")
    data = _get_snapshot_data()
    fundamentals = data.get("fundamentals", [])
    insider_items = [
        f for f in fundamentals
        if ("insider" in f.get("metric", "").lower() or "transaction" in f.get("metric", "").lower())
        and (not curr_date or (
            _fundamental_available_date(f)
            and _fundamental_available_date(f) <= str(curr_date).split("T")[0].split(" ")[0]
        ))
    ]
    if not insider_items:
        return f"No insider transaction data available for {ticker}."
    return _format_fundamental_items(insider_items)


# --------------------------------------------------------------------------
# Fundamental data
# --------------------------------------------------------------------------

def snapshot_get_fundamentals(ticker: str, curr_date: str) -> str:
    """Return comprehensive fundamental data from snapshot."""
    data = _get_snapshot_data()
    fundamentals = data.get("fundamentals", [])
    if not fundamentals:
        return f"No fundamental data available for {ticker}."

    # Filter by available_date <= curr_date
    cutoff = _effective_cutoff(curr_date)
    filtered = [
        f for f in fundamentals
        if cutoff and _fundamental_available_date(f) and _fundamental_available_date(f) <= cutoff
    ]
    if not filtered:
        return f"No fundamental data available for {ticker} up to {curr_date}."
    return _format_fundamental_items(filtered)


def snapshot_get_balance_sheet(
    ticker: str,
    freq: str = "quarterly",
    curr_date: Optional[str] = None,
) -> str:
    """Return balance sheet data from snapshot fundamentals."""
    data = _get_snapshot_data()
    fundamentals = data.get("fundamentals", [])
    keywords = ["balance", "asset", "liability", "equity", "debt"]
    items = [
        f for f in fundamentals
        if any(kw in f.get("metric", "").lower() for kw in keywords)
    ]
    cutoff = curr_date or _runtime_trade_date()
    if cutoff:
        cutoff = _effective_cutoff(cutoff)
        items = [
            f for f in items
            if _fundamental_available_date(f) and _fundamental_available_date(f) <= cutoff
        ]
    if not items:
        return f"No balance sheet data available for {ticker}."
    return _format_fundamental_items(items)


def snapshot_get_cashflow(
    ticker: str,
    freq: str = "quarterly",
    curr_date: Optional[str] = None,
) -> str:
    """Return cash flow statement from snapshot fundamentals."""
    data = _get_snapshot_data()
    fundamentals = data.get("fundamentals", [])
    keywords = ["cash", "flow", "operating", "investing", "financing"]
    items = [
        f for f in fundamentals
        if any(kw in f.get("metric", "").lower() for kw in keywords)
    ]
    cutoff = curr_date or _runtime_trade_date()
    if cutoff:
        cutoff = _effective_cutoff(cutoff)
        items = [
            f for f in items
            if _fundamental_available_date(f) and _fundamental_available_date(f) <= cutoff
        ]
    if not items:
        return f"No cashflow data available for {ticker}."
    return _format_fundamental_items(items)


def snapshot_get_income_statement(
    ticker: str,
    freq: str = "quarterly",
    curr_date: Optional[str] = None,
) -> str:
    """Return income statement from snapshot fundamentals."""
    data = _get_snapshot_data()
    fundamentals = data.get("fundamentals", [])
    keywords = ["income", "revenue", "profit", "eps", "earnings", "margin"]
    items = [
        f for f in fundamentals
        if any(kw in f.get("metric", "").lower() for kw in keywords)
    ]
    cutoff = curr_date or _runtime_trade_date()
    if cutoff:
        cutoff = _effective_cutoff(cutoff)
        items = [
            f for f in items
            if _fundamental_available_date(f) and _fundamental_available_date(f) <= cutoff
        ]
    if not items:
        return f"No income statement data available for {ticker}."
    return _format_fundamental_items(items)
