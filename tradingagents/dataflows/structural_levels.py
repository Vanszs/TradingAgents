"""Quantitative market price structural levels (Multi-Horizon Daily Structure)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .stockstats_utils import compute_atr, load_ohlcv

logger = logging.getLogger(__name__)


def compute_1h_micro_levels(df_1h: pd.DataFrame, trade_date: str) -> Dict[str, Any]:
    """Calculate 1H 24-bar swing levels and EMA 20/50 trend alignment."""
    if df_1h is None or df_1h.empty:
        return {}

    work_df = df_1h.copy()
    if isinstance(work_df.index, pd.DatetimeIndex):
        work_df = work_df.reset_index()

    col_map = {c: str(c).strip().title() for c in work_df.columns}
    work_df = work_df.rename(columns=col_map)

    date_candidates = [c for c in work_df.columns if str(c).lower() in ("date", "datetime", "timestamp", "index", "trade_date")]
    if date_candidates:
        date_col = date_candidates[0]
        work_df["ts_utc"] = pd.to_datetime(work_df[date_col], errors="coerce", utc=True)
        work_df = work_df.dropna(subset=["ts_utc"])
        if len(str(trade_date)) > 10:
            cutoff_ts = pd.Timestamp(trade_date, tz="UTC")
        else:
            cutoff_ts = pd.Timestamp(f"{str(trade_date)[:10]} 23:59:59", tz="UTC")
        history = work_df[work_df["ts_utc"] <= cutoff_ts].sort_values("ts_utc").copy()
    else:
        history = work_df.copy()

    for col in ["Open", "High", "Low", "Close"]:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")

    history = history.dropna(subset=["Close"])
    if len(history) < 24:
        return {}

    # 1H 24-bar Swing High / Low (~3-4 trading sessions or 24h crypto)
    w24 = history.tail(24)
    h24 = float(w24["High"].max()) if "High" in w24 else float(w24["Close"].max())
    l24 = float(w24["Low"].min()) if "Low" in w24 else float(w24["Close"].min())

    # 1H EMA 20 and EMA 50
    ema20_series = history["Close"].ewm(span=20, adjust=False).mean()
    ema50_series = history["Close"].ewm(span=50, adjust=False).mean()
    ema20 = float(ema20_series.iloc[-1])
    ema50 = float(ema50_series.iloc[-1])
    last_1h_close = float(history["Close"].iloc[-1])

    if last_1h_close > ema20 > ema50:
        trend_bias = "BULLISH (Close > EMA20 > EMA50)"
    elif last_1h_close < ema20 < ema50:
        trend_bias = "BEARISH (Close < EMA20 < EMA50)"
    else:
        trend_bias = "NEUTRAL / MIXED (Consolidation or Pullback)"

    return {
        "1h_last_close": round(last_1h_close, 2),
        "1h_24bar_swing_high": round(h24, 2),
        "1h_24bar_swing_low": round(l24, 2),
        "1h_ema_20": round(ema20, 2),
        "1h_ema_50": round(ema50, 2),
        "1h_trend_bias": trend_bias,
    }


def compute_structural_levels(
    df: pd.DataFrame,
    trade_date: str,
    df_1h: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Calculate Multi-Horizon Daily Structure (Macro 52W, Intermediate 60D Swings/Fib, Tactical 20D/ATR)."""
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
    history = work_df[work_df["date_str"] <= str(trade_date)[:10]].sort_values("date_str").copy()
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

    # 52-Week (~252 trading days) or Available History High / Low
    n_bars = len(history)
    w52 = history.tail(252)
    h52 = float(w52["High"].max()) if "High" in w52 else float(w52["Close"].max())
    l52 = float(w52["Low"].min()) if "Low" in w52 else float(w52["Close"].min())
    is_full_52w = n_bars >= 180

    # Multi-month Swing Lows & Highs (60D and 20D)
    w60 = history.tail(60).reset_index(drop=True)
    l60 = float(w60["Low"].min()) if "Low" in w60 else float(w60["Close"].min())
    h60 = float(w60["High"].max()) if "High" in w60 else float(w60["Close"].max())
    w20 = history.tail(20)
    l20 = float(w20["Low"].min()) if "Low" in w20 else float(w20["Close"].min())
    h20 = float(w20["High"].max()) if "High" in w20 else float(w20["Close"].max())

    # Fibonacci calculation between 60D High and Low with directional awareness
    pos_h60 = int(w60["High"].argmax()) if "High" in w60 else int(w60["Close"].argmax())
    pos_l60 = int(w60["Low"].argmin()) if "Low" in w60 else int(w60["Close"].argmin())

    fib_range = h60 - l60
    if fib_range > 0:
        if pos_l60 < pos_h60:
            # Bullish impulse (Low -> High): Pullback demand support levels
            fib_50 = round(h60 - (0.50 * fib_range), 2)
            fib_618 = round(h60 - (0.618 * fib_range), 2)
        else:
            # Bearish leg (High -> Low): Counter-trend bounce resistance levels
            fib_50 = round(l60 + (0.50 * fib_range), 2)
            fib_618 = round(l60 + (0.618 * fib_range), 2)
    else:
        fib_50 = fib_618 = last_close

    # Volatility context only; exit levels remain static after entry.
    atr_14_s = compute_atr(history, period=14) if len(history) >= 14 else None
    atr_20_s = compute_atr(history, period=20) if len(history) >= 20 else None

    atr_14_val = round(float(atr_14_s.iloc[-1]), 2) if atr_14_s is not None and not pd.isna(atr_14_s.iloc[-1]) else None
    atr_20_val = round(float(atr_20_s.iloc[-1]), 2) if atr_20_s is not None and not pd.isna(atr_20_s.iloc[-1]) else None
    atr_14_pct = round((atr_14_val / last_close) * 100, 2) if atr_14_val is not None and last_close > 0 else None

    # Fibonacci Extension Targets (1.272x and 1.618x above 60D High for Breakouts)
    fib_ext_1272 = round(h60 + (0.272 * fib_range), 2) if fib_range > 0 else round(last_close * 1.05, 2)
    fib_ext_1618 = round(h60 + (0.618 * fib_range), 2) if fib_range > 0 else round(last_close * 1.10, 2)

    # Forward ATR Volatility Target Channels
    atr_val = atr_14_val or (last_close * 0.02)
    atr_half_val = round(0.5 * atr_val, 2)
    atr_target_2x = round(last_close + (2.0 * atr_val), 2)
    atr_target_3x = round(last_close + (3.0 * atr_val), 2)

    result: Dict[str, Any] = {
        "trade_date": str(latest["date_str"]),
        "last_close": round(last_close, 2),
        "is_full_52w": is_full_52w,
        "history_bars": n_bars,
        "52_week_high": round(h52, 2),
        "52_week_low": round(l52, 2),
        "60d_swing_low": round(l60, 2),
        "60d_swing_high": round(h60, 2),
        "20d_swing_low": round(l20, 2),
        "20d_swing_high": round(h20, 2),
        "fib_50_level": fib_50,
        "fib_618_level": fib_618,
        "fib_ext_1272": fib_ext_1272,
        "fib_ext_1618": fib_ext_1618,
        "atr_14": atr_14_val,
        "atr_20": atr_20_val,
        "atr_14_pct": atr_14_pct,
        "atr_half": atr_half_val,
        "atr_target_2x": atr_target_2x,
        "atr_target_3x": atr_target_3x,
    }

    if df_1h is not None and not df_1h.empty:
        micro = compute_1h_micro_levels(df_1h, trade_date)
        if micro:
            result["micro_1h"] = micro

    return result


def get_market_structural_summary(
    symbol: str,
    trade_date: str,
    df_1h: Optional[pd.DataFrame] = None,
) -> str:
    """Format quantitative structural levels (Multi-Horizon Daily Structure) into clean prompt text."""
    try:
        df_1d = load_ohlcv(symbol, trade_date)
    except Exception as exc:
        logger.warning("Could not calculate structural levels for %s: %s", symbol, exc)
        return ""

    if df_1h is None:
        from .config import get_config, is_point_in_time_mode
        if is_point_in_time_mode():
            snap = get_config().get("snapshot_data", {})
            df_1h = snap.get("ohlcv_1h") if isinstance(snap, dict) else None

    levels = compute_structural_levels(df_1d, trade_date, df_1h=df_1h)
    if not levels:
        return ""

    range_label = (
        "52-Week Range"
        if levels.get("is_full_52w", True)
        else f"Available History Range ({levels.get('history_bars', len(df_1d))} bars)"
    )

    summary = (
        f"Quantitative Structural Price Levels (as of {levels['trade_date']}):\n"
        f"1. **Macro Horizon (52-Week Range & Major Regimes)**:\n"
        f"   - Last Close: {levels['last_close']}\n"
        f"   - {range_label}: Low = {levels['52_week_low']} | High = {levels['52_week_high']}\n"
        f"2. **Intermediate Horizon (60D Swings & Fibonacci)**:\n"
        f"   - 60D Swing Range: Low = {levels['60d_swing_low']} | High = {levels['60d_swing_high']}\n"
        f"   - Pullback Support Floors (Dips/Consolidations Only - Do NOT bid below market on Breakouts): Fib 50% = {levels['fib_50_level']} | Fib 61.8% = {levels['fib_618_level']}\n"
        f"   - Forward Expansion Targets (Breakout Upside): Fib 1.272x = {levels.get('fib_ext_1272')} | Fib 1.618x = {levels.get('fib_ext_1618')}\n"
        f"3. **Tactical Horizon (20D Momentum & ATR Volatility)**:\n"
        f"   - 20D Swing Range: Low = {levels['20d_swing_low']} | High = {levels['20d_swing_high']}\n"
        f"   - Expected Volatility Target Channels: +2x ATR = {levels.get('atr_target_2x')} | +3x ATR = {levels.get('atr_target_3x')}\n"
    )

    if levels.get("atr_14") is not None:
        summary += (
            f"   - Volatility (ATR 14): {levels['atr_14']} ({levels.get('atr_14_pct', 'N/A')}% of price) | 0.5x ATR Buffer: {levels.get('atr_half', 'N/A')} | ATR 20: {levels.get('atr_20', 'N/A')}\n"
        )

    if "micro_1h" in levels:
        m = levels["micro_1h"]
        summary += (
            f"4. **Intraday Supplement (1H Micro)**:\n"
            f"   - 1H 24-Bar Swing Range: Low = {m['1h_24bar_swing_low']} | High = {m['1h_24bar_swing_high']}\n"
            f"   - 1H Momentum EMAs: 20 EMA = {m['1h_ema_20']} | 50 EMA = {m['1h_ema_50']}\n"
            f"   - 1H Trend Alignment: {m['1h_trend_bias']}\n"
        )

    return summary
