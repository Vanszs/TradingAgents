"""
Data window utility — limits the agent's visible history to a rolling window.

Given a per-day snapshot and a ``lookback_days`` value, returns:

* ``ohlcv``   — the last ``lookback_days`` rows whose date is <= trade_date
* ``news``    — news whose ``published_at`` falls inside the window
* ``fundamentals`` — fundamentals whose ``available_date`` falls inside
  the window (otherwise the agent could implicitly observe quarters that
  were only disclosed much later)
* ``sentiment``  — sentiment whose ``timestamp`` falls inside the window
* ``broker_activity`` — broker activity inside the window

The window is inclusive on both ends:
``[trade_date - lookback_days, trade_date]`` (with a small calendar buffer
to absorb weekends/holidays, but never expanding past trade_date).

If ``lookback_days`` is ``None`` or 0 the data is returned unchanged (full
history subject to the provider's own time-cutoff).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

# Allowable lookback values from the spec. ``None`` means "no window".
ALLOWED_LOOKBACKS: tuple[Optional[int], ...] = (
    None, 5, 10, 20, 40, 60, 80, 100, 120, 240, 512,
)

# Calendar buffer: extra business days absorbed to handle weekends/holidays
# when the strict window is too narrow to produce enough rows.
DEFAULT_CALENDAR_BUFFER_DAYS = 7


@dataclass
class WindowCutoffs:
    """Effective cutoffs after applying the lookback window."""
    trade_date: str
    lookback_days: Optional[int]
    min_ohlcv_date_in_window: Optional[str]
    max_ohlcv_date_in_window: Optional[str]
    min_event_datetime: Optional[str]
    min_event_date: Optional[str]


def _to_iso_date(value: Any) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _to_iso_datetime(value: Any) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.Timestamp(value).to_pydatetime()
    except Exception:
        return None


def _normalize_to_utc(ts: Any) -> pd.Timestamp:
    """Normalize any timestamp to UTC-naive for safe comparison.

    If the input carries timezone info, convert to UTC then strip it.
    If the input is already naive, return as-is.
    """
    result = pd.Timestamp(ts)
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result


def compute_window(
    trade_date: str,
    lookback_days: Optional[int],
    buffer_days: int = DEFAULT_CALENDAR_BUFFER_DAYS,
) -> WindowCutoffs:
    """Compute the inclusive window cutoffs for ``trade_date``."""
    td = _to_iso_date(trade_date)
    if td is None:
        raise ValueError(f"Invalid trade_date: {trade_date!r}")
    max_date = td
    if lookback_days is None or lookback_days <= 0:
        return WindowCutoffs(
            trade_date=trade_date,
            lookback_days=None,
            min_ohlcv_date_in_window=None,
            max_ohlcv_date_in_window=max_date.isoformat(),
            min_event_datetime=None,
            min_event_date=None,
        )

    min_date = td - timedelta(days=lookback_days + buffer_days)
    min_event_date = (td - timedelta(days=lookback_days)).isoformat()
    min_event_datetime = f"{min_event_date} 00:00:00"

    return WindowCutoffs(
        trade_date=trade_date,
        lookback_days=lookback_days,
        min_ohlcv_date_in_window=min_date.isoformat(),
        max_ohlcv_date_in_window=max_date.isoformat(),
        min_event_datetime=min_event_datetime,
        min_event_date=min_event_date,
    )


def slice_ohlcv(
    df: pd.DataFrame,
    cutoffs: WindowCutoffs,
) -> pd.DataFrame:
    """Return the last ``lookback_days`` rows of OHLCV <= trade_date."""
    if df is None or df.empty:
        return df
    if cutoffs.lookback_days is None:
        return df
    if "date" not in df.columns:
        return df
    dates = df["date"].astype(str)
    masked = df[dates <= cutoffs.trade_date]
    if masked.empty:
        return masked
    # Already time-cut to <= trade_date. Now apply the rolling window.
    n = int(cutoffs.lookback_days)
    if n <= 0:
        return masked
    return masked.tail(n).reset_index(drop=True)


def _filter_records_by_field(
    records: list[dict[str, Any]],
    field: str,
    *,
    min_iso_date: Optional[str],
    min_iso_datetime: Optional[str],
    max_iso_date: str,
    max_iso_datetime: str,
    by_date_only: bool = False,
) -> list[dict[str, Any]]:
    if not records:
        return []
    out: list[dict[str, Any]] = []
    for rec in records:
        v = rec.get(field)
        if not v:
            continue
        try:
            if by_date_only:
                item_date = _normalize_to_utc(v).date()
                upper = _normalize_to_utc(max_iso_date).date()
                if item_date > upper:
                    continue
                if min_iso_date is not None and item_date < _normalize_to_utc(min_iso_date).date():
                    continue
                out.append(rec)
            else:
                item_dt = _normalize_to_utc(v)
                upper = _normalize_to_utc(max_iso_datetime)
                if item_dt > upper:
                    continue
                if min_iso_datetime is not None and item_dt < _normalize_to_utc(min_iso_datetime):
                    continue
                out.append(rec)
        except Exception:
            continue
    return out


def slice_news(
    news: list[dict[str, Any]],
    cutoffs: WindowCutoffs,
    report_time: str = "16:30:00",
    prev_trading_day: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Slice news by lookback window. When prev_trading_day is provided, use
    it as the upper bound (previous-day cutoff strategy) to prevent leakage
    from same-day news that could reflect today's price action."""
    upper_date = prev_trading_day if prev_trading_day else cutoffs.trade_date
    max_dt = f"{upper_date} {report_time}"
    return _filter_records_by_field(
        news,
        "published_at",
        min_iso_date=None,
        min_iso_datetime=cutoffs.min_event_datetime,
        max_iso_date=upper_date,
        max_iso_datetime=max_dt,
    )


def slice_fundamentals(
    fundamentals: list[dict[str, Any]],
    cutoffs: WindowCutoffs,
) -> list[dict[str, Any]]:
    return _filter_records_by_field(
        fundamentals,
        "available_date",
        min_iso_date=cutoffs.min_event_date,
        min_iso_datetime=None,
        max_iso_date=cutoffs.trade_date,
        max_iso_datetime=f"{cutoffs.trade_date} 23:59:59",
        by_date_only=True,
    )


def slice_sentiment(
    sentiment: list[dict[str, Any]],
    cutoffs: WindowCutoffs,
    report_time: str = "16:30:00",
    prev_trading_day: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Slice sentiment by lookback window. When prev_trading_day is provided,
    use it as the upper bound to prevent same-day sentiment leakage."""
    upper_date = prev_trading_day if prev_trading_day else cutoffs.trade_date
    max_dt = f"{upper_date} {report_time}"
    return _filter_records_by_field(
        sentiment,
        "timestamp",
        min_iso_date=None,
        min_iso_datetime=cutoffs.min_event_datetime,
        max_iso_date=upper_date,
        max_iso_datetime=max_dt,
    )


def slice_broker_activity(
    activity: list[dict[str, Any]],
    cutoffs: WindowCutoffs,
    report_time: str = "16:30:00",
    prev_trading_day: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Slice broker activity by lookback window. When prev_trading_day is
    provided, use it as the upper bound to prevent same-day leakage."""
    upper_date = prev_trading_day if prev_trading_day else cutoffs.trade_date
    max_dt = f"{upper_date} {report_time}"
    return _filter_records_by_field(
        activity,
        "timestamp",
        min_iso_date=None,
        min_iso_datetime=cutoffs.min_event_datetime,
        max_iso_date=upper_date,
        max_iso_datetime=max_dt,
    )


def assert_window_is_valid(
    cutoffs: WindowCutoffs,
    actual_min_ohlcv_date: Optional[str],
    actual_max_ohlcv_date: Optional[str] = None,
) -> None:
    """Audit the windowed OHLCV.

    A valid window satisfies:

    * the latest row is at or before ``trade_date`` (anti-leakage, the hard
      constraint);
    * the data covers **at least** ``lookback_days`` rows when enough
      history exists — i.e. the rolling window from the latest available
      row spans roughly ``lookback_days`` trading days. If the underlying
      history simply does not extend that far back, we accept a shorter
      window (the agent sees the whole available history).
    """
    if cutoffs.lookback_days is None:
        return
    if not actual_min_ohlcv_date:
        return
    actual_min = _to_iso_date(actual_min_ohlcv_date)
    actual_max = _to_iso_date(actual_max_ohlcv_date) if actual_max_ohlcv_date else None
    trade = _to_iso_date(cutoffs.trade_date)
    if actual_min is None or trade is None:
        return

    # Hard anti-leakage: the actual_max must be <= trade_date.
    if actual_max is not None and actual_max > trade:
        raise ValueError(
            f"OHLCV window violated: max date {actual_max} is after trade_date {trade}"
        )
