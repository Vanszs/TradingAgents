"""
Helpers for fetching missing OHLCV data on demand.

The CLI uses these to download a ticker's daily history from yfinance when
the local ``data/<TICKER>/ohlcv.csv`` is missing. The downloaded file
covers a wide enough window so that the configured lookback window
(in trading days) is satisfied for every day in the backtest's
``[start_date, end_date]`` range.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# yfinance data goes back about 20 years for liquid tickers. We pad the
# start of the download by `LOOKBACK_BUFFER_DAYS` (in calendar days) so
# that even a 240-trading-day lookback window has enough history for
# every day inside the backtest range, with a comfortable safety margin.
LOOKBACK_BUFFER_DAYS = 365

# Conservative earliest date yfinance will return data for free.
YF_EARLIEST = "2010-01-01"


def _to_iso_date(value: str) -> str:
    """Normalize a date string to YYYY-MM-DD."""
    return str(value)[:10]


def compute_download_window(
    start_date: str,
    end_date: str,
    lookback_days: Optional[int] = None,
) -> tuple[str, str]:
    """
    Compute the (download_start, download_end) range to fetch from
    yfinance so the backtest can run with the requested lookback.

    The download always extends back from ``start_date`` by at least
    ``LOOKBACK_BUFFER_DAYS`` calendar days, plus the lookback (if known,
    in trading days) translated to calendar days (×7/5). The download end
    extends past ``end_date`` by a small buffer to account for timezone
    drift in yfinance's daily cut.
    """
    end_dt = datetime.strptime(_to_iso_date(end_date), "%Y-%m-%d").date()
    start_dt = datetime.strptime(_to_iso_date(start_date), "%Y-%m-%d").date()

    lookback_calendar = 0
    if lookback_days:
        # ~5/7 of trading days are calendar days; round up generously.
        lookback_calendar = int(lookback_days * 7 / 5) + 14

    pad_start = max(LOOKBACK_BUFFER_DAYS, lookback_calendar)
    pad_end = 7  # timezone / market-close buffer

    download_start = max(
        datetime.strptime(YF_EARLIEST, "%Y-%m-%d").date(),
        start_dt - timedelta(days=pad_start),
    )
    download_end = end_dt + timedelta(days=pad_end)
    return download_start.isoformat(), download_end.isoformat()


def fetch_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    output_root: str = "data",
    lookback_days: Optional[int] = None,
) -> Path:
    """
    Download ``ticker``'s daily OHLCV from yfinance and write it to
    ``<output_root>/<TICKER>/ohlcv.csv``. Also creates empty
    ``news.json``, ``fundamentals.json``, ``sentiment.json``, and
    ``broker_activity.json`` stubs so the snapshot provider can find
    them. Returns the path to the saved CSV.

    The download window is computed from the user's ``start_date``,
    ``end_date``, and ``lookback_days`` so the resulting CSV covers the
    full history the backtest will need.
    """
    import pandas as pd
    import yfinance as yf

    output_dir = Path(output_root) / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    dl_start, dl_end = compute_download_window(
        start_date, end_date, lookback_days
    )

    df = yf.download(
        tickers=ticker,
        start=dl_start,
        end=dl_end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} "
            f"between {dl_start} and {dl_end}. "
            f"Check the ticker symbol and try a wider date range."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    required_cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[required_cols]
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df.dropna(subset=["open", "high", "low", "close"])

    csv_path = output_dir / "ohlcv.csv"
    df.to_csv(csv_path, index=False)

    for filename in (
        "news.json",
        "fundamentals.json",
        "sentiment.json",
        "broker_activity.json",
    ):
        path = output_dir / filename
        if not path.exists():
            path.write_text("[]", encoding="utf-8")

    return csv_path


def ensure_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    output_root: str = "data",
    lookback_days: Optional[int] = None,
) -> tuple[Path, bool]:
    """
    Return ``(path, was_fetched)``. If the OHLCV file is missing or
    older than the requested window, download it via
    :func:`fetch_ohlcv`. Otherwise return the existing file path.

    The ``was_fetched`` flag is True when a new download happened —
    useful for printing a message in the CLI.
    """
    csv_path = Path(output_root) / ticker / "ohlcv.csv"

    needs_fetch = False
    if not csv_path.exists():
        needs_fetch = True
    else:
        # If the existing file doesn't reach ``end_date`` we re-fetch.
        # A corrupt file (no ``date`` column or unreadable) also triggers
        # a refetch.
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            if "date" not in df.columns or df.empty:
                needs_fetch = True
            else:
                last = str(df["date"].max())[:10]
                if last < _to_iso_date(end_date):
                    needs_fetch = True
        except Exception:  # noqa: BLE001
            needs_fetch = True

    if needs_fetch:
        fetch_ohlcv(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            output_root=output_root,
            lookback_days=lookback_days,
        )
        return csv_path, True

    return csv_path, False
