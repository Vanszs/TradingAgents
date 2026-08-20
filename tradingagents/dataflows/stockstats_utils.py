import logging
import os
import time
from typing import Annotated

import pandas as pd
import yfinance as yf
from stockstats import wrap
from yfinance.exceptions import YFRateLimitError

from .config import get_config
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    for col in price_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill()
    data = data.dropna(subset=price_cols)

    return data


def load_ohlcv(symbol: str, curr_date: str = None) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 15 years of data up to anchor date and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.
    """
    # Reject ticker values that would escape the cache directory when
    # interpolated into the cache filename (e.g. ``../../tmp/x``).
    safe_symbol = safe_ticker_component(symbol)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date) if curr_date else None

    # Cache uses a 15y window anchored to curr_date (or today)
    anchor_dt = pd.to_datetime(curr_date) if curr_date else pd.Timestamp.today()
    start_date = anchor_dt - pd.DateOffset(years=15)
    end_date = anchor_dt + pd.DateOffset(days=1)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    if curr_date_dt is not None:
        data = data[data["Date"] <= curr_date_dt]

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str, min_filing_lag_days: int = 45) -> pd.DataFrame:
    """Drop financial statement columns after curr_date with realistic filing lag buffer.

    yfinance financial statements use fiscal period end dates as columns.
    Enforces a realistic publication/filing lag buffer (default 45 days for quarterly)
    to prevent look-ahead bias before SEC 10-Q/10-K disclosures become public.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date) - pd.Timedelta(days=min_filing_lag_days)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate True Range and Average True Range (ATR) causally with zero lookahead.

    Uses Wilder's Exponential Moving Average (alpha = 1 / period) to compute ATR.
    """
    df = data.copy()
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_chandelier_exit(
    data: pd.DataFrame,
    period: int = 22,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Calculate Chandelier Exit bands causally with zero lookahead:
    Long Band = rolling_max(High, period) - multiplier * ATR(period)
    Short Band = rolling_min(Low, period) + multiplier * ATR(period)
    """
    df = data.copy()
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    atr = compute_atr(df, period=period)

    high_roll = high.rolling(window=period, min_periods=period).max()
    low_roll = low.rolling(window=period, min_periods=period).min()

    chandelier_long = high_roll - multiplier * atr
    chandelier_short = low_roll + multiplier * atr

    return pd.DataFrame(
        {
            "chandelier_long": chandelier_long,
            "chandelier_short": chandelier_short,
            "atr": atr,
        },
        index=df.index,
    )


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        indicator_lower = indicator.strip().lower()

        if indicator_lower in ("chandelier_long", "chandelier_short"):
            chan_df = compute_chandelier_exit(data, period=22, multiplier=3.0)
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
            data[indicator_lower] = chan_df[indicator_lower]
            curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")
            matching_rows = data[data["Date"].str.startswith(curr_date_str)]
            if not matching_rows.empty:
                return matching_rows[indicator_lower].values[0]
            return "N/A: Not a trading day (weekend or holiday)"

        if indicator_lower in ("atr_14", "atr_20"):
            period = int(indicator_lower.split("_")[1])
            atr_series = compute_atr(data, period=period)
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
            data[indicator_lower] = atr_series
            curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")
            matching_rows = data[data["Date"].str.startswith(curr_date_str)]
            if not matching_rows.empty:
                return matching_rows[indicator_lower].values[0]
            return "N/A: Not a trading day (weekend or holiday)"

        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
