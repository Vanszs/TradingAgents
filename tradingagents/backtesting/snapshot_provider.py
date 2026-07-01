"""
Snapshot provider for stock backtests.

Flat data layout:
    <data_root>/<TICKER>/
        ohlcv.csv          -- date,open,high,low,close,volume
        news.json          -- optional
        fundamentals.json  -- optional
        sentiment.json     -- optional
        broker_activity.json -- optional

Cutoff guarantees:
- OHLCV cut at trade_date inclusive
- News/fundamental/sentiment/broker data cut by their respective timestamps

Window guarantees (when ``lookback_days`` is set):
- OHLCV further restricted to the last ``lookback_days`` rows
- News/sentiment/broker_activity restricted to ``[trade_date - lookback_days, trade_date]``
- Fundamentals restricted by ``available_date`` to the same range
- ``SnapshotMetadata`` exposes the actual min/max dates inside the window for audit
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

from .data_window import (
    ALLOWED_LOOKBACKS,
    WindowCutoffs,
    assert_window_is_valid,
    compute_window,
    slice_broker_activity,
    slice_fundamentals,
    slice_news,
    slice_ohlcv,
    slice_sentiment,
)
from .decision_schema import (
    InstrumentSpec,
    MarketPoint,
    SnapshotMetadata,
    ensure_dir,
)


DateLike = Union[str, int, float, date, datetime, pd.Timestamp]


@dataclass
class DataSnapshot:
    symbol: str
    trade_date: str
    root_path: Path
    metadata: SnapshotMetadata
    ohlcv: pd.DataFrame
    news: list[dict[str, Any]]
    fundamentals: list[dict[str, Any]]
    sentiment: list[dict[str, Any]]
    broker_activity: list[dict[str, Any]]
    spec: InstrumentSpec
    lookback_days: Optional[int] = None
    window_min_ohlcv_date: Optional[str] = None
    window_max_ohlcv_date: Optional[str] = None


class SnapshotDataProvider:
    """Provider for stock backtests with flat data layout.

    When ``fetch_from_api=True``, OHLCV, news, and fundamentals are
    fetched from Yahoo Finance on first access and cached to disk
    (``api_cache_dir/<TICKER>/``).  Subsequent calls read from cache,
    preserving reproducibility.  Sentiment data cannot be fetched from
    APIs because social-media platforms do not provide historical data
    for arbitrary dates.
    """

    def __init__(
        self,
        data_root: str = "data",
        snapshot_root: str = "snapshots",
        report_time: str = "16:30:00",
        lookback_days: Optional[int] = None,
        news_cutoff_strategy: str = "previous_day",
        sentiment_cutoff_strategy: str = "previous_day",
        broker_activity_cutoff_strategy: str = "previous_day",
        fundamental_buffer_days: int = 3,
        fetch_from_api: bool = False,
        api_cache_dir: str = "api_cache",
    ):
        self.data_root = Path(data_root)
        self.snapshot_root = Path(snapshot_root)
        self.report_time = report_time
        self.lookback_days = lookback_days
        self.news_cutoff_strategy = news_cutoff_strategy
        self.sentiment_cutoff_strategy = sentiment_cutoff_strategy
        self.broker_activity_cutoff_strategy = broker_activity_cutoff_strategy
        self.fundamental_buffer_days = fundamental_buffer_days
        self.fetch_from_api = fetch_from_api
        self.api_cache_dir = Path(api_cache_dir)
        if lookback_days is not None and lookback_days not in ALLOWED_LOOKBACKS:
            raise ValueError(
                f"lookback_days must be one of {ALLOWED_LOOKBACKS}, got {lookback_days!r}"
            )
        # Cache for OHLCV DataFrames to avoid repeated CSV reads
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        # Cache for fetched API data to avoid repeated fetches
        self._news_cache: dict[str, list[dict]] = {}
        self._fundamentals_cache: dict[str, list[dict]] = {}

    def _symbol_dir(self, symbol: str) -> Path:
        return self.data_root / symbol.upper()

    @staticmethod
    def _normalize_date(value: Any) -> str:
        return str(value)[:10]

    def _get_previous_trading_day(self, symbol: str, trade_date: str) -> str:
        """Find the previous trading day from OHLCV data.

        Uses the actual OHLCV dates so holidays are handled correctly.
        Falls back to calendar heuristic if the date is not in OHLCV.
        """
        trade_date = self._normalize_date(trade_date)
        try:
            df = self.load_full_ohlcv(symbol)
            dates = sorted(df["date"].astype(str).tolist())
            idx = dates.index(trade_date)
            if idx > 0:
                return dates[idx - 1]
        except (FileNotFoundError, ValueError):
            pass
        # Fallback: step back 1 calendar day, skip weekends
        d = pd.Timestamp(trade_date) - pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d -= pd.Timedelta(days=1)
        return d.strftime("%Y-%m-%d")

    @staticmethod
    def _safe_timestamp(value: Any) -> pd.Timestamp:
        if value is None:
            raise ValueError("Date value cannot be None.")
        if pd.isna(value):
            raise ValueError("Date value cannot be NA/NaN.")
        if isinstance(value, pd.Timestamp):
            return value.normalize()
        if isinstance(value, datetime):
            return pd.Timestamp(value).normalize()
        if isinstance(value, date):
            return pd.Timestamp(value).normalize()
        if isinstance(value, str):
            return pd.Timestamp(value).normalize()
        if isinstance(value, (int, float)):
            return pd.Timestamp(value).normalize()
        return pd.Timestamp(str(value)).normalize()

    # ------------------------------------------------------------------
    # Instrument spec loading
    # ------------------------------------------------------------------
    def load_instrument_spec(
        self,
        symbol: str,
        ohlcv_df: Optional[pd.DataFrame] = None,
    ) -> InstrumentSpec:
        """Load or auto-generate an InstrumentSpec from OHLCV data."""
        if ohlcv_df is None:
            ohlcv_df = self.load_full_ohlcv(symbol)
        return InstrumentSpec.auto_from_ohlcv(symbol, ohlcv_df)

    # ------------------------------------------------------------------
    # OHLCV / bar access
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        out = df.rename(columns=rename_map).copy()
        required = {"date", "open", "high", "low", "close"}
        missing = required - set(out.columns)
        if missing:
            raise ValueError(f"OHLCV missing required columns: {sorted(missing)}")
        out.loc[:, "date"] = out["date"].map(
            lambda v: SnapshotDataProvider._safe_timestamp(v).date().isoformat()
        )
        for col in ("open", "high", "low", "close", "volume"):
            if col in out.columns:
                out.loc[:, col] = pd.to_numeric(out[col], errors="coerce")
        out = out.sort_values("date").reset_index(drop=True)
        return pd.DataFrame(out)

    def load_full_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Load full OHLCV data, with caching and optional API fetch."""
        sym = symbol.upper()
        if sym in self._ohlcv_cache:
            return self._ohlcv_cache[sym]

        if self.fetch_from_api:
            raw = self._fetch_and_cache_ohlcv(symbol)
        else:
            path = self._symbol_dir(symbol) / "ohlcv.csv"
            if not path.exists():
                raise FileNotFoundError(f"OHLCV file for {symbol} not found at {path}")
            raw = pd.read_csv(path)

        df = self._normalize_ohlcv_columns(raw)
        self._ohlcv_cache[sym] = df
        return df

    def _fetch_and_cache_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV from yfinance and cache to disk."""
        import yfinance as yf

        cache_path = self.api_cache_dir / symbol.upper() / "ohlcv.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)

        logger.info(f"[SNAPSHOT] Fetching OHLCV for {symbol} from yfinance...")
        ticker = yf.Ticker(symbol.upper())
        data = ticker.history(period="max")
        if data.empty:
            raise FileNotFoundError(f"No OHLCV data from yfinance for {symbol}")

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data = data.reset_index()
        # yfinance may name the date column "Date" or "index" depending on version
        date_col = None
        for candidate in ("Date", "date", "index", data.columns[0]):
            if candidate in data.columns:
                date_col = candidate
                break
        if date_col is None:
            date_col = data.columns[0]

        data = data.rename(columns={
            date_col: "date",
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["date", "open", "high", "low", "close", "volume"]
        data[cols].to_csv(cache_path, index=False)
        logger.info(f"[SNAPSHOT] Cached {len(data)} OHLCV bars to {cache_path}")
        return data[cols]

    def _fetch_and_cache_news(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch news from yfinance and cache to disk."""
        import yfinance as yf

        sym = symbol.upper()
        if sym in self._news_cache:
            return self._news_cache[sym]

        cache_path = self.api_cache_dir / sym / "news.json"
        if cache_path.exists():
            records = self._load_json_records(cache_path)
            self._news_cache[sym] = records
            return records

        logger.info(f"[SNAPSHOT] Fetching news for {symbol} from yfinance...")
        try:
            ticker = yf.Ticker(sym)
            raw_news = ticker.get_news(count=100)
        except Exception as exc:
            logger.warning(f"[SNAPSHOT] Failed to fetch news for {symbol}: {exc}")
            return []

        articles = []
        for article in raw_news or []:
            content = article.get("content", article)
            pub_date = content.get("pubDate", content.get("providerPublishTime", ""))
            articles.append({
                "title": content.get("title", ""),
                "summary": content.get("summary", content.get("description", "")),
                "source": (
                    content.get("provider", {}).get("displayName", "Unknown")
                    if isinstance(content.get("provider"), dict)
                    else str(content.get("provider", "Unknown"))
                ),
                "published_at": str(pub_date) if pub_date else "",
                "url": (
                    (content.get("canonicalUrl") or {}).get("url", "")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else ""
                ),
            })

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(cache_path, articles)
        self._news_cache[sym] = articles
        logger.info(f"[SNAPSHOT] Cached {len(articles)} news articles to {cache_path}")
        return articles

    def _fetch_and_cache_fundamentals(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch fundamentals from yfinance and cache to disk."""
        import yfinance as yf

        sym = symbol.upper()
        if sym in self._fundamentals_cache:
            return self._fundamentals_cache[sym]

        cache_path = self.api_cache_dir / sym / "fundamentals.json"
        if cache_path.exists():
            records = self._load_json_records(cache_path)
            self._fundamentals_cache[sym] = records
            return records

        logger.info(f"[SNAPSHOT] Fetching fundamentals for {symbol} from yfinance...")
        ticker = yf.Ticker(sym)
        records: list[dict[str, Any]] = []

        # Company info
        try:
            info = ticker.info or {}
            skip_keys = {"companyOfficers", "address1", "address2", "city", "state", "zip", "country", "phone", "website"}
            for key, value in info.items():
                if value is not None and key not in skip_keys:
                    records.append({
                        "metric": key,
                        "value": value,
                        "available_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                        "period": "latest",
                        "source": "yfinance_info",
                    })
        except Exception as exc:
            logger.warning(f"[SNAPSHOT] Failed to fetch info for {symbol}: {exc}")

        # Quarterly financial statements
        for stmt_name, stmt_attr in [
            ("balance_sheet", "quarterly_balance_sheet"),
            ("income_statement", "quarterly_income_stmt"),
            ("cashflow", "quarterly_cashflow"),
        ]:
            try:
                stmt = getattr(ticker, stmt_attr, None)
                if stmt is not None and not stmt.empty:
                    for col in stmt.columns:
                        date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
                        for idx in stmt.index:
                            val = stmt.loc[idx, col]
                            if pd.notna(val):
                                records.append({
                                    "metric": f"{stmt_name}:{idx}",
                                    "value": float(val) if isinstance(val, (int, float)) else str(val),
                                    "available_date": date_str,
                                    "period": "quarterly",
                                    "source": f"yfinance_{stmt_attr}",
                                })
            except Exception as exc:
                logger.warning(f"[SNAPSHOT] Failed to fetch {stmt_name} for {symbol}: {exc}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(cache_path, records)
        self._fundamentals_cache[sym] = records
        logger.info(f"[SNAPSHOT] Cached {len(records)} fundamental records to {cache_path}")
        return records

    def get_ohlcv(self, symbol: str, trade_date: str) -> pd.DataFrame:
        trade_date = self._normalize_date(trade_date)
        df = self.load_full_ohlcv(symbol)
        cut = df[df["date"].astype(str) <= trade_date].copy()
        if cut.empty:
            raise ValueError(f"No OHLCV data available for {symbol} up to {trade_date}")
        max_date = str(cut["date"].astype(str).max())
        if max_date > trade_date:
            raise RuntimeError("Future OHLCV leakage detected.")
        return cut.reset_index(drop=True)

    def get_market_point(
        self,
        symbol: str,
        trade_date: str,
        spec: InstrumentSpec,
    ) -> MarketPoint:
        trade_date = self._normalize_date(trade_date)
        df = self.load_full_ohlcv(symbol).copy()
        df["date"] = df["date"].astype(str)
        row = df[df["date"] == trade_date]
        if row.empty:
            raise ValueError(f"No market point for {symbol} on {trade_date}")
        r = row.iloc[0]
        return MarketPoint(
            date=str(r["date"]),
            ticker=symbol,
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r.get("volume", 0.0) or 0.0),
        )

    def get_market_dates(self, symbol: str) -> list[str]:
        df = self.load_full_ohlcv(symbol).copy()
        date_series = df["date"].astype(str)
        return sorted([str(v) for v in date_series.tolist()])

    # ------------------------------------------------------------------
    # News / fundamentals / sentiment / broker activity
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return [item for item in data["data"] if isinstance(item, dict)]
            return [data]
        return []

    @staticmethod
    def _cut_by_datetime(
        records: list[dict[str, Any]],
        field: str,
        cutoff: str,
    ) -> list[dict[str, Any]]:
        if not records:
            return []
        from .data_window import _normalize_to_utc
        cutoff_ts = _normalize_to_utc(cutoff)
        out: list[dict[str, Any]] = []
        for item in records:
            v = item.get(field)
            if not v:
                continue
            ts = _normalize_to_utc(v)
            if ts <= cutoff_ts:
                out.append(item)
        return out

    @staticmethod
    def _cut_by_date(
        records: list[dict[str, Any]],
        field: str,
        cutoff_date: str,
    ) -> list[dict[str, Any]]:
        if not records:
            return []
        from .data_window import _normalize_to_utc
        cutoff = _normalize_to_utc(cutoff_date).date()
        out: list[dict[str, Any]] = []
        for item in records:
            v = item.get(field)
            if not v:
                continue
            item_date = _normalize_to_utc(v).date()
            if item_date <= cutoff:
                out.append(item)
        return out

    def get_news(self, symbol: str, trade_date: str) -> list[dict[str, Any]]:
        trade_date = self._normalize_date(trade_date)
        if self.fetch_from_api:
            records = self._fetch_and_cache_news(symbol)
        else:
            path = self._symbol_dir(symbol) / "news.json"
            records = self._load_json_records(path)
        if self.news_cutoff_strategy == "previous_day":
            prev_day = self._get_previous_trading_day(symbol, trade_date)
            cutoff = f"{prev_day} {self.report_time}"
        else:
            cutoff = f"{trade_date} {self.report_time}"
        return self._cut_by_datetime(records, "published_at", cutoff)

    def get_fundamentals(self, symbol: str, trade_date: str) -> list[dict[str, Any]]:
        trade_date = self._normalize_date(trade_date)
        if self.fetch_from_api:
            records = self._fetch_and_cache_fundamentals(symbol)
        else:
            path = self._symbol_dir(symbol) / "fundamentals.json"
            records = self._load_json_records(path)
        if self.fundamental_buffer_days > 0:
            cutoff_date = (
                pd.Timestamp(trade_date) - pd.Timedelta(days=self.fundamental_buffer_days)
            ).strftime("%Y-%m-%d")
        else:
            cutoff_date = trade_date
        return self._cut_by_date(records, "available_date", cutoff_date)

    def get_sentiment(self, symbol: str, trade_date: str) -> list[dict[str, Any]]:
        """Get sentiment data for the given symbol and trade date.

        Note: Even when ``fetch_from_api=True``, sentiment data is NOT
        fetched from live APIs.  Social-media platforms (StockTwits, Reddit,
        Bluesky, Mastodon) do not provide historical data for arbitrary
        dates — live calls would return today's sentiment for every
        historical trade date, leaking future information and destroying
        reproducibility.  Users must pre-populate ``sentiment.json`` or
        accept empty sentiment data in backtests.
        """
        trade_date = self._normalize_date(trade_date)
        path = self._symbol_dir(symbol) / "sentiment.json"
        records = self._load_json_records(path)
        if self.sentiment_cutoff_strategy == "previous_day":
            prev_day = self._get_previous_trading_day(symbol, trade_date)
            cutoff = f"{prev_day} {self.report_time}"
        else:
            cutoff = f"{trade_date} {self.report_time}"
        return self._cut_by_datetime(records, "timestamp", cutoff)

    def get_broker_activity(self, symbol: str, trade_date: str) -> list[dict[str, Any]]:
        trade_date = self._normalize_date(trade_date)
        path = self._symbol_dir(symbol) / "broker_activity.json"
        records = self._load_json_records(path)
        if self.broker_activity_cutoff_strategy == "previous_day":
            prev_day = self._get_previous_trading_day(symbol, trade_date)
            cutoff = f"{prev_day} {self.report_time}"
        else:
            cutoff = f"{trade_date} {self.report_time}"
        return self._cut_by_datetime(records, "timestamp", cutoff)

    # ------------------------------------------------------------------
    # Snapshot creation
    # ------------------------------------------------------------------
    @staticmethod
    def _max_value(records: list[dict[str, Any]], field: str) -> Optional[str]:
        from .data_window import _normalize_to_utc
        timestamps: list[pd.Timestamp] = []
        for item in records:
            v = item.get(field)
            if v is None or pd.isna(v):
                continue
            timestamps.append(_normalize_to_utc(v))
        if not timestamps:
            return None
        return max(timestamps).isoformat(sep=" ")

    @staticmethod
    def _max_date_value(records: list[dict[str, Any]], field: str) -> Optional[str]:
        from .data_window import _normalize_to_utc
        dates: list[str] = []
        for item in records:
            v = item.get(field)
            if v is None or pd.isna(v):
                continue
            dates.append(
                _normalize_to_utc(v).date().isoformat()
            )
        if not dates:
            return None
        return max(dates)

    @staticmethod
    def _max_date_in_series(series: pd.Series) -> Optional[str]:
        if series.empty:
            return None
        return str(series.max())

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def create_snapshot(
        self,
        symbol: str,
        trade_date: str,
        config: Any = None,
        lookback_days: Optional[int] = None,
    ) -> DataSnapshot:
        trade_date = self._normalize_date(trade_date)

        # Resolve the effective lookback (caller can override provider default).
        effective_lookback = (
            lookback_days if lookback_days is not None else self.lookback_days
        )
        if effective_lookback is not None and effective_lookback not in ALLOWED_LOOKBACKS:
            raise ValueError(
                f"lookback_days must be one of {ALLOWED_LOOKBACKS}, got {effective_lookback!r}"
            )

        # Pull full time-cut data first (always <= trade_date).
        # For news/sentiment/broker with previous_day strategy, get_* already
        # applies the previous-day cutoff. We compute prev_trading_day once
        # so the slice functions can use the same upper bound.
        prev_trading_day = self._get_previous_trading_day(symbol, trade_date)

        full_ohlcv = self.get_ohlcv(symbol, trade_date)
        full_news = self.get_news(symbol, trade_date)
        full_fundamentals = self.get_fundamentals(symbol, trade_date)
        full_sentiment = self.get_sentiment(symbol, trade_date)
        full_broker = self.get_broker_activity(symbol, trade_date)

        # Apply the lookback window if requested.
        cutoffs = compute_window(trade_date, effective_lookback)
        ohlcv = slice_ohlcv(full_ohlcv, cutoffs)
        news = slice_news(
            full_news, cutoffs, report_time=self.report_time,
            prev_trading_day=prev_trading_day if self.news_cutoff_strategy == "previous_day" else None,
        )
        fundamentals = slice_fundamentals(full_fundamentals, cutoffs)
        sentiment = slice_sentiment(
            full_sentiment, cutoffs, report_time=self.report_time,
            prev_trading_day=prev_trading_day if self.sentiment_cutoff_strategy == "previous_day" else None,
        )
        broker_activity = slice_broker_activity(
            full_broker, cutoffs, report_time=self.report_time,
            prev_trading_day=prev_trading_day if self.broker_activity_cutoff_strategy == "previous_day" else None,
        )

        # Audit: hard anti-leakage check (actual max must be <= trade_date).
        actual_min_ohlcv_date = (
            str(ohlcv["date"].astype(str).min()) if not ohlcv.empty else None
        )
        actual_max_ohlcv_date = (
            str(ohlcv["date"].astype(str).max()) if not ohlcv.empty else None
        )
        assert_window_is_valid(
            cutoffs,
            actual_min_ohlcv_date,
            actual_max_ohlcv_date=actual_max_ohlcv_date,
        )

        spec = self.load_instrument_spec(symbol, ohlcv)

        snapshot_dir = ensure_dir(
            self.snapshot_root / symbol / trade_date
        )

        ohlcv.to_csv(snapshot_dir / "ohlcv.csv", index=False)
        self._write_json(snapshot_dir / "news.json", news)
        self._write_json(snapshot_dir / "fundamentals.json", fundamentals)
        self._write_json(snapshot_dir / "sentiment.json", sentiment)
        self._write_json(snapshot_dir / "broker_activity.json", broker_activity)

        ohlcv_dates = ohlcv["date"].astype(str)
        max_ohlcv_date = self._max_date_in_series(ohlcv_dates)
        min_ohlcv_date = (
            self._max_date_in_series(ohlcv_dates) and
            str(ohlcv_dates.min())
        )
        # Re-derive min explicitly (the expression above is a guard, not a value).
        min_ohlcv_date = str(ohlcv_dates.min()) if not ohlcv.empty else None

        metadata = SnapshotMetadata(
            ticker=symbol,
            trade_date=trade_date,
            snapshot_created_at=f"{trade_date} {self.report_time}",
            max_ohlcv_date=max_ohlcv_date,
            max_news_time=self._max_value(news, "published_at"),
            max_fundamental_available_date=self._max_date_value(
                fundamentals, "available_date"
            ),
            max_sentiment_time=self._max_value(sentiment, "timestamp"),
            max_broker_activity_time=self._max_value(broker_activity, "timestamp"),
            provider_mode="snapshot",
            path=str(snapshot_dir),
        )

        # Augment metadata with window audit fields.
        meta_dict = metadata.to_dict()
        meta_dict["lookback_days"] = effective_lookback
        meta_dict["window_min_ohlcv_date"] = min_ohlcv_date
        meta_dict["window_max_ohlcv_date"] = max_ohlcv_date
        meta_dict["window_expected_min_ohlcv_date"] = cutoffs.min_ohlcv_date_in_window
        self._write_json(snapshot_dir / "metadata.json", meta_dict)

        return DataSnapshot(
            symbol=symbol,
            trade_date=trade_date,
            root_path=snapshot_dir,
            metadata=metadata,
            ohlcv=ohlcv,
            news=news,
            fundamentals=fundamentals,
            sentiment=sentiment,
            broker_activity=broker_activity,
            spec=spec,
            lookback_days=effective_lookback,
            window_min_ohlcv_date=min_ohlcv_date,
            window_max_ohlcv_date=max_ohlcv_date,
        )
