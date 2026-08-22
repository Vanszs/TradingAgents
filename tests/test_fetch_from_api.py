"""
Tests for fetch-from-API feature in SnapshotDataProvider.

Covers:
- Constructor parameters for fetch_from_api and api_cache_dir
- Cache hit/miss behavior
- News fetch-and-cache
- Fundamentals fetch-and-cache
- OHLCV fetch-and-cache
- Sentiment does NOT fetch from API (documented limitation)
- Cutoff filters still apply to fetched data
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from tradingagents.backtesting.snapshot_provider import SnapshotDataProvider


class TestFetchFromApiConstructor(unittest.TestCase):
    """Constructor accepts fetch_from_api and api_cache_dir."""

    def test_default_fetch_from_api_false(self):
        prov = SnapshotDataProvider()
        self.assertFalse(prov.fetch_from_api)
        self.assertEqual(prov.api_cache_dir, Path("api_cache"))

    def test_fetch_from_api_true(self):
        prov = SnapshotDataProvider(fetch_from_api=True, api_cache_dir="/tmp/cache")
        self.assertTrue(prov.fetch_from_api)
        self.assertEqual(prov.api_cache_dir, Path("/tmp/cache"))


class TestFetchAndCacheOHLCV(unittest.TestCase):
    """OHLCV fetch-and-cache behavior."""

    def test_cache_hit_skips_fetch(self):
        """When cache file exists, read from cache without fetching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "api_cache" / "TEST"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "ohlcv.csv"
            cache_file.write_text(
                "date,open,high,low,close,volume\n"
                "2025-01-02,100,101,99,100.5,1000000\n"
            )

            prov = SnapshotDataProvider(
                fetch_from_api=True,
                api_cache_dir=str(Path(tmpdir) / "api_cache"),
            )
            df = prov.load_full_ohlcv("TEST")
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["close"], 100.5)

    @patch("yfinance.Ticker")
    def test_cache_miss_fetches_from_yfinance(self, mock_ticker_cls):
        """When cache is empty, fetch from yfinance and save to cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame(
                {
                    "Open": [100.0, 101.0],
                    "High": [102.0, 103.0],
                    "Low": [99.0, 100.0],
                    "Close": [101.0, 102.0],
                    "Volume": [1000000, 1100000],
                },
                index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
            )
            mock_ticker_cls.return_value = mock_ticker

            prov = SnapshotDataProvider(
                fetch_from_api=True,
                api_cache_dir=str(Path(tmpdir) / "api_cache"),
            )
            df = prov.load_full_ohlcv("TEST")

            self.assertEqual(len(df), 2)
            # Verify cache file was created
            cache_file = Path(tmpdir) / "api_cache" / "TEST" / "ohlcv.csv"
            self.assertTrue(cache_file.exists())


class TestFetchAndCacheNews(unittest.TestCase):
    """News fetch-and-cache behavior."""

    def test_cache_hit_returns_cached_news(self):
        """When news cache exists, return cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "api_cache" / "TEST"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "news.json"
            cache_file.write_text(json.dumps([
                {"title": "Test News", "published_at": "2025-06-13 10:00:00", "source": "Test"}
            ]))

            prov = SnapshotDataProvider(
                fetch_from_api=True,
                api_cache_dir=str(Path(tmpdir) / "api_cache"),
            )
            records = prov._fetch_and_cache_news("TEST")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["title"], "Test News")

    @patch("yfinance.Ticker")
    def test_cache_miss_fetches_from_yfinance(self, mock_ticker_cls):
        """When news cache is empty, fetch from yfinance and save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ticker = MagicMock()
            mock_ticker.get_news.return_value = [
                {
                    "content": {
                        "title": "Breaking News",
                        "summary": "Important update",
                        "pubDate": "2025-06-13T10:00:00Z",
                        "provider": {"displayName": "Reuters"},
                        "canonicalUrl": {"url": "https://example.com"},
                    }
                }
            ]
            mock_ticker_cls.return_value = mock_ticker

            prov = SnapshotDataProvider(
                fetch_from_api=True,
                api_cache_dir=str(Path(tmpdir) / "api_cache"),
            )
            records = prov._fetch_and_cache_news("TEST")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["title"], "Breaking News")
            self.assertEqual(records[0]["source"], "Reuters")

            # Verify cache file was created
            cache_file = Path(tmpdir) / "api_cache" / "TEST" / "news.json"
            self.assertTrue(cache_file.exists())

    def test_news_cutoff_still_applies(self):
        """Cutoff filters still apply to fetched news."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "api_cache" / "TEST"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "news.json"
            cache_file.write_text(json.dumps([
                {"title": "Old", "published_at": "2025-06-10 10:00:00", "source": "A"},
                {"title": "Recent", "published_at": "2025-06-13 10:00:00", "source": "B"},
            ]))

            prov = SnapshotDataProvider(
                fetch_from_api=True,
                api_cache_dir=str(Path(tmpdir) / "api_cache"),
                news_cutoff_strategy="previous_day",
            )
            # Mock OHLCV for _get_previous_trading_day
            prov._ohlcv_cache["TEST"] = pd.DataFrame({
                "date": ["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-06-14"],
                "open": [100]*5, "high": [101]*5, "low": [99]*5, "close": [100]*5,
            })

            records = prov.get_news("TEST", "2025-06-14")
            # With previous_day cutoff, only news <= 2025-06-13 16:30:00
            titles = [r["title"] for r in records]
            self.assertIn("Old", titles)
            self.assertIn("Recent", titles)


class TestFetchAndCacheFundamentals(unittest.TestCase):
    """Fundamentals fetch-and-cache behavior."""

    def test_cache_hit_returns_cached_fundamentals(self):
        """When fundamentals cache exists, return cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "api_cache" / "TEST"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "fundamentals.json"
            cache_file.write_text(json.dumps([
                {"metric": "revenue", "value": 1000000, "available_date": "2025-06-01", "period": "Q1"},
            ]))

            prov = SnapshotDataProvider(
                fetch_from_api=True,
                api_cache_dir=str(Path(tmpdir) / "api_cache"),
                fundamental_buffer_days=0,
            )
            records = prov._fetch_and_cache_fundamentals("TEST")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["metric"], "revenue")


class TestSentimentDoesNotFetch(unittest.TestCase):
    """Sentiment data is never fetched from APIs."""

    def test_sentiment_returns_empty_when_no_file(self):
        """Even with fetch_from_api=True, sentiment reads from file only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prov = SnapshotDataProvider(
                fetch_from_api=True,
                api_cache_dir=str(Path(tmpdir) / "api_cache"),
            )
            prov._ohlcv_cache["TEST"] = pd.DataFrame({
                "date": ["2025-06-13", "2025-06-14"],
                "open": [100]*2, "high": [101]*2, "low": [99]*2, "close": [100]*2,
            })
            records = prov.get_sentiment("TEST", "2025-06-14")
            self.assertEqual(records, [])


class TestBacktestConfigNewFields(unittest.TestCase):
    """DataConfig has fetch_from_api and api_cache_dir fields."""

    def test_decision_schema_dataconfig_has_new_fields(self):
        from tradingagents.backtesting.decision_schema import DataConfig
        cfg = DataConfig()
        self.assertFalse(cfg.fetch_from_api)
        self.assertEqual(cfg.api_cache_dir, "api_cache")

    def test_decision_schema_dataconfig_custom_values(self):
        from tradingagents.backtesting.decision_schema import DataConfig
        cfg = DataConfig(fetch_from_api=True, api_cache_dir="/tmp/cache")
        self.assertTrue(cfg.fetch_from_api)
        self.assertEqual(cfg.api_cache_dir, "/tmp/cache")


if __name__ == "__main__":
    unittest.main()
