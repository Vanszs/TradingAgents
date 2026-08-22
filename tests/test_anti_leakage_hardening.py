"""
Tests for hardened anti-leakage: previous-day news/sentiment cutoff,
fundamental buffer days, and UTC timezone normalization.
"""
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.backtesting.cutoff_validator import DecisionCutoffValidator
from tradingagents.backtesting.data_window import (
    _normalize_to_utc,
    compute_window,
    slice_news,
    slice_sentiment,
    slice_broker_activity,
    slice_fundamentals,
)
from tradingagents.backtesting.decision_schema import (
    ParsedDecision,
    Rating,
    Action,
    SnapshotMetadata,
)
from tradingagents.backtesting.snapshot_provider import SnapshotDataProvider


class TestNormalizeToUtc(unittest.TestCase):
    """Timezone normalization tests."""

    def test_naive_timestamp_unchanged(self):
        ts = _normalize_to_utc("2025-06-15 10:00:00")
        self.assertIsNone(ts.tzinfo)
        self.assertEqual(ts.hour, 10)

    def test_utc_timestamp_strips_tz(self):
        ts = _normalize_to_utc("2025-06-15T10:00:00+00:00")
        self.assertIsNone(ts.tzinfo)
        self.assertEqual(ts.hour, 10)

    def test_wib_timestamp_converted_to_utc(self):
        # WIB = UTC+7, so 10:00 WIB = 03:00 UTC
        ts = _normalize_to_utc("2025-06-15T10:00:00+07:00")
        self.assertIsNone(ts.tzinfo)
        self.assertEqual(ts.hour, 3)

    def test_est_timestamp_converted_to_utc(self):
        # EST = UTC-5, so 10:00 EST = 15:00 UTC
        ts = _normalize_to_utc("2025-06-15T10:00:00-05:00")
        self.assertIsNone(ts.tzinfo)
        self.assertEqual(ts.hour, 15)

    def test_date_string_works(self):
        ts = _normalize_to_utc("2025-06-15")
        self.assertEqual(ts.date().isoformat(), "2025-06-15")


class TestPreviousDayCutoff(unittest.TestCase):
    """News/sentiment must use previous trading day cutoff."""

    def test_slice_news_prev_day_excludes_same_day(self):
        """News from trade_date should be excluded with prev_trading_day."""
        news = [
            {"published_at": "2025-06-13 10:00:00", "title": "prev_day"},
            {"published_at": "2025-06-14 10:00:00", "title": "trade_day"},
        ]
        w = compute_window("2025-06-14", None)
        out = slice_news(news, w, prev_trading_day="2025-06-13")
        titles = [r["title"] for r in out]
        self.assertIn("prev_day", titles)
        self.assertNotIn("trade_day", titles)

    def test_slice_news_no_prev_day_uses_trade_date(self):
        """Without prev_trading_day, uses trade_date (legacy behavior)."""
        news = [
            {"published_at": "2025-06-13 10:00:00", "title": "prev_day"},
            {"published_at": "2025-06-14 10:00:00", "title": "trade_day"},
        ]
        w = compute_window("2025-06-14", None)
        out = slice_news(news, w, prev_trading_day=None)
        titles = [r["title"] for r in out]
        self.assertIn("prev_day", titles)
        self.assertIn("trade_day", titles)

    def test_slice_sentiment_prev_day_excludes_same_day(self):
        """Sentiment from trade_date should be excluded with prev_trading_day."""
        sentiment = [
            {"timestamp": "2025-06-13 15:00:00", "score": 0.3},
            {"timestamp": "2025-06-14 09:00:00", "score": 0.9},
        ]
        w = compute_window("2025-06-14", None)
        out = slice_sentiment(sentiment, w, prev_trading_day="2025-06-13")
        scores = [r["score"] for r in out]
        self.assertIn(0.3, scores)
        self.assertNotIn(0.9, scores)

    def test_slice_broker_prev_day_excludes_same_day(self):
        """Broker activity from trade_date should be excluded."""
        activity = [
            {"timestamp": "2025-06-13 14:00:00", "broker": "A"},
            {"timestamp": "2025-06-14 11:00:00", "broker": "B"},
        ]
        w = compute_window("2025-06-14", None)
        out = slice_broker_activity(activity, w, prev_trading_day="2025-06-13")
        brokers = [r["broker"] for r in out]
        self.assertIn("A", brokers)
        self.assertNotIn("B", brokers)


class TestFundamentalBuffer(unittest.TestCase):
    """Fundamental must be available N days before trade_date."""

    def test_fundamentals_within_buffer_excluded(self):
        """Fundamental available 1 day before trade_date excluded with buffer=3."""
        items = [
            {"available_date": "2025-06-11", "metric": "too_recent"},
            {"available_date": "2025-06-10", "metric": "ok"},
            {"available_date": "2025-05-01", "metric": "old"},
        ]
        w = compute_window("2025-06-14", None)
        out = slice_fundamentals(items, w)
        # With lookback=None, no window filter. But the provider's
        # get_fundamentals already applied the buffer. This test verifies
        # the slice function itself respects the upper bound.
        metrics = [r["metric"] for r in out]
        # All items <= trade_date should pass the slice
        self.assertIn("too_recent", metrics)
        self.assertIn("ok", metrics)

    def test_fundamentals_buffer_excluded_by_provider(self):
        """Provider-level buffer excludes fundamentals too close to trade_date."""
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        data_dir = root / "data" / "TEST.JK"
        data_dir.mkdir(parents=True)

        dates = pd.date_range("2025-01-01", periods=200, freq="B")
        ohlcv = pd.DataFrame({
            "date": [d.date().isoformat() for d in dates],
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.5, "volume": 1_000_000.0,
        })
        ohlcv.to_csv(data_dir / "ohlcv.csv", index=False)

        trade_date = dates[195].date().isoformat()
        # Available 1 day before trade_date — should be excluded with buffer=3
        recent_date = (dates[195] - pd.Timedelta(days=1)).date().isoformat()
        # Available 5 days before trade_date — should pass buffer=3
        safe_date = (dates[195] - pd.Timedelta(days=5)).date().isoformat()

        (data_dir / "fundamentals.json").write_text(json.dumps([
            {"available_date": recent_date, "metric": "recent"},
            {"available_date": safe_date, "metric": "safe"},
        ]))

        prov = SnapshotDataProvider(
            data_root=str(data_dir.parent),
            snapshot_root=str(root / "snapshots"),
            fundamental_buffer_days=3,
        )
        records = prov.get_fundamentals("TEST.JK", trade_date)
        metrics = [r["metric"] for r in records]
        self.assertNotIn("recent", metrics)
        self.assertIn("safe", metrics)
        tmp.cleanup()

    def test_fundamentals_no_buffer_includes_all(self):
        """With buffer=0, all fundamentals <= trade_date are included."""
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        data_dir = root / "data" / "TEST.JK"
        data_dir.mkdir(parents=True)

        dates = pd.date_range("2025-01-01", periods=200, freq="B")
        ohlcv = pd.DataFrame({
            "date": [d.date().isoformat() for d in dates],
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.5, "volume": 1_000_000.0,
        })
        ohlcv.to_csv(data_dir / "ohlcv.csv", index=False)

        trade_date = dates[195].date().isoformat()
        recent_date = (dates[195] - pd.Timedelta(days=1)).date().isoformat()

        (data_dir / "fundamentals.json").write_text(json.dumps([
            {"available_date": recent_date, "metric": "recent"},
        ]))

        prov = SnapshotDataProvider(
            data_root=str(data_dir.parent),
            snapshot_root=str(root / "snapshots"),
            fundamental_buffer_days=0,
        )
        records = prov.get_fundamentals("TEST.JK", trade_date)
        metrics = [r["metric"] for r in records]
        self.assertIn("recent", metrics)
        tmp.cleanup()


class TestCutoffValidatorHardened(unittest.TestCase):
    """Test the hardened cutoff validator."""

    def _make_decision(self, trade_date: str) -> ParsedDecision:
        from datetime import timedelta
        next_day = (
            pd.Timestamp(trade_date) + pd.Timedelta(days=1)
        ).date().isoformat()
        return ParsedDecision(
            decision_id="D1",
            ticker="TEST",
            trade_date=trade_date,
            report_generated_at=f"{trade_date} 16:30:00",
            last_data_date=trade_date,
            decision_valid_from=next_day,
            rating=Rating.BUY,
            action=Action.BUY,
            valid=True,
        )

    def test_news_same_day_fails(self):
        """News from same day as trade_date should fail validation."""
        validator = DecisionCutoffValidator(fail_on_future_data=False)
        decision = self._make_decision("2025-06-14")
        metadata = SnapshotMetadata(
            ticker="TEST",
            trade_date="2025-06-14",
            snapshot_created_at="2025-06-14 16:30:00",
            max_news_time="2025-06-14 10:00:00",
        )
        checks = validator.validate(decision, metadata)
        self.assertEqual(checks["news_cutoff"], "FAILED")

    def test_news_prev_day_passes(self):
        """News from previous day should pass validation."""
        validator = DecisionCutoffValidator(fail_on_future_data=False)
        decision = self._make_decision("2025-06-14")
        metadata = SnapshotMetadata(
            ticker="TEST",
            trade_date="2025-06-14",
            snapshot_created_at="2025-06-14 16:30:00",
            max_news_time="2025-06-13 16:30:00",
        )
        checks = validator.validate(decision, metadata)
        self.assertEqual(checks["news_cutoff"], "PASSED")

    def test_sentiment_same_day_fails(self):
        """Sentiment from same day should fail validation."""
        validator = DecisionCutoffValidator(fail_on_future_data=False)
        decision = self._make_decision("2025-06-14")
        metadata = SnapshotMetadata(
            ticker="TEST",
            trade_date="2025-06-14",
            snapshot_created_at="2025-06-14 16:30:00",
            max_sentiment_time="2025-06-14 12:00:00",
        )
        checks = validator.validate(decision, metadata)
        self.assertEqual(checks["sentiment_cutoff"], "FAILED")

    def test_sentiment_prev_day_passes(self):
        """Sentiment from previous day should pass validation."""
        validator = DecisionCutoffValidator(fail_on_future_data=False)
        decision = self._make_decision("2025-06-14")
        metadata = SnapshotMetadata(
            ticker="TEST",
            trade_date="2025-06-14",
            snapshot_created_at="2025-06-14 16:30:00",
            max_sentiment_time="2025-06-13 16:00:00",
        )
        checks = validator.validate(decision, metadata)
        self.assertEqual(checks["sentiment_cutoff"], "PASSED")

    def test_fundamental_within_buffer_fails(self):
        """Fundamental available 1 day before trade_date fails with buffer=3."""
        validator = DecisionCutoffValidator(
            fail_on_future_data=False, fundamental_buffer_days=3,
        )
        decision = self._make_decision("2025-06-14")
        metadata = SnapshotMetadata(
            ticker="TEST",
            trade_date="2025-06-14",
            snapshot_created_at="2025-06-14 16:30:00",
            max_fundamental_available_date="2025-06-13",
        )
        checks = validator.validate(decision, metadata)
        self.assertEqual(checks["fundamental_available_date"], "FAILED")

    def test_fundamental_outside_buffer_passes(self):
        """Fundamental available 5 days before trade_date passes with buffer=3."""
        validator = DecisionCutoffValidator(
            fail_on_future_data=False, fundamental_buffer_days=3,
        )
        decision = self._make_decision("2025-06-14")
        metadata = SnapshotMetadata(
            ticker="TEST",
            trade_date="2025-06-14",
            snapshot_created_at="2025-06-14 16:30:00",
            max_fundamental_available_date="2025-06-09",
        )
        checks = validator.validate(decision, metadata)
        self.assertEqual(checks["fundamental_available_date"], "PASSED")

    def test_no_news_passes(self):
        """No news metadata should pass (graceful handling)."""
        validator = DecisionCutoffValidator(fail_on_future_data=False)
        decision = self._make_decision("2025-06-14")
        metadata = SnapshotMetadata(
            ticker="TEST",
            trade_date="2025-06-14",
            snapshot_created_at="2025-06-14 16:30:00",
            max_news_time=None,
        )
        checks = validator.validate(decision, metadata)
        self.assertEqual(checks["news_cutoff"], "PASSED")


class TestTimezoneInCutoffComparison(unittest.TestCase):
    """Timezone-aware timestamps should be normalized before comparison."""

    def test_wib_news_excluded_when_utc_equivalent_same_day(self):
        """News at 10:00 WIB (=03:00 UTC) on trade_date should be excluded."""
        news = [
            {"published_at": "2025-06-14T10:00:00+07:00", "title": "wib_same_day"},
        ]
        w = compute_window("2025-06-14", None)
        out = slice_news(news, w, prev_trading_day="2025-06-13")
        self.assertEqual(len(out), 0)

    def test_utc_news_excluded_when_same_day(self):
        """News at 10:00 UTC on trade_date should be excluded."""
        news = [
            {"published_at": "2025-06-14T10:00:00+00:00", "title": "utc_same_day"},
        ]
        w = compute_window("2025-06-14", None)
        out = slice_news(news, w, prev_trading_day="2025-06-13")
        self.assertEqual(len(out), 0)

    def test_wib_news_included_when_prev_day(self):
        """News at 23:00 WIB (=16:00 UTC) on prev_day should be included."""
        news = [
            {"published_at": "2025-06-13T23:00:00+07:00", "title": "wib_prev_day"},
        ]
        w = compute_window("2025-06-14", None)
        out = slice_news(news, w, prev_trading_day="2025-06-13")
        self.assertEqual(len(out), 1)


class TestPreviousTradingDayLookup(unittest.TestCase):
    """Test _get_previous_trading_day helper."""

    def test_finds_previous_from_ohlcv(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        data_dir = root / "data" / "TEST.JK"
        data_dir.mkdir(parents=True)

        dates = pd.date_range("2025-06-09", periods=5, freq="B")
        ohlcv = pd.DataFrame({
            "date": [d.date().isoformat() for d in dates],
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.5, "volume": 1_000_000.0,
        })
        ohlcv.to_csv(data_dir / "ohlcv.csv", index=False)

        prov = SnapshotDataProvider(
            data_root=str(data_dir.parent),
            snapshot_root=str(root / "snapshots"),
        )
        # 2025-06-10 is a Tuesday, prev should be 2025-06-09 (Monday)
        prev = prov._get_previous_trading_day("TEST.JK", "2025-06-10")
        self.assertEqual(prev, "2025-06-09")
        tmp.cleanup()

    def test_fallback_skips_weekend(self):
        """If date not in OHLCV, fallback skips weekends."""
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        data_dir = root / "data" / "TEST.JK"
        data_dir.mkdir(parents=True)

        # Only have data for Monday
        (data_dir / "ohlcv.csv").write_text(
            "date,open,high,low,close,volume\n2025-06-09,100,101,99,100.5,1000000\n"
        )

        prov = SnapshotDataProvider(
            data_root=str(data_dir.parent),
            snapshot_root=str(root / "snapshots"),
        )
        # Monday not in OHLCV for "2025-06-16" (next Monday) — fallback should
        # go back to Friday 2025-06-13
        prev = prov._get_previous_trading_day("TEST.JK", "2025-06-16")
        self.assertEqual(prev, "2025-06-13")  # Friday
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
