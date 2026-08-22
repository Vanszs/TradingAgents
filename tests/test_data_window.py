"""
Tests for the data window utility and the SnapshotDataProvider integration.
"""
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.backtesting.data_window import (
    ALLOWED_LOOKBACKS,
    assert_window_is_valid,
    compute_window,
    slice_broker_activity,
    slice_fundamentals,
    slice_news,
    slice_ohlcv,
    slice_sentiment,
)
from tradingagents.backtesting.snapshot_provider import (
    DataSnapshot,
    SnapshotDataProvider,
)


class TestComputeWindow(unittest.TestCase):
    def test_none_lookback(self):
        w = compute_window("2026-05-25", None)
        self.assertIsNone(w.lookback_days)
        self.assertIsNone(w.min_ohlcv_date_in_window)
        self.assertEqual(w.max_ohlcv_date_in_window, "2026-05-25")

    def test_60_lookback(self):
        w = compute_window("2026-05-25", 60)
        self.assertEqual(w.lookback_days, 60)
        self.assertEqual(w.max_ohlcv_date_in_window, "2026-05-25")
        # 60 + 7 buffer
        self.assertEqual(
            w.min_ohlcv_date_in_window,
            (date(2026, 5, 25) - timedelta(days=60 + 7)).isoformat(),
        )
        self.assertEqual(
            w.min_event_date,
            (date(2026, 5, 25) - timedelta(days=60)).isoformat(),
        )

    def test_zero_lookback_treated_as_none(self):
        w = compute_window("2026-05-25", 0)
        self.assertIsNone(w.lookback_days)


class TestSliceOHLCV(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2026-01-01", periods=200, freq="D")
        self.df = pd.DataFrame({
            "date": [d.date().isoformat() for d in dates],
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000.0,
        })

    def test_none_lookback_returns_unchanged(self):
        w = compute_window("2026-07-19", None)
        out = slice_ohlcv(self.df, w)
        self.assertEqual(len(out), 200)

    def test_60_lookback_keeps_last_60(self):
        w = compute_window("2026-07-19", 60)
        out = slice_ohlcv(self.df, w)
        self.assertEqual(len(out), 60)
        self.assertEqual(out["date"].iloc[0], "2026-05-21")
        self.assertEqual(out["date"].iloc[-1], "2026-07-19")

    def test_window_clamped_to_history(self):
        # trade_date is BEFORE any data row -> empty result
        w = compute_window("2020-01-01", 60)
        out = slice_ohlcv(self.df, w)
        self.assertTrue(out.empty)

    def test_window_respects_trade_date_cutoff(self):
        w = compute_window("2026-03-15", 60)
        out = slice_ohlcv(self.df, w)
        # All dates <= trade_date
        self.assertLessEqual(out["date"].iloc[-1], "2026-03-15")
        # Last 60 dates
        self.assertEqual(len(out), 60)


class TestSliceEvents(unittest.TestCase):
    def test_news_windowed(self):
        items = [
            {"published_at": "2026-04-01 10:00:00", "title": "old"},
            {"published_at": "2026-05-20 10:00:00", "title": "in_window"},
            {"published_at": "2026-05-25 16:30:00", "title": "boundary"},
            {"published_at": "2026-05-26 10:00:00", "title": "future"},
        ]
        w = compute_window("2026-05-25", 20)
        out = slice_news(items, w)
        titles = [r["title"] for r in out]
        self.assertIn("in_window", titles)
        self.assertIn("boundary", titles)
        self.assertNotIn("old", titles)
        self.assertNotIn("future", titles)

    def test_fundamentals_windowed_by_date(self):
        items = [
            {"available_date": "2026-01-15", "name": "old"},
            {"available_date": "2026-05-10", "name": "in_window"},
            {"available_date": "2026-05-26", "name": "future"},
        ]
        w = compute_window("2026-05-25", 60)
        out = slice_fundamentals(items, w)
        names = [r["name"] for r in out]
        self.assertIn("in_window", names)
        self.assertNotIn("old", names)
        self.assertNotIn("future", names)

    def test_sentiment_windowed(self):
        items = [
            {"timestamp": "2026-05-10 12:00:00", "score": 0.1},
            {"timestamp": "2026-05-24 12:00:00", "score": 0.5},
        ]
        w = compute_window("2026-05-25", 5)
        out = slice_sentiment(items, w)
        # 2026-05-10 is more than 5 days before 2026-05-25, so it is excluded.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["score"], 0.5)

    def test_broker_activity_windowed(self):
        items = [
            {"timestamp": "2026-05-24 12:00:00", "broker": "A"},
            {"timestamp": "2026-05-25 16:30:01", "broker": "B"},
        ]
        w = compute_window("2026-05-25", 5)
        out = slice_broker_activity(items, w, report_time="16:30:00")
        brokers = [r["broker"] for r in out]
        self.assertIn("A", brokers)
        self.assertNotIn("B", brokers)


class TestAssertWindowIsValid(unittest.TestCase):
    def test_valid_window(self):
        w = compute_window("2026-05-25", 60)
        # Older min is fine — we just have more history than needed.
        assert_window_is_valid(w, "2024-04-01", actual_max_ohlcv_date="2026-05-25")

    def test_violation_raises(self):
        w = compute_window("2026-05-25", 60)
        with self.assertRaises(ValueError):
            assert_window_is_valid(
                w, "2024-04-01", actual_max_ohlcv_date="2026-06-01"  # future!
            )

    def test_no_lookback_skips_check(self):
        w = compute_window("2026-05-25", None)
        # Should not raise even with a far-past date.
        assert_window_is_valid(w, "2020-01-01")


class TestSnapshotProviderWithLookback(unittest.TestCase):
    """Integration test: real snapshot provider with a lookback window."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data_dir = root / "data" / "TEST.JK"
        self.data_dir.mkdir(parents=True)

        # 400 business days of OHLCV (more than enough for a 60-day window)
        dates = pd.date_range("2024-01-01", periods=400, freq="B")
        ohlcv = pd.DataFrame({
            "date": [d.date().isoformat() for d in dates],
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000.0,
        })
        ohlcv.to_csv(self.data_dir / "ohlcv.csv", index=False)

        (self.data_dir / "news.json").write_text(json.dumps([
            {"published_at": "2024-06-15 10:00:00", "title": "far_past"},
            {"published_at": dates[394].date().isoformat() + " 10:00:00", "title": "near"},
        ]))
        (self.data_dir / "fundamentals.json").write_text(json.dumps([
            {"available_date": "2024-03-31", "metric": "q1_old"},
            {"available_date": dates[390].date().isoformat(), "metric": "q_near"},
        ]))
        (self.data_dir / "sentiment.json").write_text(json.dumps([
            {"timestamp": dates[394].date().isoformat() + " 12:00:00", "score": 0.1},
        ]))
        (self.data_dir / "broker_activity.json").write_text(json.dumps([]))

        self.snap_root = root / "snapshots"
        self.last_data_date = dates[395].date().isoformat()

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_lookback_uses_full_history(self):
        prov = SnapshotDataProvider(
            data_root=str(self.data_dir.parent),
            snapshot_root=str(self.snap_root),
        )
        snap = prov.create_snapshot("TEST.JK", self.last_data_date)
        self.assertIsNone(snap.lookback_days)
        # 396 business days <= last_data_date (the trade_date is the 396th row).
        self.assertEqual(len(snap.ohlcv), 396)

    def test_60_lookback_truncates_ohlcv(self):
        prov = SnapshotDataProvider(
            data_root=str(self.data_dir.parent),
            snapshot_root=str(self.snap_root),
            lookback_days=60,
        )
        snap = prov.create_snapshot("TEST.JK", self.last_data_date)
        self.assertEqual(snap.lookback_days, 60)
        self.assertEqual(len(snap.ohlcv), 60)
        self.assertLessEqual(snap.ohlcv["date"].iloc[-1], self.last_data_date)
        # 60 business days is approximately 84 calendar days.
        last = pd.Timestamp(snap.ohlcv["date"].iloc[-1])
        first = pd.Timestamp(snap.ohlcv["date"].iloc[0])
        delta = (last - first).days
        self.assertLessEqual(delta, 60 + 30)  # 60 business days ≈ 84 calendar days
        # And the row count is exactly lookback.
        self.assertEqual(len(snap.ohlcv), 60)

    def test_60_lookback_truncates_news(self):
        prov = SnapshotDataProvider(
            data_root=str(self.data_dir.parent),
            snapshot_root=str(self.snap_root),
            lookback_days=60,
        )
        snap = prov.create_snapshot("TEST.JK", self.last_data_date)
        titles = [n["title"] for n in snap.news]
        self.assertIn("near", titles)
        self.assertNotIn("far_past", titles)

    def test_60_lookback_truncates_fundamentals(self):
        prov = SnapshotDataProvider(
            data_root=str(self.data_dir.parent),
            snapshot_root=str(self.snap_root),
            lookback_days=60,
        )
        snap = prov.create_snapshot("TEST.JK", self.last_data_date)
        metrics = [f["metric"] for f in snap.fundamentals]
        self.assertIn("q_near", metrics)
        self.assertNotIn("q1_old", metrics)

    def test_metadata_contains_window_info(self):
        prov = SnapshotDataProvider(
            data_root=str(self.data_dir.parent),
            snapshot_root=str(self.snap_root),
            lookback_days=60,
        )
        snap = prov.create_snapshot("TEST.JK", self.last_data_date)
        meta_path = snap.root_path / "metadata.json"
        meta = json.loads(meta_path.read_text())
        self.assertEqual(meta["lookback_days"], 60)
        self.assertIsNotNone(meta["window_min_ohlcv_date"])
        self.assertIsNotNone(meta["window_max_ohlcv_date"])
        self.assertLessEqual(meta["window_max_ohlcv_date"], self.last_data_date)

    def test_invalid_lookback_raises(self):
        with self.assertRaises(ValueError):
            SnapshotDataProvider(
                data_root=str(self.data_dir.parent),
                snapshot_root=str(self.snap_root),
                lookback_days=42,  # not in the allowed list
            )


if __name__ == "__main__":
    unittest.main()
