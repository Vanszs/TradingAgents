"""
Unit tests for SnapshotDataProvider (flat layout).
"""
import os
import tempfile
import unittest

import pandas as pd

from tradingagents.backtesting.snapshot_provider import SnapshotDataProvider


class TestSnapshotDataProvider(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_root = os.path.join(self.tmpdir, "data")
        os.makedirs(os.path.join(self.data_root, "TEST"))
        # Create minimal OHLCV
        df = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1000, 1100, 1200],
        })
        df.to_csv(os.path.join(self.data_root, "TEST", "ohlcv.csv"), index=False)

    def test_load_full_ohlcv(self):
        p = SnapshotDataProvider(data_root=self.data_root)
        df = p.load_full_ohlcv("TEST")
        self.assertEqual(len(df), 3)
        self.assertIn("open", df.columns)
        self.assertIn("close", df.columns)

    def test_get_ohlcv_cutoff(self):
        p = SnapshotDataProvider(data_root=self.data_root)
        df = p.get_ohlcv("TEST", "2024-01-03")
        self.assertEqual(len(df), 2)

    def test_get_market_dates(self):
        p = SnapshotDataProvider(data_root=self.data_root)
        dates = p.get_market_dates("TEST")
        self.assertEqual(len(dates), 3)

    def test_create_snapshot_metadata(self):
        p = SnapshotDataProvider(
            data_root=self.data_root,
            snapshot_root=os.path.join(self.tmpdir, "snapshots"),
        )
        snap = p.create_snapshot("TEST", "2024-01-03")
        self.assertEqual(snap.metadata.ticker, "TEST")
        self.assertEqual(snap.metadata.trade_date, "2024-01-03")
        self.assertIsNotNone(snap.metadata.max_ohlcv_date)

    def test_empty_news_when_no_file(self):
        p = SnapshotDataProvider(data_root=self.data_root)
        news = p.get_news("TEST", "2024-01-02")
        self.assertEqual(news, [])


if __name__ == "__main__":
    unittest.main()
