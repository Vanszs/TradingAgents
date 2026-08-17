"""
Tests for the snapshot vendor — ensures snapshot data is correctly
served through the dataflow interface for backtesting.
"""
import unittest

import pandas as pd

from tradingagents.dataflows.snapshot import (
    _format_fundamental_items,
    _format_news_items,
    snapshot_get_balance_sheet,
    snapshot_get_cashflow,
    snapshot_get_fundamentals,
    snapshot_get_global_news,
    snapshot_get_income_statement,
    snapshot_get_indicators,
    snapshot_get_news,
    snapshot_get_stock_data,
)


def _record_snapshot_data():
    data = _mock_snapshot_data()
    data["ohlcv"] = data["ohlcv"].to_dict(orient="records")
    return data


def _mock_snapshot_data():
    """Build a mock snapshot_data dict."""
    ohlcv = pd.DataFrame({
        "date": ["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-06-14"],
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [101.5, 102.5, 103.5, 104.5, 105.5],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000],
    })
    return {
        "ohlcv": ohlcv,
        "news": [
            {"published_at": "2025-06-13 10:00:00", "title": "News A", "summary": "Summary A", "source": "Reuters"},
            {"published_at": "2025-06-13 14:00:00", "title": "News B", "summary": "Summary B", "source": "Bloomberg"},
        ],
        "fundamentals": [
            {"available_date": "2025-06-01", "metric": "revenue", "value": "10B", "period": "Q2"},
            {"available_date": "2025-06-01", "metric": "net_income", "value": "2B", "period": "Q2"},
            {"available_date": "2025-06-01", "metric": "total_assets", "value": "50B", "period": "Q2"},
            {"available_date": "2025-06-01", "metric": "cash_from_operations", "value": "3B", "period": "Q2"},
        ],
        "sentiment": [
            {"timestamp": "2025-06-13 09:00:00", "source": "stocktwits", "score": 0.7, "label": "Bullish", "text": "Looking good"},
            {"timestamp": "2025-06-13 10:00:00", "source": "reddit", "score": 0.5, "label": "Neutral", "text": "Mixed feelings"},
        ],
        "broker_activity": [],
        "spec": None,
    }


class TestSnapshotGetStockData(unittest.TestCase):

    def test_returns_csv(self):
        data = _mock_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_stock_data("TEST", "2025-06-10", "2025-06-14")
            self.assertIn("date", result)
            self.assertIn("open", result)
            self.assertIn("100.0", result)
            self.assertIn("# Stock data for TEST", result)
        finally:
            cfg._config = original

    def test_filters_date_range(self):
        data = _mock_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_stock_data("TEST", "2025-06-12", "2025-06-14")
            self.assertIn("2025-06-12", result)
            self.assertIn("2025-06-13", result)
            self.assertNotIn("2025-06-10", result)
        finally:
            cfg._config = original

    def test_no_data_returns_message(self):
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": {}}
        try:
            result = snapshot_get_stock_data("TEST", "2025-06-10", "2025-06-14")
            self.assertIn("No OHLCV", result)
        finally:
            cfg._config = original


class TestSnapshotGetNews(unittest.TestCase):

    def test_returns_formatted_news(self):
        data = _mock_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_news("TEST", "2025-06-10", "2025-06-14")
            self.assertIn("News A", result)
            self.assertIn("News B", result)
            self.assertIn("Reuters", result)
        finally:
            cfg._config = original

    def test_no_news_returns_message(self):
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": {"news": []}}
        try:
            result = snapshot_get_news("TEST", "2025-06-10", "2025-06-14")
            self.assertIn("No news", result)
        finally:
            cfg._config = original


class TestSnapshotGetGlobalNews(unittest.TestCase):

    def test_returns_formatted_global_news(self):
        data = _mock_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_global_news("2025-06-14")
            self.assertIn("News A", result)
            self.assertIn("News B", result)
        finally:
            cfg._config = original


class TestSnapshotGetFundamentals(unittest.TestCase):

    def test_returns_formatted_fundamentals(self):
        data = _mock_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_fundamentals("TEST", "2025-06-14")
            self.assertIn("revenue", result)
            self.assertIn("10B", result)
            self.assertIn("Q2", result)
        finally:
            cfg._config = original


class TestSnapshotGetBalanceSheet(unittest.TestCase):

    def test_returns_balance_sheet_items(self):
        data = _mock_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_balance_sheet("TEST")
            self.assertIn("total_assets", result)
            self.assertIn("50B", result)
        finally:
            cfg._config = original


class TestSnapshotGetCashflow(unittest.TestCase):

    def test_returns_cashflow_items(self):
        data = _mock_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_cashflow("TEST")
            self.assertIn("cash_from_operations", result)
            self.assertIn("3B", result)
        finally:
            cfg._config = original


class TestSnapshotGetIncomeStatement(unittest.TestCase):

    def test_returns_income_items(self):
        data = _mock_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_income_statement("TEST")
            self.assertIn("revenue", result)
            self.assertIn("10B", result)
        finally:
            cfg._config = original


class TestSnapshotGetIndicators(unittest.TestCase):

    def test_computes_rsi(self):
        data = _mock_snapshot_data()
        # Need more data points for RSI
        import numpy as np
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2025-04-01", periods=n, freq="B")
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        ohlcv = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": [1_000_000] * n,
        })
        data["ohlcv"] = ohlcv
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_indicators("TEST", "rsi", "2025-06-02", 10)
            self.assertIn("rsi", result.lower())
            self.assertIn("2025-06-02", result)
        finally:
            cfg._config = original

    def test_record_snapshot_computes_indicator(self):
        data = _record_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_indicators("TEST", "rsi", "2025-06-14", 10)
            self.assertNotIn("list' object has no attribute", result)
        finally:
            cfg._config = original

    def test_unsupported_indicator(self):
        data = _mock_snapshot_data()
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        cfg._config = {"snapshot_data": data}
        try:
            result = snapshot_get_indicators("TEST", "invalid_ind", "2025-06-14", 10)
            self.assertIn("not supported", result)
        finally:
            cfg._config = original


class TestFormatHelpers(unittest.TestCase):

    def test_format_news_items(self):
        items = [
            {"published_at": "2025-06-13 10:00", "title": "Test", "source": "RS", "summary": "Sum"},
        ]
        result = _format_news_items(items)
        self.assertIn("Test", result)
        self.assertIn("RS", result)
        self.assertIn("Sum", result)

    def test_format_news_items_empty(self):
        self.assertIn("No news", _format_news_items([]))

    def test_format_fundamental_items(self):
        items = [{"available_date": "2025-06-01", "metric": "rev", "value": "100", "period": "Q1"}]
        result = _format_fundamental_items(items)
        self.assertIn("rev", result)
        self.assertIn("100", result)
        self.assertIn("Q1", result)

    def test_format_fundamental_items_empty(self):
        self.assertIn("No fundamental", _format_fundamental_items([]))


class TestSnapshotVendorRegistered(unittest.TestCase):

    def test_snapshot_in_vendor_methods(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        self.assertIn("snapshot", VENDOR_METHODS["get_stock_data"])
        self.assertIn("snapshot", VENDOR_METHODS["get_indicators"])
        self.assertIn("snapshot", VENDOR_METHODS["get_news"])
        self.assertIn("snapshot", VENDOR_METHODS["get_global_news"])
        self.assertIn("snapshot", VENDOR_METHODS["get_fundamentals"])
        self.assertIn("snapshot", VENDOR_METHODS["get_balance_sheet"])
        self.assertIn("snapshot", VENDOR_METHODS["get_cashflow"])
        self.assertIn("snapshot", VENDOR_METHODS["get_income_statement"])


if __name__ == "__main__":
    unittest.main()
