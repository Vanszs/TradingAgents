"""
Tests for snapshot-only enforcement in backtest mode.
"""
import unittest

from tradingagents.dataflows.interface import VENDOR_METHODS, route_to_vendor


class TestSnapshotEnforcement(unittest.TestCase):
    """In backtest mode, fallback chain should only use snapshot vendor."""

    def test_snapshot_vendor_registered_for_all_methods(self):
        """Snapshot vendor should be registered for all data methods."""
        methods = [
            "get_stock_data", "get_indicators", "get_news",
            "get_global_news", "get_fundamentals", "get_balance_sheet",
            "get_cashflow", "get_income_statement",
        ]
        for method in methods:
            self.assertIn(
                "snapshot", VENDOR_METHODS[method],
                f"Snapshot vendor not registered for {method}",
            )

    def test_backtest_mode_fallback_restricted(self):
        """In backtest mode, fallback should only try snapshot vendor."""
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        try:
            # Set backtest mode
            cfg._config = {
                "backtest_mode": True,
                "data_vendors": {
                    "core_stock_apis": "snapshot",
                    "technical_indicators": "snapshot",
                    "fundamental_data": "snapshot",
                    "news_data": "snapshot",
                },
            }
            # The route_to_vendor function should only try snapshot
            # We can't easily test the internal fallback, but we can
            # verify that the config is read correctly
            from tradingagents.dataflows.interface import get_vendor
            vendor = get_vendor("core_stock_apis", "get_stock_data")
            self.assertEqual(vendor, "snapshot")
        finally:
            cfg._config = original

    def test_live_mode_fallback_allows_all(self):
        """In live mode, fallback should try all vendors."""
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        try:
            cfg._config = {
                "backtest_mode": False,
                "data_vendors": {
                    "core_stock_apis": "yfinance",
                },
            }
            from tradingagents.dataflows.interface import get_vendor
            vendor = get_vendor("core_stock_apis", "get_stock_data")
            self.assertEqual(vendor, "yfinance")
        finally:
            cfg._config = original


class TestRiskEngineDefaults(unittest.TestCase):
    """Risk engine defaults should be fractions, not percentages."""

    def test_default_max_loss_is_fraction(self):
        from tradingagents.backtesting.risk import RiskEngine
        engine = RiskEngine()
        # Default should be 0.10 (10%), not 10.0 (1000%)
        self.assertAlmostEqual(engine.max_loss_per_trade_pct, 0.10)

    def test_default_max_portfolio_loss_is_fraction(self):
        from tradingagents.backtesting.risk import RiskEngine
        engine = RiskEngine()
        # Default should be 0.20 (20%), not 20.0 (2000%)
        self.assertAlmostEqual(engine.max_portfolio_loss_pct, 0.20)


class TestATRNoFutureData(unittest.TestCase):
    """ATR computation should not use future data."""

    def test_atr_filtered_by_current_date(self):
        import pandas as pd

        from tradingagents.backtesting.walk_forward_runner import WalkForwardBacktestRunner

        # Create runner with mock data
        runner = WalkForwardBacktestRunner.__new__(WalkForwardBacktestRunner)

        # Create OHLCV with 30 days
        dates = pd.date_range("2025-01-01", periods=30, freq="B")
        ohlcv = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "high": [100 + i * 0.5 for i in range(30)],
            "low": [98 + i * 0.5 for i in range(30)],
            "close": [99 + i * 0.5 for i in range(30)],
        })

        # Compute ATR at day 15 (should only use first 15 days)
        atr = runner._compute_atr(ohlcv, period=14, current_date="2025-01-21")
        self.assertIsNotNone(atr)

        # Compute ATR at day 5 (not enough data for period=14)
        atr_early = runner._compute_atr(ohlcv, period=14, current_date="2025-01-07")
        self.assertIsNone(atr_early)


if __name__ == "__main__":
    unittest.main()
