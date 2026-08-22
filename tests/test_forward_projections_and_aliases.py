import unittest
import pandas as pd
from unittest.mock import MagicMock, patch

from tradingagents.dataflows.structural_levels import (
    compute_structural_levels,
    get_market_structural_summary,
)
from tradingagents.dataflows.y_finance import (
    get_stock_stats_indicators_window,
    _get_stock_stats_bulk,
)
from tradingagents.agents.schemas import ResearchPlan, PortfolioRating


class TestForwardProjectionsAndAliases(unittest.TestCase):
    def test_fibonacci_extensions_and_atr_targets(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        df = pd.DataFrame({
            "Open": [100.0] * 60,
            "High": [100.0] * 60,
            "Low": [100.0] * 60,
            "Close": [100.0] * 60,
        }, index=dates)
        # 60D High = 200, 60D Low = 100 -> fib_range = 100
        df.iloc[-30, df.columns.get_loc("High")] = 200.0
        df.iloc[-40, df.columns.get_loc("Low")] = 100.0
        # Last close = 150
        df.iloc[-1, df.columns.get_loc("Close")] = 150.0

        trade_date = dates[-1].strftime("%Y-%m-%d")
        levels = compute_structural_levels(df, trade_date)

        # fib_ext_1272 = 200 + 0.272 * 100 = 227.2
        # fib_ext_1618 = 200 + 0.618 * 100 = 261.8
        self.assertEqual(levels["fib_ext_1272"], 227.2)
        self.assertEqual(levels["fib_ext_1618"], 261.8)
        self.assertIn("atr_target_2x", levels)
        self.assertIn("atr_target_3x", levels)
        self.assertGreater(levels["atr_target_2x"], 150.0)
        self.assertGreater(levels["atr_target_3x"], levels["atr_target_2x"])

    def test_market_structural_summary_text_includes_extensions(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        df = pd.DataFrame({
            "Open": [100.0] * 60,
            "High": [100.0] * 60,
            "Low": [100.0] * 60,
            "Close": [100.0] * 60,
        }, index=dates)
        df.iloc[-30, df.columns.get_loc("High")] = 200.0
        df.iloc[-40, df.columns.get_loc("Low")] = 100.0
        df.iloc[-1, df.columns.get_loc("Close")] = 150.0

        trade_date = dates[-1].strftime("%Y-%m-%d")
        with patch("tradingagents.dataflows.structural_levels.load_ohlcv", return_value=df):
            summary = get_market_structural_summary("TEST", trade_date)
            self.assertIn("Forward Expansion Targets (Breakout Upside)", summary)
            self.assertIn("Fib 1.272x =", summary)
            self.assertIn("Fib 1.618x =", summary)
            self.assertIn("Expected Volatility Target Channels", summary)
            self.assertIn("+2x ATR =", summary)
            self.assertIn("+3x ATR =", summary)

    def test_indicator_aliases_in_y_finance(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": [100.0 + i for i in range(30)],
            "High": [105.0 + i for i in range(30)],
            "Low": [95.0 + i for i in range(30)],
            "Close": [102.0 + i for i in range(30)],
            "Volume": [10000] * 30,
        })

        trade_date = dates[-1].strftime("%Y-%m-%d")
        with patch("tradingagents.dataflows.y_finance.load_ohlcv", return_value=df):
            # rsi_14 alias to rsi
            res_rsi = get_stock_stats_indicators_window("TEST", "rsi_14", trade_date, look_back_days=10)
            self.assertIn("rsi", res_rsi)

            # atr_14 alias to atr
            res_atr = get_stock_stats_indicators_window("TEST", "atr_14", trade_date, look_back_days=10)
            self.assertIn("atr", res_atr)

            # atr directly
            res_atr_direct = get_stock_stats_indicators_window("TEST", "atr", trade_date, look_back_days=10)
            self.assertIn("atr", res_atr_direct)

    def test_research_plan_schema_wns(self):
        plan = ResearchPlan(
            recommendation="WNS",
            rationale="Unfavorable R:R and choppy market.",
            strategic_actions="Wait for structural retest at support.",
        )
        self.assertEqual(plan.recommendation, PortfolioRating.WNS)
