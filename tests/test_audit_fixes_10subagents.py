import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.backtesting.decision_schema import Action, ParsedDecision, Rating
from tradingagents.backtesting.order_generator import OrderGenerator, OrderType, Position
from tradingagents.dataflows.structural_levels import compute_structural_levels
from tradingagents.default_config import DEFAULT_CONFIG, _apply_env_overrides


class TestAuditFixes10Subagents(unittest.TestCase):
    def test_default_config_env_overrides_thinking(self):
        env = {
            "TRADINGAGENTS_GOOGLE_THINKING_LEVEL": "high",
            "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "medium",
            "TRADINGAGENTS_ANTHROPIC_EFFORT": "low",
        }
        with patch.dict("os.environ", env):
            cfg = _apply_env_overrides(DEFAULT_CONFIG.copy())
            self.assertEqual(cfg.get("google_thinking_level"), "high")
            self.assertEqual(cfg.get("openai_reasoning_effort"), "medium")
            self.assertEqual(cfg.get("anthropic_effort"), "low")

    def test_structural_levels_fibonacci_range(self):
        dates = pd.date_range("2024-01-01", periods=70, freq="B")
        # 60D High = 200, 60D Low = 100
        df = pd.DataFrame({
            "Open": [150.0] * 70,
            "High": [150.0] * 70,
            "Low": [150.0] * 70,
            "Close": [150.0] * 70,
        }, index=dates)
        # Set 60D high on day 20
        df.iloc[-50, df.columns.get_loc("High")] = 200.0
        # Set 60D low on day 10
        df.iloc[-60, df.columns.get_loc("Low")] = 100.0

        levels = compute_structural_levels(df, "2024-04-05")
        self.assertEqual(levels["fib_50_level"], 150.0)
        self.assertEqual(levels["fib_618_level"], 138.2)
        # h60 = 200, fib_range = 100
        # fib_ext_1272 = 200 + 0.272 * 100 = 227.2
        # fib_ext_1618 = 200 + 0.618 * 100 = 261.8
        self.assertEqual(levels["fib_ext_1272"], 227.2)
        self.assertEqual(levels["fib_ext_1618"], 261.8)
        self.assertIn("atr_target_2x", levels)
        self.assertIn("atr_target_3x", levels)

    def test_order_generator_reverse_open_order_flag(self):
        from tradingagents.backtesting.position import BacktestConfig, ExtendedDecision
        og = OrderGenerator(BacktestConfig())
        pos = Position(ticker="AAPL", quantity=-100)  # short position
        dec = ExtendedDecision(
            decision_id="D1",
            ticker="AAPL",
            trade_date="2024-01-02",
            agent_rating="buy",
            normalized_rating="buy",
            allocation_pct=0.5,
            leverage=1.0,
            market_mode="FUTURES_STYLE_SIMULATION",
            allowed_position_sides="LONG,SHORT",
            position_intent="reverse",
            current_position_side="SHORT",
            target_position_side="LONG",
            futures_action="REVERSE_TO_LONG",
            valid=True,
        )
        orders = og.decide(dec, pos, current_equity=10000.0, reference_price=150.0)
        self.assertEqual(len(orders), 2)
        close_order, open_order = orders
        self.assertEqual(close_order.order_type, OrderType.BUY_TO_CLOSE)
        self.assertEqual(open_order.order_type, OrderType.BUY_TO_OPEN)
        self.assertFalse(open_order.is_reverse)


def test_yfinance_same_day_daily_validation():
    from tradingagents.dataflows.y_finance import get_intraday_data
    with patch("tradingagents.dataflows.y_finance.yf_retry") as mock_yf:
        mock_df = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [99.0], "Close": [102.0], "Volume": [1000]
        }, index=pd.DatetimeIndex(["2024-01-02"]))
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.side_effect = lambda fn: mock_df

        # Start == end for 1d interval should not raise ValueError("data end must be after start")
        df = get_intraday_data("AAPL", "2024-01-02", "2024-01-02", interval="1d")
        assert not df.empty
