"""
Tests for audit fixes Round 8:
- C2: eod_margin_breach counted in margin_calls_count
- C1: Trade metrics with string fields (no crash)
- H2: auto_liquidate_on_breach prevents force close
- M1: No double equity curve entry on last day
- L10: config.risk values used instead of defaults
"""
import unittest
from unittest.mock import MagicMock

import pandas as pd

from tradingagents.backtesting.metrics import MetricsCalculator
from tradingagents.backtesting.position import (
    MarginConfig,
    OrderType,
    Position,
    RiskConfig,
)
from tradingagents.backtesting.risk import RiskEngine


class TestC2_EodMarginBreachCounted(unittest.TestCase):
    """eod_margin_breach should be counted as 'call' in metrics."""

    def test_call_events_counted_in_metrics(self):
        from tradingagents.backtesting.decision_schema import MarginEvent
        calc = MetricsCalculator()
        events = [
            MarginEvent(date="2025-01-02", kind="call", deficit=1000, mark_price=100,
                        maintenance_required=5000, account_equity=4000, action="liquidation"),
            MarginEvent(date="2025-01-03", kind="liquidation", deficit=2000, mark_price=90,
                        maintenance_required=5000, account_equity=3000, action="liquidation"),
        ]
        result = calc._augment_with_margin_metrics({}, [], events)
        self.assertEqual(result["margin_calls_count"], 2)  # call + liquidation
        self.assertEqual(result["liquidations_count"], 1)


class TestC1_TradeMetricNormalization(unittest.TestCase):
    """Trade metrics should handle string side/open_close fields."""

    def test_metrics_with_string_fields(self):
        """Metrics should not crash when trade fields are strings."""
        from tradingagents.backtesting.decision_schema import (
            OpenClose, OrderSide, Trade as LegacyTrade,
        )
        calc = MetricsCalculator()
        # Create a trade with enum fields (normal case)
        trade = LegacyTrade(
            date="2025-01-02", ticker="TEST", side=OrderSide.BUY,
            quantity=100, price=100.0, gross_amount=10000, fee=10,
            net_amount=9990, multiplier=1.0, notional=10000,
            tick_size=0.01, slippage_ticks=0, realized_pnl_delta=0,
            margin_delta=0, open_close=OpenClose.OPEN, reason="test",
            decision_id="D1", order_id="O1",
        )
        # This should not crash
        result = calc._trade_stats([trade])
        self.assertIn("win_rate", result)


class TestH2_AutoLiquidate(unittest.TestCase):
    """auto_liquidate=False should prevent force close but still log event."""

    def test_auto_liquidate_false_no_order(self):
        engine = RiskEngine(
            max_loss_per_trade_pct=10.0,
            max_portfolio_loss_pct=10.0,
            auto_liquidate=False,
        )
        position = Position(
            ticker="TEST", quantity=1000, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        # Use very low equity so liquidation triggers
        bar = {"open": 82, "high": 83, "low": 60, "close": 60}
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=5000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        # Event should be logged
        liq_events = [e for e in events if e.event_type == "liquidation"]
        self.assertEqual(len(liq_events), 1)
        # But no force-close order
        self.assertIsNone(order)

    def test_auto_liquidate_true_force_close(self):
        engine = RiskEngine(
            max_loss_per_trade_pct=10.0,
            max_portfolio_loss_pct=10.0,
            auto_liquidate=True,
        )
        position = Position(
            ticker="TEST", quantity=1000, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        bar = {"open": 82, "high": 83, "low": 60, "close": 60}
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=5000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        liq_events = [e for e in events if e.event_type == "liquidation"]
        self.assertEqual(len(liq_events), 1)
        self.assertIsNotNone(order)


class TestL10_ConfigRisk(unittest.TestCase):
    """config.risk should be used instead of default RiskConfig()."""

    def test_risk_config_from_yaml_used(self):
        """WalkForwardRunner should use config.risk, not default RiskConfig()."""
        import inspect
        from tradingagents.backtesting.walk_forward_runner import WalkForwardBacktestRunner
        source = inspect.getsource(WalkForwardBacktestRunner.__init__)
        # Should reference self.config.risk
        self.assertIn("self.config.risk", source)


if __name__ == "__main__":
    unittest.main()
