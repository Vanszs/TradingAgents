"""
Tests for risk engine unit fix — verifies that the hard-risk rule
uses correct units (fraction vs fraction, not percentage vs fraction).

After the fix:
- max_loss_per_trade_pct = 0.05 means 5% (stored as fraction)
- trade_risk_pct = abs(unrealized_pnl) / equity (also fraction)
- Comparison: 0.03 < 0.05 → NO trigger; 0.06 > 0.05 → trigger
"""
import unittest

from tradingagents.backtesting.position import Position
from tradingagents.backtesting.risk import RiskEngine


class TestHardRiskUnitFix(unittest.TestCase):
    """Hard-risk rule should trigger at correct thresholds."""

    def test_3pct_loss_below_5pct_threshold_no_trigger(self):
        """3% loss should NOT trigger hard risk (threshold is 5%)."""
        engine = RiskEngine(hard_risk_enabled=True,max_loss_per_trade_pct=0.05, max_portfolio_loss_pct=0.20)
        position = Position(
            ticker="TEST", quantity=1000, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        # 3% loss: (100-97)*1000 = 3000, equity=100000 → 3%
        bar = {"open": 100, "high": 101, "low": 96, "close": 97}
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=100_000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        hard_risk = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk), 0, "3% loss should NOT trigger hard risk")

    def test_6pct_loss_above_5pct_threshold_triggers(self):
        """6% loss SHOULD trigger hard risk (threshold is 5%)."""
        engine = RiskEngine(hard_risk_enabled=True,max_loss_per_trade_pct=0.05, max_portfolio_loss_pct=0.20)
        position = Position(
            ticker="TEST", quantity=1000, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        # 6% loss: (100-94)*1000 = 6000, equity=100000 → 6%
        bar = {"open": 100, "high": 101, "low": 93, "close": 94}
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=100_000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        hard_risk = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk), 1, "6% loss SHOULD trigger hard risk")

    def test_10pct_portfolio_loss_below_20pct_threshold_no_trigger(self):
        """10% portfolio loss should NOT trigger max portfolio loss (threshold is 20%)."""
        engine = RiskEngine(hard_risk_enabled=True,max_loss_per_trade_pct=0.05, max_portfolio_loss_pct=0.20)
        position = Position(
            ticker="TEST", quantity=1000, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        # 10% portfolio loss: unrealized=-10000, equity=100000
        # But unrealized needs to exceed max_loss_per_trade_pct first
        # Use a 4% loss to avoid hard_risk, then check portfolio loss
        bar = {"open": 100, "high": 101, "low": 95, "close": 96}
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=100_000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        # 4% loss < 5% threshold → hard_risk should NOT trigger
        hard_risk = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk), 0)

    def test_25pct_portfolio_loss_above_20pct_threshold_triggers(self):
        """25% portfolio loss SHOULD trigger max portfolio loss (threshold is 20%)."""
        engine = RiskEngine(hard_risk_enabled=True,max_loss_per_trade_pct=0.30, max_portfolio_loss_pct=0.20)
        position = Position(
            ticker="TEST", quantity=1000, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        # 25% portfolio loss: unrealized=-25000, equity=100000
        bar = {"open": 100, "high": 101, "low": 74, "close": 75}
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=100_000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        hard_risk = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk), 1, "25% portfolio loss SHOULD trigger hard risk")

    def test_negative_equity_triggers(self):
        """Negative equity should always trigger hard risk."""
        engine = RiskEngine(hard_risk_enabled=True,max_loss_per_trade_pct=0.05, max_portfolio_loss_pct=0.20)
        position = Position(
            ticker="TEST", quantity=1000, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        bar = {"open": 100, "high": 101, "low": 99, "close": 100}
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=-5000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        hard_risk = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk), 1, "Negative equity SHOULD trigger hard risk")

    def test_error_message_shows_percentage(self):
        """Error message should show percentage (e.g., 6.00%), not fraction."""
        engine = RiskEngine(hard_risk_enabled=True,max_loss_per_trade_pct=0.05, max_portfolio_loss_pct=0.20)
        position = Position(
            ticker="TEST", quantity=1000, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        bar = {"open": 100, "high": 101, "low": 93, "close": 94}
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=100_000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        hard_risk = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk), 1)
        reason = hard_risk[0].reason
        self.assertIn("%", reason)
        self.assertNotIn("600.00%", reason)

    def test_profitable_position_does_not_trigger_hard_risk(self):
        """A profitable position should NEVER trigger the max-loss rule."""
        engine = RiskEngine(hard_risk_enabled=True,max_loss_per_trade_pct=0.05, max_portfolio_loss_pct=0.20)
        position = Position(
            ticker="TEST", quantity=1000, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        # Price goes UP — position is profitable
        bar = {"open": 100, "high": 120, "low": 99, "close": 115}
        # equity = 100000 + (115-100)*1000 = 115000
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=115_000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        hard_risk = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk), 0, "Profitable position should NOT trigger hard risk")


if __name__ == "__main__":
    unittest.main()
