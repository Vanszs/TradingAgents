"""
Tests for audit fixes Round 7:
- C1: _apply_close raises ValueError on over-close
- C2: _check_hard_risk handles equity=0 without crash
- H1: max_portfolio_loss_pct is separate from max_intraday_loss_pct
- H3: snapshot vendor functions filter by curr_date
"""
import unittest
from unittest.mock import MagicMock

import pandas as pd

from tradingagents.backtesting.portfolio import PortfolioV2
from tradingagents.backtesting.position import (
    Fill,
    MarginConfig,
    OrderType,
    Position,
)
from tradingagents.backtesting.risk import RiskEngine


class TestC1_ApplyCloseRaisesOnOverClose(unittest.TestCase):
    """_apply_close should raise ValueError when close_qty > abs_qty."""

    def test_over_close_raises(self):
        portfolio = PortfolioV2(
            initial_cash=100_000.0,
            ticker="TEST",
            margin_config=MarginConfig(),
        )
        # Open long position with 100 shares
        open_fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(open_fill)

        # Try to close 200 shares (more than 100 open)
        close_fill = Fill(
            fill_id="close1", order_id="o2", decision_id="d2",
            date="2025-01-03", ticker="TEST", side="SELL",
            quantity=200, price=110.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.SELL_TO_CLOSE, open_close="CLOSE",
        )
        with self.assertRaises(ValueError):
            portfolio.apply_fill(close_fill)

    def test_exact_close_works(self):
        portfolio = PortfolioV2(
            initial_cash=100_000.0,
            ticker="TEST",
            margin_config=MarginConfig(),
        )
        open_fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(open_fill)

        close_fill = Fill(
            fill_id="close1", order_id="o2", decision_id="d2",
            date="2025-01-03", ticker="TEST", side="SELL",
            quantity=100, price=110.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.SELL_TO_CLOSE, open_close="CLOSE",
        )
        portfolio.apply_fill(close_fill)
        self.assertTrue(portfolio.is_flat())


class TestC2_DivisionByZeroGuard(unittest.TestCase):
    """_check_hard_risk should not crash when equity = 0."""

    def test_zero_equity_no_crash(self):
        engine = RiskEngine(max_loss_per_trade_pct=0.05, max_portfolio_loss_pct=0.20)
        position = Position(
            ticker="TEST", quantity=100, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        bar = {"open": 100, "high": 101, "low": 99, "close": 100}
        # equity=0 should not cause ZeroDivisionError
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=0,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        # Should trigger hard risk (equity <= 0)
        hard_risk = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk), 1)

    def test_negative_equity_triggers(self):
        engine = RiskEngine(max_loss_per_trade_pct=0.05, max_portfolio_loss_pct=0.20)
        position = Position(
            ticker="TEST", quantity=100, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0,
        )
        bar = {"open": 100, "high": 101, "low": 99, "close": 100}
        order, events = engine.check_bar(
            "2025-01-02", bar, position, equity=-5000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        hard_risk = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk), 1)


class TestH1_SeparateThresholds(unittest.TestCase):
    """max_portfolio_loss_pct should be different from max_intraday_loss_pct."""

    def test_risk_engine_has_separate_thresholds(self):
        from tradingagents.backtesting.walk_forward_runner import WalkForwardBacktestRunner
        import inspect
        source = inspect.getsource(WalkForwardBacktestRunner.__init__)
        # max_portfolio_loss_pct should be set to a different value
        self.assertIn("max_portfolio_loss_pct=0.20", source)


class TestH3_SnapshotVendorDateFilter(unittest.TestCase):
    """Snapshot vendor functions should filter by curr_date."""

    def test_balance_sheet_filters_by_date(self):
        from tradingagents.dataflows.snapshot import snapshot_get_balance_sheet
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        try:
            cfg._config = {
                "snapshot_data": {
                    "fundamentals": [
                        {"metric": "total_assets", "value": 100, "available_date": "2025-06-01"},
                        {"metric": "total_assets", "value": 200, "available_date": "2025-07-01"},
                    ]
                }
            }
            # With curr_date before the second item
            result = snapshot_get_balance_sheet("TEST", curr_date="2025-06-15")
            self.assertIn("100", result)
            self.assertNotIn("200", result)

            # With curr_date after both items
            result = snapshot_get_balance_sheet("TEST", curr_date="2025-07-15")
            self.assertIn("100", result)
            self.assertIn("200", result)
        finally:
            cfg._config = original

    def test_cashflow_filters_by_date(self):
        from tradingagents.dataflows.snapshot import snapshot_get_cashflow
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        try:
            cfg._config = {
                "snapshot_data": {
                    "fundamentals": [
                        {"metric": "cash_from_operations", "value": 50, "available_date": "2025-06-01"},
                        {"metric": "cash_from_operations", "value": 75, "available_date": "2025-07-01"},
                    ]
                }
            }
            result = snapshot_get_cashflow("TEST", curr_date="2025-06-15")
            self.assertIn("50", result)
            self.assertNotIn("75", result)
        finally:
            cfg._config = original

    def test_income_statement_filters_by_date(self):
        from tradingagents.dataflows.snapshot import snapshot_get_income_statement
        import tradingagents.dataflows.config as cfg
        original = cfg._config
        try:
            cfg._config = {
                "snapshot_data": {
                    "fundamentals": [
                        {"metric": "revenue", "value": 1000, "available_date": "2025-06-01"},
                        {"metric": "revenue", "value": 2000, "available_date": "2025-07-01"},
                    ]
                }
            }
            result = snapshot_get_income_statement("TEST", curr_date="2025-06-15")
            self.assertIn("1000", result)
            self.assertNotIn("2000", result)
        finally:
            cfg._config = original


if __name__ == "__main__":
    unittest.main()
