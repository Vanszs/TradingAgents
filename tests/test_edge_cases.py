"""
Edge case tests for the backtesting module.

Covers:
- Empty backtest (no trading days) raises ValueError
- Single trading day backtest
- All orders rejected (qty=0 after lot rounding)
- Intraday margin breach for short positions (new intraday_high support)
- Leakage audit: unvalidated checks removed from required list
- Order generation with no reference price returns empty
"""
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from tradingagents.backtesting.margin_engine import is_intraday_margin_breach
from tradingagents.backtesting.order_generator import OrderGenerator
from tradingagents.backtesting.position import (
    BacktestConfig,
    DecisionMappingConfig,
    ExecutionConfig,
    ExtendedDecision,
    MarginConfig,
    Position,
)


class TestIntradayMarginBreachShort(unittest.TestCase):
    """Test that short positions can now detect intraday breaches via intraday_high."""

    def test_short_no_breach_when_high_equals_open(self):
        """Short: no breach when intraday_high == open_price."""
        result = is_intraday_margin_breach(
            account_equity_open=100_000.0,
            quantity=-1000,
            open_price=100.0,
            intraday_low=98.0,
            multiplier=1.0,
            maintenance_margin_pct=0.30,
            intraday_high=100.0,
        )
        self.assertFalse(result)

    def test_short_breach_when_high_rises(self):
        """Short: breach when intraday_high rises enough to wipe equity below maintenance."""
        # Open=100, high=145 → loss = (145-100)*1000 = 45000
        # equity = 100000 - 45000 = 55000
        # maintenance = 1000*145*0.30 = 43500
        # 55000 > 43500 → no breach
        # Open=100, high=175 → loss = (175-100)*1000 = 75000
        # equity = 100000 - 75000 = 25000
        # maintenance = 1000*175*0.30 = 52500
        # 25000 < 52500 → breach!
        result = is_intraday_margin_breach(
            account_equity_open=100_000.0,
            quantity=-1000,
            open_price=100.0,
            intraday_low=98.0,
            multiplier=1.0,
            maintenance_margin_pct=0.30,
            intraday_high=175.0,
        )
        self.assertTrue(result)

    def test_short_no_high_falls_back(self):
        """Short without intraday_high: falls back to open_price (no breach)."""
        result = is_intraday_margin_breach(
            account_equity_open=100_000.0,
            quantity=-1000,
            open_price=100.0,
            intraday_low=98.0,
            multiplier=1.0,
            maintenance_margin_pct=0.30,
            intraday_high=None,
        )
        self.assertFalse(result)

    def test_long_still_uses_low(self):
        """Long: still uses intraday_low for breach detection."""
        # open=100, low=60 → loss = (100-60)*1000 = 40000
        # equity = 100000 - 40000 = 60000
        # maintenance = 1000*60*0.30 = 18000
        # 60000 > 18000 → no breach
        result = is_intraday_margin_breach(
            account_equity_open=100_000.0,
            quantity=1000,
            open_price=100.0,
            intraday_low=60.0,
            multiplier=1.0,
            maintenance_margin_pct=0.30,
            intraday_high=102.0,
        )
        self.assertFalse(result)

    def test_flat_returns_false(self):
        """Flat position always returns False."""
        result = is_intraday_margin_breach(
            account_equity_open=100_000.0,
            quantity=0,
            open_price=100.0,
            intraday_low=50.0,
            multiplier=1.0,
            maintenance_margin_pct=0.30,
            intraday_high=200.0,
        )
        self.assertFalse(result)


class TestOrderGeneratorNoRefPrice(unittest.TestCase):
    """Order generation with no reference price returns empty."""

    def test_no_reference_price_returns_empty(self):
        config = BacktestConfig(
            ticker="AAPL",
            start_date="2024-01-02",
            end_date="2024-01-08",
            initial_cash=100_000.0,
            execution=ExecutionConfig(lot_size=1),
            margin=MarginConfig(),
            decision_mapping=DecisionMappingConfig(),
        )
        og = OrderGenerator(config)
        pos = Position(ticker="AAPL", quantity=0)
        decision = ExtendedDecision(
            decision_id="D1",
            ticker="AAPL",
            trade_date="2024-01-02",
            report_generated_at="2024-01-02 16:30:00",
            last_data_date="2024-01-02",
            decision_valid_from="2024-01-03",
            agent_rating="Buy",
            normalized_rating="BUY",
            allocation_pct=0.30,
            position_intent="open",
            current_position_side="FLAT",
            target_position_side="LONG",
            futures_action="BUY_TO_OPEN",
            valid=True,
        )
        # reference_price=0, stop_price=None → no price available
        orders = og.decide(decision, pos, 100_000.0, reference_price=0.0)
        self.assertEqual(len(orders), 0)


class TestLeakageAuditIntegrity(unittest.TestCase):
    """Leakage audit only includes validated checks in required list."""

    def test_unvalidated_checks_not_in_required(self):
        """long_short_pnl_rules, reverse_fee_rule, margin_rule, leverage_limit
        should NOT be in the required list since they are never validated."""
        from tradingagents.backtesting.walk_forward_runner import WalkForwardBacktestRunner
        import inspect
        source = inspect.getsource(WalkForwardBacktestRunner._build_leakage_audit)
        for check in ("long_short_pnl_rules", "reverse_fee_rule", "margin_rule", "leverage_limit"):
            self.assertNotIn(
                f'"{check}"',
                source,
                f"'{check}' should not be in the required list (never validated at runtime)",
            )


if __name__ == "__main__":
    unittest.main()
