"""
Tests for audit fixes — verifies that all critical/high/medium issues
from the audit are correctly resolved.
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from tradingagents.backtesting.broker import SimulatedBroker
from tradingagents.backtesting.margin_engine import is_intraday_margin_breach
from tradingagents.backtesting.portfolio import PortfolioV2
from tradingagents.backtesting.position import (
    BacktestConfig,
    ExecutionConfig,
    Fill,
    MarginConfig,
    Order,
    OrderType,
    Position,
    PositionSide,
)
from tradingagents.backtesting.risk import RiskEngine


class TestC1_DatetimeImport(unittest.TestCase):
    """C1: _next_valid_placeholder should not crash with NameError."""

    def test_next_valid_placeholder_runs(self):
        from tradingagents.backtesting.agent_runner import TradingAgentsRunner
        result = TradingAgentsRunner._next_valid_placeholder("2025-06-14")
        # 2025-06-14 is Saturday → should skip to Monday 2025-06-16
        self.assertEqual(result, "2025-06-16")

    def test_next_valid_placeholder_weekday(self):
        from tradingagents.backtesting.agent_runner import TradingAgentsRunner
        result = TradingAgentsRunner._next_valid_placeholder("2025-06-13")
        # Friday → Monday
        self.assertEqual(result, "2025-06-16")


class TestH1_FeeDoubleCharge(unittest.TestCase):
    """H1: Reverse order fee should be split 50/50 between legs."""

    def test_reverse_fee_split(self):
        """Verify that _apply_reverse splits fee between close and open legs."""
        portfolio = PortfolioV2(
            initial_cash=100_000.0,
            ticker="TEST",
            margin_config=MarginConfig(),
        )

        # Open a long position
        open_fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=100.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(open_fill)
        cash_after_open = portfolio.cash  # 100000 - (100*100 + 100) = 89900

        # Now reverse to short with total fee=200
        reverse_fill = Fill(
            fill_id="rev1", order_id="o2", decision_id="d2",
            date="2025-01-03", ticker="TEST", side="SELL",
            quantity=100, price=105.0, fee=200.0, slippage_amount=0.0,
            order_type=OrderType.REVERSE_TO_SHORT, open_close="CLOSE",
        )
        portfolio.apply_fill(reverse_fill)

        # With fee split (correct): close leg fee=100, open leg fee=100
        # Close (SELL_TO_CLOSE): cash += (105*100 - 100) = 10400
        # Open (SELL_TO_OPEN): cash += (105*100 - 100) = 10400
        # Total: cash_after_open + 10400 + 10400 = 89900 + 20800 = 110700
        expected_with_split = cash_after_open + (105*100 - 100) + (105*100 - 100)

        # With double-charge (bug): close leg fee=200, open leg fee=200
        # Close: cash += (105*100 - 200) = 10300
        # Open: cash += (105*100 - 200) = 10300
        # Total: cash_after_open + 10300 + 10300 = 89900 + 20600 = 110500
        expected_with_double = cash_after_open + (105*100 - 200) + (105*100 - 200)

        # Verify we get the split-fee result, not the double-charge result
        self.assertAlmostEqual(portfolio.cash, expected_with_split, delta=1.0)
        self.assertNotAlmostEqual(portfolio.cash, expected_with_double, delta=1.0)


class TestH2_FlatPositionReverseGuard(unittest.TestCase):
    """H2: Reverse on flat position should be rejected."""

    def test_reverse_on_flat_is_rejected(self):
        broker = SimulatedBroker(
            execution_config=ExecutionConfig(lot_size=1),
            margin_config=MarginConfig(),
        )
        portfolio = PortfolioV2(
            initial_cash=100_000.0,
            ticker="TEST",
            margin_config=MarginConfig(),
        )
        from tradingagents.backtesting.position import InstrumentSpec, MarketPoint
        spec = InstrumentSpec(ticker="TEST", multiplier=1.0, tick_size=0.01)
        market_point = MarketPoint(
            date="2025-01-02", ticker="TEST",
            open=100.0, high=101.0, low=99.0, close=100.5,
        )

        reverse_order = Order(
            order_id="rev1", decision_id="d1", ticker="TEST",
            order_type=OrderType.REVERSE_TO_LONG, quantity=100,
            execution_date="2025-01-02", is_reverse=True,
        )
        broker.add_pending_orders([reverse_order])

        # Should not raise (caught internally), but order should be rejected
        trades = broker.execute_pending_orders(
            date="2025-01-02",
            market_point=market_point,
            portfolio=portfolio,
            spec=spec,
        )
        self.assertEqual(len(trades), 0)
        self.assertEqual(len(broker.rejected_orders), 1)
        self.assertIn("flat", broker.rejected_orders[0].rejection_reason.lower())


class TestH3_FreshPnLInRisk(unittest.TestCase):
    """H3: Risk engine should use fresh PnL, not cached field."""

    def test_hard_risk_uses_close_price_pnl(self):
        engine = RiskEngine(max_loss_per_trade_pct=0.05)
        position = Position(
            ticker="TEST", quantity=100, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0, unrealized_pnl=0.0,
        )
        # Simulate stale cached PnL (from open price = 100)
        position.unrealized_pnl = 0.0

        bar = {"open": 100.0, "high": 101.0, "low": 90.0, "close": 90.0}
        # equity at close: 100_000 + (90-100)*100 = 99_000
        equity = 99_000.0

        order, events = engine.check_bar(
            date="2025-01-02", bar=bar, position=position,
            equity=equity, margin_rate=0.5, maintenance_rate=0.35,
            max_leverage=2.0, multiplier=1.0,
        )
        # Unrealized loss = (90-100)*100 = -1000
        # 1000/99000 = 0.0101 (1.01%) < 0.05 (5%) → should NOT trigger
        hard_risk_events = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk_events), 0)

    def test_hard_risk_triggers_on_big_move(self):
        engine = RiskEngine(max_loss_per_trade_pct=0.05)
        position = Position(
            ticker="TEST", quantity=100, avg_entry_price=100.0,
            multiplier=1.0, mark_price=100.0, unrealized_pnl=0.0,
        )
        # Stale cached PnL
        position.unrealized_pnl = 0.0

        bar = {"open": 100.0, "high": 101.0, "low": 40.0, "close": 40.0}
        # equity at close: 100_000 + (40-100)*100 = 94_000
        equity = 94_000.0

        order, events = engine.check_bar(
            date="2025-01-02", bar=bar, position=position,
            equity=equity, margin_rate=0.5, maintenance_rate=0.35,
            max_leverage=2.0, multiplier=1.0,
        )
        # Unrealized loss = (40-100)*100 = -6000
        # 6000/94000 = 0.0638 (6.38%) > 0.05 (5%) → SHOULD trigger
        hard_risk_events = [e for e in events if e.event_type == "hard_risk"]
        self.assertEqual(len(hard_risk_events), 1)


class TestM6_UniqueOrderIds(unittest.TestCase):
    """M6: Risk-generated orders should have unique IDs."""

    def test_unique_ids_for_same_day_orders(self):
        engine = RiskEngine(max_loss_per_trade_pct=5.0)
        position = Position(
            ticker="TEST", quantity=100, avg_entry_price=100.0,
            multiplier=1.0, stop_price=95.0, take_profit=110.0,
        )

        bar = {"open": 100.0, "high": 111.0, "low": 94.0, "close": 94.0}
        order1, _ = engine.check_bar(
            date="2025-01-02", bar=bar, position=position,
            equity=9400.0, margin_rate=0.5, maintenance_rate=0.35,
            max_leverage=2.0, multiplier=1.0,
        )
        order2, _ = engine.check_bar(
            date="2025-01-02", bar=bar, position=position,
            equity=9400.0, margin_rate=0.5, maintenance_rate=0.35,
            max_leverage=2.0, multiplier=1.0,
        )
        if order1 and order2:
            self.assertNotEqual(order1.order_id, order2.order_id)


class TestH6_MarginEngineDeadImport(unittest.TestCase):
    """H6: margin_engine should not have dead import."""

    def test_no_shadowed_optional_import(self):
        import inspect
        source = inspect.getsource(is_intraday_margin_breach)
        self.assertNotIn("_Optional", source)


class TestSnapshotPandasImport(unittest.TestCase):
    """M1: pandas should be importable at module level in snapshot.py."""

    def test_pandas_available(self):
        import tradingagents.dataflows.snapshot as snap
        self.assertTrue(hasattr(snap, 'pd'))


class TestInterfaceExceptionHandling(unittest.TestCase):
    """H4: interface.py should catch network errors for fallback."""

    def test_catches_connection_error(self):
        import inspect
        from tradingagents.dataflows.interface import route_to_vendor
        source = inspect.getsource(route_to_vendor)
        self.assertIn("ConnectionError", source)
        self.assertIn("TimeoutError", source)


if __name__ == "__main__":
    unittest.main()
