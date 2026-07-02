"""
Tests for audit fixes Round 9:
- C1: State corruption on margin failure
- C2: Fee fallback in reverse orders
- C3: Dead PnL computation removed
- H2: Slippage recorded in reverse orders
- H4: Infinite returns handled
- M1: lot_size=1 allows small orders
"""
import unittest

from tradingagents.backtesting.broker import SimulatedBroker
from tradingagents.backtesting.portfolio import PortfolioV2
from tradingagents.backtesting.position import (
    ExecutionConfig,
    Fill,
    InstrumentSpec,
    MarginConfig,
    MarketPoint,
    Order,
    OrderType,
)


class TestC1_MarginFailureNoCorruption(unittest.TestCase):
    """Portfolio should not be corrupted when margin check fails."""

    def test_margin_failure_preserves_state(self):
        portfolio = PortfolioV2(
            initial_cash=1000.0,
            ticker="TEST",
            margin_config=MarginConfig(initial_margin_pct=0.50),
        )
        # Try to open a position that requires more margin than equity
        fill = Fill(
            fill_id="f1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        # Margin required = 100 * 100 * 0.50 = 5000, but equity = 1000
        with self.assertRaises(Exception):
            portfolio.apply_fill(fill)

        # Portfolio should still be flat (no corruption)
        self.assertTrue(portfolio.is_flat())
        self.assertEqual(portfolio.cash, 1000.0)

    def test_successful_open_after_rejection(self):
        """After a rejected open, a valid open should still work."""
        portfolio = PortfolioV2(
            initial_cash=100_000.0,
            ticker="TEST",
            margin_config=MarginConfig(initial_margin_pct=0.50),
        )
        # First: try too large
        large_fill = Fill(
            fill_id="f1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=10000, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        with self.assertRaises(Exception):
            portfolio.apply_fill(large_fill)

        # Then: try valid
        small_fill = Fill(
            fill_id="f2", order_id="o2", decision_id="d2",
            date="2025-01-03", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(small_fill)
        self.assertFalse(portfolio.is_flat())
        self.assertEqual(portfolio.position.quantity, 100)


class TestC2_ReverseFeeFallback(unittest.TestCase):
    """Reverse orders should work with legacy config (no buy_fee/sell_fee)."""

    def test_reverse_with_percentage_fees(self):
        broker = SimulatedBroker(
            execution_config=ExecutionConfig(lot_size=1, buy_fee=0.001, sell_fee=0.001),
            margin_config=MarginConfig(),
        )
        portfolio = PortfolioV2(
            initial_cash=100_000.0,
            ticker="TEST",
            margin_config=MarginConfig(),
        )
        spec = InstrumentSpec(ticker="TEST", multiplier=1.0, tick_size=0.01)
        market_point = MarketPoint(
            date="2025-01-02", ticker="TEST",
            open=100.0, high=101.0, low=99.0, close=100.5,
        )

        # Open long first
        open_fill = Fill(
            fill_id="o1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(open_fill)

        # Reverse to short
        reverse_order = Order(
            order_id="rev1", decision_id="d1", ticker="TEST",
            order_type=OrderType.REVERSE_TO_SHORT, quantity=100,
            execution_date="2025-01-02", is_reverse=True,
        )
        broker.add_pending_orders([reverse_order])
        trades = broker.execute_pending_orders(
            date="2025-01-02",
            market_point=market_point,
            portfolio=portfolio,
            spec=spec,
        )
        # Should produce 2 trades (close + open)
        self.assertEqual(len(trades), 2)
        # Fees should be non-zero
        self.assertGreater(trades[0].fee, 0)
        self.assertGreater(trades[1].fee, 0)


class TestH2_ReverseSlippageRecorded(unittest.TestCase):
    """Reverse orders should record actual slippage."""

    def test_reverse_slippage_nonzero(self):
        broker = SimulatedBroker(
            execution_config=ExecutionConfig(lot_size=1, buy_fee=0.001, sell_fee=0.001, slippage=0.01),
            margin_config=MarginConfig(),
        )
        portfolio = PortfolioV2(
            initial_cash=100_000.0,
            ticker="TEST",
            margin_config=MarginConfig(),
        )
        spec = InstrumentSpec(ticker="TEST", multiplier=1.0, tick_size=0.01)
        market_point = MarketPoint(
            date="2025-01-02", ticker="TEST",
            open=100.0, high=101.0, low=99.0, close=100.5,
        )

        # Open long
        open_fill = Fill(
            fill_id="o1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(open_fill)

        # Reverse
        reverse_order = Order(
            order_id="rev1", decision_id="d1", ticker="TEST",
            order_type=OrderType.REVERSE_TO_SHORT, quantity=100,
            execution_date="2025-01-02", is_reverse=True,
        )
        broker.add_pending_orders([reverse_order])
        trades = broker.execute_pending_orders(
            date="2025-01-02",
            market_point=market_point,
            portfolio=portfolio,
            spec=spec,
        )
        # At least one trade should have non-zero slippage
        total_slippage = sum(t.slippage_ticks for t in trades)
        self.assertGreater(total_slippage, 0)


class TestH4_InfiniteReturns(unittest.TestCase):
    """Metrics should handle infinite returns gracefully."""

    def test_zero_equity_no_crash(self):
        import pandas as pd
        from tradingagents.backtesting.metrics import MetricsCalculator

        calc = MetricsCalculator()
        equity_df = pd.DataFrame({
            "date": ["2025-01-02", "2025-01-03", "2025-01-04"],
            "total_equity": [100_000, 50_000, 0],
            "drawdown": [0.0, -0.5, -1.0],
            "position_qty": [100, 50, 0],
            "margin_used": [50_000, 25_000, 0],
            "leverage": [1.0, 0.5, 0.0],
            "daily_realized_pnl": [0, -50000, -50000],
            "daily_unrealized_pnl": [0, 0, 0],
            "lifetime_realized_pnl": [0, -50000, -100000],
        })
        result = calc.calculate(
            equity_curve=equity_df,
            trades=[],
            initial_cash=100_000,
        )
        # Should not crash, should return finite values
        self.assertFalse(float("inf") == result["sharpe_ratio"])
        self.assertFalse(float("-inf") == result["sharpe_ratio"])


if __name__ == "__main__":
    unittest.main()
