"""
Unit tests for reverse order execution (PRD §12.6).

Tests cover:
- Two-leg reverse: close + open
- Double fee and double slippage
- Portfolio state after reverse
"""
import unittest

from tradingagents.backtesting.broker import SimulatedBroker
from tradingagents.backtesting.decision_schema import MarketPoint
from tradingagents.backtesting.portfolio import Portfolio
from tradingagents.backtesting.position import (
    BacktestConfig,
    ExecutionConfig,
    InstrumentSpec,
    MarginConfig,
    MarketMode,
    Order,
    OrderType,
)


def _make_config() -> BacktestConfig:
    return BacktestConfig(
        ticker="AAPL",
        start_date="2024-01-02",
        end_date="2024-01-08",
        initial_cash=100_000.0,
        execution=ExecutionConfig(buy_fee=0.0015, sell_fee=0.0025, slippage=0.001),
        margin=MarginConfig(initial_margin_pct=0.50, maintenance_margin_pct=0.30, max_leverage=2.0),
    )


def _make_bar(date: str = "2024-01-03", open_: float = 150.0, close: float = 152.0) -> MarketPoint:
    return MarketPoint(date=date, ticker="AAPL", open=open_, high=155.0, low=149.0, close=close)


def _make_spec() -> InstrumentSpec:
    return InstrumentSpec(ticker="AAPL", multiplier=1.0, tick_size=0.01)


class TestReverseExecution(unittest.TestCase):

    def test_reverse_long_to_short(self):
        """REVERSE_TO_LONG closes long then opens short."""
        config = _make_config()
        broker = SimulatedBroker(config.execution, config.margin)
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(0.50, 0.30)

        # Open long first
        open_order = Order(
            order_id="O1", decision_id="D1", ticker="AAPL",
            order_type=OrderType.BUY_TO_OPEN, quantity=100,
            execution_date="2024-01-02",
        )
        broker.add_pending_orders([open_order])
        broker.execute_pending_orders("2024-01-02", _make_bar("2024-01-02"), p, _make_spec())
        self.assertEqual(p.position.quantity, 100)

        # Reverse to short
        reverse_order = Order(
            order_id="O2", decision_id="D2", ticker="AAPL",
            order_type=OrderType.REVERSE_TO_SHORT, quantity=100,
            execution_date="2024-01-03", is_reverse=True,
        )
        broker.add_pending_orders([reverse_order])
        trades = broker.execute_pending_orders("2024-01-03", _make_bar("2024-01-03"), p, _make_spec())

        # Should produce 2 trades: close leg + open leg
        self.assertEqual(len(trades), 2)
        self.assertEqual(p.position.quantity, -100)

    def test_reverse_short_to_long(self):
        """REVERSE_TO_LONG covers short then opens long."""
        config = _make_config()
        broker = SimulatedBroker(config.execution, config.margin)
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(0.50, 0.30)

        # Open short first
        open_order = Order(
            order_id="O1", decision_id="D1", ticker="AAPL",
            order_type=OrderType.SELL_TO_OPEN, quantity=100,
            execution_date="2024-01-02",
        )
        broker.add_pending_orders([open_order])
        broker.execute_pending_orders("2024-01-02", _make_bar("2024-01-02"), p, _make_spec())
        self.assertEqual(p.position.quantity, -100)

        # Reverse to long
        reverse_order = Order(
            order_id="O2", decision_id="D2", ticker="AAPL",
            order_type=OrderType.REVERSE_TO_LONG, quantity=100,
            execution_date="2024-01-03", is_reverse=True,
        )
        broker.add_pending_orders([reverse_order])
        trades = broker.execute_pending_orders("2024-01-03", _make_bar("2024-01-03"), p, _make_spec())

        self.assertEqual(len(trades), 2)
        self.assertEqual(p.position.quantity, 100)

    def test_reverse_double_fee(self):
        """Reverse should charge fees on both legs."""
        config = _make_config()
        broker = SimulatedBroker(config.execution, config.margin)
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(0.50, 0.30)

        # Open long
        open_order = Order(
            order_id="O1", decision_id="D1", ticker="AAPL",
            order_type=OrderType.BUY_TO_OPEN, quantity=100,
            execution_date="2024-01-02",
        )
        broker.add_pending_orders([open_order])
        broker.execute_pending_orders("2024-01-02", _make_bar("2024-01-02"), p, _make_spec())

        # Reverse
        reverse_order = Order(
            order_id="O2", decision_id="D2", ticker="AAPL",
            order_type=OrderType.REVERSE_TO_SHORT, quantity=100,
            execution_date="2024-01-03", is_reverse=True,
        )
        broker.add_pending_orders([reverse_order])
        trades = broker.execute_pending_orders("2024-01-03", _make_bar("2024-01-03"), p, _make_spec())

        # Both trades should have fees > 0
        self.assertGreater(trades[0].fee, 0)
        self.assertGreater(trades[1].fee, 0)

    def test_reverse_double_slippage(self):
        """Reverse should apply slippage on both legs."""
        config = _make_config()
        broker = SimulatedBroker(config.execution, config.margin)
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(0.50, 0.30)

        # Open long
        open_order = Order(
            order_id="O1", decision_id="D1", ticker="AAPL",
            order_type=OrderType.BUY_TO_OPEN, quantity=100,
            execution_date="2024-01-02",
        )
        broker.add_pending_orders([open_order])
        broker.execute_pending_orders("2024-01-02", _make_bar("2024-01-02"), p, _make_spec())

        # Reverse
        reverse_order = Order(
            order_id="O2", decision_id="D2", ticker="AAPL",
            order_type=OrderType.REVERSE_TO_SHORT, quantity=100,
            execution_date="2024-01-03", is_reverse=True,
        )
        broker.add_pending_orders([reverse_order])
        trades = broker.execute_pending_orders("2024-01-03", _make_bar("2024-01-03"), p, _make_spec())

        # Close leg: sell at open*(1-slippage) = 150*0.999 = 149.85
        # Open leg: sell at open*(1-slippage) = 150*0.999 = 149.85
        self.assertAlmostEqual(trades[0].price, 150.0 * 0.999, places=2)
        self.assertAlmostEqual(trades[1].price, 150.0 * 0.999, places=2)


if __name__ == "__main__":
    unittest.main()
