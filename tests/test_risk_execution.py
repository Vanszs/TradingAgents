"""
Tests for risk order immediate execution — verifies that stop-loss,
take-profit, and liquidation orders are executed at the current bar's
price, not deferred to the next trading day.
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
    Position,
)


class TestRiskOrderImmediateExecution(unittest.TestCase):
    """Risk orders should execute immediately, not be deferred."""

    def _make_broker_and_portfolio(self):
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
        return broker, portfolio, spec

    def test_stop_loss_executes_immediately(self):
        """Stop-loss order should execute at current bar's price."""
        broker, portfolio, spec = self._make_broker_and_portfolio()

        # Open a long position
        open_fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(open_fill)

        # Create a stop-loss order
        stop_order = Order(
            order_id="stop1", decision_id="d1", ticker="TEST",
            order_type=OrderType.SELL_TO_CLOSE, quantity=100,
            execution_date="2025-01-02", price=95.0, reason="stop",
        )

        market_point = MarketPoint(
            date="2025-01-02", ticker="TEST",
            open=96.0, high=97.0, low=94.0, close=95.0,
        )

        # Execute immediately
        trades = broker.execute_pending_orders_immediate(
            order=stop_order,
            market_point=market_point,
            portfolio=portfolio,
            spec=spec,
        )

        self.assertEqual(len(trades), 1)
        self.assertTrue(portfolio.is_flat())
        self.assertEqual(stop_order.status, "FILLED")

    def test_liquidation_executes_immediately(self):
        """Liquidation order should execute at current bar's price."""
        broker, portfolio, spec = self._make_broker_and_portfolio()

        # Open a long position
        open_fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(open_fill)

        # Create a liquidation order
        liq_order = Order(
            order_id="liq1", decision_id="", ticker="TEST",
            order_type=OrderType.SELL_TO_CLOSE, quantity=100,
            execution_date="2025-01-02", price=80.0, reason="liquidation",
        )

        market_point = MarketPoint(
            date="2025-01-02", ticker="TEST",
            open=82.0, high=83.0, low=79.0, close=80.0,
        )

        trades = broker.execute_pending_orders_immediate(
            order=liq_order,
            market_point=market_point,
            portfolio=portfolio,
            spec=spec,
        )

        self.assertEqual(len(trades), 1)
        self.assertTrue(portfolio.is_flat())

    def test_immediate_execution_does_not_add_to_pending(self):
        """Immediate execution should NOT add order to pending queue."""
        broker, portfolio, spec = self._make_broker_and_portfolio()

        # Open a long position
        open_fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(open_fill)

        stop_order = Order(
            order_id="stop1", decision_id="d1", ticker="TEST",
            order_type=OrderType.SELL_TO_CLOSE, quantity=100,
            execution_date="2025-01-02", price=95.0, reason="stop",
        )

        market_point = MarketPoint(
            date="2025-01-02", ticker="TEST",
            open=96.0, high=97.0, low=94.0, close=95.0,
        )

        initial_pending = len(broker.pending_orders)
        broker.execute_pending_orders_immediate(
            order=stop_order,
            market_point=market_point,
            portfolio=portfolio,
            spec=spec,
        )

        # Pending orders should not increase
        self.assertEqual(len(broker.pending_orders), initial_pending)


class TestAllocationPctNormalization(unittest.TestCase):
    """Allocation pct > 1.0 should be normalized to fraction."""

    def test_pct_above_100_normalized(self):
        from tradingagents.backtesting.order_generator import OrderGenerator
        from tradingagents.backtesting.position import (
            BacktestConfig,
            DecisionMappingConfig,
            ExtendedDecision,
        )
        config = BacktestConfig(
            ticker="TEST", start_date="2025-01-02", end_date="2025-01-10",
            initial_cash=100_000.0,
            decision_mapping=DecisionMappingConfig(mode="strict_5tier"),
        )
        og = OrderGenerator(config)
        pos = Position(ticker="TEST", quantity=0)

        decision = ExtendedDecision(
            decision_id="D1", ticker="TEST", trade_date="2025-01-02",
            report_generated_at="2025-01-02 16:30:00",
            last_data_date="2025-01-02", decision_valid_from="2025-01-03",
            agent_rating="Buy", normalized_rating="BUY",
            allocation_pct=30.0,  # 30 instead of 0.30
            position_intent="open", current_position_side="FLAT",
            target_position_side="LONG", futures_action="BUY_TO_OPEN",
            valid=True,
        )

        orders = og.decide(decision, pos, 100_000.0, reference_price=100.0)
        # Should normalize 30.0 → 0.30, giving 30000/100 = 300 shares
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].quantity, 300)

    def test_fraction_below_1_kept(self):
        from tradingagents.backtesting.order_generator import OrderGenerator
        from tradingagents.backtesting.position import (
            BacktestConfig,
            DecisionMappingConfig,
            ExtendedDecision,
        )
        config = BacktestConfig(
            ticker="TEST", start_date="2025-01-02", end_date="2025-01-10",
            initial_cash=100_000.0,
            decision_mapping=DecisionMappingConfig(mode="strict_5tier"),
        )
        og = OrderGenerator(config)
        pos = Position(ticker="TEST", quantity=0)

        decision = ExtendedDecision(
            decision_id="D1", ticker="TEST", trade_date="2025-01-02",
            report_generated_at="2025-01-02 16:30:00",
            last_data_date="2025-01-02", decision_valid_from="2025-01-03",
            agent_rating="Buy", normalized_rating="BUY",
            allocation_pct=0.30,  # correct fraction
            position_intent="open", current_position_side="FLAT",
            target_position_side="LONG", futures_action="BUY_TO_OPEN",
            valid=True,
        )

        orders = og.decide(decision, pos, 100_000.0, reference_price=100.0)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].quantity, 300)


class TestFillRealizedPnlDelta(unittest.TestCase):
    """Fill.realized_pnl_delta should be updated on close."""

    def test_close_updates_realized_pnl(self):
        from tradingagents.backtesting.portfolio import PortfolioV2
        portfolio = PortfolioV2(
            initial_cash=100_000.0, ticker="TEST",
            margin_config=MarginConfig(),
        )

        # Open long
        open_fill = Fill(
            fill_id="o1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(open_fill)

        # Close long with profit
        close_fill = Fill(
            fill_id="c1", order_id="o2", decision_id="d2",
            date="2025-01-03", ticker="TEST", side="SELL",
            quantity=100, price=110.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.SELL_TO_CLOSE, open_close="CLOSE",
        )
        portfolio.apply_fill(close_fill)

        # realized_pnl_delta should be updated
        self.assertEqual(close_fill.realized_pnl_delta, 1000.0)  # (110-100)*100


if __name__ == "__main__":
    unittest.main()
