"""
Unit tests for position.py — PositionSide, Position, Fill, Order types.
"""
import unittest

from tradingagents.backtesting.position import (
    BacktestConfig,
    DecisionMappingConfig,
    ExecutionConfig,
    ExtendedDecision,
    Fill,
    InstrumentSpec,
    MarketMode,
    Order,
    OrderType,
    ParsedDecision,
    Position,
    PositionIntent,
    PositionSide,
    RiskConfig,
)


class TestPositionSide(unittest.TestCase):

    def test_side_from_quantity(self):
        p = Position(ticker="AAPL", quantity=100)
        self.assertEqual(p.side, PositionSide.LONG)

    def test_side_from_negative_quantity(self):
        p = Position(ticker="AAPL", quantity=-50)
        self.assertEqual(p.side, PositionSide.SHORT)

    def test_side_flat(self):
        p = Position(ticker="AAPL", quantity=0)
        self.assertEqual(p.side, PositionSide.FLAT)


class TestPosition(unittest.TestCase):

    def test_notional(self):
        p = Position(ticker="AAPL", quantity=100, multiplier=1.0)
        self.assertAlmostEqual(p.notional(150.0), 15000.0)

    def test_unrealized_pnl_long(self):
        p = Position(ticker="AAPL", quantity=100, avg_entry_price=150.0, multiplier=1.0)
        self.assertAlmostEqual(p.unrealized_pnl_calc(160.0), 1000.0)
        self.assertAlmostEqual(p.unrealized_pnl_calc(140.0), -1000.0)

    def test_unrealized_pnl_short(self):
        p = Position(ticker="AAPL", quantity=-100, avg_entry_price=150.0, multiplier=1.0)
        self.assertAlmostEqual(p.unrealized_pnl_calc(140.0), 1000.0)
        self.assertAlmostEqual(p.unrealized_pnl_calc(160.0), -1000.0)

    def test_mark_to_market(self):
        p = Position(ticker="AAPL", quantity=100, avg_entry_price=150.0, multiplier=1.0)
        p.mark_to_market(160.0)
        self.assertAlmostEqual(p.mark_price, 160.0)
        self.assertAlmostEqual(p.notional_value, 16000.0)
        self.assertAlmostEqual(p.unrealized_pnl, 1000.0)

    def test_side_transitions(self):
        """Side changes only through external quantity updates, not internal state."""
        p = Position(ticker="AAPL", quantity=100)
        self.assertEqual(p.side, PositionSide.LONG)
        p.quantity = -50
        self.assertEqual(p.side, PositionSide.SHORT)
        p.quantity = 0
        self.assertEqual(p.side, PositionSide.FLAT)


class TestOrderType(unittest.TestCase):

    def test_all_order_types_exist(self):
        expected = [
            "BUY_TO_OPEN", "BUY_TO_ADD", "SELL_TO_REDUCE", "SELL_TO_CLOSE",
            "SELL_TO_OPEN", "SELL_TO_ADD", "BUY_TO_REDUCE", "BUY_TO_CLOSE",
            "REVERSE_TO_LONG", "REVERSE_TO_SHORT", "NO_ORDER",
        ]
        actual = [ot.value for ot in OrderType]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_order_type_from_string(self):
        self.assertEqual(OrderType("BUY_TO_OPEN"), OrderType.BUY_TO_OPEN)


class TestExecutionConfig(unittest.TestCase):

    def test_defaults(self):
        ec = ExecutionConfig()
        self.assertEqual(ec.lot_size, 1)
        self.assertEqual(ec.buy_fee, 0.0015)
        self.assertEqual(ec.sell_fee, 0.0025)
        self.assertAlmostEqual(ec.slippage, 0.001)

    def test_invalid_lot_size_raises(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(lot_size=0)


class TestRiskConfig(unittest.TestCase):

    def test_defaults(self):
        rc = RiskConfig()
        self.assertAlmostEqual(rc.default_stop_pct, 0.08)
        self.assertAlmostEqual(rc.default_take_profit_pct, 0.20)


class TestDecisionMappingConfig(unittest.TestCase):

    def test_conservative_mode(self):
        dm = DecisionMappingConfig(mode="conservative")
        self.assertEqual(dm.mode, "conservative")

    def test_aggressive_mode(self):
        dm = DecisionMappingConfig(mode="aggressive")
        self.assertEqual(dm.mode, "aggressive")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            DecisionMappingConfig(mode="invalid")


class TestFill(unittest.TestCase):

    def test_to_dict(self):
        f = Fill(
            fill_id="F1", order_id="O1", decision_id="D1",
            date="2024-01-02", ticker="AAPL", side="BUY",
            quantity=100, price=150.0, fee=22.5, slippage_amount=15.0,
            order_type="BUY_TO_OPEN", open_close="OPEN",
        )
        d = f.to_dict()
        self.assertEqual(d["fill_id"], "F1")
        self.assertEqual(d["side"], "BUY")
        self.assertAlmostEqual(d["fee"], 22.5)


class TestOrder(unittest.TestCase):

    def test_to_dict(self):
        o = Order(
            order_id="O1", decision_id="D1", ticker="AAPL",
            order_type=OrderType.BUY_TO_OPEN, quantity=100,
            execution_date="2024-01-02",
        )
        d = o.to_dict()
        self.assertEqual(d["order_type"], "BUY_TO_OPEN")
        self.assertEqual(d["status"], "PENDING")


if __name__ == "__main__":
    unittest.main()
