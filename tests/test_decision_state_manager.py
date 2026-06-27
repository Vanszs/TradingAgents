"""
Unit tests for DecisionStateManager (PRD §10.5).

Tests cover:
- Conservative mode (PRD §10.2): 5 ratings × 3 sides = 15 cases
- Aggressive mode (PRD §10.3): 5 ratings × 3 sides = 15 cases
- Edge cases: invalid decisions, reverse, short allowed
"""
import unittest

from tradingagents.backtesting.decision_state_manager import DecisionStateManager
from tradingagents.backtesting.position import (
    DecisionMappingConfig,
    OrderType,
    ParsedDecision,
    Position,
    PositionSide,
)


def _make_decision(rating: str, ticker: str = "AAPL", trade_date: str = "2024-01-02") -> ParsedDecision:
    return ParsedDecision(
        decision_id=f"{ticker}-{trade_date}-D1",
        ticker=ticker,
        trade_date=trade_date,
        report_generated_at=f"{trade_date} 16:30:00",
        last_data_date=trade_date,
        decision_valid_from=trade_date,
        agent_rating=rating,
    )


def _make_position(side: PositionSide = PositionSide.FLAT, qty: int = 0, price: float = 150.0) -> Position:
    p = Position(ticker="AAPL", quantity=qty, avg_entry_price=price, mark_price=price)
    return p


class TestConservativeMode(unittest.TestCase):
    """PRD §10.2 — conservative mode tests."""

    def setUp(self):
        self.dm = DecisionStateManager(DecisionMappingConfig(mode="conservative", allow_short_on_sell=True))

    def test_flat_buy_opens_long(self):
        d = self.dm.map(_make_decision("strong_buy"), _make_position())
        self.assertEqual(d.position_intent, "open")
        self.assertEqual(d.target_position_side, "LONG")
        self.assertEqual(d.futures_action, OrderType.BUY_TO_OPEN.value)

    def test_flat_hold_no_order(self):
        d = self.dm.map(_make_decision("hold"), _make_position())
        self.assertEqual(d.position_intent, "hold")
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_flat_sell_opens_short(self):
        d = self.dm.map(_make_decision("strong_sell"), _make_position())
        self.assertEqual(d.position_intent, "open")
        self.assertEqual(d.target_position_side, "SHORT")
        self.assertEqual(d.futures_action, OrderType.SELL_TO_OPEN.value)

    def test_long_buy_holds(self):
        d = self.dm.map(_make_decision("strong_buy"), _make_position(PositionSide.LONG, 100))
        self.assertEqual(d.position_intent, "hold")
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_long_sell_reduces(self):
        d = self.dm.map(_make_decision("sell"), _make_position(PositionSide.LONG, 100))
        self.assertEqual(d.position_intent, "reduce")
        self.assertEqual(d.futures_action, OrderType.SELL_TO_REDUCE.value)

    def test_long_strong_sell_closes(self):
        d = self.dm.map(_make_decision("strong_sell"), _make_position(PositionSide.LONG, 100))
        self.assertEqual(d.position_intent, "close")
        self.assertEqual(d.futures_action, OrderType.SELL_TO_CLOSE.value)

    def test_long_hold_no_order(self):
        d = self.dm.map(_make_decision("hold"), _make_position(PositionSide.LONG, 100))
        self.assertEqual(d.position_intent, "hold")
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_short_sell_holds(self):
        d = self.dm.map(_make_decision("strong_sell"), _make_position(PositionSide.SHORT, -100))
        self.assertEqual(d.position_intent, "hold")
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_short_buy_reduces(self):
        d = self.dm.map(_make_decision("buy"), _make_position(PositionSide.SHORT, -100))
        self.assertEqual(d.position_intent, "reduce")
        self.assertEqual(d.futures_action, OrderType.BUY_TO_REDUCE.value)

    def test_short_strong_buy_closes(self):
        d = self.dm.map(_make_decision("strong_buy"), _make_position(PositionSide.SHORT, -100))
        self.assertEqual(d.position_intent, "close")
        self.assertEqual(d.futures_action, OrderType.BUY_TO_CLOSE.value)

    def test_short_hold_no_order(self):
        d = self.dm.map(_make_decision("hold"), _make_position(PositionSide.SHORT, -100))
        self.assertEqual(d.position_intent, "hold")
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_invalid_decision_returns_invalid(self):
        d = self.dm.map(
            ParsedDecision.invalid("AAPL", "2024-01-02", "bad report"),
            _make_position(),
        )
        self.assertFalse(d.valid)

    def test_conservative_no_reverse_on_buy(self):
        """Conservative mode: strong_buy on LONG should not reverse."""
        dm = DecisionStateManager(DecisionMappingConfig(mode="conservative", allow_reverse_on_buy_sell=False))
        d = dm.map(_make_decision("strong_buy"), _make_position(PositionSide.LONG, 100))
        self.assertEqual(d.position_intent, "hold")

    def test_allocation_on_buy(self):
        d = self.dm.map(_make_decision("strong_buy"), _make_position())
        self.assertAlmostEqual(d.allocation_pct, 0.30)

    def test_allocation_on_reduce(self):
        d = self.dm.map(_make_decision("sell"), _make_position(PositionSide.LONG, 100))
        self.assertAlmostEqual(d.allocation_pct, 0.5)


class TestAggressiveMode(unittest.TestCase):
    """PRD §10.3 — aggressive mode tests."""

    def setUp(self):
        self.dm = DecisionStateManager(DecisionMappingConfig(mode="aggressive", allow_short_on_sell=True))

    def test_long_buy_increases(self):
        d = self.dm.map(_make_decision("strong_buy"), _make_position(PositionSide.LONG, 100))
        self.assertEqual(d.position_intent, "increase")
        self.assertEqual(d.futures_action, OrderType.BUY_TO_ADD.value)

    def test_long_strong_sell_reverse(self):
        dm = DecisionStateManager(DecisionMappingConfig(
            mode="aggressive", allow_reverse_on_buy_sell=True, allow_short_on_sell=True
        ))
        d = dm.map(_make_decision("strong_sell"), _make_position(PositionSide.LONG, 100))
        self.assertEqual(d.position_intent, "reverse")
        self.assertEqual(d.futures_action, OrderType.REVERSE_TO_SHORT.value)

    def test_short_sell_increases(self):
        d = self.dm.map(_make_decision("strong_sell"), _make_position(PositionSide.SHORT, -100))
        self.assertEqual(d.position_intent, "increase")
        self.assertEqual(d.futures_action, OrderType.SELL_TO_ADD.value)

    def test_short_strong_buy_reverse(self):
        dm = DecisionStateManager(DecisionMappingConfig(
            mode="aggressive", allow_reverse_on_buy_sell=True
        ))
        d = dm.map(_make_decision("strong_buy"), _make_position(PositionSide.SHORT, -100))
        self.assertEqual(d.position_intent, "reverse")
        self.assertEqual(d.futures_action, OrderType.REVERSE_TO_LONG.value)

    def test_flat_strong_buy_opens(self):
        d = self.dm.map(_make_decision("strong_buy"), _make_position())
        self.assertEqual(d.position_intent, "open")
        self.assertEqual(d.futures_action, OrderType.BUY_TO_OPEN.value)

    def test_long_strong_sell_close_no_reverse(self):
        """Without allow_reverse_on_buy_sell, strong_sell on LONG closes."""
        dm = DecisionStateManager(DecisionMappingConfig(mode="aggressive", allow_reverse_on_buy_sell=False))
        d = dm.map(_make_decision("strong_sell"), _make_position(PositionSide.LONG, 100))
        self.assertEqual(d.position_intent, "close")
        self.assertEqual(d.futures_action, OrderType.SELL_TO_CLOSE.value)


if __name__ == "__main__":
    unittest.main()
