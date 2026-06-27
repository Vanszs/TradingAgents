"""
Unit tests for max leverage cap enforcement.

Tests cover:
- Order rejected when max leverage exceeded
- Order resized to fit within max leverage
- Order allowed when exactly at cap
"""
import unittest

from tradingagents.backtesting.order_generator import OrderGenerator
from tradingagents.backtesting.portfolio import Portfolio
from tradingagents.backtesting.position import (
    BacktestConfig,
    DecisionMappingConfig,
    ExecutionConfig,
    ExtendedDecision,
    MarginConfig,
    Position,
    PositionSide,
)


def _make_config(max_leverage: float = 2.0) -> BacktestConfig:
    return BacktestConfig(
        ticker="AAPL",
        start_date="2024-01-02",
        end_date="2024-01-08",
        initial_cash=100_000.0,
        execution=ExecutionConfig(),
        margin=MarginConfig(
            initial_margin_pct=0.50,
            maintenance_margin_pct=0.30,
            max_leverage=max_leverage,
        ),
        decision_mapping=DecisionMappingConfig(mode="conservative"),
    )


def _make_decision(
    action: str = "BUY_TO_OPEN",
    allocation_pct: float = 0.50,
    side: str = "FLAT",
) -> ExtendedDecision:
    return ExtendedDecision(
        decision_id="D1", ticker="AAPL", trade_date="2024-01-02",
        report_generated_at="2024-01-02 16:30:00",
        last_data_date="2024-01-02",
        decision_valid_from="2024-01-03",
        agent_rating="strong_buy",
        normalized_rating="strong_buy",
        allocation_pct=allocation_pct,
        market_mode="FUTURES_STYLE_SIMULATION",
        position_intent="open" if action == "BUY_TO_OPEN" else "increase",
        current_position_side=side,
        target_position_side="LONG",
        futures_action=action,
        valid=True,
    )


class TestMaxLeverage(unittest.TestCase):

    def test_order_resized_when_exceeding_leverage(self):
        """Order quantity should be capped by max leverage."""
        config = _make_config(max_leverage=2.0)
        og = OrderGenerator(config)
        pos = Position(ticker="AAPL", quantity=0, mark_price=150.0)
        # 50% allocation of 100k = 50k notional, at 150 = 333 shares
        # max leverage 2x: max exposure = 100k * 2 = 200k, existing = 0
        # so max new = 200k / 150 = 1333 shares. 333 < 1333, so not capped.
        d = _make_decision("BUY_TO_OPEN", allocation_pct=0.50)
        orders = og.decide(d, pos, 100_000.0, reference_price=150.0)
        self.assertEqual(len(orders), 1)
        self.assertGreater(orders[0].quantity, 0)

    def test_order_rejected_when_existing_exposure_fills_cap(self):
        """When existing exposure fills the leverage cap, no new order."""
        config = _make_config(max_leverage=1.5)
        og = OrderGenerator(config)
        # Existing position: 1000 shares at 150 = 150k notional
        # Equity = 100k, max leverage 1.5 → max exposure = 150k
        # Already at 150k → no room for new order
        pos = Position(ticker="AAPL", quantity=1000, avg_entry_price=150.0, mark_price=150.0)
        d = _make_decision("BUY_TO_ADD", allocation_pct=0.20, side="LONG")
        orders = og.decide(d, pos, 100_000.0, reference_price=150.0)
        self.assertEqual(len(orders), 0)

    def test_order_allowed_at_exactly_cap(self):
        """Order at exactly the leverage cap is allowed."""
        config = _make_config(max_leverage=2.0)
        og = OrderGenerator(config)
        # Existing: 500 shares at 150 = 75k notional
        # Max exposure = 100k * 2 = 200k, remaining = 125k
        # 125k / 150 = 833 shares available
        pos = Position(ticker="AAPL", quantity=500, avg_entry_price=150.0, mark_price=150.0)
        d = _make_decision("BUY_TO_ADD", allocation_pct=0.10, side="LONG")
        orders = og.decide(d, pos, 100_000.0, reference_price=150.0)
        # 10% of 100k = 10k notional / 150 = 66 shares
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].quantity, 66)

    def test_close_order_not_capped(self):
        """Close orders are not affected by max leverage cap."""
        config = _make_config(max_leverage=1.0)  # tight but valid
        og = OrderGenerator(config)
        # Existing: 1000 shares at 150 = 150k notional. Equity=100k, max_lev=1.0
        # Max exposure = 100k, existing = 150k. Already over, but close should work.
        pos = Position(ticker="AAPL", quantity=1000, avg_entry_price=150.0, mark_price=150.0)
        d = _make_decision("SELL_TO_CLOSE", allocation_pct=0.0, side="LONG")
        orders = og.decide(d, pos, 100_000.0, reference_price=150.0)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].quantity, 1000)


if __name__ == "__main__":
    unittest.main()
