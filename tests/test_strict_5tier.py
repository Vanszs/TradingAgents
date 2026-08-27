"""BUY/WNS spot long-only decision mapping tests."""
import unittest

from tradingagents.backtesting.decision_state_manager import (
    RATING_BUY,
    RATING_HOLD,
    RATING_OVERWEIGHT,
    RATING_SELL,
    RATING_UNDERWEIGHT,
    DecisionStateManager,
    _canonical_rating,
)
from tradingagents.backtesting.position import (
    DecisionMappingConfig,
    OrderType,
    ParsedDecision,
    Position,
)


def _dec(rating: str, *, allow_new_position: bool = True) -> ParsedDecision:
    return ParsedDecision(
        decision_id="AAPL-2024-01-02-D1",
        ticker="AAPL",
        trade_date="2024-01-02",
        report_generated_at="2024-01-02 16:30:00",
        last_data_date="2024-01-02",
        decision_valid_from="2024-01-03",
        agent_rating=rating,
        allow_new_position=allow_new_position,
    )


class TestSpotLongOnlyMapping(unittest.TestCase):
    def setUp(self):
        self.dm = DecisionStateManager(DecisionMappingConfig())

    def test_legacy_labels_normalize_to_buy_or_wns(self):
        self.assertEqual(_canonical_rating("buy"), RATING_BUY)
        self.assertEqual(_canonical_rating("overweight"), RATING_BUY)
        for label in ("hold", "underweight", "sell", "wns"):
            self.assertEqual(_canonical_rating(label), "WNS")

    def test_buy_flat_opens_long(self):
        decision = self.dm.map(_dec("BUY"), Position(ticker="AAPL"))
        self.assertEqual(decision.futures_action, OrderType.BUY_TO_OPEN.value)
        self.assertEqual(decision.market_mode, "SPOT_LONG_ONLY")
        self.assertEqual(decision.allowed_position_sides, "LONG")
        self.assertFalse(decision.short_allowed)

    def test_buy_long_does_not_pyramid(self):
        decision = self.dm.map(_dec("BUY"), Position(ticker="AAPL", quantity=100))
        self.assertEqual(decision.futures_action, OrderType.NO_ORDER.value)
        self.assertEqual(decision.position_intent, "hold")
        self.assertEqual(decision.target_position_side, "LONG")

    def test_wns_never_orders(self):
        for rating in ("WNS", "HOLD", "SELL", "UNDERWEIGHT"):
            decision = self.dm.map(_dec(rating), Position(ticker="AAPL", quantity=100))
            self.assertEqual(decision.futures_action, OrderType.NO_ORDER.value)

    def test_buy_respects_no_new_position(self):
        decision = self.dm.map(
            _dec("BUY", allow_new_position=False), Position(ticker="AAPL")
        )
        self.assertEqual(decision.futures_action, OrderType.NO_ORDER.value)
        self.assertEqual(decision.position_intent, "hold")

    def test_no_short_open_add_or_reverse(self):
        for rating in ("BUY", "WNS", "SELL", "UNDERWEIGHT"):
            decision = self.dm.map(_dec(rating), Position(ticker="AAPL"))
            self.assertNotIn(
                decision.futures_action,
                {OrderType.SELL_TO_OPEN.value, OrderType.SELL_TO_ADD.value,
                 OrderType.REVERSE_TO_LONG.value, OrderType.REVERSE_TO_SHORT.value},
            )


if __name__ == "__main__":
    unittest.main()
