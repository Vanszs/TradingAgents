"""
Unit tests for DecisionStateManager in ``strict_5tier`` mode.

The strict 5-tier table (with pyramiding and gradual exit):

    Rating \\ Pos | FLAT          | LONG           | SHORT
    --------------+---------------+----------------+-----------------
    Buy           | OPEN_LONG     | HOLD_LONG      | CLOSE_SHORT
    Overweight    | OPEN_LONG     | INCREASE_LONG  | REDUCE_SHORT
    Hold          | NO_ACTION     | HOLD_LONG      | HOLD_SHORT
    Underweight   | NO_ACTION     | REDUCE_LONG    | INCREASE_SHORT
    Sell          | OPEN_SHORT    | CLOSE_LONG     | HOLD_SHORT

Invariants:
* Strict 5tier never auto-reverses LONG <-> SHORT.
* Buy when Long = HOLD_LONG (no pyramiding for Buy).
* Overweight when Long = INCREASE_LONG (pyramiding +10%).
* Underweight when Short = INCREASE_SHORT (pyramiding +10%).
* Underweight when Long = REDUCE_LONG (gradual exit -10%).
* Overweight when Short = REDUCE_SHORT (gradual exit -10%).
* Hold preserves the existing position side.
"""
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
    PositionIntent,
    PositionSide,
)


def _dec(rating: str, ticker: str = "AAPL", trade_date: str = "2024-01-02") -> ParsedDecision:
    return ParsedDecision(
        decision_id=f"{ticker}-{trade_date}-D1",
        ticker=ticker,
        trade_date=trade_date,
        report_generated_at=f"{trade_date} 16:30:00",
        last_data_date=trade_date,
        decision_valid_from=trade_date,
        agent_rating=rating,
    )


def _pos(side: PositionSide = PositionSide.FLAT, qty: int = 0, price: float = 150.0) -> Position:
    return Position(ticker="AAPL", quantity=qty, avg_entry_price=price, mark_price=price)


class TestStrictCanonical(unittest.TestCase):
    def test_canonical_lowercase(self):
        for r in ("buy", "overweight", "hold", "underweight", "sell"):
            self.assertEqual(_canonical_rating(r), r.capitalize())

    def test_canonical_capitalised(self):
        self.assertEqual(_canonical_rating("Buy"), "Buy")
        self.assertEqual(_canonical_rating("Overweight"), "Overweight")
        self.assertEqual(_canonical_rating("Underweight"), "Underweight")

    def test_canonical_invalid(self):
        self.assertIsNone(_canonical_rating("strong_buy"))
        self.assertIsNone(_canonical_rating(""))
        self.assertIsNone(_canonical_rating(None))


class TestStrictFlat(unittest.TestCase):
    def setUp(self):
        self.dm = DecisionStateManager(DecisionMappingConfig(
            mode="strict_5tier",
            allow_short_on_sell=True,
            allow_short_on_underweight=False,
        ))

    def test_flat_buy_opens_long(self):
        d = self.dm.map(_dec(RATING_BUY), _pos())
        self.assertEqual(d.position_intent, PositionIntent.OPEN.value)
        self.assertEqual(d.target_position_side, PositionSide.LONG.value)
        self.assertEqual(d.futures_action, OrderType.BUY_TO_OPEN.value)

    def test_flat_overweight_opens_long(self):
        d = self.dm.map(_dec(RATING_OVERWEIGHT), _pos())
        self.assertEqual(d.position_intent, PositionIntent.OPEN.value)
        self.assertEqual(d.target_position_side, PositionSide.LONG.value)
        self.assertEqual(d.futures_action, OrderType.BUY_TO_OPEN.value)
        # Uses initial_entry_pct (30%)
        self.assertAlmostEqual(d.allocation_pct, 0.3)

    def test_flat_hold_no_action(self):
        d = self.dm.map(_dec(RATING_HOLD), _pos())
        self.assertEqual(d.position_intent, PositionIntent.HOLD.value)
        self.assertEqual(d.target_position_side, PositionSide.FLAT.value)
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)
        self.assertEqual(d.allocation_pct, 0.0)

    def test_flat_underweight_no_action(self):
        d = self.dm.map(_dec(RATING_UNDERWEIGHT), _pos())
        self.assertEqual(d.position_intent, PositionIntent.HOLD.value)
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_flat_sell_opens_short(self):
        d = self.dm.map(_dec(RATING_SELL), _pos())
        self.assertEqual(d.position_intent, PositionIntent.OPEN.value)
        self.assertEqual(d.target_position_side, PositionSide.SHORT.value)
        self.assertEqual(d.futures_action, OrderType.SELL_TO_OPEN.value)


class TestStrictLong(unittest.TestCase):
    def setUp(self):
        self.dm = DecisionStateManager(DecisionMappingConfig(
            mode="strict_5tier",
            allow_short_on_sell=True,
        ))

    def _long(self) -> Position:
        return _pos(PositionSide.LONG, 100, 150.0)

    def test_long_buy_holds_no_pyramiding(self):
        d = self.dm.map(_dec(RATING_BUY), self._long())
        self.assertEqual(d.position_intent, PositionIntent.HOLD.value)
        self.assertEqual(d.target_position_side, PositionSide.LONG.value)
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_long_overweight_increases_pyramiding(self):
        d = self.dm.map(_dec(RATING_OVERWEIGHT), self._long())
        self.assertEqual(d.position_intent, PositionIntent.INCREASE.value)
        self.assertEqual(d.target_position_side, PositionSide.LONG.value)
        self.assertEqual(d.futures_action, OrderType.BUY_TO_ADD.value)
        # Uses pyramid_pct (10%)
        self.assertAlmostEqual(d.allocation_pct, 0.1)

    def test_long_hold_holds(self):
        d = self.dm.map(_dec(RATING_HOLD), self._long())
        self.assertEqual(d.position_intent, PositionIntent.HOLD.value)
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_long_underweight_reduces(self):
        d = self.dm.map(_dec(RATING_UNDERWEIGHT), self._long())
        self.assertEqual(d.position_intent, PositionIntent.REDUCE.value)
        self.assertEqual(d.futures_action, OrderType.SELL_TO_REDUCE.value)

    def test_long_sell_closes(self):
        d = self.dm.map(_dec(RATING_SELL), self._long())
        self.assertEqual(d.position_intent, PositionIntent.CLOSE.value)
        self.assertEqual(d.target_position_side, PositionSide.FLAT.value)
        self.assertEqual(d.futures_action, OrderType.SELL_TO_CLOSE.value)
        # Never auto-reverse to short
        self.assertNotEqual(d.futures_action, OrderType.REVERSE_TO_SHORT.value)


class TestStrictShort(unittest.TestCase):
    def setUp(self):
        self.dm = DecisionStateManager(DecisionMappingConfig(
            mode="strict_5tier",
            allow_short_on_sell=True,
        ))

    def _short(self) -> Position:
        return _pos(PositionSide.SHORT, -100, 150.0)

    def test_short_buy_closes(self):
        d = self.dm.map(_dec(RATING_BUY), self._short())
        self.assertEqual(d.position_intent, PositionIntent.CLOSE.value)
        self.assertEqual(d.target_position_side, PositionSide.FLAT.value)
        self.assertEqual(d.futures_action, OrderType.BUY_TO_CLOSE.value)
        # Never auto-reverse to long
        self.assertNotEqual(d.futures_action, OrderType.REVERSE_TO_LONG.value)

    def test_short_overweight_reduces_gradual_exit(self):
        d = self.dm.map(_dec(RATING_OVERWEIGHT), self._short())
        self.assertEqual(d.position_intent, PositionIntent.REDUCE.value)
        self.assertEqual(d.futures_action, OrderType.BUY_TO_REDUCE.value)
        # Uses reduce_step_pct (10%)
        self.assertAlmostEqual(d.allocation_pct, 0.1)

    def test_short_hold_holds(self):
        d = self.dm.map(_dec(RATING_HOLD), self._short())
        self.assertEqual(d.position_intent, PositionIntent.HOLD.value)
        self.assertEqual(d.target_position_side, PositionSide.SHORT.value)
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_short_underweight_adds(self):
        d = self.dm.map(_dec(RATING_UNDERWEIGHT), self._short())
        self.assertEqual(d.position_intent, PositionIntent.INCREASE.value)
        self.assertEqual(d.futures_action, OrderType.SELL_TO_ADD.value)

    def test_short_sell_holds(self):
        d = self.dm.map(_dec(RATING_SELL), self._short())
        self.assertEqual(d.position_intent, PositionIntent.HOLD.value)
        self.assertEqual(d.target_position_side, PositionSide.SHORT.value)
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)


class TestStrictNoReverse(unittest.TestCase):
    """The mapper must never produce REVERSE_TO_* in strict_5tier mode."""

    def setUp(self):
        self.dm = DecisionStateManager(DecisionMappingConfig(
            mode="strict_5tier",
            allow_short_on_sell=True,
            allow_explicit_reverse=False,
        ))

    def test_sell_when_long_does_not_reverse(self):
        d = self.dm.map(_dec(RATING_SELL), _pos(PositionSide.LONG, 100))
        self.assertNotEqual(d.futures_action, OrderType.REVERSE_TO_SHORT.value)
        self.assertEqual(d.futures_action, OrderType.SELL_TO_CLOSE.value)

    def test_buy_when_short_does_not_reverse(self):
        d = self.dm.map(_dec(RATING_BUY), _pos(PositionSide.SHORT, -100))
        self.assertNotEqual(d.futures_action, OrderType.REVERSE_TO_LONG.value)
        self.assertEqual(d.futures_action, OrderType.BUY_TO_CLOSE.value)

    def test_overweight_when_short_does_not_reverse(self):
        d = self.dm.map(_dec(RATING_OVERWEIGHT), _pos(PositionSide.SHORT, -100))
        self.assertNotEqual(d.futures_action, OrderType.REVERSE_TO_LONG.value)

    def test_underweight_when_long_does_not_reverse(self):
        d = self.dm.map(_dec(RATING_UNDERWEIGHT), _pos(PositionSide.LONG, 100))
        self.assertNotEqual(d.futures_action, OrderType.REVERSE_TO_SHORT.value)


class TestStrictUnderweightShortAllowed(unittest.TestCase):
    def test_underweight_short_allowed_flag_default(self):
        """By default, Underweight+FLAT is NO_ORDER (no auto-open short)."""
        dm = DecisionStateManager(DecisionMappingConfig(
            mode="strict_5tier",
        ))
        d = dm.map(_dec(RATING_UNDERWEIGHT), _pos())
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)

    def test_underweight_short_allowed_flag_opt_in(self):
        """With allow_short_on_underweight=True, Underweight+FLAT opens short."""
        dm = DecisionStateManager(DecisionMappingConfig(
            mode="strict_5tier",
            allow_short_on_underweight=True,
        ))
        d = dm.map(_dec(RATING_UNDERWEIGHT), _pos())
        self.assertEqual(d.futures_action, OrderType.SELL_TO_OPEN.value)
        self.assertEqual(d.target_position_side, "SHORT")


class TestStrictDefaults(unittest.TestCase):
    """When mode is unspecified the default must be strict_5tier."""

    def test_default_mode_is_strict(self):
        cfg = DecisionMappingConfig()
        self.assertEqual(cfg.mode, "strict_5tier")

    def test_default_does_not_pyramid(self):
        dm = DecisionStateManager(DecisionMappingConfig())
        d = dm.map(_dec(RATING_BUY), _pos(PositionSide.LONG, 100))
        self.assertEqual(d.futures_action, OrderType.NO_ORDER.value)


if __name__ == "__main__":
    unittest.main()
