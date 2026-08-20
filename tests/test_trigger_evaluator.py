"""
Unit tests for the Trigger-Based Execution Agent (TriggerEvaluator).
"""
import unittest

from tradingagents.backtesting.position import (
    ExtendedDecision,
    Order,
    OrderType,
    Position,
    PositionSide,
)
from tradingagents.backtesting.trigger_evaluator import (
    TriggerConfig,
    TriggerEvaluator,
)


def _dec(
    rating: str,
    trade_date: str = "2024-01-02",
    confidence: float = 0.5,
    raw_text: str = "",
) -> ExtendedDecision:
    return ExtendedDecision(
        decision_id=f"T-{trade_date}",
        ticker="AAPL",
        trade_date=trade_date,
        report_generated_at=f"{trade_date} 16:30:00",
        last_data_date=trade_date,
        decision_valid_from=trade_date,
        agent_rating=rating,
        confidence=confidence,
        raw_text_excerpt=raw_text,
    )


def _pos(side: PositionSide = PositionSide.FLAT, qty: int = 0) -> Position:
    return Position(ticker="AAPL", quantity=qty, avg_entry_price=100.0)


def _risk_order(reason: str) -> Order:
    return Order(
        order_id="risk_1",
        decision_id="",
        ticker="AAPL",
        order_type=OrderType.SELL_TO_CLOSE,
        quantity=100,
        execution_date="2024-01-02",
        reason=reason,
    )


class TestTriggerGating(unittest.TestCase):
    def test_disabled_always_fires(self):
        ev = TriggerEvaluator(TriggerConfig(enabled=False))
        # Even Hold should fire when gate is disabled.
        r = ev.evaluate(_dec("Hold"), None, _pos())
        self.assertTrue(r.triggered)
        self.assertIn("triggers_disabled", r.reasons)


class TestTpSlHit(unittest.TestCase):
    def test_stop_loss_fires(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Buy"), _dec("Hold"), _pos(), risk_order=_risk_order("stop_loss"))
        self.assertIn("tp_sl_hit", r.reasons)

    def test_take_profit_fires(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Buy"), _dec("Hold"), _pos(), risk_order=_risk_order("take_profit"))
        self.assertIn("tp_sl_hit", r.reasons)

    def test_no_risk_order_does_not_fire(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Buy"), _dec("Hold"), _pos(), risk_order=None)
        self.assertNotIn("tp_sl_hit", r.reasons)


class TestSetupInvalid(unittest.TestCase):
    def test_two_keywords_fire(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(
            _dec("Buy", raw_text="Thesis broken. Setup invalid. Tidak valid lagi."),
            _dec("Hold"),
            _pos(),
        )
        self.assertIn("setup_invalid", r.reasons)

    def test_one_keyword_does_not_fire(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(
            _dec("Buy", raw_text="Setup invalid."),
            _dec("Hold"),
            _pos(),
        )
        self.assertNotIn("setup_invalid", r.reasons)

    def test_no_text_does_not_fire(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Buy"), _dec("Hold"), _pos())
        self.assertNotIn("setup_invalid", r.reasons)


class TestRrDeteriorated(unittest.TestCase):
    def test_confidence_drop_fires(self):
        ev = TriggerEvaluator(TriggerConfig(rr_deterioration_pct=0.30))
        r = ev.evaluate(
            _dec("Buy", confidence=0.4),
            _dec("Buy", confidence=0.8),
            _pos(),
        )
        self.assertIn("rr_deteriorated", r.reasons)

    def test_small_drop_does_not_fire(self):
        ev = TriggerEvaluator(TriggerConfig(rr_deterioration_pct=0.30))
        r = ev.evaluate(
            _dec("Buy", confidence=0.75),
            _dec("Buy", confidence=0.8),
            _pos(),
        )
        self.assertNotIn("rr_deteriorated", r.reasons)

    def test_no_confidence_does_not_fire(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Buy", confidence=None), _dec("Buy", confidence=None), _pos())
        self.assertNotIn("rr_deteriorated", r.reasons)


class TestStrongExitSignal(unittest.TestCase):
    def test_sell_when_long_fires(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Sell"), _dec("Hold"), _pos(PositionSide.LONG, 100))
        self.assertIn("strong_exit_signal", r.reasons)

    def test_buy_when_short_fires(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Buy"), _dec("Hold"), _pos(PositionSide.SHORT, -100))
        self.assertIn("strong_exit_signal", r.reasons)

    def test_sell_when_short_does_not_fire(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Sell"), _dec("Hold"), _pos(PositionSide.SHORT, -100))
        self.assertNotIn("strong_exit_signal", r.reasons)

    def test_buy_when_long_does_not_fire(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Buy"), _dec("Hold"), _pos(PositionSide.LONG, 100))
        self.assertNotIn("strong_exit_signal", r.reasons)


class TestBetterCandidate(unittest.TestCase):
    def test_confidence_ratio_fires(self):
        ev = TriggerEvaluator(TriggerConfig(
            better_candidate_confidence_ratio=1.5,
            better_candidate_min_confidence=0.6,
        ))
        r = ev.evaluate(
            _dec("Buy", confidence=0.9),
            _dec("Hold", confidence=0.5),
            _pos(),
        )
        self.assertIn("better_candidate", r.reasons)

    def test_below_min_confidence_does_not_fire(self):
        ev = TriggerEvaluator(TriggerConfig(
            better_candidate_confidence_ratio=1.5,
            better_candidate_min_confidence=0.95,
        ))
        r = ev.evaluate(
            _dec("Buy", confidence=0.9),
            _dec("Hold", confidence=0.5),
            _pos(),
        )
        self.assertNotIn("better_candidate", r.reasons)


class TestEntryConditionChanged(unittest.TestCase):
    def test_flat_and_rating_changed_fires(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Buy"), _dec("Hold"), _pos())
        self.assertIn("entry_condition_changed", r.reasons)

    def test_flat_hold_does_not_fire(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Hold"), _dec("Hold"), _pos())
        self.assertNotIn("entry_condition_changed", r.reasons)

    def test_with_position_does_not_fire(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(_dec("Buy"), _dec("Hold"), _pos(PositionSide.LONG, 100))
        self.assertNotIn("entry_condition_changed", r.reasons)


class TestCombination(unittest.TestCase):
    def test_multiple_reasons_aggregate(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(
            _dec("Sell", confidence=0.4, raw_text="Thesis broken. Setup invalid."),
            _dec("Buy", confidence=0.8),
            _pos(PositionSide.LONG, 100),
            risk_order=_risk_order("stop_loss"),
        )
        # tp_sl_hit, setup_invalid, rr_deteriorated, strong_exit_signal
        self.assertTrue(r.triggered)
        self.assertGreaterEqual(len(r.reasons), 3)

    def test_no_trigger_when_all_silent(self):
        ev = TriggerEvaluator()
        r = ev.evaluate(
            _dec("Hold", confidence=0.7),
            _dec("Hold", confidence=0.7),
            _pos(),
        )
        self.assertFalse(r.triggered)
        self.assertEqual(r.reasons, [])


class TestPerTriggerToggle(unittest.TestCase):
    def test_disable_each_condition_independently(self):
        for cond in (
            "trigger_on_tp_sl_hit",
            "trigger_on_setup_invalid",
            "trigger_on_rr_deteriorated",
            "trigger_on_strong_exit_signal",
            "trigger_on_better_candidate",
            "trigger_on_entry_condition_changed",
        ):
            cfg = TriggerConfig(**{cond: False})
            ev = TriggerEvaluator(cfg)
            # Pick a setup that would normally fire all of them.
            r = ev.evaluate(
                _dec("Sell", confidence=0.4, raw_text="Setup invalid. Thesis broken."),
                _dec("Buy", confidence=0.8),
                _pos(PositionSide.LONG, 100),
                risk_order=_risk_order("stop_loss"),
            )
            self.assertNotIn(cond.replace("trigger_on_", ""), r.reasons)


class TestPriceLevelTouched(unittest.TestCase):
    def test_price_touch_triggers_when_flat(self):
        ev = TriggerEvaluator()
        d = _dec("Hold")
        d.planned_entry_price = 150.0

        class MockBar:
            low = 148.0
            high = 152.0

        r = ev.evaluate(d, _dec("Hold"), _pos(), current_bar=MockBar())
        self.assertTrue(r.triggered)
        self.assertIn("price_level_touched", r.reasons)

    def test_price_not_touch_does_not_trigger(self):
        ev = TriggerEvaluator()
        d = _dec("Hold")
        d.planned_entry_price = 150.0

        class MockBar:
            low = 155.0
            high = 160.0

        r = ev.evaluate(d, _dec("Hold"), _pos(), current_bar=MockBar())
        self.assertFalse(r.triggered)
        self.assertNotIn("price_level_touched", r.reasons)


if __name__ == "__main__":
    unittest.main()
