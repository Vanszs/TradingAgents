"""
Unit tests for ExtendedDecision and ParsedDecision from position.py.
"""
import unittest

from tradingagents.backtesting.position import (
    ExtendedDecision,
    ParsedDecision,
)


class TestParsedDecision(unittest.TestCase):

    def test_invalid_factory(self):
        pd = ParsedDecision.invalid("AAPL", "2024-01-02", "missing report")
        self.assertFalse(pd.valid)
        self.assertEqual(pd.invalid_reason, "missing report")
        self.assertEqual(pd.agent_rating, "")

    def test_to_dict(self):
        pd = ParsedDecision(
            decision_id="D1", ticker="AAPL", trade_date="2024-01-02",
            report_generated_at="2024-01-02 16:30:00",
            last_data_date="2024-01-02",
            decision_valid_from="2024-01-02",
            agent_rating="strong_buy",
            allocation_pct=0.25,
            leverage=1.5,
            short_allowed=True,
        )
        d = pd.to_dict()
        self.assertEqual(d["agent_rating"], "strong_buy")
        self.assertAlmostEqual(d["allocation_pct"], 0.25)
        self.assertAlmostEqual(d["leverage"], 1.5)
        self.assertTrue(d["short_allowed"])


class TestExtendedDecision(unittest.TestCase):

    def test_invalid_factory(self):
        ed = ExtendedDecision.invalid("AAPL", "2024-01-02", "parse error")
        self.assertFalse(ed.valid)
        self.assertEqual(ed.invalid_reason, "parse error")

    def test_full_decision_to_dict(self):
        ed = ExtendedDecision(
            decision_id="D1", ticker="AAPL", trade_date="2024-01-02",
            report_generated_at="2024-01-02 16:30:00",
            last_data_date="2024-01-02",
            decision_valid_from="2024-01-02",
            agent_rating="strong_buy",
            normalized_rating="strong_buy",
            allocation_pct=0.30,
            leverage=2.0,
            short_allowed=True,
            market_mode="FUTURES_STYLE_SIMULATION",
            allowed_position_sides="LONG,SHORT",
            position_intent="open",
            current_position_side="FLAT",
            target_position_side="LONG",
            futures_action="BUY_TO_OPEN",
            stop_price=140.0,
            take_profit=180.0,
        )
        d = ed.to_dict()
        self.assertEqual(d["futures_action"], "BUY_TO_OPEN")
        self.assertEqual(d["position_intent"], "open")
        self.assertEqual(d["target_position_side"], "LONG")
        self.assertAlmostEqual(d["allocation_pct"], 0.30)
        self.assertAlmostEqual(d["stop_price"], 140.0)

    def test_reverse_decision_fields(self):
        ed = ExtendedDecision(
            decision_id="D1", ticker="AAPL", trade_date="2024-01-02",
            report_generated_at="2024-01-02 16:30:00",
            last_data_date="2024-01-02",
            decision_valid_from="2024-01-02",
            agent_rating="strong_sell",
            normalized_rating="strong_sell",
            position_intent="reverse",
            current_position_side="LONG",
            target_position_side="SHORT",
            futures_action="REVERSE_TO_SHORT",
        )
        self.assertEqual(ed.futures_action, "REVERSE_TO_SHORT")
        self.assertEqual(ed.current_position_side, "LONG")
        self.assertEqual(ed.target_position_side, "SHORT")


if __name__ == "__main__":
    unittest.main()
