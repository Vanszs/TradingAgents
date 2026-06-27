"""
Unit tests for margin engine (generic math).
"""
import unittest

from tradingagents.backtesting.margin_engine import (
    excess_margin,
    initial_margin,
    is_margin_call,
    is_intraday_margin_breach,
    leverage,
    maintenance_margin,
    max_contracts_by_margin,
    notional_value,
)


class TestMarginEngine(unittest.TestCase):

    def test_notional_value(self):
        self.assertAlmostEqual(notional_value(100, 50.0, 1.0), 5000.0)

    def test_initial_margin(self):
        self.assertAlmostEqual(initial_margin(100, 50.0, 1.0, 0.5), 2500.0)

    def test_maintenance_margin(self):
        self.assertAlmostEqual(maintenance_margin(100, 50.0, 1.0, 0.3), 1500.0)

    def test_max_contracts_by_margin(self):
        # 10000 cash, 50 price, 1.0 multiplier, 0.50 margin
        # per_contract = 50*1.0*0.50 = 25
        # max = 10000/25 = 400
        self.assertEqual(max_contracts_by_margin(10000, 50.0, 1.0, 0.5), 400)

    def test_excess_margin(self):
        # equity=10000, qty=100, mark=50, mult=1.0, mm_pct=0.30
        # MM = 100*50*1.0*0.30 = 1500
        # excess = 10000 - 1500 = 8500
        self.assertAlmostEqual(excess_margin(10000, 100, 50.0, 1.0, 0.30), 8500.0)

    def test_is_margin_call_true(self):
        # equity=1400, qty=100, mark=50, mult=1.0, mm=0.30
        # MM = 1500, 1400 < 1500 -> True
        self.assertTrue(is_margin_call(1400, 100, 50.0, 1.0, 0.30))

    def test_is_margin_call_false(self):
        # equity=2000, MM=1500, 2000 >= 1500 -> False
        self.assertFalse(is_margin_call(2000, 100, 50.0, 1.0, 0.30))

    def test_leverage(self):
        # notional=5000, equity=10000, leverage=0.5
        self.assertAlmostEqual(leverage(100, 50.0, 1.0, 10000.0), 0.5)


if __name__ == "__main__":
    unittest.main()
