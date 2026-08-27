"""Spot long-only execution rejects short positions."""
import unittest

from tradingagents.backtesting.portfolio import PortfolioV2
from tradingagents.backtesting.position import Fill, MarginConfig, OrderType


class TestSpotLongOnlyExecution(unittest.TestCase):
    def test_short_open_is_rejected(self):
        portfolio = PortfolioV2(
            initial_cash=100_000.0,
            ticker="TEST",
            margin_config=MarginConfig(initial_margin_pct=0.50, maintenance_margin_pct=0.35),
        )
        fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1", date="2025-01-02",
            ticker="TEST", side="SELL", quantity=100, price=100.0, fee=0.0,
            slippage_amount=0.0, order_type=OrderType.SELL_TO_OPEN, open_close="OPEN",
        )
        with self.assertRaises(ValueError, msg="short open must be rejected"):
            portfolio.apply_fill(fill)

    def test_long_intraday_breach_uses_low(self):
        portfolio = PortfolioV2(initial_cash=100_000.0, ticker="TEST")
        fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1", date="2025-01-02",
            ticker="TEST", side="BUY", quantity=100, price=100.0, fee=0.0,
            slippage_amount=0.0, order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(fill)
        assert portfolio.is_intraday_breach(100.0, 95.0, 102.0) is False

    def test_flat_intraday_is_safe(self):
        assert PortfolioV2(initial_cash=100_000.0, ticker="TEST").is_intraday_breach(
            100.0, 50.0, 200.0
        ) is False
