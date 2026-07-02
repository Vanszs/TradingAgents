"""
Tests for short position intraday margin breach detection.
"""
import unittest

from tradingagents.backtesting.portfolio import PortfolioV2
from tradingagents.backtesting.position import (
    Fill,
    MarginConfig,
    OrderType,
)


class TestShortIntradayBreach(unittest.TestCase):
    """Short positions should detect intraday margin breaches via intraday_high."""

    def _make_portfolio_with_short(self, cash=100_000.0, qty=100, entry=100.0):
        portfolio = PortfolioV2(
            initial_cash=cash, ticker="TEST",
            margin_config=MarginConfig(initial_margin_pct=0.50, maintenance_margin_pct=0.35),
        )
        fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="SELL",
            quantity=qty, price=entry, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.SELL_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(fill)
        return portfolio

    def test_short_breach_when_high_rises(self):
        """Short: intraday_high rises enough to breach maintenance margin."""
        portfolio = self._make_portfolio_with_short(cash=100_000.0, qty=100, entry=100.0)
        # maintenance_margin = 100 * 200 * 0.35 = 7000
        # loss from open to high = (200-100)*100 = 10000
        # worst_equity ≈ 100000 - 10000 = 90000 > 7000 → no breach
        # Use extreme high to trigger breach
        # maintenance_margin = 100 * 500 * 0.35 = 17500
        # loss = (500-100)*100 = 40000
        # worst_equity = 100000 - 40000 = 60000 > 17500 → still no breach
        # Need: worst_equity < maintenance
        # With qty=100, entry=100, open=100:
        # loss = (high-100)*100
        # worst_equity = 100000 - (high-100)*100
        # maintenance = 100 * high * 0.35
        # Need: 100000 - (high-100)*100 < 100 * high * 0.35
        # 100000 - 100*high + 10000 < 35*high
        # 110000 < 135*high
        # high > 814.8
        result = portfolio.is_intraday_breach(
            open_price=100.0,
            intraday_low=98.0,
            intraday_high=900.0,
        )
        self.assertTrue(result)

    def test_short_no_breach_when_high_equals_open(self):
        """Short: intraday_high == open_price → no breach."""
        portfolio = self._make_portfolio_with_short(cash=100_000.0, qty=100, entry=100.0)
        result = portfolio.is_intraday_breach(
            open_price=100.0,
            intraday_low=98.0,
            intraday_high=100.0,
        )
        self.assertFalse(result)

    def test_short_no_breach_small_rise(self):
        """Short: small price rise → no breach."""
        portfolio = self._make_portfolio_with_short(cash=100_000.0, qty=100, entry=100.0)
        result = portfolio.is_intraday_breach(
            open_price=100.0,
            intraday_low=98.0,
            intraday_high=105.0,
        )
        self.assertFalse(result)

    def test_short_no_high_returns_false(self):
        """Short without intraday_high → returns False."""
        portfolio = self._make_portfolio_with_short(cash=100_000.0, qty=100, entry=100.0)
        result = portfolio.is_intraday_breach(
            open_price=100.0,
            intraday_low=98.0,
            intraday_high=None,
        )
        self.assertFalse(result)

    def test_long_still_uses_low(self):
        """Long: still uses intraday_low for breach detection."""
        portfolio = PortfolioV2(
            initial_cash=100_000.0, ticker="TEST",
            margin_config=MarginConfig(initial_margin_pct=0.50, maintenance_margin_pct=0.35),
        )
        fill = Fill(
            fill_id="open1", order_id="o1", decision_id="d1",
            date="2025-01-02", ticker="TEST", side="BUY",
            quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
            order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
        )
        portfolio.apply_fill(fill)

        # Long: small loss, no breach
        result = portfolio.is_intraday_breach(
            open_price=100.0,
            intraday_low=95.0,
            intraday_high=102.0,
        )
        self.assertFalse(result)

    def test_flat_returns_false(self):
        """Flat position always returns False."""
        portfolio = PortfolioV2(
            initial_cash=100_000.0, ticker="TEST",
            margin_config=MarginConfig(),
        )
        result = portfolio.is_intraday_breach(
            open_price=100.0,
            intraday_low=50.0,
            intraday_high=200.0,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
