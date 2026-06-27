"""
Unit tests for Portfolio (long, short, settlement, margin breach).
"""
import unittest

from tradingagents.backtesting.decision_schema import (
    OpenClose,
    OrderSide,
    Trade,
)
from tradingagents.backtesting.portfolio import InsufficientMarginError, Portfolio


def _make_trade(
    side: OrderSide,
    quantity: int,
    price: float,
    multiplier: float = 1.0,
    open_close: OpenClose = OpenClose.AUTO,
    reason: str = "agent_open",
) -> Trade:
    return Trade(
        date="2024-01-02",
        ticker="TEST",
        side=side,
        quantity=quantity,
        price=price,
        gross_amount=price * quantity * multiplier,
        fee=0.0,
        net_amount=price * quantity * multiplier,
        multiplier=multiplier,
        notional=price * quantity * multiplier,
        tick_size=0.01,
        slippage_ticks=0,
        realized_pnl_delta=0.0,
        margin_delta=0.0,
        open_close=open_close,
        reason=reason,
        decision_id="D1",
        order_id="O1",
    )


class TestStockPortfolio(unittest.TestCase):

    def test_open_long_then_close_long(self):
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(initial_margin_pct=0.50, maintenance_margin_pct=0.30)
        # Buy 100 shares at $150, margin = 100*150*1.0*0.50 = 7500
        p.apply_trade(_make_trade(OrderSide.BUY, 100, 150.0))
        self.assertEqual(p.position.quantity, 100)
        self.assertAlmostEqual(p.position.avg_price, 150.0)
        self.assertAlmostEqual(p.margin_posted, 7500.0)
        self.assertAlmostEqual(p.cash, 92_500.0)

        # Price moves to 160, then close
        p.apply_daily_settlement(prev_close=150.0, today_close=160.0)
        # Settlement PnL: (160-150)*100*1.0 = 1000
        self.assertAlmostEqual(p.lifetime_realized_pnl, 1000.0)
        p.apply_trade(_make_trade(OrderSide.SELL, 100, 160.0))
        # Close realized: (160-150)*100*1.0 = 1000 (via apply_trade)
        self.assertAlmostEqual(p.lifetime_realized_pnl, 2000.0)
        self.assertEqual(p.position.quantity, 0)
        self.assertAlmostEqual(p.margin_posted, 0.0)

    def test_open_short_then_cover(self):
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(initial_margin_pct=0.50, maintenance_margin_pct=0.30)
        p.apply_trade(_make_trade(OrderSide.SELL, 100, 150.0))
        self.assertEqual(p.position.quantity, -100)
        self.assertAlmostEqual(p.position.avg_price, 150.0)

        # Price falls to 140, short gains
        p.apply_daily_settlement(prev_close=150.0, today_close=140.0)
        # Settlement PnL: (140-150)*(-100)*1.0 = 1000
        self.assertAlmostEqual(p.lifetime_realized_pnl, 1000.0)

        # Cover
        p.apply_trade(_make_trade(OrderSide.BUY, 100, 140.0))
        self.assertEqual(p.position.quantity, 0)
        # Total realized: 1000 from settlement + 1000 from cover = 2000
        self.assertAlmostEqual(p.lifetime_realized_pnl, 2000.0)

    def test_insufficient_margin_rejected(self):
        p = Portfolio(initial_cash=100.0, ticker="AAPL")
        p.configure_margin(initial_margin_pct=0.50, maintenance_margin_pct=0.30)
        # 100 shares at 150, margin needed = 7500, cash = 100
        with self.assertRaises(InsufficientMarginError):
            p.apply_trade(_make_trade(OrderSide.BUY, 100, 150.0))

    def test_margin_breach_detection(self):
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(initial_margin_pct=0.50, maintenance_margin_pct=0.30)
        p.apply_trade(_make_trade(OrderSide.BUY, 100, 150.0))
        # Cash = 92500, margin posted = 7500
        # At mark=150, unrealized=0, equity=92500
        # MM = 100*150*0.30 = 4500. 92500 > 4500, no breach.
        self.assertFalse(p.is_margin_breach(mark_price=150.0))
        # Price drops to 50: unrealized = (50-150)*100 = -10000
        # equity = 92500 - 10000 = 82500, MM = 100*50*0.30 = 1500, no breach
        self.assertFalse(p.is_margin_breach(mark_price=50.0))
        # Use a very large position with small cash to trigger breach
        p2 = Portfolio(initial_cash=10_000.0, ticker="AAPL")
        p2.configure_margin(initial_margin_pct=0.50, maintenance_margin_pct=0.30)
        # 50 shares at 150: margin = 50*150*0.50 = 3750, cash left = 6250
        p2.apply_trade(_make_trade(OrderSide.BUY, 50, 150.0))
        # equity = 6250 + 0 = 6250, MM = 50*150*0.30 = 2250, no breach
        self.assertFalse(p2.is_margin_breach(mark_price=150.0))
        # Price drops to 100: unrealized = (100-150)*50 = -2500
        # equity = 6250 - 2500 = 3750, MM = 50*100*0.30 = 1500, no breach
        self.assertFalse(p2.is_margin_breach(mark_price=100.0))
        # Price drops to 50: unrealized = (50-150)*50 = -5000
        # equity = 6250 - 5000 = 1250, MM = 50*50*0.30 = 750, no breach
        self.assertFalse(p2.is_margin_breach(mark_price=50.0))
        # Price drops to 10: unrealized = (10-150)*50 = -7000
        # equity = 6250 - 7000 = -750, MM = 50*10*0.30 = 150, breach!
        self.assertTrue(p2.is_margin_breach(mark_price=10.0))

    def test_force_liquidate_clears_position(self):
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(initial_margin_pct=0.50, maintenance_margin_pct=0.30)
        p.apply_trade(_make_trade(OrderSide.BUY, 100, 150.0))
        trade = p.force_liquidate(price=130.0, reason="margin_call")
        self.assertEqual(p.position.quantity, 0)
        self.assertAlmostEqual(p.margin_posted, 0.0)
        # Realized PnL: (130-150)*100*1.0 = -2000
        self.assertAlmostEqual(p.lifetime_realized_pnl, -2000.0)
        self.assertEqual(trade.reason, "margin_call")

    def test_daily_settlement_long_profit(self):
        p = Portfolio(initial_cash=50_000.0, ticker="AAPL")
        p.configure_margin(initial_margin_pct=0.50, maintenance_margin_pct=0.30)
        p.apply_trade(_make_trade(OrderSide.BUY, 100, 150.0))
        # Settlement: price rises from 150 to 160
        pnl = p.apply_daily_settlement(prev_close=150.0, today_close=160.0)
        self.assertAlmostEqual(pnl, 1000.0)
        self.assertAlmostEqual(p.cash, 42_500.0 + 1000.0)

    def test_daily_settlement_short_profit(self):
        p = Portfolio(initial_cash=50_000.0, ticker="AAPL")
        p.configure_margin(initial_margin_pct=0.50, maintenance_margin_pct=0.30)
        p.apply_trade(_make_trade(OrderSide.SELL, 100, 150.0))
        # Settlement: price falls from 150 to 140, short gains
        pnl = p.apply_daily_settlement(prev_close=150.0, today_close=140.0)
        self.assertAlmostEqual(pnl, 1000.0)


if __name__ == "__main__":
    unittest.main()
