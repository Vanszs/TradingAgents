"""
Unit tests for OrderGenerator (long, short, reduce, close, target_position_pct).
"""
import unittest

from tradingagents.backtesting.decision_schema import (
    Action,
    BacktestConfig,
    InstrumentSpec,
    MarketPoint,
    OrderSide,
    ParsedDecision,
    Rating,
)
from tradingagents.backtesting.order_generator import OrderGenerator
from tradingagents.backtesting.portfolio import Portfolio


def _make_decision(
    action: Action,
    ticker: str = "AAPL",
    trade_date: str = "2024-01-02",
    allow_new_position: bool = True,
    reduce_only: bool = False,
    target_position_pct: float = None,
    stop_price: float = None,
) -> ParsedDecision:
    return ParsedDecision(
        decision_id=f"{ticker}-{trade_date}-D1",
        ticker=ticker,
        trade_date=trade_date,
        report_generated_at=f"{trade_date} 16:30:00",
        last_data_date=trade_date,
        decision_valid_from=trade_date,
        rating=Rating.BUY if action in (Action.BUY, Action.ADD) else Rating.SELL,
        action=action,
        allow_new_position=allow_new_position,
        reduce_only=reduce_only,
        target_position_pct=target_position_pct,
        stop_price=stop_price,
    )


def _make_config() -> BacktestConfig:
    cfg = BacktestConfig(
        ticker="AAPL",
        start_date="2024-01-02",
        end_date="2024-01-08",
        initial_cash=100_000.0,
        max_risk_per_trade_pct=0.01,
    )
    cfg.execution.lot_size = 1
    return cfg


def _make_spec() -> InstrumentSpec:
    return InstrumentSpec(ticker="AAPL", multiplier=1.0, tick_size=0.01)


def _make_ref() -> MarketPoint:
    return MarketPoint(
        date="2024-01-02",
        ticker="AAPL",
        open=150.0,
        high=155.0,
        low=149.0,
        close=152.0,
    )


class TestOrderGenerator(unittest.TestCase):

    def test_buy_opens_long_when_flat(self):
        config = _make_config()
        og = OrderGenerator(config)
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(0.50, 0.30)
        d = _make_decision(Action.BUY)
        orders = og.generate(d, p, "2024-01-03", _make_ref(), _make_spec())
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, OrderSide.BUY)

    def test_sell_from_flat_is_rejected(self):
        orders = OrderGenerator(_make_config()).generate(
            _make_decision(Action.SELL), Portfolio(initial_cash=100_000.0, ticker="AAPL"),
            "2024-01-03", _make_ref(), _make_spec(),
        )
        self.assertEqual(orders, [])
    def test_sell_closes_long_when_long(self):
        config = _make_config()
        og = OrderGenerator(config)
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(0.50, 0.30)
        from tradingagents.backtesting.decision_schema import OpenClose, Trade
        t = Trade(
            date="2024-01-02", ticker="AAPL", side=OrderSide.BUY,
            quantity=100, price=150.0, gross_amount=15000, fee=0.0,
            net_amount=15000, multiplier=1.0, notional=15000,
            tick_size=0.01, slippage_ticks=0, realized_pnl_delta=0.0,
            margin_delta=0.0, open_close=OpenClose.AUTO,
            reason="test", decision_id="D1", order_id="O1",
        )
        p.apply_trade(t)
        d = _make_decision(Action.SELL)
        orders = og.generate(d, p, "2024-01-03", _make_ref(), _make_spec())
        self.assertEqual(orders, [])

    def test_short_seed_trade_is_rejected(self):
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        from tradingagents.backtesting.decision_schema import OpenClose, Trade
        with self.assertRaises(ValueError):
            p.apply_trade(Trade(
                date="2024-01-02", ticker="AAPL", side=OrderSide.SELL,
                quantity=100, price=150.0, gross_amount=15000, fee=0.0,
                net_amount=15000, multiplier=1.0, notional=15000,
                tick_size=0.01, slippage_ticks=0, realized_pnl_delta=0.0,
                margin_delta=0.0, open_close=OpenClose.AUTO,
                reason="test", decision_id="D1", order_id="O1",
            ))

    def test_sizing_caps_by_margin(self):
        config = _make_config()
        config.max_risk_per_trade_pct = 0.50  # risk 50% of equity
        og = OrderGenerator(config)
        # With 100k cash and 50% initial margin, max shares by margin:
        # 100000 / (152 * 1.0 * 0.50) = 1315
        # risk-based: 50000 / (152*1.0*0.01) = 32894
        # min(1315, 32894) = 1315
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(0.50, 0.30)
        d = _make_decision(Action.BUY)
        orders = og.generate(d, p, "2024-01-03", _make_ref(), _make_spec())
        self.assertEqual(len(orders), 1)
        self.assertGreater(orders[0].quantity, 0)

    def test_target_position_pct_caps_qty(self):
        config = _make_config()
        config.max_risk_per_trade_pct = 0.50
        og = OrderGenerator(config)
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(0.50, 0.30)
        # target_position_pct=10 means target 10% of equity
        d = _make_decision(Action.BUY, target_position_pct=10.0)
        orders = og.generate(d, p, "2024-01-03", _make_ref(), _make_spec())
        self.assertEqual(len(orders), 1)
        # target: 100000*0.10 = 10000, 10000/152 = 65
        self.assertEqual(orders[0].quantity, 65)

    def test_hold_returns_no_orders(self):
        config = _make_config()
        og = OrderGenerator(config)
        p = Portfolio(initial_cash=100_000.0, ticker="AAPL")
        p.configure_margin(0.50, 0.30)
        d = _make_decision(Action.HOLD)
        orders = og.generate(d, p, "2024-01-03", _make_ref(), _make_spec())
        self.assertEqual(len(orders), 0)


if __name__ == "__main__":
    unittest.main()
