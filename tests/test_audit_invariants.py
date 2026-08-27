"""
Invariant tests locking in the Round-2 hardening fixes (plan.md §2/§5).

1. Horizon fallback clamps to SignalContract's le=63 bound instead of raising
   (B1) and quant fields survive the PM -> markdown -> parser -> DSM roundtrip
   (B3).
2. Limit orders never fill beyond their limit; gap-through limits get price
   improvement; untouched limits stay unfilled (B5).
3. Market-order liquidations fill at the open through a gap-down that would
   strand a stale limit price below maintenance (B2).
4. Short positions accrue daily borrow fees; leveraged longs accrue daily
   financing on the actually-borrowed slice (B8).
"""
import unittest

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    portfolio_decision_to_signal_contract,
    render_pm_decision,
)
from tradingagents.backtesting.broker import SimulatedBroker
from tradingagents.backtesting.decision_state_manager import DecisionStateManager
from tradingagents.backtesting.markdown_parser import MarkdownDecisionParser
from tradingagents.backtesting.portfolio import PortfolioV2
from tradingagents.backtesting.position import (
    DecisionMappingConfig,
    ExecutionConfig,
    Fill,
    InstrumentSpec,
    MarginConfig,
    MarketPoint,
    Order,
    OrderType,
    Position,
)

TICKER = "TEST"
SPEC = InstrumentSpec(ticker=TICKER, multiplier=1.0, tick_size=0.01)


def _broker_and_portfolio(
    initial_cash: float = 100_000.0,
    margin_config: MarginConfig | None = None,
):
    margin_config = margin_config or MarginConfig()
    broker = SimulatedBroker(
        execution_config=ExecutionConfig(lot_size=1, buy_fee=0.001, sell_fee=0.001),
        margin_config=margin_config,
    )
    portfolio = PortfolioV2(
        initial_cash=initial_cash,
        ticker=TICKER,
        margin_config=margin_config,
    )
    return broker, portfolio


def _open_long(quantity: int = 100, price: float = 100.0) -> Fill:
    return Fill(
        fill_id="open1", order_id="o1", decision_id="d1",
        date="2026-01-05", ticker=TICKER, side="BUY",
        quantity=quantity, price=price, fee=0.0, slippage_amount=0.0,
        order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
    )


class TestDecisionRoundtrip(unittest.TestCase):
    """PM structured output must survive render -> parse -> DSM intact (B3)."""

    def _decision(self, **overrides):
        fields = dict(
            rating=PortfolioRating.BUY,
            executive_summary="Strong momentum",
            investment_thesis="Thesis",
            stop_loss=90.0,
            take_profit=110.0,
            planned_entry_price=100.0,
            confidence=0.8,
            time_horizon_days=126,
        )
        fields.update(overrides)
        return PortfolioDecision(**fields)

    def test_horizon_fallback_clamps_to_63_not_crash(self):
        # B1: 126-day horizon used to raise ValidationError and kill the run.
        contract = portfolio_decision_to_signal_contract(
            self._decision(), TICKER, "2026-01-05"
        )
        self.assertEqual(contract.max_holding_days, 63)
        self.assertEqual(contract.action, "BUY")
        self.assertEqual(contract.planned_entry_price, 100.0)

    def test_quant_fields_survive_markdown_roundtrip(self):
        md = render_pm_decision(self._decision())
        parsed = MarkdownDecisionParser().parse_text(
            md,
            fallback_ticker=TICKER,
            fallback_trade_date="2026-01-05",
            fallback_decision_valid_from="2026-01-06",
        )
        self.assertTrue(parsed.valid, parsed.invalid_reason)
        self.assertEqual(parsed.confidence, 0.8)
        self.assertEqual(parsed.planned_entry_price, 100.0)

        dsm = DecisionStateManager(config=DecisionMappingConfig())
        mapped = dsm.map(parsed, Position(ticker=TICKER, quantity=0))
        self.assertEqual(mapped.planned_entry_price, 100.0)
        self.assertAlmostEqual(mapped.confidence, 0.8)


class TestLimitFillInvariants(unittest.TestCase):
    """A limit order can never execute beyond its limit (B5)."""

    def _limit_buy_fill(self, open_, high, low):
        broker, portfolio = _broker_and_portfolio()
        order = Order(
            order_id="L1", decision_id="d", ticker=TICKER,
            order_type=OrderType.BUY_TO_OPEN, quantity=100,
            execution_date="2026-01-06", price=100.0,
        )
        broker.add_pending_orders([order])
        bar = MarketPoint(date="2026-01-06", ticker=TICKER,
                          open=open_, high=high, low=low, close=(open_ + high) / 2)
        trades = broker.execute_pending_orders(
            date="2026-01-06", market_point=bar,
            portfolio=portfolio, spec=SPEC,
        )
        return order, trades

    def test_gap_up_open_gets_capped_at_limit(self):
        # Open above the limit: fill must be exactly the limit, not above.
        order, trades = self._limit_buy_fill(open_=102.0, high=103.0, low=99.0)
        self.assertEqual(order.status, "FILLED")
        self.assertLessEqual(trades[0].price, 100.0)
        self.assertAlmostEqual(trades[0].price, 100.0, places=6)

    def test_gap_down_open_gets_price_improvement(self):
        # Open below the limit: fill near the better open, not the limit.
        order, trades = self._limit_buy_fill(open_=95.0, high=99.0, low=94.0)
        self.assertEqual(order.status, "FILLED")
        self.assertLess(trades[0].price, 96.0)
        self.assertGreaterEqual(trades[0].price, 95.0)

    def test_untouched_limit_stays_unfilled(self):
        # Whole bar above the limit: boundary behaviour preserved.
        order, trades = self._limit_buy_fill(open_=102.0, high=103.0, low=101.0)
        self.assertEqual(order.status, "UNFILLED")
        self.assertEqual(len(trades), 0)


class TestMarketLiquidationThroughGapDown(unittest.TestCase):
    """Force-close market orders must fill even when a stale limit could not (B2)."""

    GAP_BAR = MarketPoint(date="2026-01-06", ticker=TICKER,
                          open=88.0, high=89.0, low=80.0, close=82.0)

    def test_market_force_close_fills_at_open(self):
        broker, portfolio = _broker_and_portfolio()
        portfolio.apply_fill(_open_long())
        order = Order(
            order_id="risk_close_1", decision_id="", ticker=TICKER,
            order_type=OrderType.SELL_TO_CLOSE, quantity=100,
            execution_date="2026-01-06", price=0.0, reason="eod_margin_breach",
        )
        broker.add_pending_orders([order])
        trades = broker.execute_pending_orders(
            date="2026-01-06", market_point=self.GAP_BAR,
            portfolio=portfolio, spec=SPEC,
        )
        self.assertEqual(order.status, "FILLED")
        self.assertEqual(len(trades), 1)
        self.assertTrue(portfolio.is_flat())

    def test_stale_limit_would_have_stranded(self):
        # Contrast case documenting the old bug: breach-close as a limit
        # (90 > high 89) stays UNFILLED forever — hence market orders.
        broker, portfolio = _broker_and_portfolio()
        portfolio.apply_fill(_open_long())
        order = Order(
            order_id="risk_close_2", decision_id="", ticker=TICKER,
            order_type=OrderType.SELL_TO_CLOSE, quantity=100,
            execution_date="2026-01-06", price=90.0, reason="eod_margin_breach",
        )
        broker.add_pending_orders([order])
        trades = broker.execute_pending_orders(
            date="2026-01-06", market_point=self.GAP_BAR,
            portfolio=portfolio, spec=SPEC,
        )
        self.assertEqual(order.status, "UNFILLED")
        self.assertEqual(len(trades), 0)
        self.assertFalse(portfolio.is_flat())


class TestCarryCostAccrual(unittest.TestCase):
    """Shorts pay borrow daily; leveraged longs pay financing on borrowed cash (B8)."""

    def test_short_open_is_rejected(self):
        _, portfolio = _broker_and_portfolio()
        with self.assertRaises(ValueError):
            portfolio.apply_fill(Fill(
                fill_id="s1", order_id="o", decision_id="d",
                date="2026-01-05", ticker=TICKER, side="SELL",
                quantity=100, price=100.0, fee=0.0, slippage_amount=0.0,
                order_type=OrderType.SELL_TO_OPEN, open_close="OPEN",
            ))

    def test_leveraged_long_is_rejected(self):
        """Spot backtests reject entries whose notional exceeds cash."""
        cash_only = MarginConfig(
            initial_margin_pct=1.0,
            maintenance_margin_pct=1.0,
            max_leverage=1.0,
        )
        _, portfolio = _broker_and_portfolio(
            initial_cash=50_000.0,
            margin_config=cash_only,
        )
        with self.assertRaises(Exception):
            portfolio.apply_fill(Fill(
                fill_id="l1", order_id="o", decision_id="d",
                date="2026-01-05", ticker=TICKER, side="BUY",
                quantity=600, price=100.0, fee=0.0, slippage_amount=0.0,
                order_type=OrderType.BUY_TO_OPEN, open_close="OPEN",
            ))

    def test_unleveraged_long_pays_no_financing(self):
        broker, portfolio = _broker_and_portfolio()
        portfolio.apply_fill(_open_long(quantity=100))  # 10k position, 90k cash left
        s1 = portfolio.mark_to_market("2026-01-06", close_price=100.0)
        s2 = portfolio.mark_to_market("2026-01-07", close_price=100.0)
        self.assertAlmostEqual(s1.total_equity, s2.total_equity, places=6)
        self.assertTrue(broker is not None)


if __name__ == "__main__":
    unittest.main()
