"""
Tests for risk.py — PRD §11, §16.
"""
from __future__ import annotations

import pytest

from tradingagents.backtesting.risk import RiskEngine, RiskEvent
from tradingagents.backtesting.position import Position, PositionSide, OrderType


class TestRiskEngine:
    def setup_method(self):
        self.engine = RiskEngine(
            max_loss_per_trade_pct=10.0,
            max_portfolio_loss_pct=20.0,
            liquidation_enabled=True,
            stop_loss_enabled=True,
            take_profit_enabled=True,
        )

    def test_flat_position_no_events(self):
        position = Position(ticker="TEST", quantity=0, avg_entry_price=0)
        bar = {"open": 100, "high": 105, "low": 95, "close": 102}
        order, events = self.engine.check_bar(
            "2026-01-01", bar, position, equity=100000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        assert order is None
        assert len(events) == 0

    def test_long_stop_triggered(self):
        position = Position(ticker="TEST", quantity=100, avg_entry_price=100)
        position.stop_price = 95.0
        bar = {"open": 98, "high": 99, "low": 94, "close": 96}
        order, events = self.engine.check_bar(
            "2026-01-01", bar, position, equity=100000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        assert order is not None
        assert order.order_type == OrderType.SELL_TO_CLOSE
        assert order.quantity == 100
        assert len(events) == 1
        assert events[0].event_type == "stop"

    def test_short_stop_triggered(self):
        position = Position(ticker="TEST", quantity=-100, avg_entry_price=100)
        position.stop_price = 105.0
        bar = {"open": 103, "high": 106, "low": 102, "close": 104}
        order, events = self.engine.check_bar(
            "2026-01-01", bar, position, equity=100000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        assert order is not None
        assert order.order_type == OrderType.BUY_TO_CLOSE
        assert order.quantity == 100
        assert len(events) == 1
        assert events[0].event_type == "stop"

    def test_long_take_profit(self):
        position = Position(ticker="TEST", quantity=100, avg_entry_price=100)
        position.take_profit = 110.0
        bar = {"open": 108, "high": 111, "low": 107, "close": 109}
        order, events = self.engine.check_bar(
            "2026-01-01", bar, position, equity=100000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        assert order is not None
        assert order.order_type == OrderType.SELL_TO_CLOSE
        assert order.quantity == 100
        assert len(events) == 1
        assert events[0].event_type == "take_profit"

    def test_short_take_profit(self):
        position = Position(ticker="TEST", quantity=-100, avg_entry_price=100)
        position.take_profit = 90.0
        bar = {"open": 92, "high": 93, "low": 89, "close": 91}
        order, events = self.engine.check_bar(
            "2026-01-01", bar, position, equity=100000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        assert order is not None
        assert order.order_type == OrderType.BUY_TO_CLOSE
        assert order.quantity == 100
        assert len(events) == 1
        assert events[0].event_type == "take_profit"

    def test_intraday_ambiguity_stop_wins(self):
        """PRD §16.3: If both stop and target triggered same bar, stop wins (conservative)."""
        position = Position(ticker="TEST", quantity=100, avg_entry_price=100)
        position.stop_price = 95.0
        position.take_profit = 105.0
        # Both stop (low <= 95) and target (high >= 105) triggered
        bar = {"open": 100, "high": 106, "low": 94, "close": 98}
        order, events = self.engine.check_bar(
            "2026-01-01", bar, position, equity=100000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        # Stop should win (first in priority)
        assert order is not None
        assert order.order_type == OrderType.SELL_TO_CLOSE
        assert events[0].event_type == "stop"

    def test_liquidation_long(self):
        """PRD §15.3: Liquidation if equity < maintenance margin."""
        # Use high thresholds so hard risk doesn't trigger first
        engine = RiskEngine(max_loss_per_trade_pct=100.0, max_portfolio_loss_pct=100.0)
        position = Position(ticker="TEST", quantity=1000, avg_entry_price=100)
        # notional = 1000 * 80 * 1 = 80000, maintenance = 80000 * 0.35 = 28000
        # equity = 25000 < 28000 → liquidation
        bar = {"open": 82, "high": 83, "low": 79, "close": 80}
        order, events = engine.check_bar(
            "2026-01-01", bar, position, equity=25000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        assert order is not None
        assert order.order_type == OrderType.SELL_TO_CLOSE
        assert any(e.event_type == "liquidation" for e in events)

    def test_liquidation_short(self):
        """PRD §15.3: Short liquidation check uses high price."""
        # Use high thresholds so hard risk doesn't trigger first
        engine = RiskEngine(max_loss_per_trade_pct=100.0, max_portfolio_loss_pct=100.0)
        position = Position(ticker="TEST", quantity=-1000, avg_entry_price=100)
        # notional = 1000 * 120 * 1 = 120000, maintenance = 120000 * 0.35 = 42000
        # equity = 40000 < 42000 → liquidation
        bar = {"open": 118, "high": 121, "low": 117, "close": 119}
        order, events = engine.check_bar(
            "2026-01-01", bar, position, equity=40000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        assert order is not None
        assert order.order_type == OrderType.BUY_TO_CLOSE
        assert any(e.event_type == "liquidation" for e in events)

    def test_no_stop_no_target_no_risk(self):
        position = Position(ticker="TEST", quantity=100, avg_entry_price=100)
        bar = {"open": 101, "high": 102, "low": 100, "close": 101}
        order, events = self.engine.check_bar(
            "2026-01-01", bar, position, equity=100000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        assert order is None
        assert len(events) == 0

    def test_margin_call_event_generated(self):
        """Margin call should generate event but not force close."""
        position = Position(ticker="TEST", quantity=1000, avg_entry_price=100)
        # worst_equity with low=98: 35000 + (98-100)*1000 = 33000
        # maintenance at worst_price=98: 98*1000*0.35 = 34300
        # 33000 < 34300, still liquidation. Use low=99:
        # worst_equity = 35000 + (99-100)*1000 = 34000
        # maintenance = 99*1000*0.35 = 34650. Still liquidation.
        # Use equity=40000, low=99:
        # worst_equity = 40000 + (99-100)*1000 = 39000
        # maintenance = 99*1000*0.35 = 34650. 39000 > 34650, no liquidation.
        # margin_call threshold = 34650*1.4 = 48510. 39000 < 48510, margin_call!
        bar = {"open": 100, "high": 101, "low": 99, "close": 100}
        order, events = self.engine.check_bar(
            "2026-01-01", bar, position, equity=40000,
            margin_rate=0.5, maintenance_rate=0.35, max_leverage=2.0,
        )
        assert any(e.event_type == "margin_call" for e in events)
