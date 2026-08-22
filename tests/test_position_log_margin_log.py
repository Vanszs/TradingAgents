"""
Tests for position_log.csv and margin_log.csv outputs — PRD §19.4, §19.5.
"""
from __future__ import annotations

import pytest

from tradingagents.backtesting.portfolio import PortfolioV2
from tradingagents.backtesting.position import (
    Fill,
    MarginConfig,
    OrderType,
    Position,
    PositionSide,
)
from tradingagents.backtesting.margin import MarginAccount


class TestPositionLog:
    def test_position_log_entry_flat(self):
        portfolio = PortfolioV2(
            initial_cash=100000,
            ticker="TEST",
            margin_config=MarginConfig(),
        )
        snapshot = portfolio.mark_to_market("2026-01-01", 100.0)
        assert snapshot.position_qty == 0
        assert portfolio.position_log == []

    def test_position_log_after_trade(self):
        portfolio = PortfolioV2(
            initial_cash=100000,
            ticker="TEST",
            margin_config=MarginConfig(initial_margin_pct=0.5),
        )
        fill = Fill(
            fill_id="f1",
            order_id="o1",
            decision_id="d1",
            date="2026-01-01",
            ticker="TEST",
            side="LONG",
            quantity=100,
            price=100.0,
            fee=15.0,
            slippage_amount=0.1,
            order_type=OrderType.BUY_TO_OPEN,
            open_close="OPEN",
        )
        portfolio.apply_fill(fill)
        snapshot = portfolio.mark_to_market("2026-01-01", 105.0)
        assert len(portfolio.position_log) == 1
        entry = portfolio.position_log[0]
        assert entry["side"] == "LONG"
        assert entry["quantity"] == 100
        assert entry["avg_entry_price"] == 100.0

    def test_margin_log_after_trade(self):
        portfolio = PortfolioV2(
            initial_cash=100000,
            ticker="TEST",
            margin_config=MarginConfig(initial_margin_pct=0.5),
        )
        fill = Fill(
            fill_id="f1",
            order_id="o1",
            decision_id="d1",
            date="2026-01-01",
            ticker="TEST",
            side="LONG",
            quantity=100,
            price=100.0,
            fee=15.0,
            slippage_amount=0.1,
            order_type=OrderType.BUY_TO_OPEN,
            open_close="OPEN",
        )
        portfolio.apply_fill(fill)
        snapshot = portfolio.mark_to_market("2026-01-01", 100.0)
        assert len(portfolio.margin_log) == 1
        entry = portfolio.margin_log[0]
        assert entry["date"] == "2026-01-01"
        assert entry["equity"] == snapshot.total_equity
        assert entry["margin_utilization"] >= 0.0


class TestMarginAccountIntegration:
    def test_leverage_capped(self):
        """Leverage should not exceed max_leverage."""
        margin_cfg = MarginConfig(max_leverage=2.0)
        notional = 250000
        equity = 100000
        assert MarginAccount.is_leverage_capped(notional, equity, margin_cfg.max_leverage)

    def test_liquidation_guard(self):
        """Equity below maintenance should trigger liquidation."""
        equity = 30000
        maintenance = 35000
        assert MarginAccount.liquidation_guard(equity, maintenance)
