"""
Tests for margin.py — PRD §15.
"""
from __future__ import annotations

import pytest

from tradingagents.backtesting.margin import MarginAccount, MarginResult


class TestMarginAccount:
    def test_initial_margin(self):
        assert MarginAccount.initial_margin(100000, 0.5) == 50000.0

    def test_initial_margin_zero_notional(self):
        assert MarginAccount.initial_margin(0, 0.5) == 0.0

    def test_maintenance_margin(self):
        assert MarginAccount.maintenance_margin(100000, 0.35) == 35000.0

    def test_maintenance_margin_zero_notional(self):
        assert MarginAccount.maintenance_margin(0, 0.35) == 0.0

    def test_leverage(self):
        assert MarginAccount.leverage(200000, 100000) == 2.0

    def test_leverage_zero_equity(self):
        assert MarginAccount.leverage(200000, 0) == 0.0

    def test_leverage_negative_equity(self):
        assert MarginAccount.leverage(200000, -10000) == 0.0

    def test_is_leverage_capped_true(self):
        assert MarginAccount.is_leverage_capped(300000, 100000, 2.0) is True

    def test_is_leverage_capped_false(self):
        assert MarginAccount.is_leverage_capped(150000, 100000, 2.0) is False

    def test_is_leverage_capped_exact(self):
        assert MarginAccount.is_leverage_capped(200000, 100000, 2.0) is False

    def test_is_leverage_capped_zero_equity(self):
        assert MarginAccount.is_leverage_capped(100000, 0, 2.0) is True

    def test_liquidation_guard_below_maintenance(self):
        assert MarginAccount.liquidation_guard(30000, 35000) is True

    def test_liquidation_guard_above_maintenance(self):
        assert MarginAccount.liquidation_guard(40000, 35000) is False

    def test_margin_call(self):
        assert MarginAccount.margin_call(35000, 35000, 0.40) is True

    def test_margin_call_above_threshold(self):
        assert MarginAccount.margin_call(50000, 35000, 0.40) is False

    def test_check_returns_margin_result(self):
        result = MarginAccount.check(
            equity=100000,
            notional_value=150000,
            margin_rate=0.5,
            maintenance_rate=0.35,
            max_leverage=2.0,
        )
        assert isinstance(result, MarginResult)
        assert result.initial_margin == 75000.0
        assert result.maintenance_margin == 52500.0
        assert result.leverage == 1.5
        assert result.is_liquidation is False

    def test_accrue_financing(self):
        assert MarginAccount.accrue_financing(100000, 0.0001) == 10.0

    def test_accrue_borrow(self):
        assert MarginAccount.accrue_borrow(50000, 0.0002) == 10.0
