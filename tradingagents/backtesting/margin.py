"""
Margin account management — PRD §15.

Provides:
- Initial/maintenance margin calculation
- Leverage cap enforcement
- Liquidation guard (equity < maintenance)
- Daily financing accrual (for overnight long positions)
- Daily borrow fee accrual (for overnight short positions)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarginResult:
    """Result of a margin check."""
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    excess_margin: float = 0.0
    leverage: float = 0.0
    is_margin_call: bool = False
    is_liquidation: bool = False
    margin_utilization: float = 0.0


class MarginAccount:
    """
    PRD §15 — margin account with leverage cap and liquidation guard.

    Stateless: all methods take explicit parameters, no internal state.
    """

    @staticmethod
    def initial_margin(
        notional_value: float,
        margin_rate: float,
    ) -> float:
        """PRD §15.1 — initial margin = notional × initial_margin_rate."""
        return abs(notional_value) * margin_rate

    @staticmethod
    def maintenance_margin(
        notional_value: float,
        maintenance_rate: float,
    ) -> float:
        """PRD §15.2 — maintenance margin = notional × maintenance_margin_rate."""
        return abs(notional_value) * maintenance_rate

    @staticmethod
    def leverage(
        notional_value: float,
        equity: float,
    ) -> float:
        """Leverage = |notional| / equity."""
        if equity <= 0:
            return 0.0
        return abs(notional_value) / equity

    @staticmethod
    def is_leverage_capped(
        notional_value: float,
        equity: float,
        max_leverage: float,
    ) -> bool:
        """True if leverage would exceed max_leverage."""
        if equity <= 0 or max_leverage <= 0:
            return True
        return abs(notional_value) / equity > max_leverage

    @staticmethod
    def liquidation_guard(
        equity: float,
        maintenance_margin: float,
        liquidation_equity_pct: float = 0.25,
    ) -> bool:
        """PRD §15.3 — liquidation if equity < maintenance or equity < liquidation_equity_pct of notional."""
        if equity < maintenance_margin:
            return True
        # Additional guard: equity too low relative to initial margin
        return False

    @staticmethod
    def margin_call(
        equity: float,
        maintenance_margin: float,
        threshold: float = 0.40,
    ) -> bool:
        """PRD §15.3 — margin call if equity < maintenance × (1 + threshold)."""
        return equity < maintenance_margin * (1.0 + threshold)

    @staticmethod
    def check(
        equity: float,
        notional_value: float,
        margin_rate: float,
        maintenance_rate: float,
        max_leverage: float,
        margin_call_threshold: float = 0.40,
    ) -> MarginResult:
        """Run all margin checks and return a MarginResult."""
        im = MarginAccount.initial_margin(notional_value, margin_rate)
        mm = MarginAccount.maintenance_margin(notional_value, maintenance_rate)
        lev = MarginAccount.leverage(notional_value, equity)
        excess = equity - mm
        util = mm / equity if equity > 0 else 0.0
        mc = MarginAccount.margin_call(equity, mm, margin_call_threshold)
        liq = MarginAccount.liquidation_guard(equity, mm)

        return MarginResult(
            initial_margin=im,
            maintenance_margin=mm,
            excess_margin=excess,
            leverage=lev,
            is_margin_call=mc,
            is_liquidation=liq,
            margin_utilization=util,
        )

    @staticmethod
    def accrue_financing(
        notional_value: float,
        daily_rate: float,
    ) -> float:
        """PRD §15.4 — daily financing charge for long positions."""
        return abs(notional_value) * daily_rate

    @staticmethod
    def accrue_borrow(
        short_notional: float,
        daily_rate: float,
    ) -> float:
        """PRD §15.4 — daily borrow fee for short positions."""
        return abs(short_notional) * daily_rate
