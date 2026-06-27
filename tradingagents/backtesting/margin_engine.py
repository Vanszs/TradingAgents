"""
Pure margin math helpers for futures backtests.

All functions are stateless and side-effect free. They operate on scalars or
plain dicts and are reused by ``Portfolio``, ``Broker``, and ``OrderGenerator``.

Margin model (Reg-T style, simplified):
- Initial margin:    IM = notional * initial_margin_pct
- Maintenance margin: MM = notional * maintenance_margin_pct
- Excess margin:      account_equity - MM
- Margin call:        account_equity < MM  (or account_equity - MM < buffer)
- Liquidation:        when account_equity < MM - liquidation_buffer

The "notional" is the absolute current market value of the open position:
    notional = |quantity| * mark_price * multiplier
"""
from __future__ import annotations

from typing import Any


def notional_value(
    quantity: int,
    mark_price: float,
    multiplier: float,
) -> float:
    """Return the absolute notional value of a position at mark_price."""
    return abs(int(quantity)) * float(mark_price) * float(multiplier)


def initial_margin(
    quantity: int,
    mark_price: float,
    multiplier: float,
    initial_margin_pct: float,
) -> float:
    """Return the initial margin requirement in dollars."""
    if quantity == 0:
        return 0.0
    return notional_value(quantity, mark_price, multiplier) * float(initial_margin_pct)


def maintenance_margin(
    quantity: int,
    mark_price: float,
    multiplier: float,
    maintenance_margin_pct: float,
) -> float:
    """Return the maintenance margin requirement in dollars."""
    if quantity == 0:
        return 0.0
    return notional_value(quantity, mark_price, multiplier) * float(maintenance_margin_pct)


def max_contracts_by_margin(
    available_cash: float,
    reference_price: float,
    multiplier: float,
    initial_margin_pct: float,
) -> int:
    """
    Compute the maximum integer number of contracts that can be opened
    given the available cash as the initial margin deposit.
    """
    if reference_price <= 0 or multiplier <= 0 or initial_margin_pct <= 0:
        return 0
    per_contract_margin = reference_price * multiplier * initial_margin_pct
    if per_contract_margin <= 0:
        return 0
    return int(available_cash // per_contract_margin)


def excess_margin(
    account_equity: float,
    quantity: int,
    mark_price: float,
    multiplier: float,
    maintenance_margin_pct: float,
) -> float:
    """Return the excess margin (account_equity - maintenance_margin)."""
    if quantity == 0:
        return float(account_equity)
    return float(account_equity) - maintenance_margin(
        quantity, mark_price, multiplier, maintenance_margin_pct
    )


def is_margin_call(
    account_equity: float,
    quantity: int,
    mark_price: float,
    multiplier: float,
    maintenance_margin_pct: float,
    buffer_pct: float = 0.0,
) -> bool:
    """
    Return True if a margin call should be triggered.

    A margin call happens when excess margin (account_equity - MM) falls below
    the configured buffer (default 0).
    """
    if quantity == 0:
        return False
    mm = maintenance_margin(
        quantity, mark_price, multiplier, maintenance_margin_pct
    )
    buffer = mm * float(buffer_pct)
    return float(account_equity) < mm - buffer


def is_intraday_margin_breach(
    account_equity_open: float,
    quantity: int,
    open_price: float,
    intraday_low: float,
    multiplier: float,
    maintenance_margin_pct: float,
) -> bool:
    """
    Conservative intraday check: assumes price moved from open to intraday_low.

    For long positions, low < open means losses; for shorts, low < open is favorable.
    We compute the worst-case equity (long loss or short loss depending on side)
    and check whether maintenance margin would be violated.

    To keep this conservative, we always use the side that loses money:
        - if long:  loss = (open - low) * qty * mult
        - if short: loss = (low - open) * |qty| * mult is wrong; shorts lose
                    when price rises. Since we don't have intraday_high, we
                    conservatively assume the worst side that would have hit
                    our stop. We use a heuristic: only check the long-loss side
                    when position is long, only check the short-loss side when
                    position is short. We do not have intraday_high here, so
                    for short positions we conservatively check (intraday_low
                    implies short is *profitable* and is not a breach on this
                    side). This is a known limitation; for shorts, the EOD
                    check is the authoritative one. We still flag a breach
                    if account equity at the close (open) already breaches.
    """
    if quantity == 0:
        return False

    if quantity > 0:
        worst_price = float(intraday_low)
        worst_equity = float(account_equity_open) - (
            (float(open_price) - worst_price) * quantity * multiplier
        )
    else:
        # For short, we don't have intraday_high here. Assume no intraday breach
        # from the low side; rely on EOD check.
        worst_equity = float(account_equity_open)

    return is_margin_call(
        account_equity=worst_equity,
        quantity=quantity,
        mark_price=worst_price,
        multiplier=multiplier,
        maintenance_margin_pct=maintenance_margin_pct,
    )


def leverage(
    quantity: int,
    mark_price: float,
    multiplier: float,
    account_equity: float,
) -> float:
    """Return leverage = abs(notional) / account_equity."""
    if account_equity <= 0:
        return 0.0
    return notional_value(quantity, mark_price, multiplier) / account_equity


def margin_utilization(
    quantity: int,
    mark_price: float,
    multiplier: float,
    account_equity: float,
    maintenance_margin_pct: float,
) -> float:
    """
    Return margin utilization as a fraction in [0, +inf):
        MM / account_equity
    """
    if quantity == 0 or account_equity <= 0:
        return 0.0
    mm = maintenance_margin(
        quantity, mark_price, multiplier, maintenance_margin_pct
    )
    return mm / account_equity


__all__ = [
    "notional_value",
    "initial_margin",
    "maintenance_margin",
    "max_contracts_by_margin",
    "excess_margin",
    "is_margin_call",
    "is_intraday_margin_breach",
    "leverage",
    "margin_utilization",
]
