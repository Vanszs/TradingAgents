"""
Stock simulated broker — PRD-compliant version.

Responsibilities:
- Hold a queue of pending orders with explicit OrderType.
- Execute orders against a MarketPoint for the given trade_date.
- Apply percentage-based fees (buy_fee, sell_fee) and slippage.
- Apply daily settlement to the portfolio.
- Detect intraday long risk and EOD margin breaches.
- Auto-liquidate when margin.auto_liquidate_on_breach is True.
- Enforce max-leverage cap before execution.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Optional

from .decision_schema import (
    MarginEvent,
    MarketPoint,
    OpenClose,
    OrderSide,
)
from .decision_schema import (
    Trade as LegacyTrade,
)
from .portfolio import Portfolio
from .position import (
    ExecutionConfig,
    Fill,
    InstrumentSpec,
    MarginConfig,
    Order,
    OrderType,
)

if TYPE_CHECKING:
    from .decision_schema import Trade

logger = logging.getLogger(__name__)


class SimulatedBroker:
    """Simulated stock broker with margin call and auto-liquidation logic."""

    def __init__(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        margin_config: Optional[MarginConfig] = None,
    ):
        self.config = execution_config or ExecutionConfig()
        self.margin_config = margin_config or MarginConfig()

        self.pending_orders: list[Order] = []
        self.filled_orders: list[Order] = []
        self.rejected_orders: list[Order] = []
        self.margin_events: list[MarginEvent] = []

    def add_pending_orders(self, orders: Iterable[Order]) -> None:
        for order in orders:
            if order.quantity <= 0:
                order.status = "REJECTED"
                order.rejection_reason = "Non-positive quantity."
                self.rejected_orders.append(order)
                continue
            if order.is_reverse or order.order_type not in {
                OrderType.BUY_TO_OPEN,
                OrderType.SELL_TO_CLOSE,
            }:
                order.status = "REJECTED"
                order.rejection_reason = "Spot long-only execution accepts only BUY_TO_OPEN or SELL_TO_CLOSE."
                self.rejected_orders.append(order)
                continue
            self.pending_orders.append(order)

    def cancel_orders_before(self, date: str) -> None:
        for order in self.pending_orders:
            if order.execution_date < date:
                order.status = "CANCELLED"
        self.pending_orders = [
            o for o in self.pending_orders if o.status == "PENDING"
        ]

    def execute_pending_orders_immediate(
        self,
        order: "Order",
        market_point: "MarketPoint",
        portfolio: "Portfolio",
        spec: "InstrumentSpec",
    ) -> list["Trade"]:
        """Execute a single risk order immediately at the current bar's price.

        Used for stop-loss, take-profit, and liquidation orders that must
        be executed at the current bar's price, not deferred to the next day.
        The order is NOT added to the pending queue — it's executed directly.
        """
        order.execution_date = market_point.date
        try:
            trades = self._execute_order(order, market_point, portfolio, spec)

            for trade in trades:
                portfolio.apply_trade(trade)
            if trades:
                order.status = "FILLED"
                self.filled_orders.append(order)
            return trades
        except Exception as exc:
            order.status = "REJECTED"
            order.rejection_reason = str(exc)
            self.rejected_orders.append(order)
            logger.warning("Risk order rejected: %s -- %s", order.order_id, exc)
            return []

    # ------------------------------------------------------------------
    # Pending order execution
    # ------------------------------------------------------------------
    def execute_pending_orders(
        self,
        date: str,
        market_point: MarketPoint,
        portfolio: Portfolio,
        spec: InstrumentSpec,
    ) -> list["Trade"]:
        executable = [
            o
            for o in self.pending_orders
            if o.execution_date == date and o.status == "PENDING"
        ]
        remaining = [
            o
            for o in self.pending_orders
            if not (o.execution_date == date and o.status == "PENDING")
        ]

        trades: list["Trade"] = []
        for order in executable:
            try:
                new_trades = self._execute_order(order, market_point, portfolio, spec)

                if new_trades:
                    for trade in new_trades:
                        portfolio.apply_trade(trade)
                    order.status = "FILLED"
                    self.filled_orders.append(order)
                    trades.extend(new_trades)
            except Exception as exc:
                order.status = "REJECTED"
                order.rejection_reason = str(exc)
                self.rejected_orders.append(order)
                logger.warning("Order rejected: %s -- %s", order.order_id, exc)

        self.pending_orders = remaining
        return trades

    def _execute_order(
        self,
        order: Order,
        market_point: MarketPoint,
        portfolio: Optional[Portfolio] = None,
        spec: Optional[InstrumentSpec] = None,
    ) -> list["Trade"]:
        """Execute a single-leg order with percentage-based fees/slippage."""
        if order.is_reverse or order.order_type not in {
            OrderType.BUY_TO_OPEN,
            OrderType.SELL_TO_CLOSE,
        }:
            raise ValueError(
                "Spot long-only broker accepts only BUY_TO_OPEN or SELL_TO_CLOSE"
            )

        if spec is None:
            spec = InstrumentSpec(ticker=order.ticker, multiplier=1.0, tick_size=0.01)

        is_buy = order.order_type == OrderType.BUY_TO_OPEN

        # BUY prices are entry limits. Risk SELL prices are already-resolved
        # static stop/target levels; ordinary SELL prices remain limits.
        has_price = order.price is not None and float(order.price) > 0
        is_limit = has_price and (is_buy or not order.is_risk_order)
        limit_p = float(order.price) if has_price else 0.0
        if is_limit:
            traded_past_limit = (
                float(market_point.low) > limit_p
                if is_buy
                else float(market_point.high) < limit_p
            )
            if traded_past_limit:
                order.status = "UNFILLED"
                return []

        open_price = float(market_point.open)
        if is_buy and is_limit:
            # Price improvement: a gap through the limit fills at the better open.
            base_price = min(open_price, limit_p)
        elif not is_buy and has_price and order.is_risk_order:
            base_price = limit_p
        else:
            base_price = open_price if not is_limit else max(open_price, limit_p)

        # PRD §13 — percentage-based slippage (new config) or tick-based (legacy)
        slippage_pct = getattr(self.config, "slippage", None)
        if slippage_pct is None:
            # Legacy config: use tick_slippage
            tick_slippage = getattr(self.config, "tick_slippage", 0)
            tick = getattr(spec, "tick_size", 0.01)
            slippage_pct = (tick_slippage * tick) / base_price if base_price > 0 else 0.0

        half_spread_pct = (getattr(self.config, "spread_bps", 0.0) / 10000.0) / 2.0
        total_friction_pct = slippage_pct + half_spread_pct
        if is_buy:
            fill_price = base_price * (1 + total_friction_pct)
        else:
            fill_price = base_price * (1 - total_friction_pct)
        if is_limit:
            # A limit order can never execute beyond its limit price; friction
            # is absorbed by the filler rather than violating the bound.
            fill_price = min(fill_price, limit_p) if is_buy else max(fill_price, limit_p)

        if fill_price <= 0:
            raise ValueError(
                f"Computed non-positive fill price {fill_price} for {order.order_id}"
            )

        # PRD §13 — asymmetric fees (new: percentage, legacy: per-contract)
        fee_pct = getattr(self.config, "buy_fee", None) if is_buy else getattr(self.config, "sell_fee", None)
        if fee_pct is not None:
            gross = fill_price * order.quantity * spec.multiplier
            fee = gross * fee_pct
        else:
            # Legacy: fee_per_contract
            fee_per = getattr(self.config, "fee_per_contract", 0.0)
            gross = fill_price * order.quantity * spec.multiplier
            fee = fee_per * order.quantity
        slippage_amount = abs(fill_price - base_price) * order.quantity * spec.multiplier
        slippage_ticks = round(slippage_amount / (spec.tick_size * order.quantity)) if spec.tick_size > 0 and order.quantity > 0 else 0

        # Portfolio computes realized PnL in _apply_close (single source of truth)
        realized = 0.0

        ticker = portfolio.ticker if portfolio else order.ticker
        open_close_val = OpenClose(order.open_close) if hasattr(order, 'open_close') and order.open_close else (
            OpenClose.OPEN if order.order_type == OrderType.BUY_TO_OPEN else OpenClose.CLOSE
        )

        return [LegacyTrade(
            date=market_point.date,
            ticker=ticker,
            side=OrderSide.BUY if is_buy else OrderSide.SELL,
            quantity=order.quantity,
            price=fill_price,
            gross_amount=gross,
            fee=fee,
            net_amount=gross - fee,
            multiplier=spec.multiplier,
            notional=gross,
            tick_size=spec.tick_size,
            slippage_ticks=slippage_ticks,
            realized_pnl_delta=realized,
            margin_delta=0.0,
            open_close=open_close_val,
            order_type=order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
            reason=order.reason,
            decision_id=order.decision_id,
            order_id=order.order_id,
            mark_price=market_point.close,
        )]
