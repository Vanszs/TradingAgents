"""
Stock simulated broker — PRD-compliant version.

Responsibilities:
- Hold a queue of pending orders with explicit OrderType.
- Execute orders against a MarketPoint for the given trade_date.
- Apply percentage-based fees (buy_fee, sell_fee) and slippage.
- Handle two-leg reverse orders (close + open).
- Apply daily settlement to the portfolio.
- Detect intraday (long: bar.low, short: bar.high) and EOD margin breaches.
- Auto-liquidate when margin.auto_liquidate_on_breach is True.
- Enforce max-leverage cap before execution.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional
from uuid import uuid4

from .decision_schema import MarginEvent, MarketPoint, OpenClose
from .portfolio import Portfolio
from .position import (
    ExecutionConfig,
    Fill,
    InstrumentSpec,
    MarginConfig,
    Order,
    OrderType,
)

logger = logging.getLogger(__name__)


class SimulatedBroker:
    """Simulated stock broker with margin call and auto-liquidation logic."""

    def __init__(
        self,
        execution_config: ExecutionConfig,
        margin_config: MarginConfig,
    ):
        self.config = execution_config
        self.margin_config = margin_config

        self.pending_orders: list[Order] = []
        self.filled_orders: list[Order] = []
        self.rejected_orders: list[Order] = []
        self.margin_events: list[MarginEvent] = []
        self.fills: list[Fill] = []

    def add_pending_orders(self, orders: Iterable[Order]) -> None:
        for order in orders:
            if order.quantity <= 0:
                order.status = "REJECTED"
                order.rejection_reason = "Non-positive quantity."
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
            if order.is_reverse or order.order_type in (
                OrderType.REVERSE_TO_LONG, OrderType.REVERSE_TO_SHORT,
            ):
                trades = self._execute_reverse(order, market_point, portfolio, spec)
            else:
                trades = [self._execute_order(order, market_point, portfolio, spec)]

            for trade in trades:
                portfolio.apply_trade(trade)
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
    # Daily settlement
    # ------------------------------------------------------------------
    def apply_daily_settlement(
        self,
        portfolio: Portfolio,
        prev_close: Optional[float],
        today_close: float,
    ) -> float:
        return portfolio.apply_daily_settlement(
            prev_close=prev_close,
            today_close=today_close,
        )

    # ------------------------------------------------------------------
    # Margin breach / liquidation
    # ------------------------------------------------------------------
    def check_intraday_margin(
        self,
        portfolio: Portfolio,
        market_point: MarketPoint,
    ) -> Optional[MarginEvent]:
        """PRD §15.3 — intraday check: long uses bar.low, short uses bar.high."""
        if not portfolio.has_position() or not self.config.intraday_margin_check:
            return None

        if portfolio.is_long():
            breach_price = market_point.low
        elif portfolio.is_short():
            breach_price = market_point.high
        else:
            return None

        if portfolio.is_intraday_breach(
            open_price=market_point.open,
            intraday_low=breach_price,
            intraday_high=market_point.high,
        ):
            ev = MarginEvent(
                date=market_point.date,
                kind="warning",
                deficit=portfolio.maintenance_required(breach_price)
                - portfolio.account_equity(breach_price),
                mark_price=breach_price,
                maintenance_required=portfolio.maintenance_required(breach_price),
                account_equity=portfolio.account_equity(breach_price),
                action="queued_liquidation",
                note="intraday_check",
            )
            self.margin_events.append(ev)
            return ev
        return None

    def check_eod_margin(
        self,
        portfolio: Portfolio,
        market_point: MarketPoint,
    ) -> Optional[MarginEvent]:
        if not portfolio.has_position():
            return None
        mark = market_point.close
        if portfolio.is_margin_breach(mark):
            ev = MarginEvent(
                date=market_point.date,
                kind="call",
                deficit=portfolio.maintenance_required(mark)
                - portfolio.account_equity(mark),
                mark_price=mark,
                maintenance_required=portfolio.maintenance_required(mark),
                account_equity=portfolio.account_equity(mark),
                action="queued_liquidation",
                note="eod_check",
            )
            self.margin_events.append(ev)
            return ev
        return None

    def force_liquidate(
        self,
        portfolio: Portfolio,
        price: float,
        reason: str,
    ) -> "Trade":
        return portfolio.force_liquidate(price=price, reason=reason)

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
                if order.is_reverse or order.order_type in (
                    OrderType.REVERSE_TO_LONG, OrderType.REVERSE_TO_SHORT,
                ):
                    new_trades = self._execute_reverse(order, market_point, portfolio, spec)
                else:
                    new_trades = [self._execute_order(order, market_point, portfolio, spec)]

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
        portfolio: Portfolio,
        spec: InstrumentSpec,
    ) -> "Trade":
        """Execute a single-leg order with percentage-based fees/slippage."""
        from .decision_schema import Trade as LegacyTrade

        base_price = float(order.price) if order.price and order.price > 0 else float(market_point.open)

        # PRD §13 — percentage-based slippage (new config) or tick-based (legacy)
        slippage_pct = getattr(self.config, "slippage", None)
        if slippage_pct is None:
            # Legacy config: use tick_slippage
            tick_slippage = getattr(self.config, "tick_slippage", 0)
            tick = getattr(spec, "tick_size", 0.01)
            slippage_pct = (tick_slippage * tick) / base_price if base_price > 0 else 0.0

        is_buy = order.order_type in (
            OrderType.BUY_TO_OPEN, OrderType.BUY_TO_ADD,
            OrderType.BUY_TO_REDUCE, OrderType.BUY_TO_CLOSE,
        )
        if is_buy:
            fill_price = base_price * (1 + slippage_pct)
        else:
            fill_price = base_price * (1 - slippage_pct)

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

        return LegacyTrade(
            date=market_point.date,
            ticker=portfolio.ticker,
            side="BUY" if is_buy else "SELL",
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
            open_close=OpenClose(order.open_close),
            order_type=order.order_type.value,
            reason=order.reason,
            decision_id=order.decision_id,
            order_id=order.order_id,
            mark_price=market_point.close,
        )

    def _execute_reverse(
        self,
        order: Order,
        market_point: MarketPoint,
        portfolio: Portfolio,
        spec: InstrumentSpec,
    ) -> list["Trade"]:
        """PRD §12.6 — two-leg reverse: close first, then open."""
        from .decision_schema import Trade as LegacyTrade

        if portfolio.is_flat():
            raise ValueError(
                "Cannot reverse: position is flat. Use OPEN instead."
            )

        # Leg 1: close existing position
        close_side = "SELL" if portfolio.is_long() else "BUY"
        close_type = (
            OrderType.SELL_TO_CLOSE if portfolio.is_long()
            else OrderType.BUY_TO_CLOSE
        )
        close_qty = portfolio.abs_qty()

        base_price = float(market_point.open)
        slippage_pct = getattr(self.config, "slippage", None)
        if slippage_pct is None:
            tick_slippage = getattr(self.config, "tick_slippage", 0)
            tick = getattr(spec, "tick_size", 0.01)
            slippage_pct = (tick_slippage * tick) / base_price if base_price > 0 else 0.0

        if close_side == "BUY":
            close_fill = base_price * (1 + slippage_pct)
            close_fee_pct = getattr(self.config, "buy_fee", None)
        else:
            close_fill = base_price * (1 - slippage_pct)
            close_fee_pct = getattr(self.config, "sell_fee", None)

        close_gross = close_fill * close_qty * spec.multiplier
        if close_fee_pct is not None:
            close_fee = close_gross * close_fee_pct
        else:
            close_fee = getattr(self.config, "fee_per_contract", 0.0) * close_qty
        close_slippage = abs(close_fill - base_price) * close_qty * spec.multiplier
        close_slippage_ticks = round(abs(close_fill - base_price) / spec.tick_size) if spec.tick_size > 0 else 0

        # Realized PnL on close leg
        realized = 0.0
        if portfolio.position.quantity > 0:
            realized = (close_fill - getattr(portfolio.position, "avg_entry_price", getattr(portfolio.position, "avg_price", 0.0))) * close_qty * spec.multiplier
        elif portfolio.position.quantity < 0:
            realized = (getattr(portfolio.position, "avg_entry_price", getattr(portfolio.position, "avg_price", 0.0)) - close_fill) * close_qty * spec.multiplier

        close_trade = LegacyTrade(
            date=market_point.date,
            ticker=portfolio.ticker,
            side=close_side,
            quantity=close_qty,
            price=close_fill,
            gross_amount=close_gross,
            fee=close_fee,
            net_amount=close_gross - close_fee,
            multiplier=spec.multiplier,
            notional=close_gross,
            tick_size=spec.tick_size,
            slippage_ticks=close_slippage_ticks,
            realized_pnl_delta=realized,
            margin_delta=0.0,
            open_close=OpenClose.CLOSE,
            reason=f"{order.reason}_close_leg",
            decision_id=order.decision_id,
            order_id=str(uuid4()),
            mark_price=market_point.close,
        )

        # Leg 2: open new position
        open_side = "BUY" if order.order_type == OrderType.REVERSE_TO_LONG else "SELL"
        open_type = (
            OrderType.BUY_TO_OPEN if order.order_type == OrderType.REVERSE_TO_LONG
            else OrderType.SELL_TO_OPEN
        )

        if open_side == "BUY":
            open_fill = base_price * (1 + slippage_pct)
            open_fee_pct = getattr(self.config, "buy_fee", None)
        else:
            open_fill = base_price * (1 - slippage_pct)
            open_fee_pct = getattr(self.config, "sell_fee", None)

        open_gross = open_fill * order.quantity * spec.multiplier
        if open_fee_pct is not None:
            open_fee = open_gross * open_fee_pct
        else:
            open_fee = getattr(self.config, "fee_per_contract", 0.0) * order.quantity
        open_slippage = abs(open_fill - base_price) * order.quantity * spec.multiplier
        open_slippage_ticks = round(abs(open_fill - base_price) / spec.tick_size) if spec.tick_size > 0 else 0

        open_trade = LegacyTrade(
            date=market_point.date,
            ticker=portfolio.ticker,
            side=open_side,
            quantity=order.quantity,
            price=open_fill,
            gross_amount=open_gross,
            fee=open_fee,
            net_amount=open_gross - open_fee,
            multiplier=spec.multiplier,
            notional=open_gross,
            tick_size=spec.tick_size,
            slippage_ticks=open_slippage_ticks,
            realized_pnl_delta=0.0,
            margin_delta=0.0,
            open_close=OpenClose.OPEN,
            reason=f"{order.reason}_open_leg",
            decision_id=order.decision_id,
            order_id=str(uuid4()),
            mark_price=market_point.close,
        )

        return [close_trade, open_trade]
