"""
Portfolio state for stock margin backtests.

Tracks:
- Cash (free cash + posted margin)
- Signed position (positive = long, negative = short)
- Daily mark-to-market settlement
- Realized PnL and unrealized PnL
- Margin posted, maintenance requirement, excess margin
- Equity curve and drawdown
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

from .decision_schema import (
    OpenClose,
    OrderSide,
    PortfolioSnapshot,
    Trade,
)
from .decision_schema import (
    Position as LegacyPosition,
)
from .margin import MarginAccount
from .margin_engine import (
    excess_margin,
    initial_margin,
    leverage,
    maintenance_margin,
    margin_utilization,
    notional_value,
)
from .position import (
    Fill,
    MarginConfig,
    OrderType,
)
from .position import (
    Position as V2Position,
)


class InsufficientMarginError(Exception):
    pass


class ReduceOnlyViolationError(Exception):
    pass


class Portfolio:
    """
    Margin-aware portfolio for stock backtests.

    Invariants:
    - ``position.quantity`` is signed: positive = long, negative = short.
    - Cash is decremented/credited by realized PnL, fees, margin postings.
    - Daily settlement is applied between trading days, not within a day.
    """

    def __init__(
        self,
        initial_cash: float,
        ticker: str,
        initial_position_qty: int = 0,
        initial_position_price: float = 0.0,
        initial_multiplier: float = 1.0,
        initial_tick_size: float = 0.01,
        enable_daily_settlement: bool = False,
    ):
        self.initial_cash = float(initial_cash)
        self.ticker = ticker
        self.enable_daily_settlement = enable_daily_settlement
        if initial_position_qty < 0:
            raise ValueError("spot long-only portfolio cannot start with a short position")

        self.position = LegacyPosition(
            ticker=ticker,
            quantity=int(initial_position_qty),
            avg_price=float(initial_position_price),
            multiplier=float(initial_multiplier),
            tick_size=float(initial_tick_size),
            realized_pnl=0.0,
            lifetime_fees=0.0,
        )

        self.cash: float = float(initial_cash)
        self.margin_posted: float = 0.0
        self.lifetime_realized_pnl: float = 0.0
        self.daily_realized_pnl: float = 0.0
        self.daily_unrealized_pnl: float = 0.0
        self.last_close: Optional[float] = None
        self.last_mark: Optional[float] = None

        self.trades: list[Trade] = []
        self.equity_curve: list[PortfolioSnapshot] = []
        self.peak_equity: float = float(initial_cash)

        self._initial_margin_pct: float = 0.5
        self._maintenance_margin_pct: float = 0.3

    # ------------------------------------------------------------------
    # Position state helpers
    # ------------------------------------------------------------------
    def has_position(self) -> bool:
        return self.position.quantity != 0

    def is_long(self) -> bool:
        return self.position.quantity > 0

    def is_short(self) -> bool:
        return self.position.quantity < 0

    def is_flat(self) -> bool:
        return self.position.quantity == 0

    def abs_qty(self) -> int:
        return abs(self.position.quantity)

    def notional(self, price: float) -> float:
        return notional_value(self.position.quantity, price, self.position.multiplier)

    def position_value(self, price: float) -> float:
        return self.position.quantity * float(price) * self.position.multiplier

    def account_equity(self, mark_price: Optional[float] = None) -> float:
        if mark_price is None:
            mark_price = self.last_mark or self.position.avg_price or 0.0
        return self.cash + self.position.unrealized_pnl(mark_price)

    def margin_used(self) -> float:
        return self.margin_posted

    def margin_available(self, mark_price: float) -> float:
        return excess_margin(
            account_equity=self.account_equity(mark_price),
            quantity=self.position.quantity,
            mark_price=mark_price,
            multiplier=self.position.multiplier,
            maintenance_margin_pct=self._maintenance_margin_pct,
        )

    def maintenance_required(self, mark_price: float) -> float:
        return maintenance_margin(
            quantity=self.position.quantity,
            mark_price=mark_price,
            multiplier=self.position.multiplier,
            maintenance_margin_pct=self._maintenance_margin_pct,
        )

    def initial_required(self, mark_price: float) -> float:
        return initial_margin(
            quantity=self.position.quantity,
            mark_price=mark_price,
            multiplier=self.position.multiplier,
            initial_margin_pct=self._initial_margin_pct,
        )

    def leverage(self, mark_price: float) -> float:
        return leverage(
            quantity=self.position.quantity,
            mark_price=mark_price,
            multiplier=self.position.multiplier,
            account_equity=self.account_equity(mark_price),
        )

    def margin_util(self, mark_price: float) -> float:
        return margin_utilization(
            quantity=self.position.quantity,
            mark_price=mark_price,
            multiplier=self.position.multiplier,
            account_equity=self.account_equity(mark_price),
            maintenance_margin_pct=self._maintenance_margin_pct,
        )

    def configure_margin(
        self,
        initial_margin_pct: float,
        maintenance_margin_pct: float,
    ) -> None:
        self._initial_margin_pct = float(initial_margin_pct)
        self._maintenance_margin_pct = float(maintenance_margin_pct)

    # ------------------------------------------------------------------
    # Trade application
    # ------------------------------------------------------------------
    def apply_trade(self, trade: Trade) -> None:
        if trade.quantity <= 0:
            raise ValueError("Trade quantity must be positive")
        if trade.open_close == OpenClose.OPEN:
            if trade.side != OrderSide.BUY:
                raise ValueError("Spot long-only portfolio rejects short-opening trades")
            self._apply_open(trade)
        elif trade.open_close == OpenClose.CLOSE:
            if trade.side != OrderSide.SELL:
                raise ValueError("Spot long-only portfolio closes with SELL only")
            self._apply_close(trade)
        elif trade.open_close == OpenClose.AUTO:
            if self.is_flat() and trade.side == OrderSide.BUY:
                self._apply_open(trade)
            elif self.is_long() and trade.side == OrderSide.SELL:
                self._apply_close(trade)
            else:
                raise ValueError("Spot long-only portfolio accepts BUY open or SELL close only")
        else:
            raise ValueError(f"Unsupported trade phase: {trade.open_close}")

        self.trades.append(trade)
        self.cash -= trade.fee
        self.position.lifetime_fees += trade.fee
        self.position.realized_pnl += trade.realized_pnl_delta
        self.lifetime_realized_pnl += trade.realized_pnl_delta
        self.daily_realized_pnl += trade.realized_pnl_delta
        if self.enable_daily_settlement:
            prev_mark = self.last_mark if self.last_mark is not None else float(trade.price)
            close_pnl = (
                (float(trade.price) - prev_mark) * trade.quantity * trade.multiplier
                if trade.side == OrderSide.SELL
                else 0.0
            )
            self.cash += close_pnl
        else:
            self.cash += trade.realized_pnl_delta

    def _apply_open(self, trade: Trade) -> None:
        if self.position.quantity != 0:
            raise ValueError(
                f"Cannot OPEN: existing position {self.position.quantity}. "
                "Spot long-only execution does not pyramid."
            )
        if trade.side != OrderSide.BUY:
            raise ValueError("Spot long-only OPEN requires BUY")

        self.position.quantity = int(trade.quantity)
        self.position.avg_price = float(trade.price)
        self.position.multiplier = float(trade.multiplier)
        self.position.tick_size = float(trade.tick_size)
        self._post_margin(trade)
        self.last_mark = float(trade.price)

    def _apply_close(self, trade: Trade) -> None:
        if not self.is_long():
            raise ValueError("Spot long-only CLOSE requires an open long position")

        close_qty = int(trade.quantity)
        if close_qty > self.position.quantity:
            raise ValueError(
                f"Cannot CLOSE: requested {close_qty} but only {self.position.quantity} open."
            )
        if trade.side != OrderSide.SELL:
            raise ValueError("Spot long-only CLOSE requires SELL")

        realized = (float(trade.price) - self.position.avg_price) * close_qty * self.position.multiplier
        released = self.margin_posted * (close_qty / self.position.quantity)
        self.margin_posted -= released
        self.cash += released
        trade.realized_pnl_delta = float(realized)

        self.position.quantity -= close_qty
        if self.position.quantity == 0:
            self.position.avg_price = 0.0

    def _post_margin(self, trade: Trade) -> None:
        post = float(trade.price) * int(trade.quantity) * float(trade.multiplier) * self._initial_margin_pct
        if post > self.cash + 1e-9:
            raise InsufficientMarginError(
                f"Insufficient cash to post initial margin: need {post:.2f}, "
                f"have {self.cash:.2f}."
            )
        self.cash -= post
        self.margin_posted += post
        trade.margin_delta = post

    # ------------------------------------------------------------------
    # Daily settlement
    # ------------------------------------------------------------------
    def apply_daily_settlement(self, prev_close: Optional[float], today_close: float) -> float:
        """
        Apply daily mark-to-market settlement for stocks.

        Realized PnL for the day = (today_close - prev_close) * position * multiplier.
        Cash is credited/debited, margin_posted is unchanged.
        """
        if prev_close is None or self.position.quantity == 0:
            self.last_close = float(today_close)
            return 0.0

        pnl = (
            (float(today_close) - float(prev_close))
            * self.position.quantity
            * self.position.multiplier
        )
        self.cash += pnl
        self.daily_realized_pnl += pnl
        self.lifetime_realized_pnl += pnl
        self.position.realized_pnl += pnl
        self.last_close = float(today_close)
        self.last_mark = float(today_close)
        return pnl

    # ------------------------------------------------------------------
    # Mark to market / snapshot
    # ------------------------------------------------------------------
    def mark_to_market(
        self,
        date: str,
        mark_price: float,
    ) -> PortfolioSnapshot:
        mark = float(mark_price)

        position_value = self.position_value(mark)
        unrealized = self.position.unrealized_pnl(mark)
        equity = self.cash + unrealized

        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (equity / self.peak_equity) - 1.0

        mm_req = self.maintenance_required(mark)
        im_req = self.initial_required(mark)
        excess = equity - mm_req
        lev = self.leverage(mark)
        util = self.margin_util(mark)

        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            position_qty=self.position.quantity,
            position_avg_price=self.position.avg_price,
            position_value=position_value,
            mark_price=mark,
            total_equity=equity,
            drawdown=drawdown,
            margin_used=self.margin_posted,
            margin_available=excess,
            excess_margin=excess,
            maintenance_margin_required=mm_req,
            initial_margin_required=im_req,
            leverage=lev,
            daily_realized_pnl=self.daily_realized_pnl,
            daily_unrealized_pnl=unrealized,
            lifetime_realized_pnl=self.lifetime_realized_pnl,
        )

        self.equity_curve.append(snapshot)
        self.daily_realized_pnl = 0.0
        self.daily_unrealized_pnl = unrealized
        self.last_mark = mark
        return snapshot

    # ------------------------------------------------------------------
    # Margin breach helpers
    # ------------------------------------------------------------------
    def is_margin_breach(self, mark_price: float) -> bool:
        if self.position.quantity == 0:
            return False
        equity = self.cash + self.position.unrealized_pnl(mark_price)
        mm = maintenance_margin(
            quantity=self.position.quantity,
            mark_price=mark_price,
            multiplier=self.position.multiplier,
            maintenance_margin_pct=self._maintenance_margin_pct,
        )
        return equity < mm

    def is_intraday_breach(self, open_price: float, intraday_low: float, intraday_high: Optional[float] = None) -> bool:
        if self.position.quantity == 0:
            return False
        if self.position.quantity < 0:
            raise ValueError("spot long-only portfolio rejects short positions")

        equity_open = self.cash + self.position.unrealized_pnl(open_price)
        worst_equity = equity_open - (
            (float(open_price) - float(intraday_low))
            * self.position.quantity
            * self.position.multiplier
        )
        mm = maintenance_margin(
            quantity=self.position.quantity,
            mark_price=intraday_low,
            multiplier=self.position.multiplier,
            maintenance_margin_pct=self._maintenance_margin_pct,
        )
        return worst_equity < mm

    # ------------------------------------------------------------------
    # Force liquidation
    # ------------------------------------------------------------------
    def force_liquidate(self, price: float, reason: str = "margin_liquidation", date: str = "", slippage_pct: float = 0.0) -> Trade:
        if not self.is_long():
            raise ValueError("Spot long-only liquidation requires an open long position.")
        qty = self.position.quantity
        adjusted_price = float(price) * (1 - slippage_pct) if slippage_pct > 0 else float(price)
        realized = (
            (adjusted_price - self.position.avg_price)
            * self.position.quantity
            * self.position.multiplier
        )
        self.cash += self.margin_posted + realized
        self.lifetime_realized_pnl += realized
        self.daily_realized_pnl += realized
        self.position.realized_pnl += realized
        self.margin_posted = 0.0
        trade = Trade(
            date=date,
            ticker=self.ticker,
            side=OrderSide.SELL,
            quantity=qty,
            price=adjusted_price,
            gross_amount=adjusted_price * qty * self.position.multiplier,
            fee=0.0,
            net_amount=adjusted_price * qty * self.position.multiplier,
            multiplier=self.position.multiplier,
            notional=adjusted_price * qty * self.position.multiplier,
            tick_size=self.position.tick_size,
            slippage_ticks=0,
            realized_pnl_delta=realized,
            margin_delta=0.0,
            open_close=OpenClose.CLOSE,
            reason=reason,
            decision_id="",
            order_id="",
            mark_price=float(price),
        )
        self.position.quantity = 0
        self.position.avg_price = 0.0
        self.trades.append(trade)
        return trade


class PortfolioV2:
    """
    PRD §17 — Margin-aware portfolio using new Position model.

    Tracks:
    - Cash (free cash + posted margin)
    - Signed position (positive = long, negative = short)
    - Daily mark-to-market settlement
    - Realized PnL and unrealized PnL
    - Margin posted, maintenance requirement, excess margin
    - Equity curve and drawdown

    Uses Position from position.py (PRD §7.3) and MarginAccount from margin.py (PRD §15).
    """

    def __init__(
        self,
        initial_cash: float,
        ticker: str,
        margin_config: Optional[MarginConfig] = None,
        initial_position_qty: int = 0,
        initial_position_price: float = 0.0,
        initial_multiplier: float = 1.0,
        margin_cfg: Optional[MarginConfig] = None,
    ):
        self.initial_cash = float(initial_cash)
        self.ticker = ticker
        self.margin_cfg = margin_config or margin_cfg or MarginConfig()
        if initial_position_qty < 0:
            raise ValueError("spot long-only portfolio cannot start with a short position")

        self.position = V2Position(
            ticker=ticker,
            quantity=int(initial_position_qty),
            avg_entry_price=float(initial_position_price),
            multiplier=float(initial_multiplier),
        )

        self.cash: float = float(initial_cash)
        self.margin_posted: float = 0.0
        self.lifetime_realized_pnl: float = 0.0
        self.daily_realized_pnl: float = 0.0
        self.daily_unrealized_pnl: float = 0.0
        self.last_close: Optional[float] = None
        self.last_mark: Optional[float] = None

        self.trades: list[Trade] = []
        self.equity_curve: list[PortfolioSnapshot] = []
        self.peak_equity: float = float(initial_cash)

        self.position_log: list[dict] = []
        self.margin_log: list[dict] = []

    # ------------------------------------------------------------------
    # Position state helpers
    # ------------------------------------------------------------------
    def has_position(self) -> bool:
        return self.position.quantity != 0

    def is_long(self) -> bool:
        return self.position.is_long()

    def is_short(self) -> bool:
        return self.position.is_short()

    def is_flat(self) -> bool:
        return self.position.is_flat()

    def abs_qty(self) -> int:
        return self.position.abs_qty()

    def notional(self, price: float) -> float:
        return self.position.abs_qty() * float(price) * self.position.multiplier

    def position_value(self, price: float) -> float:
        return self.position.quantity * float(price) * self.position.multiplier

    def account_equity(self, mark_price: Optional[float] = None) -> float:
        if mark_price is None:
            mark_price = self.last_mark or self.position.avg_entry_price or 0.0
        # In stock-like cash accounting where notional cash is deducted on long open,
        # total equity is cash + signed position market value (positive for long, negative liability for short).
        return self.cash + self.position_value(float(mark_price))

    def margin_used(self) -> float:
        return self.margin_posted

    def margin_available(self, mark_price: float) -> float:
        mm = MarginAccount.maintenance_margin(
            self.notional(mark_price), self.margin_cfg.maintenance_margin_pct
        )
        return max(0.0, self.account_equity(mark_price) - mm)

    def maintenance_required(self, mark_price: float) -> float:
        return MarginAccount.maintenance_margin(
            self.notional(mark_price), self.margin_cfg.maintenance_margin_pct
        )

    def initial_required(self, mark_price: float) -> float:
        return MarginAccount.initial_margin(
            self.notional(mark_price), self.margin_cfg.initial_margin_pct
        )

    def leverage(self, mark_price: float) -> float:
        return MarginAccount.leverage(
            self.notional(mark_price), self.account_equity(mark_price)
        )

    def margin_util(self, mark_price: float) -> float:
        mm = MarginAccount.maintenance_margin(
            self.notional(mark_price), self.margin_cfg.maintenance_margin_pct
        )
        equity = self.account_equity(mark_price)
        return mm / equity if equity > 0 else 0.0

    def is_margin_breach(self, mark_price: float) -> bool:
        mm = MarginAccount.maintenance_margin(
            self.notional(mark_price), self.margin_cfg.maintenance_margin_pct
        )
        return self.account_equity(mark_price) < mm

    def is_intraday_breach(self, open_price: float, intraday_low: float, intraday_high: Optional[float] = None) -> bool:
        """Check long inventory against the intraday low."""
        if self.position.quantity == 0:
            return False
        if self.position.quantity < 0:
            raise ValueError("spot long-only portfolio rejects short positions")

        equity_open = self.account_equity(open_price)
        worst_equity = equity_open - (
            (float(open_price) - float(intraday_low))
            * self.position.quantity
            * self.position.multiplier
        )
        mm = MarginAccount.maintenance_margin(
            self.position.quantity * float(intraday_low) * self.position.multiplier,
            self.margin_cfg.maintenance_margin_pct,
        )
        return worst_equity < mm

    # ------------------------------------------------------------------
    # Trade application
    # ------------------------------------------------------------------
    def apply_trade(self, trade) -> None:
        """Bridge a legacy trade into the canonical spot fill contract."""
        from .decision_schema import OpenClose as LegacyOC
        from .decision_schema import OrderSide
        from .position import OrderType as V2OT

        side = trade.side if isinstance(trade.side, OrderSide) else OrderSide(str(trade.side))
        open_close = (
            trade.open_close
            if isinstance(trade.open_close, LegacyOC)
            else LegacyOC(str(trade.open_close))
        )

        if side == OrderSide.BUY and open_close in {LegacyOC.OPEN, LegacyOC.AUTO}:
            order_type = V2OT.BUY_TO_OPEN
        elif side == OrderSide.SELL and open_close == LegacyOC.CLOSE:
            order_type = V2OT.SELL_TO_CLOSE
        elif side == OrderSide.SELL and open_close == LegacyOC.AUTO and self.position.is_long():
            order_type = V2OT.SELL_TO_CLOSE
        else:
            raise ValueError(
                "Spot long-only portfolio accepts BUY open or SELL close only"
            )

        if order_type == V2OT.SELL_TO_CLOSE and not self.position.is_long():
            raise ValueError("Spot long-only portfolio cannot close a flat position")

        side_str = side.value
        fill = Fill(
            fill_id=f"trade_{trade.order_id}",
            order_id=trade.order_id,
            decision_id=trade.decision_id,
            date=trade.date,
            ticker=trade.ticker,
            side=side_str,
            quantity=trade.quantity,
            price=trade.price,
            fee=trade.fee,
            slippage_amount=trade.slippage_ticks * trade.tick_size * trade.quantity,
            order_type=order_type,
            open_close=open_close.value,
            realized_pnl_delta=trade.realized_pnl_delta,
            margin_delta=trade.margin_delta,
        )
        self.apply_fill(fill)
        self.trades.append(trade)

    def apply_fill(self, fill: Fill) -> None:
        """Apply a fill under the spot long-only execution contract."""
        if fill.quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if fill.order_type not in {OrderType.BUY_TO_OPEN, OrderType.SELL_TO_CLOSE}:
            raise ValueError(
                "Spot long-only portfolio accepts only BUY_TO_OPEN or SELL_TO_CLOSE"
            )
        side = str(getattr(fill.side, "value", fill.side)).upper()
        if fill.order_type == OrderType.BUY_TO_OPEN:
            if side not in {"BUY", "LONG"}:
                raise ValueError("BUY_TO_OPEN fill requires BUY side")
            if not self.position.is_flat():
                raise ValueError("Spot long-only portfolio cannot add to an existing position")
            self._apply_open(fill)
        else:
            if side != "SELL":
                raise ValueError("SELL_TO_CLOSE fill requires SELL side")
            self._apply_close(fill)

        # Record position log
        self.position_log.append({
            "date": fill.date,
            "order_type": fill.order_type.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "side": self.position.side.value,
            "position_qty": self.position.quantity,
            "avg_entry_price": self.position.avg_entry_price,
            "cash": self.cash,
            "unrealized_pnl": self.position.unrealized_pnl_calc(fill.price),
            "realized_pnl": self.position.realized_pnl,
        })

    def _apply_open(self, fill: Fill) -> None:
        """Apply an open long fill.

        Stock-like cash model: BUY deducts notional plus fee; margin is tracked
        for reporting but is not a separate cash flow.
        """
        signed_qty = fill.quantity

        # 1. Compute margin requirement FIRST (before mutating state)
        add_notional_val = fill.quantity * fill.price * self.position.multiplier
        margin_required = MarginAccount.initial_margin(
            add_notional_val, self.margin_cfg.initial_margin_pct
        )
        equity = self.account_equity(fill.price)
        if margin_required > equity + 1e-9:
            raise InsufficientMarginError(
                f"Insufficient equity for margin: need {margin_required:.2f}, "
                f"have {equity:.2f}."
            )

        total_cost = add_notional_val + fill.fee
        if total_cost > self.cash + 1e-7:
            raise InsufficientMarginError(
                f"Spot cash insufficient: requires {total_cost:.2f} IDR, available {self.cash:.2f} IDR"
            )

        if self.position.is_flat():
            self.position.quantity = signed_qty
            self.position.avg_entry_price = fill.price
            self.position.opened_at = fill.date
        else:
            raise ValueError("Spot long-only portfolio cannot add to an existing position")

        # 3. THEN cash flow
        notional = fill.quantity * fill.price * self.position.multiplier
        self.cash -= (notional + fill.fee)

        # 4. THEN post margin
        self.margin_posted += margin_required
        self.last_mark = fill.price

    def _apply_close(self, fill: Fill) -> None:
        """Apply a full or partial long close with SELL_TO_CLOSE."""
        if self.position.is_flat():
            raise ValueError("Spot long-only portfolio cannot close a flat position")

        close_qty = fill.quantity
        if close_qty > self.abs_qty():
            raise ValueError(
                f"Cannot close {close_qty}: only {self.abs_qty()} open."
            )

        pre_close_abs_qty = self.abs_qty()

        # Calculate realized PnL for the long close.
        realized = (fill.price - self.position.avg_entry_price) * close_qty * self.position.multiplier
        self.position.quantity -= close_qty

        notional = close_qty * fill.price * self.position.multiplier
        self.cash += (notional - fill.fee)

        # Release margin proportionally (tracked only, not cash flow)
        if pre_close_abs_qty > 0:
            released = self.margin_posted * (close_qty / pre_close_abs_qty)
        else:
            released = 0.0
        self.margin_posted -= released

        self.lifetime_realized_pnl += realized
        self.daily_realized_pnl += realized
        self.position.realized_pnl += realized
        fill.realized_pnl_delta = realized

        if self.position.quantity == 0:
            self.position.avg_entry_price = 0.0
            self.position.opened_at = None
            self.position.stop_price = None
            self.position.take_profit = None

        # Warn if cash went negative after close
        if self.cash < -1e-9:
            logger.warning(f"Negative cash after close: {self.cash:.2f}")

        self.last_mark = fill.price

    # ------------------------------------------------------------------
    # Daily settlement
    # ------------------------------------------------------------------
    def mark_to_market(self, date: str, close_price: float) -> PortfolioSnapshot:
        """PRD §17 step 3 — mark-to-market at daily close."""
        mark = float(close_price)

        # Daily carrying costs are zero in cash-only spot mode.
        equity = self.account_equity(mark)
        unrealized = self.position.unrealized_pnl_calc(mark)

        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (equity / self.peak_equity - 1.0) if self.peak_equity > 0 else 0.0

        mm_req = self.maintenance_required(mark)
        im_req = self.initial_required(mark)
        excess = equity - mm_req
        lev = self.leverage(mark)
        util = self.margin_util(mark)

        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            position_qty=self.position.quantity,
            position_avg_price=self.position.avg_entry_price,
            position_value=self.position_value(mark),
            mark_price=mark,
            total_equity=equity,
            drawdown=drawdown,
            margin_used=self.margin_posted,
            margin_available=excess,
            excess_margin=excess,
            maintenance_margin_required=mm_req,
            initial_margin_required=im_req,
            leverage=lev,
            daily_realized_pnl=self.daily_realized_pnl,
            daily_unrealized_pnl=unrealized,
            lifetime_realized_pnl=self.lifetime_realized_pnl,
        )

        self.equity_curve.append(snapshot)

        # Record margin log
        self.margin_log.append({
            "date": date,
            "cash": self.cash,
            "margin_posted": self.margin_posted,
            "equity": equity,
            "maintenance_margin": mm_req,
            "initial_margin": im_req,
            "excess_margin": excess,
            "leverage": lev,
            "margin_utilization": util,
        })

        self.daily_realized_pnl = 0.0
        self.daily_unrealized_pnl = unrealized
        self.last_mark = mark
        return snapshot

    def get_position_snapshot(self) -> dict:
        """Return current position state for logging."""
        return {
            "side": self.position.side.value,
            "qty": self.position.quantity,
            "avg_price": self.position.avg_entry_price,
            "stop_price": self.position.stop_price,
            "take_profit": self.position.take_profit,
            "realized_pnl": self.position.realized_pnl,
            "unrealized_pnl": self.position.unrealized_pnl_calc(self.last_mark or 0.0),
        }
