"""
Risk engine — PRD §11, §16.

Implements the 8-step daily event priority:
1. Hard risk rule
2. Liquidation / margin call guard
3. Stop-loss
4. Forced close / forced cover
5. Existing stop/target management
6. New agent rating mapping
7. Take-profit / scale-out
8. Hold / no order

Bar-by-bar stop/target processing with intraday ambiguity handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from .margin import MarginAccount
from .position import Order, OrderType, Position, PositionSide


@dataclass
class RiskEvent:
    """A risk event that occurred during bar-by-bar processing."""
    date: str
    event_type: str  # "hard_risk", "liquidation", "stop", "forced_close", "margin_call"
    side: str  # "LONG", "SHORT", "FLAT"
    price: float
    reason: str


class RiskEngine:
    """
    PRD §11 — 8-step daily priority risk engine.

    Stateless: all state comes from Portfolio and position.
    """

    def __init__(
        self,
        max_loss_per_trade_pct: float = 0.10,
        max_portfolio_loss_pct: float = 0.20,
        max_exposure_pct: float = 0.50,
        max_single_trade_risk_pct: float = 0.02,
        liquidation_enabled: bool = True,
        stop_loss_enabled: bool = True,
        take_profit_enabled: bool = True,
        margin_call_threshold: float = 0.40,
        auto_liquidate: bool = True,
    ):
        self.max_loss_per_trade_pct = max_loss_per_trade_pct
        self.max_portfolio_loss_pct = max_portfolio_loss_pct
        self.max_exposure_pct = max_exposure_pct
        self.max_single_trade_risk_pct = max_single_trade_risk_pct
        self.liquidation_enabled = liquidation_enabled
        self.stop_loss_enabled = stop_loss_enabled
        self.take_profit_enabled = take_profit_enabled
        self.margin_call_threshold = margin_call_threshold
        self.auto_liquidate = auto_liquidate

    def check_bar(
        self,
        date: str,
        bar: dict,
        position: Position,
        equity: float,
        margin_rate: float,
        maintenance_rate: float,
        max_leverage: float,
        multiplier: float = 1.0,
    ) -> tuple[Optional[Order], list[RiskEvent]]:
        """
        PRD §11 — check risk events for a single bar.

        Returns (order_to_execute, events).
        """
        events: list[RiskEvent] = []
        order: Optional[Order] = None

        if position.is_flat():
            return None, events

        open_price = float(bar.get("open", bar.get("close", 0)))
        close_price = float(bar.get("close", 0))
        low_price = float(bar.get("low", close_price))
        high_price = float(bar.get("high", close_price))

        notional = position.abs_qty() * close_price * multiplier

        # Step 1: Hard risk rule
        if self._check_hard_risk(date, position, equity, events, close_price):
            order = self._force_close_position(date, position, close_price, "hard_risk")
            return order, events

        # Step 2: Liquidation / margin call guard
        if self.liquidation_enabled:
            liq_order = self._check_liquidation(
                date, bar, position, equity,
                margin_rate, maintenance_rate, multiplier, events
            )
            if liq_order is not None:
                return liq_order, events

        # Step 3 & 4: Stop-loss and forced close (bar-by-bar)
        if self.stop_loss_enabled:
            stop_order = self._check_stop_loss(
                date, bar, position, close_price, multiplier, events
            )
            if stop_order is not None:
                return stop_order, events

        # Step 5: Existing stop/target management (already handled above)

        # Step 6: New agent rating mapping (not here — in decision_state_manager)

        # Step 7: Take-profit / scale-out
        if self.take_profit_enabled:
            tp_order = self._check_take_profit(
                date, bar, position, close_price, multiplier, events
            )
            if tp_order is not None:
                return tp_order, events

        # Step 8: Hold / no order
        return None, events

    def _check_hard_risk(
        self,
        date: str,
        position: Position,
        equity: float,
        events: list[RiskEvent],
        close_price: float = 0.0,
    ) -> bool:
        """PRD §11 step 1 — hard risk rule (max loss per trade, max portfolio loss)."""
        if position.is_flat():
            return False

        # Use fresh PnL calculation from current bar's close price
        # instead of the cached position.unrealized_pnl field
        if close_price > 0:
            unrealized_pnl = position.unrealized_pnl_calc(close_price)
        else:
            unrealized_pnl = position.unrealized_pnl

        # Only check max loss when position is LOSING (negative PnL).
        # Profitable positions should not trigger the "max loss" rule.
        if position.abs_qty() > 0 and equity > 0 and unrealized_pnl < 0:
            trade_risk_pct = abs(unrealized_pnl) / equity
            if trade_risk_pct > self.max_loss_per_trade_pct:
                events.append(RiskEvent(
                    date=date,
                    event_type="hard_risk",
                    side=position.side.value,
                    price=position.avg_entry_price,
                    reason=f"Max single trade risk exceeded: {trade_risk_pct * 100:.2f}%",
                ))
                return True

        # Check max portfolio loss
        if equity <= 0 or (unrealized_pnl < 0 and equity > 0 and abs(unrealized_pnl) / equity > self.max_portfolio_loss_pct):
            events.append(RiskEvent(
                date=date,
                event_type="hard_risk",
                side=position.side.value,
                price=position.avg_entry_price,
                reason="Max portfolio loss exceeded",
            ))
            return True

        return False

    def _check_liquidation(
        self,
        date: str,
        bar: dict,
        position: Position,
        equity: float,
        margin_rate: float,
        maintenance_rate: float,
        multiplier: float,
        events: list[RiskEvent],
    ) -> Optional[Order]:
        """PRD §11 step 2 — liquidation / margin call guard."""
        if position.is_flat():
            return None

        close_price = float(bar.get("close", 0))
        open_price = float(bar.get("open", close_price))
        low_price = float(bar.get("low", close_price))
        high_price = float(bar.get("high", close_price))

        # PRD §16.1/§16.2: Long uses low, short uses high
        if position.side == PositionSide.LONG:
            worst_price = low_price
        else:
            worst_price = high_price

        # Calculate worst-case equity from open-price equity.
        # equity = account_equity(open) = cash + unrealized_pnl_at_open
        # worst_equity = equity + (worst_price - open_price) * qty * mult
        # This avoids double-counting the PnL from entry to open.
        if position.side == PositionSide.LONG:
            worst_pnl_from_open = (worst_price - open_price) * position.abs_qty() * multiplier
        else:
            worst_pnl_from_open = (open_price - worst_price) * position.abs_qty() * multiplier

        worst_equity = equity + worst_pnl_from_open

        # Check maintenance margin
        notional = position.abs_qty() * worst_price * multiplier
        maintenance = MarginAccount.maintenance_margin(notional, maintenance_rate)

        if MarginAccount.liquidation_guard(worst_equity, maintenance):
            events.append(RiskEvent(
                date=date,
                event_type="liquidation",
                side=position.side.value,
                price=worst_price,
                reason=f"Liquidation triggered: equity {worst_equity:.2f} < maintenance {maintenance:.2f}",
            ))
            if self.auto_liquidate:
                return self._force_close_position(date, position, worst_price, "liquidation")
            return None

        # Check margin call
        if MarginAccount.margin_call(worst_equity, maintenance, self.margin_call_threshold):
            events.append(RiskEvent(
                date=date,
                event_type="margin_call",
                side=position.side.value,
                price=worst_price,
                reason=f"Margin call: equity {worst_equity:.2f} < maintenance*(1+threshold) {maintenance*(1+self.margin_call_threshold):.2f}",
            ))

        return None

    def _check_stop_loss(
        self,
        date: str,
        bar: dict,
        position: Position,
        close_price: float,
        multiplier: float,
        events: list[RiskEvent],
    ) -> Optional[Order]:
        """PRD §11 step 3 — stop-loss check (bar-by-bar, intraday ambiguity)."""
        if position.is_flat() or position.stop_price is None:
            return None

        low_price = float(bar.get("low", close_price))
        high_price = float(bar.get("high", close_price))
        open_price = float(bar.get("open", close_price))

        stop_price = position.stop_price

        if position.side == PositionSide.LONG:
            # PRD §16.1: Long stop triggered if Low <= stop_price
            stop_triggered = low_price <= stop_price
            # PRD §16.3: If both stop and target triggered, stop wins (conservative)
            if stop_triggered:
                events.append(RiskEvent(
                    date=date,
                    event_type="stop",
                    side="LONG",
                    price=stop_price,
                    reason=f"Long stop triggered: low {low_price} <= stop {stop_price}",
                ))
                return self._force_close_position(date, position, stop_price, "stop")

        elif position.side == PositionSide.SHORT:
            # PRD §16.2: Short stop triggered if High >= stop_price
            stop_triggered = high_price >= stop_price
            if stop_triggered:
                events.append(RiskEvent(
                    date=date,
                    event_type="stop",
                    side="SHORT",
                    price=stop_price,
                    reason=f"Short stop triggered: high {high_price} >= stop {stop_price}",
                ))
                return self._force_close_position(date, position, stop_price, "stop")

        return None

    def _check_take_profit(
        self,
        date: str,
        bar: dict,
        position: Position,
        close_price: float,
        multiplier: float,
        events: list[RiskEvent],
    ) -> Optional[Order]:
        """PRD §11 step 7 — take-profit check."""
        if position.is_flat() or position.take_profit is None:
            return None

        low_price = float(bar.get("low", close_price))
        high_price = float(bar.get("high", close_price))

        tp_price = position.take_profit

        if position.side == PositionSide.LONG:
            # PRD §16.1: Long target triggered if High >= take_profit
            if high_price >= tp_price:
                events.append(RiskEvent(
                    date=date,
                    event_type="take_profit",
                    side="LONG",
                    price=tp_price,
                    reason=f"Long target triggered: high {high_price} >= target {tp_price}",
                ))
                return self._force_close_position(date, position, tp_price, "take_profit")

        elif position.side == PositionSide.SHORT:
            # PRD §16.2: Short target triggered if Low <= take_profit
            if low_price <= tp_price:
                events.append(RiskEvent(
                    date=date,
                    event_type="take_profit",
                    side="SHORT",
                    price=tp_price,
                    reason=f"Short target triggered: low {low_price} <= target {tp_price}",
                ))
                return self._force_close_position(date, position, tp_price, "take_profit")

        return None

    def _force_close_position(
        self,
        date: str,
        position: Position,
        price: float,
        reason: str,
    ) -> Order:
        """Generate a forced close order."""
        uid = uuid4().hex[:8]
        if position.side == PositionSide.LONG:
            return Order(
                order_id=f"risk_close_{date}_{reason}_{uid}",
                decision_id="",
                ticker=position.ticker,
                order_type=OrderType.SELL_TO_CLOSE,
                quantity=abs(position.quantity),
                execution_date=date,
                price=price,
                reason=reason,
            )
        elif position.side == PositionSide.SHORT:
            return Order(
                order_id=f"risk_close_{date}_{reason}_{uid}",
                decision_id="",
                ticker=position.ticker,
                order_type=OrderType.BUY_TO_CLOSE,
                quantity=abs(position.quantity),
                execution_date=date,
                price=price,
                reason=reason,
            )
        else:
            return Order(
                order_id=f"risk_noop_{date}_{uid}",
                decision_id="",
                ticker=position.ticker,
                order_type=OrderType.NO_ORDER,
                quantity=0,
                execution_date=date,
                price=price,
                reason="no_position",
            )


def compute_atr(ohlcv_df, period: int = 14, current_date: str = None) -> Optional[float]:
    """Compute ATR (Average True Range) from OHLCV DataFrame.

    Returns the latest ATR value, or None if insufficient data.
    When current_date is provided, filters OHLCV to prevent future data leakage.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return None

    import pandas as pd
    # Filter to current_date to prevent future data leakage
    if current_date is not None:
        df = ohlcv_df[ohlcv_df["date"].astype(str) <= current_date].copy()
    else:
        df = ohlcv_df.copy()

    if len(df) < period + 1:
        return None

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean().iloc[-1]
    if pd.isna(atr):
        return None
    return float(atr)


def update_position_risk_levels(
    position: Position,
    decision: Any,
    risk_config: Any,
    ohlcv_df: Any = None,
    current_date: str = None,
) -> None:
    """Update position stop/take_profit from decision and risk config.

    When holding a position (LONG/SHORT), always ensure stop_price and
    take_profit are set.
    """
    if not getattr(decision, "valid", True):
        return
    if position.is_flat():
        return

    entry_price = position.avg_entry_price
    agent_stop = getattr(decision, "stop_price", None)
    agent_tp = getattr(decision, "take_profit", None)

    if entry_price > 0:
        if position.is_short():
            if agent_stop is not None and agent_tp is not None:
                if agent_stop < entry_price and agent_tp > entry_price:
                    agent_stop, agent_tp = agent_tp, agent_stop
            if agent_stop is not None and agent_stop <= entry_price:
                agent_stop = None
            if agent_tp is not None and agent_tp >= entry_price:
                agent_tp = None
        elif position.is_long():
            if agent_stop is not None and agent_tp is not None:
                if agent_stop > entry_price and agent_tp < entry_price:
                    agent_stop, agent_tp = agent_tp, agent_stop
            if agent_stop is not None and agent_stop >= entry_price:
                agent_stop = None
            if agent_tp is not None and agent_tp <= entry_price:
                agent_tp = None

    cached_atr = None
    if getattr(risk_config, "use_atr_based_stops", False) and ohlcv_df is not None:
        cached_atr = compute_atr(ohlcv_df, getattr(risk_config, "atr_period", 14), current_date=current_date)

    if agent_stop is not None:
        position.stop_price = agent_stop
    elif entry_price > 0:
        if cached_atr is not None:
            mult = getattr(risk_config, "atr_stop_multiplier", 2.0)
            if position.is_long():
                position.stop_price = entry_price - (cached_atr * mult)
            else:
                position.stop_price = entry_price + (cached_atr * mult)
        else:
            stop_pct = getattr(risk_config, "default_stop_pct", 0.05)
            if position.is_long():
                position.stop_price = entry_price * (1 - stop_pct)
            else:
                position.stop_price = entry_price * (1 + stop_pct)

    if agent_tp is not None:
        position.take_profit = agent_tp
    elif entry_price > 0:
        if cached_atr is not None:
            mult = getattr(risk_config, "atr_tp_multiplier", 3.0)
            if position.is_long():
                position.take_profit = entry_price + (cached_atr * mult)
            else:
                position.take_profit = entry_price - (cached_atr * mult)
        else:
            tp_pct = getattr(risk_config, "default_take_profit_pct", 0.10)
            if position.is_long():
                position.take_profit = entry_price * (1 + tp_pct)
            else:
                position.take_profit = entry_price * (1 - tp_pct)
