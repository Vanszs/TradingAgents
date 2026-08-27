"""
Risk engine — PRD §11, §16.

Implements static spot-long risk handling:
1. Static stop-loss
2. Static take-profit
3. Time-stop/end-horizon closure
4. Hold / no order

Bar-by-bar stop/target processing with intraday ambiguity handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from .margin import MarginAccount
from .position import Order, OrderType, Position


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
        liquidation_enabled: bool = False,
        hard_risk_enabled: bool = False,
        stop_loss_enabled: bool = True,
        take_profit_enabled: bool = True,
        margin_call_threshold: float = 0.40,
        auto_liquidate: bool = False,
    ):
        self.max_loss_per_trade_pct = max_loss_per_trade_pct
        self.max_portfolio_loss_pct = max_portfolio_loss_pct
        self.max_exposure_pct = max_exposure_pct
        self.max_single_trade_risk_pct = max_single_trade_risk_pct
        self.liquidation_enabled = liquidation_enabled
        self.hard_risk_enabled = hard_risk_enabled
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
        if position.quantity < 0:
            raise ValueError("spot long-only risk engine rejects short positions")
        close_price = float(bar.get("close", 0))
        low_price = float(bar.get("low", close_price))
        high_price = float(bar.get("high", close_price))

        notional = position.abs_qty() * close_price * multiplier

        # Step 1: Hard risk rule
        if self.hard_risk_enabled and self._check_hard_risk(date, position, equity, events, close_price):
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

        # Spot long-only: worst case is the intraday low.
        worst_price = low_price

        # Calculate worst-case equity from open-price equity without double-counting
        # PnL from entry to the bar open.
        worst_pnl_from_open = (worst_price - open_price) * position.abs_qty() * multiplier

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
        open_price = float(bar.get("open", close_price))
        stop_price = position.stop_price

        # Long stop triggers on the intraday low; stop wins on ambiguity.
        if low_price <= stop_price:
            fill_price = min(open_price, stop_price)
            events.append(RiskEvent(
                date=date,
                event_type="stop",
                side="LONG",
                price=fill_price,
                reason=f"Long stop triggered: low {low_price} <= stop {stop_price} (fill: {fill_price})",
            ))
            return self._force_close_position(date, position, fill_price, "stop")

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

        high_price = float(bar.get("high", close_price))
        tp_price = position.take_profit
        open_price = float(bar.get("open", close_price))

        # Long target triggers on the intraday high.
        if high_price >= tp_price:
            fill_price = max(open_price, tp_price)
            events.append(RiskEvent(
                date=date,
                event_type="take_profit",
                side="LONG",
                price=fill_price,
                reason=f"Long target triggered: high {high_price} >= target {tp_price} (fill: {fill_price})",
            ))
            return self._force_close_position(date, position, fill_price, "take_profit")

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
        return Order(
            order_id=f"risk_close_{date}_{reason}_{uid}",
            decision_id="",
            ticker=position.ticker,
            order_type=OrderType.SELL_TO_CLOSE,
            quantity=position.quantity,
            execution_date=date,
            price=price,
            reason=reason,
            is_risk_order=True,
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

    When holding a long position, keep static stop_price and take_profit levels.
    Never ratchet either level after entry.
    """
    if not getattr(decision, "valid", True):
        return
    if position.is_flat():
        return

    entry_price = position.avg_entry_price
    agent_stop = getattr(decision, "stop_price", None)
    agent_tp = getattr(decision, "take_profit", None)

    current_mark = 0.0
    if ohlcv_df is not None and hasattr(ohlcv_df, "empty") and not ohlcv_df.empty:
        df_curr = ohlcv_df
        if current_date and "date" in ohlcv_df.columns:
            df_curr = ohlcv_df[ohlcv_df["date"].astype(str) <= str(current_date)]
        elif current_date and "Date" in ohlcv_df.columns:
            df_curr = ohlcv_df[ohlcv_df["Date"].astype(str) <= str(current_date)]

        if not df_curr.empty:
            if "Close" in df_curr.columns:
                current_mark = float(df_curr["Close"].iloc[-1])
            elif "close" in df_curr.columns:
                current_mark = float(df_curr["close"].iloc[-1])

    ref_price = current_mark if current_mark > 0 else entry_price

    # Risk levels are entry-time invariants. Never replace a level after it
    # has been initialized; later reports cannot ratchet or widen exits.
    if position.stop_price is not None:
        agent_stop = None
    if position.take_profit is not None:
        agent_tp = None

    if ref_price > 0 and position.is_long():
        if agent_stop is not None and agent_stop >= ref_price:
            agent_stop = None
        if agent_tp is not None and agent_tp <= ref_price:
            agent_tp = None

    cached_atr = None
    if getattr(risk_config, "use_atr_based_stops", False) and ohlcv_df is not None:
        cached_atr = compute_atr(ohlcv_df, getattr(risk_config, "atr_period", 14), current_date=current_date)

    if agent_stop is not None:
        position.stop_price = agent_stop
    elif entry_price > 0 and position.stop_price is None:
        # Initialize one static stop; never move it after initialization.
        if cached_atr is not None:
            mult = getattr(risk_config, "atr_stop_multiplier", 2.0)
            position.stop_price = entry_price - (cached_atr * mult)
        else:
            stop_pct = getattr(risk_config, "default_stop_pct", 0.05)
            position.stop_price = entry_price * (1 - stop_pct)

    if agent_tp is not None:
        position.take_profit = agent_tp
    elif entry_price > 0 and position.take_profit is None:
        if cached_atr is not None:
            mult = getattr(risk_config, "atr_tp_multiplier", 3.0)
            position.take_profit = entry_price + (cached_atr * mult)
        else:
            tp_pct = getattr(risk_config, "default_take_profit_pct", 0.10)
            position.take_profit = entry_price * (1 + tp_pct)
