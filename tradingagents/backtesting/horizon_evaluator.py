"""
Forward Horizon Evaluator for Single-Shot Trading Signals.

Evaluates price trajectory from entry day T1 across horizon H trading days
against Take Profit (TP) and Stop Loss (SL) barrier levels.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, List, Optional

import pandas as pd


class EvaluationOutcome(str, Enum):
    HIT_TAKE_PROFIT = "HIT_TAKE_PROFIT"
    HIT_STOP_LOSS = "HIT_STOP_LOSS"
    HIT_TRAILING_STOP = "HIT_TRAILING_STOP"
    HIT_BREAK_EVEN = "HIT_BREAK_EVEN"
    HIT_TIME_STOP = "HIT_TIME_STOP"
    EXPIRED = "EXPIRED"
    NO_ORDER = "NO_ORDER"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_FILL = "NO_FILL"


@dataclass
class DailyExcursionBar:
    """Tracking metrics for a single bar during the holding period."""
    bar_index: int
    date: str
    open: float
    high: float
    low: float
    close: float
    unrealized_return_close_pct: float
    unrealized_mfe_pct: float
    unrealized_mae_pct: float


@dataclass
class EvaluationResult:
    """Evaluation with explicit planned and actual execution prices."""
    ticker: str
    signal_date: str
    entry_date: Optional[str]
    actual_entry_price: Optional[float]
    exit_date: Optional[str]
    exit_price: float
    outcome: EvaluationOutcome
    side: str  # "LONG" or "SHORT"
    take_profit: Optional[float]
    stop_loss: Optional[float]
    planned_time_horizon_days: int
    actual_holding_days: int
    realized_return_pct: float
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    planned_entry_price: Optional[float] = None
    entry_policy: str = "T1_OPEN"
    signal_timestamp: Optional[str] = None
    entry_timestamp: Optional[str] = None
    reference_price_at_signal: Optional[float] = None
    planned_rr_ratio: Optional[float] = None
    realized_rr_ratio: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    break_even_trigger_pct: Optional[float] = None
    max_holding_days: Optional[int] = None
    mfe_efficiency: Optional[float] = None
    trajectory: List[DailyExcursionBar] = field(default_factory=list)

    @property
    def entry_price(self) -> Optional[float]:
        """Backward-compatible alias for the actual simulated fill price."""
        return self.actual_entry_price

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


class HorizonEvaluator:
    """
    Stateless evaluator for single-shot trading signals.

    Rules:
    1. ``actual_entry_price`` controls the simulated fill; callers may set it to
       the AI's planned price for an assumed-entry evaluation.
    2. Without an explicit actual price, entry fills at T1 Open (first trading
       session following T0).
    3. Evaluates daily High/Low extremes against TP/SL levels.
    4. Conservative conflict rule: if both Stop Loss and Take Profit are touched
       within the same bar, Stop Loss triggers first.
    5. If neither barrier is reached by T_H, position exits at T_H Close.
    """

    @staticmethod
    def evaluate(
        ticker: str,
        signal_date: str,
        side: str,
        take_profit: Optional[float],
        stop_loss: Optional[float],
        time_horizon_days: int,
        ohlcv_df: pd.DataFrame,
        planned_entry_price: Optional[float] = None,
        actual_entry_price: Optional[float] = None,
        actual_entry_date: Optional[str] = None,
        entry_policy: str = "T1_OPEN",
        signal_timestamp: Optional[str] = None,
        entry_timestamp: Optional[str] = None,
        reference_price_at_signal: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
        break_even_trigger_pct: Optional[float] = None,
        max_holding_days: Optional[int] = None,
    ) -> EvaluationResult:
        if not 1 <= time_horizon_days <= 252:
            raise ValueError("time_horizon_days must be between 1 and 252")

        signal_date = pd.to_datetime(signal_date, errors="raise", utc=True).strftime("%Y-%m-%d")
        side_norm = side.upper()
        if side_norm not in ("LONG", "BUY", "SHORT", "SELL"):
            return EvaluationResult(
                ticker=ticker,
                signal_date=signal_date,
                entry_date=None,
                actual_entry_price=None,
                exit_date=None,
                exit_price=0.0,
                outcome=EvaluationOutcome.NO_ORDER,
                side="FLAT",
                take_profit=take_profit,
                stop_loss=stop_loss,
                planned_time_horizon_days=time_horizon_days,
                actual_holding_days=0,
                realized_return_pct=0.0,
                max_favorable_excursion_pct=0.0,
                max_adverse_excursion_pct=0.0,
                planned_entry_price=planned_entry_price,
            )

        is_long = side_norm in ("LONG", "BUY")
        if take_profit is None or stop_loss is None:
            raise ValueError("active orders require both take_profit and stop_loss")

        required_columns = {"date", "open", "high", "low", "close"}
        missing_columns = required_columns.difference(ohlcv_df.columns)
        if missing_columns:
            raise ValueError(
                f"ohlcv_df is missing required columns: {sorted(missing_columns)}"
            )

        fill_timestamp = None
        if entry_timestamp is not None:
            fill_timestamp = pd.Timestamp(entry_timestamp)
            if fill_timestamp.tzinfo is None or fill_timestamp.utcoffset() is None:
                raise ValueError("entry_timestamp must include a timezone")
            fill_timestamp = fill_timestamp.tz_convert("UTC")

        entry_date = None
        if actual_entry_date is not None:
            entry_date = pd.to_datetime(actual_entry_date, errors="raise", utc=True).strftime("%Y-%m-%d")
        if fill_timestamp is not None:
            fill_date = fill_timestamp.strftime("%Y-%m-%d")
            if entry_date is not None and entry_date != fill_date:
                raise ValueError("entry_timestamp date does not match actual_entry_date")
            entry_date = fill_date
        assumed_entry = entry_policy == "ASSUMED_AI_ENTRY"
        if assumed_entry and entry_date not in (None, signal_date):
            raise ValueError("ASSUMED_AI_ENTRY entry date must match signal_date")
        if assumed_entry and entry_date is None:
            entry_date = signal_date

        df = ohlcv_df.copy()
        parsed_dates = pd.to_datetime(df["date"], errors="raise", utc=True)
        if parsed_dates.duplicated().any():
            raise ValueError("ohlcv_df contains duplicate dates")
        df["date_str"] = parsed_dates.dt.strftime("%Y-%m-%d")
        numeric = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="raise")
        if not numeric.apply(lambda column: column.map(math.isfinite).all()).all() or (numeric <= 0).any().any():
            raise ValueError("ohlcv_df prices must be finite positive numbers")
        df[["open", "high", "low", "close"]] = numeric
        future_df = df[df["date_str"] > signal_date].sort_values("date_str").reset_index(drop=True)
        if entry_date is not None and not assumed_entry:
            if entry_date not in set(future_df["date_str"]):
                raise ValueError("actual entry date is not present in future daily data")
            future_df = future_df[future_df["date_str"] >= entry_date].reset_index(drop=True)

        if future_df.empty:
            return EvaluationResult(
                ticker=ticker,
                signal_date=signal_date,
                entry_date=None,
                actual_entry_price=None,
                exit_date=None,
                exit_price=0.0,
                outcome=EvaluationOutcome.INSUFFICIENT_DATA,
                side="LONG" if is_long else "SHORT",
                take_profit=take_profit,
                stop_loss=stop_loss,
                planned_time_horizon_days=time_horizon_days,
                actual_holding_days=0,
                realized_return_pct=0.0,
                max_favorable_excursion_pct=0.0,
                max_adverse_excursion_pct=0.0,
                planned_entry_price=planned_entry_price,
            )

        # T1 Entry execution
        entry_row = future_df.iloc[0]
        entry_date = entry_date or entry_row["date_str"]
        b_open_first = float(entry_row["open"])
        b_high_first = float(entry_row["high"])
        b_low_first = float(entry_row["low"])

        if actual_entry_price is not None:
            if is_long:
                if b_low_first <= actual_entry_price <= b_high_first:
                    entry_price = actual_entry_price
                elif b_open_first < actual_entry_price:  # Favorable gap-down open for buy
                    entry_price = b_open_first
                else:
                    return EvaluationResult(
                        ticker=ticker,
                        signal_date=signal_date,
                        entry_date=None,
                        actual_entry_price=None,
                        exit_date=None,
                        exit_price=0.0,
                        outcome=EvaluationOutcome.NO_FILL,
                        side="LONG",
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        planned_time_horizon_days=time_horizon_days,
                        actual_holding_days=0,
                        realized_return_pct=0.0,
                        max_favorable_excursion_pct=0.0,
                        max_adverse_excursion_pct=0.0,
                        planned_entry_price=planned_entry_price,
                        entry_policy=entry_policy,
                    )
            else:
                if b_low_first <= actual_entry_price <= b_high_first:
                    entry_price = actual_entry_price
                elif b_open_first > actual_entry_price:  # Favorable gap-up open for short
                    entry_price = b_open_first
                else:
                    return EvaluationResult(
                        ticker=ticker,
                        signal_date=signal_date,
                        entry_date=None,
                        actual_entry_price=None,
                        exit_date=None,
                        exit_price=0.0,
                        outcome=EvaluationOutcome.NO_FILL,
                        side="SHORT",
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        planned_time_horizon_days=time_horizon_days,
                        actual_holding_days=0,
                        realized_return_pct=0.0,
                        max_favorable_excursion_pct=0.0,
                        max_adverse_excursion_pct=0.0,
                        planned_entry_price=planned_entry_price,
                        entry_policy=entry_policy,
                    )
        else:
            entry_price = b_open_first

        if entry_price <= 0:
            entry_price = float(entry_row["close"])

        # Validate barrier order sanity (TP must be above SL for LONG, and below SL for SHORT)
        if (is_long and take_profit <= stop_loss) or (not is_long and take_profit >= stop_loss):
            raise ValueError("take_profit and stop_loss do not match entry direction")

        # Gap open barrier evaluation at T1 Open
        if is_long:
            if take_profit is not None and entry_price >= take_profit:
                return EvaluationResult(
                    ticker=ticker,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    actual_entry_price=entry_price,
                    exit_date=entry_date,
                    exit_price=entry_price,
                    outcome=EvaluationOutcome.HIT_TAKE_PROFIT,
                    side="LONG",
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    planned_time_horizon_days=time_horizon_days,
                    actual_holding_days=1,
                    realized_return_pct=0.0,
                    max_favorable_excursion_pct=0.0,
                    max_adverse_excursion_pct=0.0,
                    planned_entry_price=planned_entry_price,
                    entry_policy=entry_policy,
                )
            if stop_loss is not None and entry_price <= stop_loss:
                return EvaluationResult(
                    ticker=ticker,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    actual_entry_price=entry_price,
                    exit_date=entry_date,
                    exit_price=entry_price,
                    outcome=EvaluationOutcome.HIT_STOP_LOSS,
                    side="LONG",
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    planned_time_horizon_days=time_horizon_days,
                    actual_holding_days=1,
                    realized_return_pct=0.0,
                    max_favorable_excursion_pct=0.0,
                    max_adverse_excursion_pct=0.0,
                    planned_entry_price=planned_entry_price,
                    entry_policy=entry_policy,
                )
        else:  # SHORT
            if take_profit is not None and entry_price <= take_profit:
                return EvaluationResult(
                    ticker=ticker,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    actual_entry_price=entry_price,
                    exit_date=entry_date,
                    exit_price=entry_price,
                    outcome=EvaluationOutcome.HIT_TAKE_PROFIT,
                    side="SHORT",
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    planned_time_horizon_days=time_horizon_days,
                    actual_holding_days=1,
                    realized_return_pct=0.0,
                    max_favorable_excursion_pct=0.0,
                    max_adverse_excursion_pct=0.0,
                    planned_entry_price=planned_entry_price,
                    entry_policy=entry_policy,
                )
            if stop_loss is not None and entry_price >= stop_loss:
                return EvaluationResult(
                    ticker=ticker,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    actual_entry_price=entry_price,
                    exit_date=entry_date,
                    exit_price=entry_price,
                    outcome=EvaluationOutcome.HIT_STOP_LOSS,
                    side="SHORT",
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    planned_time_horizon_days=time_horizon_days,
                    actual_holding_days=1,
                    realized_return_pct=0.0,
                    max_favorable_excursion_pct=0.0,
                    max_adverse_excursion_pct=0.0,
                    planned_entry_price=planned_entry_price,
                    entry_policy=entry_policy,
                )

        if (is_long and not stop_loss < entry_price < take_profit) or (
            not is_long and not take_profit < entry_price < stop_loss
        ):
            raise ValueError("take_profit and stop_loss do not match entry direction")

        # Calculate planned R:R
        if is_long:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        planned_rr = round(reward / risk, 2)

        # Determine max evaluation window based on available bars
        has_full_horizon = len(future_df) >= time_horizon_days
        horizon_bars = (
            future_df.iloc[:time_horizon_days].reset_index(drop=True)
            if has_full_horizon
            else future_df.copy().reset_index(drop=True)
        )

        trajectory: List[DailyExcursionBar] = []
        max_mfe = 0.0
        max_mae = 0.0

        outcome = EvaluationOutcome.EXPIRED if has_full_horizon else EvaluationOutcome.INSUFFICIENT_DATA
        exit_date = horizon_bars.iloc[-1]["date_str"] if has_full_horizon else None
        exit_price = float(horizon_bars.iloc[-1]["close"]) if has_full_horizon else 0.0
        actual_days = len(horizon_bars)

        peak_high = entry_price
        peak_low = entry_price
        current_sl = stop_loss
        is_break_even_active = False
        is_trailing_active = False

        effective_max_holding = max_holding_days or time_horizon_days

        for idx, row in horizon_bars.iterrows():
            bar_date = row["date_str"]
            b_open = float(row["open"])
            b_high = float(row["high"])
            b_low = float(row["low"])
            b_close = float(row["close"])

            if is_long:
                # 1. Gap Open Take Profit
                if take_profit is not None and b_open >= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = b_open
                    actual_days = idx + 1
                    exit_mfe = (exit_price - entry_price) / entry_price
                    max_mfe = max(max_mfe, exit_mfe)
                    exit_mae = min(0.0, (b_open - entry_price) / entry_price)
                    max_mae = min(max_mae, exit_mae)
                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mfe * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 2. Gap Open Stop Loss
                if current_sl is not None and b_open <= current_sl:
                    if is_trailing_active:
                        outcome = EvaluationOutcome.HIT_TRAILING_STOP
                    elif is_break_even_active:
                        outcome = EvaluationOutcome.HIT_BREAK_EVEN
                    else:
                        outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = b_open
                    actual_days = idx + 1
                    exit_mae = (exit_price - entry_price) / entry_price
                    max_mae = min(max_mae, exit_mae)
                    exit_mfe = max(0.0, (b_open - entry_price) / entry_price)
                    max_mfe = max(max_mfe, exit_mfe)
                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mae * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 3. Intraday Stop loss / Trailing stop / Break-even hit
                if current_sl is not None and b_low <= current_sl:
                    if is_trailing_active:
                        outcome = EvaluationOutcome.HIT_TRAILING_STOP
                    elif is_break_even_active:
                        outcome = EvaluationOutcome.HIT_BREAK_EVEN
                    else:
                        outcome = EvaluationOutcome.HIT_STOP_LOSS

                    exit_date = bar_date
                    exit_price = current_sl
                    actual_days = idx + 1

                    exit_mae = (exit_price - entry_price) / entry_price
                    max_mae = min(max_mae, exit_mae)
                    exit_mfe = max(0.0, (b_open - entry_price) / entry_price)
                    max_mfe = max(max_mfe, exit_mfe)

                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mae * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 4. Intraday Take profit barrier hit
                if take_profit is not None and b_high >= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = take_profit
                    actual_days = idx + 1

                    exit_mfe = (exit_price - entry_price) / entry_price
                    max_mfe = max(max_mfe, exit_mfe)
                    exit_mae = min(0.0, (b_open - entry_price) / entry_price)
                    max_mae = min(max_mae, exit_mae)

                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mfe * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 5. Time stop exit
                if max_holding_days is not None and (idx + 1) >= max_holding_days and max_holding_days < time_horizon_days:
                    outcome = EvaluationOutcome.HIT_TIME_STOP
                    exit_date = bar_date
                    exit_price = b_close
                    actual_days = idx + 1
                    close_ret = (b_close - entry_price) / entry_price
                    mfe_bar = (b_high - entry_price) / entry_price
                    mae_bar = (b_low - entry_price) / entry_price
                    max_mfe = max(max_mfe, mfe_bar)
                    max_mae = min(max_mae, mae_bar)

                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=b_close,
                            unrealized_return_close_pct=round(close_ret * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # Normal un-triggered Long bar
                mfe_bar = (b_high - entry_price) / entry_price
                mae_bar = (b_low - entry_price) / entry_price
                close_ret = (b_close - entry_price) / entry_price
                max_mfe = max(max_mfe, mfe_bar)
                max_mae = min(max_mae, mae_bar)

                peak_high = max(peak_high, b_high)
                current_runup_pct = (peak_high - entry_price) / entry_price

                # Break-Even Ratchet
                if break_even_trigger_pct and current_runup_pct >= break_even_trigger_pct:
                    be_price = entry_price * 1.001
                    if current_sl is None or be_price > current_sl:
                        current_sl = be_price
                        is_break_even_active = True

                # Trailing Stop Ratchet
                if trailing_stop_pct and (current_runup_pct >= (trailing_stop_pct * 0.8) or is_trailing_active):
                    trail_price = peak_high * (1.0 - trailing_stop_pct)
                    if current_sl is None or trail_price > current_sl:
                        current_sl = trail_price
                        is_trailing_active = True

            else:  # SHORT
                # 1. Gap Open Take Profit (Short)
                if take_profit is not None and b_open <= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = b_open
                    actual_days = idx + 1
                    exit_mfe = (entry_price - exit_price) / entry_price
                    max_mfe = max(max_mfe, exit_mfe)
                    exit_mae = min(0.0, (entry_price - b_open) / entry_price)
                    max_mae = min(max_mae, exit_mae)
                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mfe * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 2. Gap Open Stop Loss (Short)
                if current_sl is not None and b_open >= current_sl:
                    if is_trailing_active:
                        outcome = EvaluationOutcome.HIT_TRAILING_STOP
                    elif is_break_even_active:
                        outcome = EvaluationOutcome.HIT_BREAK_EVEN
                    else:
                        outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = b_open
                    actual_days = idx + 1
                    exit_mae = (entry_price - exit_price) / entry_price
                    max_mae = min(max_mae, exit_mae)
                    exit_mfe = max(0.0, (entry_price - b_open) / entry_price)
                    max_mfe = max(max_mfe, exit_mfe)
                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mae * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 3. Intraday Stop loss / Trailing stop / Break-even hit (Short)
                if current_sl is not None and b_high >= current_sl:
                    if is_trailing_active:
                        outcome = EvaluationOutcome.HIT_TRAILING_STOP
                    elif is_break_even_active:
                        outcome = EvaluationOutcome.HIT_BREAK_EVEN
                    else:
                        outcome = EvaluationOutcome.HIT_STOP_LOSS

                    exit_date = bar_date
                    exit_price = current_sl
                    actual_days = idx + 1

                    exit_mae = (entry_price - exit_price) / entry_price
                    max_mae = min(max_mae, exit_mae)
                    exit_mfe = max(0.0, (entry_price - b_open) / entry_price)
                    max_mfe = max(max_mfe, exit_mfe)

                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mae * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 4. Intraday Take profit barrier hit (Short)
                if take_profit is not None and b_low <= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = take_profit
                    actual_days = idx + 1

                    exit_mfe = (entry_price - exit_price) / entry_price
                    max_mfe = max(max_mfe, exit_mfe)
                    exit_mae = min(0.0, (entry_price - b_open) / entry_price)
                    max_mae = min(max_mae, exit_mae)

                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mfe * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 5. Time stop exit (Short)
                if max_holding_days is not None and (idx + 1) >= max_holding_days and max_holding_days < time_horizon_days:
                    outcome = EvaluationOutcome.HIT_TIME_STOP
                    exit_date = bar_date
                    exit_price = b_close
                    actual_days = idx + 1
                    close_ret = (entry_price - b_close) / entry_price
                    mfe_bar = (entry_price - b_low) / entry_price
                    mae_bar = (entry_price - b_high) / entry_price
                    max_mfe = max(max_mfe, mfe_bar)
                    max_mae = min(max_mae, mae_bar)

                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=b_close,
                            unrealized_return_close_pct=round(close_ret * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # Normal un-triggered Short bar
                mfe_bar = (entry_price - b_low) / entry_price
                mae_bar = (entry_price - b_high) / entry_price
                close_ret = (entry_price - b_close) / entry_price
                max_mfe = max(max_mfe, mfe_bar)
                max_mae = min(max_mae, mae_bar)

                peak_low = min(peak_low, b_low)
                current_drop_pct = (entry_price - peak_low) / entry_price

                # Break-Even Ratchet
                if break_even_trigger_pct and current_drop_pct >= break_even_trigger_pct:
                    be_price = entry_price * 0.999
                    if current_sl is None or be_price < current_sl:
                        current_sl = be_price
                        is_break_even_active = True

                # Trailing Stop Ratchet
                if trailing_stop_pct and (current_drop_pct >= (trailing_stop_pct * 0.8) or is_trailing_active):
                    trail_price = peak_low * (1.0 + trailing_stop_pct)
                    if current_sl is None or trail_price < current_sl:
                        current_sl = trail_price
                        is_trailing_active = True

            trajectory.append(
                DailyExcursionBar(
                    bar_index=idx + 1,
                    date=bar_date,
                    open=b_open,
                    high=b_high,
                    low=b_low,
                    close=b_close,
                    unrealized_return_close_pct=round(close_ret * 100.0, 2),
                    unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                    unrealized_mae_pct=round(max_mae * 100.0, 2),
                )
            )

        # Calculate Realized Return
        is_terminal_fill = outcome in (
            EvaluationOutcome.HIT_TAKE_PROFIT,
            EvaluationOutcome.HIT_STOP_LOSS,
            EvaluationOutcome.HIT_TRAILING_STOP,
            EvaluationOutcome.HIT_BREAK_EVEN,
            EvaluationOutcome.HIT_TIME_STOP,
            EvaluationOutcome.EXPIRED,
        )

        if is_terminal_fill:
            if is_long:
                realized_return = (exit_price - entry_price) / entry_price
            else:
                realized_return = (entry_price - exit_price) / entry_price
            risk = abs(entry_price - stop_loss)
            realized_rr = round(realized_return / (risk / entry_price), 2) if risk > 0 else None
            mfe_eff = round(realized_return / max(max_mfe, 0.0001), 2)
        else:
            realized_return = 0.0
            realized_rr = None
            actual_entry_price = None
            mfe_eff = None

        return EvaluationResult(
            ticker=ticker,
            signal_date=signal_date,
            entry_date=entry_date,
            actual_entry_price=round(entry_price, 2) if is_terminal_fill else None,
            exit_date=exit_date,
            exit_price=round(exit_price, 2),
            outcome=outcome,
            side="LONG" if is_long else "SHORT",
            take_profit=take_profit,
            stop_loss=stop_loss,
            planned_time_horizon_days=time_horizon_days,
            actual_holding_days=actual_days,
            realized_return_pct=round(realized_return * 100.0, 2),
            max_favorable_excursion_pct=round(max_mfe * 100.0, 2),
            max_adverse_excursion_pct=round(max_mae * 100.0, 2),
            planned_entry_price=planned_entry_price,
            entry_policy=entry_policy,
            signal_timestamp=signal_timestamp,
            entry_timestamp=entry_timestamp,
            reference_price_at_signal=reference_price_at_signal,
            planned_rr_ratio=planned_rr,
            realized_rr_ratio=realized_rr,
            trailing_stop_pct=trailing_stop_pct,
            break_even_trigger_pct=break_even_trigger_pct,
            max_holding_days=max_holding_days,
            mfe_efficiency=mfe_eff,
            trajectory=trajectory,
        )
