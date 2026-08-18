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
        # Drop incomplete rows (e.g. active intraday sessions where prices are NaN or 0)
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if len(price_cols) == 4:
            numeric_check = df[price_cols].apply(pd.to_numeric, errors="coerce")
            valid_mask = numeric_check.notna().all(axis=1) & (numeric_check > 0).all(axis=1)
            df = df[valid_mask].copy()

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

        # T1 Entry execution at Open
        entry_row = future_df.iloc[0]
        entry_date = entry_date or entry_row["date_str"]
        entry_price = actual_entry_price if actual_entry_price is not None else float(entry_row["open"])

        if entry_price <= 0:
            entry_price = float(entry_row["close"])

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

        for idx, row in horizon_bars.iterrows():
            bar_date = row["date_str"]
            b_open = float(row["open"])
            b_high = float(row["high"])
            b_low = float(row["low"])
            b_close = float(row["close"])

            # Excursion calculation relative to entry
            if is_long:
                mfe_bar = (b_high - entry_price) / entry_price
                mae_bar = (b_low - entry_price) / entry_price
                close_ret = (b_close - entry_price) / entry_price
            else:
                mfe_bar = (entry_price - b_low) / entry_price
                mae_bar = (entry_price - b_high) / entry_price
                close_ret = (entry_price - b_close) / entry_price

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
                    unrealized_mfe_pct=round(mfe_bar * 100.0, 2),
                    unrealized_mae_pct=round(mae_bar * 100.0, 2),
                )
            )

            # Barrier evaluation: check gap open precedence first
            if is_long:
                if take_profit is not None and b_open >= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = b_open
                    actual_days = idx + 1
                    break
                if stop_loss is not None and b_open <= stop_loss:
                    outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = b_open
                    actual_days = idx + 1
                    break
                if stop_loss is not None and b_low <= stop_loss:
                    outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = stop_loss
                    actual_days = idx + 1
                    break
                if take_profit is not None and b_high >= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = take_profit
                    actual_days = idx + 1
                    break
            else:  # SHORT
                if take_profit is not None and b_open <= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = b_open
                    actual_days = idx + 1
                    break
                if stop_loss is not None and b_open >= stop_loss:
                    outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = b_open
                    actual_days = idx + 1
                    break
                if stop_loss is not None and b_high >= stop_loss:
                    outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = stop_loss
                    actual_days = idx + 1
                    break
                if take_profit is not None and b_low <= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = take_profit
                    actual_days = idx + 1
                    break

        # Calculate Realized Return
        if outcome in (EvaluationOutcome.HIT_TAKE_PROFIT, EvaluationOutcome.HIT_STOP_LOSS, EvaluationOutcome.EXPIRED):
            if is_long:
                realized_return = (exit_price - entry_price) / entry_price
            else:
                realized_return = (entry_price - exit_price) / entry_price
            risk = abs(entry_price - stop_loss)
            realized_rr = round(realized_return / (risk / entry_price), 2) if risk > 0 else None
        else:
            realized_return = 0.0
            realized_rr = None
            actual_entry_price = None

        return EvaluationResult(
            ticker=ticker,
            signal_date=signal_date,
            entry_date=entry_date,
            actual_entry_price=round(entry_price, 2) if outcome in (EvaluationOutcome.HIT_TAKE_PROFIT, EvaluationOutcome.HIT_STOP_LOSS, EvaluationOutcome.EXPIRED) else None,
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
            trajectory=trajectory,
        )
