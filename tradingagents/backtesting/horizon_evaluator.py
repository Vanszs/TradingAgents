"""Point-in-time forward evaluator for BUY signals.

Only spot long entries are evaluated. Exits are static take-profit/stop-loss,
a configured time stop, or the end of the available horizon.
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
    HIT_TIME_STOP = "HIT_TIME_STOP"
    EXPIRED = "EXPIRED"
    NO_ORDER = "NO_ORDER"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_FILL = "NO_FILL"


@dataclass
class DailyExcursionBar:
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
    ticker: str
    signal_date: str
    entry_date: Optional[str]
    actual_entry_price: Optional[float]
    exit_date: Optional[str]
    exit_price: float
    outcome: EvaluationOutcome
    side: str
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
    max_holding_days: Optional[int] = None
    mfe_efficiency: Optional[float] = None
    trajectory: List[DailyExcursionBar] = field(default_factory=list)

    @property
    def entry_price(self) -> Optional[float]:
        return self.actual_entry_price

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outcome"] = self.outcome.value
        return data


def _no_order(
    ticker: str,
    signal_date: str,
    horizon: int,
    take_profit: Optional[float],
    stop_loss: Optional[float],
    planned_entry_price: Optional[float],
    entry_policy: str = "T1_OPEN",
) -> EvaluationResult:
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
        planned_time_horizon_days=horizon,
        actual_holding_days=0,
        realized_return_pct=0.0,
        max_favorable_excursion_pct=0.0,
        max_adverse_excursion_pct=0.0,
        planned_entry_price=planned_entry_price,
        entry_policy=entry_policy,
    )


class HorizonEvaluator:
    """Evaluate one causal BUY signal against future daily bars."""

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
        max_holding_days: Optional[int] = None,
    ) -> EvaluationResult:
        if not 1 <= time_horizon_days <= 252:
            raise ValueError("time_horizon_days must be between 1 and 252")

        signal_date = pd.to_datetime(signal_date, utc=True, errors="raise").strftime("%Y-%m-%d")
        side_norm = side.upper()
        if side_norm in {"FLAT", "WNS", "HOLD", "SELL"}:
            return _no_order(
                ticker, signal_date, time_horizon_days, take_profit, stop_loss,
                planned_entry_price, entry_policy,
            )
        if side_norm not in {"LONG", "BUY"}:
            raise ValueError("only LONG/BUY evaluation is supported")
        if take_profit is None or stop_loss is None:
            raise ValueError("BUY evaluation requires both take_profit and stop_loss")

        required = {"date", "open", "high", "low", "close"}
        missing = required.difference(ohlcv_df.columns)
        if missing:
            raise ValueError(f"ohlcv_df is missing required columns: {sorted(missing)}")

        fill_timestamp = None
        if entry_timestamp is not None:
            fill_timestamp = pd.Timestamp(entry_timestamp)
            if fill_timestamp.tzinfo is None or fill_timestamp.utcoffset() is None:
                raise ValueError("entry_timestamp must include a timezone")
            fill_timestamp = fill_timestamp.tz_convert("UTC")

        requested_entry_date = None
        if actual_entry_date is not None:
            requested_entry_date = pd.to_datetime(actual_entry_date, utc=True, errors="raise").strftime("%Y-%m-%d")
        if fill_timestamp is not None:
            timestamp_date = fill_timestamp.strftime("%Y-%m-%d")
            if requested_entry_date is not None and requested_entry_date != timestamp_date:
                raise ValueError("entry_timestamp date does not match actual_entry_date")
            requested_entry_date = timestamp_date

        assumed_entry = entry_policy == "ASSUMED_AI_ENTRY"
        if assumed_entry and requested_entry_date not in (None, signal_date):
            raise ValueError("ASSUMED_AI_ENTRY entry date must match signal_date")

        frame = ohlcv_df.copy()
        parsed = pd.to_datetime(frame["date"], utc=True, errors="raise")
        if parsed.duplicated().any():
            raise ValueError("ohlcv_df contains duplicate dates")
        frame["date_str"] = parsed.dt.strftime("%Y-%m-%d")
        numeric = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="raise")
        if not numeric.map(math.isfinite).all().all() or (numeric <= 0).any().any():
            raise ValueError("ohlcv_df prices must be finite positive numbers")
        frame[["open", "high", "low", "close"]] = numeric
        future = frame[frame["date_str"] > signal_date].sort_values("date_str").reset_index(drop=True)
        if requested_entry_date is not None and not assumed_entry:
            if requested_entry_date not in set(future["date_str"]):
                raise ValueError("actual entry date is not present in future daily data")
            future = future[future["date_str"] >= requested_entry_date].reset_index(drop=True)
        if future.empty:
            return EvaluationResult(
                ticker=ticker, signal_date=signal_date, entry_date=None,
                actual_entry_price=None, exit_date=None, exit_price=0.0,
                outcome=EvaluationOutcome.INSUFFICIENT_DATA, side="LONG",
                take_profit=take_profit, stop_loss=stop_loss,
                planned_time_horizon_days=time_horizon_days, actual_holding_days=0,
                realized_return_pct=0.0, max_favorable_excursion_pct=0.0,
                max_adverse_excursion_pct=0.0, planned_entry_price=planned_entry_price,
                entry_policy=entry_policy,
            )

        first = future.iloc[0]
        entry_date = requested_entry_date or first["date_str"]
        first_open = float(first["open"])
        first_high = float(first["high"])
        first_low = float(first["low"])
        if actual_entry_price is None:
            entry_price = first_open
        else:
            requested_price = float(actual_entry_price)
            if first_low <= requested_price <= first_high:
                entry_price = requested_price
            elif first_open < requested_price:
                entry_price = first_open
            else:
                return EvaluationResult(
                    ticker=ticker, signal_date=signal_date, entry_date=None,
                    actual_entry_price=None, exit_date=None, exit_price=0.0,
                    outcome=EvaluationOutcome.NO_FILL, side="LONG",
                    take_profit=take_profit, stop_loss=stop_loss,
                    planned_time_horizon_days=time_horizon_days, actual_holding_days=0,
                    realized_return_pct=0.0, max_favorable_excursion_pct=0.0,
                    max_adverse_excursion_pct=0.0, planned_entry_price=planned_entry_price,
                    entry_policy=entry_policy,
                )
        if stop_loss >= take_profit:
            raise ValueError("BUY requires stop_loss < entry_price < take_profit")
        gap_open_exit = (
            actual_entry_price is None
            and (first_open <= stop_loss or first_open >= take_profit)
        )
        if entry_price <= 0 or (not gap_open_exit and not stop_loss < entry_price < take_profit):
            raise ValueError("BUY requires stop_loss < entry_price < take_profit")

        horizon = future.iloc[:time_horizon_days].reset_index(drop=True)
        full_horizon = len(future) >= time_horizon_days
        if horizon.empty:
            raise ValueError("future OHLCV window is empty")
        effective_time_stop = max_holding_days
        if effective_time_stop is not None and not 1 <= effective_time_stop <= 252:
            raise ValueError("max_holding_days must be between 1 and 252")

        planned_rr = round((take_profit - entry_price) / (entry_price - stop_loss), 2)
        trajectory: list[DailyExcursionBar] = []
        max_mfe = 0.0
        max_mae = 0.0
        outcome = EvaluationOutcome.EXPIRED if full_horizon else EvaluationOutcome.INSUFFICIENT_DATA
        exit_date = horizon.iloc[-1]["date_str"] if full_horizon else None
        exit_price = float(horizon.iloc[-1]["close"]) if full_horizon else 0.0
        actual_days = len(horizon)

        for idx, row in horizon.iterrows():
            bar_date = row["date_str"]
            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])
            limit_fill_below_open = idx == 0 and actual_entry_price is not None and entry_price < bar_open

            if not limit_fill_below_open and bar_open <= stop_loss:
                outcome, exit_date, exit_price, actual_days = EvaluationOutcome.HIT_STOP_LOSS, bar_date, bar_open, idx + 1
            elif not limit_fill_below_open and bar_open >= take_profit:
                outcome, exit_date, exit_price, actual_days = EvaluationOutcome.HIT_TAKE_PROFIT, bar_date, bar_open, idx + 1
            elif bar_low <= stop_loss:
                outcome, exit_date, exit_price, actual_days = EvaluationOutcome.HIT_STOP_LOSS, bar_date, stop_loss, idx + 1
            elif bar_high >= take_profit:
                outcome, exit_date, exit_price, actual_days = EvaluationOutcome.HIT_TAKE_PROFIT, bar_date, take_profit, idx + 1
            elif effective_time_stop is not None and idx + 1 >= effective_time_stop:
                outcome, exit_date, exit_price, actual_days = EvaluationOutcome.HIT_TIME_STOP, bar_date, bar_close, idx + 1
            else:
                exit_ret = (bar_close - entry_price) / entry_price
                max_mfe = max(max_mfe, (bar_high - entry_price) / entry_price)
                max_mae = min(max_mae, (bar_low - entry_price) / entry_price)
                trajectory.append(DailyExcursionBar(
                    bar_index=idx + 1, date=bar_date, open=bar_open, high=bar_high,
                    low=bar_low, close=bar_close,
                    unrealized_return_close_pct=round(exit_ret * 100, 2),
                    unrealized_mfe_pct=round(max_mfe * 100, 2),
                    unrealized_mae_pct=round(max_mae * 100, 2),
                ))
                continue

            max_mfe = max(max_mfe, (bar_high - entry_price) / entry_price)
            max_mae = min(max_mae, (bar_low - entry_price) / entry_price)
            close_ret = (exit_price - entry_price) / entry_price
            trajectory.append(DailyExcursionBar(
                bar_index=idx + 1, date=bar_date, open=bar_open, high=bar_high,
                low=bar_low, close=exit_price,
                unrealized_return_close_pct=round(close_ret * 100, 2),
                unrealized_mfe_pct=round(max_mfe * 100, 2),
                unrealized_mae_pct=round(max_mae * 100, 2),
            ))
            break

        terminal = outcome in {
            EvaluationOutcome.HIT_TAKE_PROFIT,
            EvaluationOutcome.HIT_STOP_LOSS,
            EvaluationOutcome.HIT_TIME_STOP,
            EvaluationOutcome.EXPIRED,
        }
        realized = (exit_price - entry_price) / entry_price if terminal else 0.0
        realized_rr = round(realized / ((entry_price - stop_loss) / entry_price), 2) if terminal else None
        mfe_efficiency = round(realized / max_mfe, 2) if terminal and max_mfe > 0 else 0.0 if terminal else None
        return EvaluationResult(
            ticker=ticker, signal_date=signal_date, entry_date=entry_date,
            actual_entry_price=round(entry_price, 2) if terminal else None,
            exit_date=exit_date, exit_price=round(exit_price, 2), outcome=outcome,
            side="LONG", take_profit=take_profit, stop_loss=stop_loss,
            planned_time_horizon_days=time_horizon_days, actual_holding_days=actual_days,
            realized_return_pct=round(realized * 100, 2),
            max_favorable_excursion_pct=round(max_mfe * 100, 2),
            max_adverse_excursion_pct=round(max_mae * 100, 2),
            planned_entry_price=planned_entry_price, entry_policy=entry_policy,
            signal_timestamp=signal_timestamp, entry_timestamp=entry_timestamp,
            reference_price_at_signal=reference_price_at_signal,
            planned_rr_ratio=planned_rr, realized_rr_ratio=realized_rr,
            max_holding_days=max_holding_days, mfe_efficiency=mfe_efficiency,
            trajectory=trajectory,
        )
