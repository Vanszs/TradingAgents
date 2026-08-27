"""Unit tests for the Forward Horizon Evaluator."""
from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.backtesting.horizon_evaluator import (
    EvaluationOutcome,
    HorizonEvaluator,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    data = [
        {"date": "2025-01-02", "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 1000},
        {"date": "2025-01-03", "open": 102.0, "high": 108.0, "low": 101.0, "close": 106.0, "volume": 1200},
        {"date": "2025-01-06", "open": 106.0, "high": 115.0, "low": 105.0, "close": 112.0, "volume": 1500},
        {"date": "2025-01-07", "open": 112.0, "high": 114.0, "low": 94.0, "close": 96.0, "volume": 1800},
        {"date": "2025-01-08", "open": 96.0, "high": 99.0, "low": 92.0, "close": 95.0, "volume": 1100},
    ]
    return pd.DataFrame(data)


def test_planned_entry_is_distinct_from_actual_t1_open(sample_ohlcv):
    result = HorizonEvaluator.evaluate(
        ticker="TEST.JK",
        signal_date="2025-01-01",
        side="LONG",
        take_profit=110.0,
        stop_loss=95.0,
        time_horizon_days=5,
        ohlcv_df=sample_ohlcv,
        planned_entry_price=98.0,
    )

    assert result.actual_entry_price == 100.0
    assert result.planned_entry_price == 98.0
    assert result.entry_policy == "T1_OPEN"
    serialized = result.to_dict()
    assert serialized["actual_entry_price"] == 100.0
    assert serialized["planned_entry_price"] == 98.0


def test_assumed_ai_entry_uses_planned_price_without_market_fill(sample_ohlcv):
    result = HorizonEvaluator.evaluate(
        ticker="TEST.JK",
        signal_date="2025-01-01",
        side="LONG",
        take_profit=110.0,
        stop_loss=95.0,
        time_horizon_days=5,
        ohlcv_df=sample_ohlcv,
        planned_entry_price=98.0,
        actual_entry_price=98.0,
        actual_entry_date="2025-01-01",
        entry_timestamp="2025-01-01T23:59:59+00:00",
        entry_policy="ASSUMED_AI_ENTRY",
    )

    assert result.actual_entry_price == 98.0
    assert result.planned_entry_price == 98.0
    assert result.entry_date == "2025-01-01"
    assert result.entry_policy == "ASSUMED_AI_ENTRY"
    assert result.outcome == EvaluationOutcome.HIT_TAKE_PROFIT


def test_long_hit_take_profit(sample_ohlcv):
    # Signal at 2025-01-01 -> Entry at 2025-01-02 Open (100.0)
    # TP: 110.0, SL: 95.0 -> Touches 115.0 on 2025-01-06 (bar 3)
    res = HorizonEvaluator.evaluate(
        ticker="TEST.JK",
        signal_date="2025-01-01",
        side="LONG",
        take_profit=110.0,
        stop_loss=95.0,
        time_horizon_days=5,
        ohlcv_df=sample_ohlcv,
    )
    assert res.outcome == EvaluationOutcome.HIT_TAKE_PROFIT
    assert res.entry_price == 100.0
    assert res.exit_price == 110.0
    assert res.exit_date == "2025-01-06"
    assert res.realized_return_pct == 10.0
    assert res.actual_holding_days == 3


def test_long_hit_stop_loss(sample_ohlcv):
    # Signal at 2025-01-01 -> Entry 100.0
    # TP: 120.0, SL: 97.0 -> Touches low 98.0 on day 1 (no hit), low 94.0 on day 4 (hit SL 97)
    res = HorizonEvaluator.evaluate(
        ticker="TEST.JK",
        signal_date="2025-01-01",
        side="LONG",
        take_profit=120.0,
        stop_loss=97.0,
        time_horizon_days=5,
        ohlcv_df=sample_ohlcv,
    )
    assert res.outcome == EvaluationOutcome.HIT_STOP_LOSS
    assert res.entry_price == 100.0
    assert res.exit_price == 97.0
    assert res.exit_date == "2025-01-07"
    assert res.realized_return_pct == -3.0
    assert res.actual_holding_days == 4


def test_conservative_same_bar_conflict(sample_ohlcv):
    # On 2025-01-07, High is 114, Low is 94.
    # If TP is 113 and SL is 95, both touched on same day -> Stop loss wins
    res = HorizonEvaluator.evaluate(
        ticker="TEST.JK",
        signal_date="2025-01-01",
        side="LONG",
        take_profit=113.0,
        stop_loss=95.0,
        time_horizon_days=5,
        ohlcv_df=sample_ohlcv,
    )
    # Note: On 2025-01-06 High was 115 >= 113, so it hit TP on 2025-01-06 first!
    assert res.outcome == EvaluationOutcome.HIT_TAKE_PROFIT
    assert res.exit_date == "2025-01-06"


def test_expired_without_barrier(sample_ohlcv):
    # TP 130, SL 80 -> Never touched in 5 days -> Exit at last close (95.0)
    res = HorizonEvaluator.evaluate(
        ticker="TEST.JK",
        signal_date="2025-01-01",
        side="LONG",
        take_profit=130.0,
        stop_loss=80.0,
        time_horizon_days=5,
        ohlcv_df=sample_ohlcv,
    )
    assert res.outcome == EvaluationOutcome.EXPIRED
    assert res.exit_date == "2025-01-08"
    assert res.exit_price == 95.0
    assert res.realized_return_pct == -5.0
    assert res.actual_holding_days == 5


def test_short_input_is_rejected(sample_ohlcv):
    with pytest.raises(ValueError, match="only LONG/BUY"):
        HorizonEvaluator.evaluate(
            ticker="TEST.JK", signal_date="2025-01-01", side="SHORT",
            take_profit=95.0, stop_loss=110.0, time_horizon_days=5,
            ohlcv_df=sample_ohlcv,
        )

def test_rejects_naive_fill_timestamp(sample_ohlcv):
    with pytest.raises(ValueError, match="entry_timestamp.*timezone"):
        HorizonEvaluator.evaluate(
            "TEST", "2025-01-01", "LONG", 110, 95, 1, sample_ohlcv,
            actual_entry_price=102.0,
            actual_entry_date="2025-01-02",
            entry_timestamp="2025-01-02T13:35:00",
        )


def test_rejects_inconsistent_fill_dates(sample_ohlcv):
    with pytest.raises(ValueError, match="does not match actual_entry_date"):
        HorizonEvaluator.evaluate(
            "TEST", "2025-01-01", "LONG", 110, 95, 1, sample_ohlcv,
            actual_entry_price=102.0,
            actual_entry_date="2025-01-03",
            entry_timestamp="2025-01-02T13:35:00+00:00",
        )


def test_no_order_and_insufficient_data_have_no_actual_entry_price(sample_ohlcv):
    no_order = HorizonEvaluator.evaluate("TEST", "2025-01-01", "WNS", None, None, 1, sample_ohlcv)
    insufficient = HorizonEvaluator.evaluate(
        "TEST", "2025-01-01", "LONG", 130, 80, 10, sample_ohlcv,
        actual_entry_price=100.0,
        actual_entry_date="2025-01-02",
    )

    assert no_order.actual_entry_price is None
    assert insufficient.actual_entry_price is None


def test_rejects_invalid_horizon(sample_ohlcv):
    with pytest.raises(ValueError, match="between 1 and 252"):
        HorizonEvaluator.evaluate("TEST", "2025-01-01", "LONG", 110, 95, 0, sample_ohlcv)


def test_rejects_missing_columns(sample_ohlcv):
    with pytest.raises(ValueError, match="missing required columns"):
        HorizonEvaluator.evaluate(
            "TEST", "2025-01-01", "LONG", 110, 95, 5, sample_ohlcv.drop(columns=["high"])
        )


def test_rejects_duplicate_dates(sample_ohlcv):
    duplicate = pd.concat([sample_ohlcv, sample_ohlcv.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate dates"):
        HorizonEvaluator.evaluate("TEST", "2025-01-01", "LONG", 110, 95, 5, duplicate)


def test_rejects_invalid_barriers(sample_ohlcv):
    with pytest.raises(ValueError, match="stop_loss < entry_price < take_profit"):
        HorizonEvaluator.evaluate("TEST", "2025-01-01", "LONG", 90, 110, 5, sample_ohlcv)


def test_incomplete_horizon_still_triggers_early_take_profit(sample_ohlcv):
    # Horizon is 10 days, but we only provide 5 days of data.
    # TP 110 is touched on Day 3 (2025-01-06 High 115) -> Should resolve to HIT_TAKE_PROFIT, not INSUFFICIENT_DATA
    result = HorizonEvaluator.evaluate("TEST.JK", "2025-01-01", "LONG", 110, 95, 10, sample_ohlcv)
    assert result.outcome == EvaluationOutcome.HIT_TAKE_PROFIT
    assert result.exit_date == "2025-01-06"
    assert result.exit_price == 110.0


def test_rejects_incomplete_horizon_when_no_barrier_hit(sample_ohlcv):
    result = HorizonEvaluator.evaluate("TEST", "2025-01-01", "LONG", 130, 80, 10, sample_ohlcv)
    assert result.outcome == EvaluationOutcome.INSUFFICIENT_DATA
    assert result.exit_date is None


def test_realized_rr_is_signed_on_loss(sample_ohlcv):
    result = HorizonEvaluator.evaluate("TEST", "2025-01-01", "LONG", 130, 80, 5, sample_ohlcv)
    assert result.realized_rr_ratio < 0
