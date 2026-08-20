"""
Unit tests for Dynamic Exit Engine (Break-Even, Trailing Stop, Gap-Open Priority, Limit Touch, Time Stop).
"""
import pandas as pd
import pytest

from tradingagents.backtesting.horizon_evaluator import (
    EvaluationOutcome,
    HorizonEvaluator,
)


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        "open": [100.0, 100.0, 105.0, 101.0, 95.0],
        "high": [102.0, 106.0, 108.0, 103.0, 98.0],
        "low": [99.0, 99.0, 100.0, 96.0, 92.0],
        "close": [101.0, 105.0, 102.0, 98.0, 93.0],
    })


def test_break_even_ratchet_protects_profit(base_df):
    """Stock rises +8% (trigger at +3%), then drops below entry. Must exit at break-even (~+0.1%)."""
    res = HorizonEvaluator.evaluate(
        ticker="TEST",
        signal_date="2026-01-01",
        side="LONG",
        take_profit=120.0,
        stop_loss=90.0,
        time_horizon_days=10,
        ohlcv_df=base_df,
        planned_entry_price=100.0,
        actual_entry_price=100.0,
        break_even_trigger_pct=0.03,
    )
    assert res.outcome == EvaluationOutcome.HIT_BREAK_EVEN
    assert res.exit_price >= 100.0
    assert res.realized_return_pct >= 0.0


def test_trailing_stop_ratchet_avgo_scenario():
    """Simulate AVGO runup to +9.27% then reversal. 5% trailing stop should lock in profit."""
    df = pd.DataFrame({
        "date": [
            "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-07", "2025-02-08",
            "2025-02-11", "2025-02-12", "2025-02-13", "2025-02-14", "2025-02-15"
        ],
        "open": [100.0, 100.0, 106.0, 108.0, 109.0, 105.0, 102.0, 98.0, 92.0, 88.0],
        "high": [102.0, 105.0, 108.0, 109.27, 109.27, 106.0, 103.0, 99.0, 93.0, 89.0],
        "low": [99.0, 99.0, 105.0, 107.0, 104.0, 101.0, 98.0, 91.0, 87.0, 85.0],
        "close": [101.0, 104.0, 107.0, 109.0, 105.0, 102.0, 98.0, 92.0, 88.0, 86.0],
    })
    res = HorizonEvaluator.evaluate(
        ticker="AVGO",
        signal_date="2025-02-04",
        side="LONG",
        take_profit=130.0,
        stop_loss=88.0,
        time_horizon_days=20,
        ohlcv_df=df,
        planned_entry_price=100.0,
        actual_entry_price=100.0,
        trailing_stop_pct=0.05,
    )
    assert res.outcome == EvaluationOutcome.HIT_TRAILING_STOP
    # Peak was 109.27, 5% trail is 109.27 * 0.95 = 103.80 (+3.8%)
    assert res.realized_return_pct > 3.0


def test_gap_open_priority_take_profit():
    """Bar opens above TP and low touches SL on same bar. Must execute HIT_TAKE_PROFIT at Open."""
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "open": [100.0, 115.0],  # Gap open over TP 110
        "high": [102.0, 116.0],
        "low": [99.0, 94.0],    # Low touches SL 95
        "close": [101.0, 105.0],
    })
    res = HorizonEvaluator.evaluate(
        ticker="TEST",
        signal_date="2026-01-01",
        side="LONG",
        take_profit=110.0,
        stop_loss=95.0,
        time_horizon_days=10,
        ohlcv_df=df,
    )
    assert res.outcome == EvaluationOutcome.HIT_TAKE_PROFIT
    assert res.exit_price == 115.0


def test_limit_fill_touch_validation():
    """Limit buy is 90.0, but entry bar range is 100-105. Must not fill (NO_FILL)."""
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "open": [100.0, 102.0],
        "high": [102.0, 105.0],
        "low": [99.0, 100.0],
        "close": [101.0, 104.0],
    })
    res = HorizonEvaluator.evaluate(
        ticker="TEST",
        signal_date="2026-01-01",
        side="LONG",
        take_profit=120.0,
        stop_loss=80.0,
        time_horizon_days=10,
        ohlcv_df=df,
        planned_entry_price=90.0,
        actual_entry_price=90.0,
    )
    assert res.outcome == EvaluationOutcome.NO_FILL
    assert res.actual_entry_price is None


def test_time_stop_exit():
    """Price hovers within SL/TP for max_holding_days. Must exit HIT_TIME_STOP."""
    df = pd.DataFrame({
        "date": [f"2026-01-{i:02d}" for i in range(1, 10)],
        "open": [100.0] * 9,
        "high": [102.0] * 9,
        "low": [98.0] * 9,
        "close": [101.0] * 9,
    })
    res = HorizonEvaluator.evaluate(
        ticker="TEST",
        signal_date="2026-01-01",
        side="LONG",
        take_profit=130.0,
        stop_loss=80.0,
        time_horizon_days=20,
        max_holding_days=5,
        ohlcv_df=df,
    )
    assert res.outcome == EvaluationOutcome.HIT_TIME_STOP
    assert res.actual_holding_days == 5
