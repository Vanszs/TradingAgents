"""
Unit and regression tests for Kronos K-Line Foundation Model (shiyu-coder/Kronos, AAAI 2026).

Verifies:
1. Point-in-time causal isolation: future bars post-T0 do not leak into T0 prediction.
2. Causal anchor normalization and volume standardization math.
3. Determinism: T=0.0 produces identical deterministic forecast contracts.
4. Dual-tier caching (Memory LRU + Disk JSON).
5. LangGraph tool @tool get_kronos_forecast integration and schema parity.
"""
import glob
import os
import shutil
import tempfile

import pandas as pd
import pytest

from tradingagents.agents.utils.core_stock_tools import get_kronos_forecast
from tradingagents.dataflows.kronos import (
    KronosCacheManager,
    KronosForecastContract,
    KronosModelManager,
    _compute_causal_normalization,
    _generate_synthetic_forecast,
    generate_kronos_forecast,
)


def _make_sample_ohlcv():
    dates = pd.date_range(start="2025-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
    data = []
    price = 100.0
    for i, d in enumerate(dates):
        o = price
        h = price + 2.0
        low = price - 1.5
        c = price + 0.5
        v = 100000.0 + (i * 500)
        data.append({"date": d, "open": o, "high": h, "low": low, "close": c, "volume": v})
        price = c
    return pd.DataFrame(data)


@pytest.fixture
def sample_ohlcv_df():
    """Create a synthetic 100-bar daily price dataframe."""
    return _make_sample_ohlcv()


def test_causal_normalization_anchor_math(sample_ohlcv_df):
    """Ensure causal normalization divides strictly by Close_T0 and computes rolling volume z-scores."""
    df_norm, p0, v_mean, v_std = _compute_causal_normalization(sample_ohlcv_df, lookback_bars=50)
    last_close = sample_ohlcv_df["close"].iloc[-1]
    
    assert p0 == last_close
    assert df_norm["close"].iloc[-1] == 0.0  # (Close_T0 - P0) / P0 == 0.0
    assert len(df_norm) == 50
    assert df_norm["volume"].mean() == pytest.approx(0.0, abs=1e-5)


def test_point_in_time_causal_isolation(sample_ohlcv_df):
    """Verify that adding future bars post-T0 does not alter the forecast at T0."""
    t0_date = sample_ohlcv_df["date"].iloc[70]
    
    # Slice 1: Strictly up to T0
    df_t0 = sample_ohlcv_df[sample_ohlcv_df["date"] <= t0_date].copy()
    forecast_clean = generate_kronos_forecast(
        symbol="TEST.US",
        df_bars=df_t0,
        cutoff_date=t0_date,
        pred_days=20,
    )
    
    # Slice 2: Full dataframe containing future bars > T0
    forecast_with_future = generate_kronos_forecast(
        symbol="TEST.US",
        df_bars=sample_ohlcv_df,
        cutoff_date=t0_date,
        pred_days=20,
    )
    
    assert forecast_clean.last_close == forecast_with_future.last_close
    assert forecast_clean.forecast_end_close == forecast_with_future.forecast_end_close
    assert forecast_clean.forecast_return_pct == forecast_with_future.forecast_return_pct
    assert forecast_clean.directional_bias == forecast_with_future.directional_bias


def test_deterministic_forecast_reproducibility(sample_ohlcv_df):
    """Ensure greedy deterministic decoding returns identical results across repeated calls."""
    t0_date = sample_ohlcv_df["date"].iloc[-1]
    
    fc1 = generate_kronos_forecast(
        symbol="MSFT",
        df_bars=sample_ohlcv_df,
        cutoff_date=t0_date,
        pred_days=20,
    )
    fc2 = generate_kronos_forecast(
        symbol="MSFT",
        df_bars=sample_ohlcv_df,
        cutoff_date=t0_date,
        pred_days=20,
    )
    
    assert fc1.last_close == fc2.last_close
    assert fc1.forecast_high == fc2.forecast_high
    assert fc1.forecast_low == fc2.forecast_low
    assert fc1.directional_bias == fc2.directional_bias
    assert len(fc1.forecast_bars) == len(fc2.forecast_bars) == 20


def test_dual_tier_cache_isolation(sample_ohlcv_df):
    """Verify in-memory LRU and disk JSON cache persistence."""
    temp_dir = tempfile.mkdtemp()
    try:
        t0_date = sample_ohlcv_df["date"].iloc[-1]
        cfg = {"data_cache_dir": temp_dir}
        
        fc = generate_kronos_forecast(
            symbol="NVDA",
            df_bars=sample_ohlcv_df,
            cutoff_date=t0_date,
            pred_days=10,
            config=cfg,
        )
        
        # Verify disk cache JSON file was created with a snapshot-bound key.
        cache_files = glob.glob(os.path.join(temp_dir, "kronos", "kronos-*.json"))
        assert len(cache_files) == 1
        expected_json = cache_files[0]
        cache_key = os.path.splitext(os.path.basename(expected_json))[0]

        # Verify read from cache
        cached = KronosCacheManager.get(cache_key, expected_json)
        assert cached is not None
        assert cached["symbol"] == "NVDA"
        assert cached["forecast_end_close"] == fc.forecast_end_close
    finally:
        shutil.rmtree(temp_dir)


def test_langgraph_tool_invocation(sample_ohlcv_df, monkeypatch):
    """Verify LangGraph @tool get_kronos_forecast returns structured markdown."""
    csv_str = sample_ohlcv_df.to_csv(index=False)
    
    # Mock data router to return CSV
    def mock_route(method, *args, **kwargs):
        if method == "get_stock_data":
            return csv_str
        elif method == "get_kronos_forecast":
            return _generate_synthetic_forecast(
                symbol=args[0],
                df_clean=args[1],
                cutoff_date=args[2],
                pred_days=args[3],
                model_name="NeoQuasar/Kronos-base",
                device_used="cpu (Test)",
            )
        raise ValueError(f"Unknown method {method}")
    
    monkeypatch.setattr("tradingagents.agents.utils.core_stock_tools.route_to_vendor", mock_route)
    
    res = get_kronos_forecast.invoke({"symbol": "AAPL", "curr_date": "2025-05-01", "pred_days": 10})
    assert isinstance(res, str)
    assert "Kronos K-Line Forecast" in res
    assert "Directional Bias" in res
    assert "Forecast Resistance" in res


if __name__ == "__main__":
    df = _make_sample_ohlcv()
    test_causal_normalization_anchor_math(df)
    test_point_in_time_causal_isolation(df)
    test_deterministic_forecast_reproducibility(df)
    test_dual_tier_cache_isolation(df)
    print("All Kronos contract and regression tests passed 100%!")
