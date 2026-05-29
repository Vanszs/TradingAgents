"""Regression tests: the market analyst data path (OHLCV + technical
indicators) must return valid data for crypto ``-USD`` tickers.

Crypto market analysis is not a separate code path — it relies on yfinance
natively supporting ``BTC-USD``-style tickers and on the default vendor
being yfinance. These tests lock that contract so a future change to ticker
normalization or vendor defaults can't silently break crypto.

All network access is mocked at the yfinance boundary; Alpha Vantage is
never touched.
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest

import tradingagents.dataflows.config as config_mod
import tradingagents.dataflows.y_finance as yf_mod
import tradingagents.default_config as default_config
from tradingagents.dataflows.interface import route_to_vendor


@pytest.fixture(autouse=True)
def _force_yfinance_vendor():
    """Pin the stock/indicator vendor to yfinance for these tests.

    Other test modules mutate the global config — including method-level
    ``tool_vendors`` (checked before category vendors) — and don't reset it.
    ``set_config`` merges dicts one level deep and never removes keys, so we
    swap in a clean default config directly and restore the original after.
    """
    original = config_mod._config
    config_mod._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_mod._config = original


@pytest.fixture
def _ohlcv_df():
    """8 days of synthetic BTC-USD OHLCV with a tz-aware index (24/7, incl. weekend)."""
    idx = pd.date_range("2026-05-20", periods=8, freq="D", tz="UTC")
    base = [77457, 77539, 75488, 76900, 78100, 77200, 76500, 77000]
    return pd.DataFrame(
        {
            "Open": base,
            "High": [p + 600 for p in base],
            "Low": [p - 700 for p in base],
            "Close": [p + 100 for p in base],
            "Volume": [26_000_000_000 + i for i in range(8)],
            "Dividends": [0.0] * 8,
            "Stock Splits": [0.0] * 8,
        },
        index=idx,
    )


@pytest.fixture
def _indicator_df():
    """50 days of synthetic OHLCV (Date column, as load_ohlcv returns) so RSI is finite."""
    dates = pd.date_range("2026-04-09", periods=50, freq="D")
    # Oscillating price so RSI lands in a normal mid-range, not NaN.
    close = [70000 + (i % 7) * 300 - (i % 3) * 250 for i in range(50)]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": [c + 400 for c in close],
            "Low": [c - 400 for c in close],
            "Close": close,
            "Volume": [25_000_000_000 + i for i in range(50)],
        }
    )


@pytest.mark.unit
def test_crypto_ohlcv_returns_valid_data(monkeypatch, _ohlcv_df):
    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end):
            return _ohlcv_df

    monkeypatch.setattr(yf_mod.yf, "Ticker", _FakeTicker)

    out = route_to_vendor("get_stock_data", "BTC-USD", "2026-05-20", "2026-05-28")

    assert "BTC-USD" in out
    assert "No data found" not in out
    assert "77557" in out  # first Close (77457 + 100), confirms real rows rendered


@pytest.mark.unit
def test_crypto_indicator_returns_valid_values(monkeypatch, _indicator_df):
    monkeypatch.setattr(yf_mod, "load_ohlcv", lambda symbol, curr_date: _indicator_df.copy())

    out = route_to_vendor("get_indicators", "BTC-USD", "rsi", "2026-05-28", 5)

    assert "rsi values" in out
    assert "2026-05-28:" in out
    # An actual numeric RSI (not "N/A"/empty) must be present for the curr date.
    line = next(ln for ln in out.splitlines() if ln.startswith("2026-05-28:"))
    value = line.split(":", 1)[1].strip()
    assert value not in ("", "N/A")
    assert 0.0 <= float(value) <= 100.0
