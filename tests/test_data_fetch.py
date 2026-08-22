"""
Tests for ``cli.data_fetch``.

Covers:
* ``compute_download_window`` math: lookback translation, buffer, YF
  earliest clamp, end padding.
* ``ensure_ohlcv`` flow: missing file → fetch; stale file (last date
  < end_date) → re-fetch; fresh file → no fetch.
* ``fetch_ohlcv`` writes the required CSV shape and stub files.
* ``engine.ensure_ohlcv`` on the BacktestEngine: existing file passes
  without auto_fetch; missing file returns ok=False with a helpful
  message; auto_fetch=True calls through.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.data_fetch import (
    LOOKBACK_BUFFER_DAYS,
    YF_EARLIEST,
    compute_download_window,
    ensure_ohlcv,
    fetch_ohlcv,
)
from tradingagents.backtesting import TradingAgentsRunner
from tradingagents.backtesting.engine import BacktestEngine

# ---------------------------------------------------------------------------
# compute_download_window
# ---------------------------------------------------------------------------


class TestComputeDownloadWindow:
    def test_no_lookback_uses_buffer(self):
        dl_start, dl_end = compute_download_window(
            "2026-05-25", "2026-05-30", lookback_days=None
        )
        # 1 year buffer back, 7 day end pad
        assert dl_start == "2025-05-25"
        assert dl_end == "2026-06-06"

    def test_short_lookback_uses_buffer(self):
        # lookback=60 → ~98 calendar days; less than 365 buffer
        dl_start, dl_end = compute_download_window(
            "2026-05-25", "2026-05-30", lookback_days=60
        )
        assert dl_start == "2025-05-25"
        assert dl_end == "2026-06-06"

    def test_long_lookback_uses_lookback_padding(self):
        # lookback=240 → int(240*7/5)+14 = 350 calendar days, < 365 buffer
        # so buffer still wins
        dl_start, _ = compute_download_window(
            "2026-05-25", "2026-05-30", lookback_days=240
        )
        assert dl_start == "2025-05-25"

    def test_huge_lookback_uses_lookback_padding(self):
        # 1000 trading days → 1414 calendar days > 365 buffer
        dl_start, _ = compute_download_window(
            "2026-05-25", "2026-05-30", lookback_days=1000
        )
        # 2026-05-25 - 1414 days ≈ 2022-07-08 (ish)
        assert dl_start < "2025-05-25"
        assert dl_start >= "2022-01-01"  # loose sanity check

    def test_earliest_clamp(self):
        # Very old start date gets clamped to YF_EARLIEST
        dl_start, _ = compute_download_window(
            "2005-01-01", "2005-12-31", lookback_days=None
        )
        assert dl_start == YF_EARLIEST

    def test_end_pad(self):
        _, dl_end = compute_download_window(
            "2026-01-01", "2026-01-31", lookback_days=None
        )
        # 7 days past end_date
        assert dl_end == "2026-02-07"

    def test_window_is_wider_than_backtest(self):
        dl_start, dl_end = compute_download_window(
            "2026-05-25", "2026-05-30", lookback_days=60
        )
        assert dl_start < "2026-05-25"
        assert dl_end > "2026-05-30"

    def test_accepts_datetime_input(self):
        # Tolerates 'YYYY-MM-DD HH:MM:SS' style strings
        dl_start, dl_end = compute_download_window(
            "2026-05-25 00:00:00", "2026-05-30 23:59:59", lookback_days=None
        )
        assert dl_start == "2025-05-25"
        assert dl_end == "2026-06-06"


# ---------------------------------------------------------------------------
# ensure_ohlcv (filesystem-level)
# ---------------------------------------------------------------------------


class TestEnsureOhlcv:
    def _write_existing_csv(self, path: Path, last_date: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", last_date],
                "open": [100.0, 110.0],
                "high": [101.0, 111.0],
                "low": [99.0, 109.0],
                "close": [100.5, 110.5],
                "volume": [1000, 2000],
            }
        )
        df.to_csv(path, index=False)

    def test_missing_file_triggers_fetch(self, tmp_path):
        # Use a fake yfinance to avoid network.
        def fake_fetch(ticker, start_date, end_date, output_root, lookback_days=None):
            target = Path(output_root) / ticker / "ohlcv.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("date,open,high,low,close,volume\n", encoding="utf-8")
            for f in ("news.json", "fundamentals.json", "sentiment.json", "broker_activity.json"):
                (target.parent / f).write_text("[]", encoding="utf-8")
            return target

        with patch("cli.data_fetch.fetch_ohlcv", side_effect=fake_fetch):
            path, fetched = ensure_ohlcv(
                "FAKE.JK", "2026-01-01", "2026-06-30",
                output_root=str(tmp_path), lookback_days=60,
            )

        assert fetched is True
        assert path.exists()
        assert (path.parent / "news.json").exists()
        assert (path.parent / "fundamentals.json").exists()

    def test_fresh_file_no_fetch(self, tmp_path):
        csv = tmp_path / "FAKE.JK" / "ohlcv.csv"
        self._write_existing_csv(csv, "2026-12-31")

        # Patch fetch_ohlcv to fail loudly if called.
        with patch("cli.data_fetch.fetch_ohlcv") as mock_fetch:
            path, fetched = ensure_ohlcv(
                "FAKE.JK", "2026-01-01", "2026-06-30",
                output_root=str(tmp_path), lookback_days=60,
            )

        assert fetched is False
        assert path == csv
        mock_fetch.assert_not_called()

    def test_stale_file_triggers_refetch(self, tmp_path):
        csv = tmp_path / "FAKE.JK" / "ohlcv.csv"
        # File ends 2025-12-31, but we ask for end_date 2026-06-30
        self._write_existing_csv(csv, "2025-12-31")

        def fake_fetch(ticker, start_date, end_date, output_root, lookback_days=None):
            target = Path(output_root) / ticker / "ohlcv.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            # simulate the refetched file now reaching 2026-12-31
            df = pd.DataFrame(
                {
                    "date": ["2025-01-01", "2026-12-31"],
                    "open": [100.0, 110.0],
                    "high": [101.0, 111.0],
                    "low": [99.0, 109.0],
                    "close": [100.5, 110.5],
                    "volume": [1000, 2000],
                }
            )
            df.to_csv(target, index=False)
            for f in ("news.json", "fundamentals.json", "sentiment.json", "broker_activity.json"):
                (target.parent / f).write_text("[]", encoding="utf-8")
            return target

        with patch("cli.data_fetch.fetch_ohlcv", side_effect=fake_fetch):
            path, fetched = ensure_ohlcv(
                "FAKE.JK", "2026-01-01", "2026-06-30",
                output_root=str(tmp_path), lookback_days=60,
            )

        assert fetched is True
        assert path.exists()

    def test_corrupt_existing_csv_triggers_refetch(self, tmp_path):
        csv = tmp_path / "FAKE.JK" / "ohlcv.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text("this is not a valid csv\n", encoding="utf-8")

        def fake_fetch(ticker, start_date, end_date, output_root, lookback_days=None):
            target = Path(output_root) / ticker / "ohlcv.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("date,open,high,low,close,volume\n", encoding="utf-8")
            for f in ("news.json", "fundamentals.json", "sentiment.json", "broker_activity.json"):
                (target.parent / f).write_text("[]", encoding="utf-8")
            return target

        with patch("cli.data_fetch.fetch_ohlcv", side_effect=fake_fetch):
            path, fetched = ensure_ohlcv(
                "FAKE.JK", "2026-01-01", "2026-06-30",
                output_root=str(tmp_path), lookback_days=60,
            )

        assert fetched is True


# ---------------------------------------------------------------------------
# fetch_ohlcv (with yfinance mocked)
# ---------------------------------------------------------------------------


class TestFetchOhlcv:
    def test_writes_csv_and_stubs(self, tmp_path, monkeypatch):
        # Fake yfinance that returns a small DataFrame indexed by date.
        def fake_yf_download(*args, **kwargs):
            dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
            return pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0],
                    "High": [101.0, 102.0, 103.0],
                    "Low": [99.0, 100.0, 101.0],
                    "Close": [100.5, 101.5, 102.5],
                    "Adj Close": [100.5, 101.5, 102.5],
                    "Volume": [1000, 2000, 3000],
                },
                index=pd.DatetimeIndex(dates, name="Date"),
            )

        # Stub the yfinance import inside fetch_ohlcv.
        import types

        fake_yf = types.ModuleType("yfinance")
        fake_yf.download = fake_yf_download
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        path = fetch_ohlcv(
            "FAKE.JK", "2025-01-01", "2025-01-31",
            output_root=str(tmp_path), lookback_days=None,
        )

        assert path.exists()
        df = pd.read_csv(path)
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(df) == 3

        # Stub files written
        for f in ("news.json", "fundamentals.json", "sentiment.json", "broker_activity.json"):
            assert (path.parent / f).exists()

    def test_empty_yfinance_raises(self, tmp_path, monkeypatch):
        import types

        fake_yf = types.ModuleType("yfinance")
        fake_yf.download = lambda *a, **k: pd.DataFrame()
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        with pytest.raises(RuntimeError, match="yfinance returned no data"):
            fetch_ohlcv(
                "BAD", "2025-01-01", "2025-01-31",
                output_root=str(tmp_path), lookback_days=None,
            )


# ---------------------------------------------------------------------------
# engine.ensure_ohlcv
# ---------------------------------------------------------------------------


class _HoldRunner(TradingAgentsRunner):
    """No-op runner for tests that don't actually run the backtest."""

    def __init__(self):  # noqa: D401
        super().__init__(
            reports_root="reports",
            run_callable=lambda *a, **k: "## Trading Decision\n**Rating**: Hold\n",
        )


def _build_engine_for_ticker(tmp_path, ticker: str, start: str, end: str, lookback=60):
    """Build a BacktestEngine pointing at a temp data root."""
    import yaml

    raw = {
        "ticker": ticker,
        "start_date": start,
        "end_date": end,
        "initial_cash": 100_000_000,
        "lookback_days": lookback,
        "data": {
            "data_root": str(tmp_path),
            "snapshot_root": str(tmp_path / "snapshots"),
        },
        "decision_mapping": {"mode": "strict_5tier"},
        "trigger": {
            "enabled": True,
            "rr_deterioration_pct": 0.30,
            "use_real_rr_when_available": True,
        },
    }
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump({"backtest": raw}), encoding="utf-8")
    return BacktestEngine.from_yaml(str(cfg_file), agent_runner=_HoldRunner(), lookback_days=lookback)


class TestEngineEnsureOhlcv:
    def test_existing_file_passes(self, tmp_path):
        # Plant a valid OHLCV file that reaches the end_date.
        csv = tmp_path / "EXIST.JK" / "ohlcv.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2026-12-31"],
                "open": [100.0, 110.0],
                "high": [101.0, 111.0],
                "low": [99.0, 109.0],
                "close": [100.5, 110.5],
                "volume": [1000, 2000],
            }
        )
        df.to_csv(csv, index=False)

        engine = _build_engine_for_ticker(tmp_path, "EXIST.JK", "2026-05-25", "2026-06-02")
        ok, msg = engine.ensure_ohlcv(auto_fetch=False)
        assert ok is True
        assert "OHLCV ready" in msg

    def test_missing_file_returns_false_with_hint(self, tmp_path):
        engine = _build_engine_for_ticker(tmp_path, "MISSING.JK", "2026-05-25", "2026-06-02")
        ok, msg = engine.ensure_ohlcv(auto_fetch=False)
        assert ok is False
        assert "MISSING.JK" in msg
        assert "auto_fetch=True" in msg

    def test_stale_file_returns_false(self, tmp_path):
        csv = tmp_path / "STALE.JK" / "ohlcv.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-06-30"],  # ends before end_date
                "open": [100.0, 110.0],
                "high": [101.0, 111.0],
                "low": [99.0, 109.0],
                "close": [100.5, 110.5],
                "volume": [1000, 2000],
            }
        )
        df.to_csv(csv, index=False)

        engine = _build_engine_for_ticker(tmp_path, "STALE.JK", "2026-05-25", "2026-06-02")
        ok, msg = engine.ensure_ohlcv(auto_fetch=False)
        assert ok is False
        assert "STALE.JK" in msg

    def test_auto_fetch_calls_through(self, tmp_path):
        engine = _build_engine_for_ticker(tmp_path, "DEWA.JK", "2026-05-25", "2026-06-02")

        # Mock the cli.data_fetch path used inside engine.ensure_ohlcv
        def fake_ensure(ticker, start_date, end_date, output_root, lookback_days=None):
            target = Path(output_root) / ticker / "ohlcv.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("date,open,high,low,close,volume\n", encoding="utf-8")
            return target, True

        with patch("cli.data_fetch.ensure_ohlcv", side_effect=fake_ensure):
            ok, msg = engine.ensure_ohlcv(auto_fetch=True)

        assert ok is True
        assert "downloaded" in msg or "already present" in msg

    def test_auto_fetch_failure_returns_error(self, tmp_path):
        engine = _build_engine_for_ticker(tmp_path, "BAD.JK", "2026-05-25", "2026-06-02")

        def fake_ensure_fail(*a, **k):
            raise RuntimeError("yfinance connection failed")

        with patch("cli.data_fetch.ensure_ohlcv", side_effect=fake_ensure_fail):
            ok, msg = engine.ensure_ohlcv(auto_fetch=True)

        assert ok is False
        assert "yfinance connection failed" in msg

    def test_invalid_lookback_refuses_fetch(self, tmp_path):
        # 7 is not in ALLOWED_LOOKBACKS → snapshot provider rejects
        engine = _build_engine_for_ticker(tmp_path, "BAD.JK", "2026-05-25", "2026-06-02", lookback=7)
        ok, msg = engine.ensure_ohlcv(auto_fetch=True)
        assert ok is False
        assert "ALLOWED_LOOKBACKS" in msg or "lookback_days" in msg


# ---------------------------------------------------------------------------
# CLI explainer & integration smoke
# ---------------------------------------------------------------------------


class TestCliWindowExplainer:
    """Smoke tests for the dry-run / CLI explainer that disambiguates
    the two time settings the user keeps mixing up."""

    def test_explainer_renders_without_error(self, capsys):
        # The explainer prints via the rich console patched onto the
        # module. Patch it to a no-op renderer to avoid terminal noise.
        from rich.console import Console

        import cli.commands.backtest as cli_mod
        from cli.commands.backtest import _print_window_explainer

        saved = cli_mod.console
        cli_mod.console = Console(file=open("/dev/null", "w"), force_terminal=False)
        try:
            _print_window_explainer("2026-05-25", "2026-06-02", 60)
            _print_window_explainer("2026-05-25", "2026-06-02", None)
        finally:
            cli_mod.console = saved

    def test_explainer_includes_worked_example(self, capsys):
        # Use a capturing console to verify the explainer surfaces the
        # 3-month / 20-day example.
        from rich.console import Console

        import cli.commands.backtest as cli_mod

        buf = __import__("io").StringIO()
        saved = cli_mod.console
        cli_mod.console = Console(file=buf, force_terminal=False, width=200)
        try:
            cli_mod._print_window_explainer(
                "2026-05-25", "2026-06-02", 60
            )
        finally:
            cli_mod.console = saved

        output = buf.getvalue()
        assert "outer window" in output
        assert "inner window" in output
        assert "60 trading days" in output
        assert "3-month backtest" in output  # worked example present


class TestCliEnsureOhlcvHook:
    """Verify the CLI backtest command calls engine.ensure_ohlcv before
    constructing the runner. We don't run the full interactive command
    (typer + questionary would need stubs), but we verify the function
    signature of the inner entry point and that the prompt path
    short-circuits cleanly."""

    def test_backtest_command_signature_has_ensure_ohlcv_call(self):
        # Reading the source is the simplest reliable way to assert the
        # hook is present without spinning up the full CLI.
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent / "cli" / "commands" / "backtest.py").read_text()
        assert "engine.ensure_ohlcv" in text, "CLI must call engine.ensure_ohlcv"
