import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from cli.commands import evaluate_results
from cli.commands.evaluate_results import sanitize_config, write_result_bundle
from tradingagents.dataflows.utils import safe_ticker_component


def test_write_result_bundle_persists_pit_snapshot_and_evaluation(tmp_path):
    snapshot = {
        "ohlcv": [
            {"date": "2025-05-24", "open": 99, "high": 101, "low": 98, "close": 100},
            {"date": "2025-05-25", "open": 100, "high": 102, "low": 99, "close": 101},
        ],
        "news": [{"published_at": "2025-05-25", "title": "Before T0"}],
        "fundamentals": [],
        "sentiment": [],
        "broker_activity": [],
    }
    signal = {
        "ticker": "TSM",
        "signal_date": "2025-05-25",
        "action": "BUY",
        "planned_entry_price": 4400.0,
        "time_horizon_days": 252,
        "time_horizon_label": "6-12 months",
    }
    evaluation = {
        "outcome": "EXPIRED",
        "planned_time_horizon_days": 252,
        "actual_entry_price": 4500.0,
        "planned_entry_price": 4400.0,
        "entry_policy": "T1_OPEN",
        "signal_timestamp": "2025-05-25T23:59:59+00:00",
    }

    output_dir = write_result_bundle(
        root=tmp_path / "result_backtest",
        ticker="TSM",
        trade_date="2025-05-25",
        snapshot_data=snapshot,
        agent_report="# Agent report",
        signal=signal,
        evaluation=evaluation,
        config={"backend_url": "https://example.test/v1", "OPENAI_API_KEY": "do-not-write"},
    )

    assert output_dir == tmp_path / "result_backtest" / "TSM" / "2025-05-25" / "v1.0"
    expected = {
        "snapshot/ohlcv.csv",
        "snapshot/news.json",
        "snapshot/fundamentals.json",
        "snapshot/sentiment.json",
        "snapshot/broker_activity.json",
        "agent_report.md",
        "signal.json",
        "evaluation.json",
        "execution_data.json",
        "leakage_audit.json",
        "config.json",
        "summary.json",
    }
    assert {str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()} == expected

    saved_ohlcv = pd.read_csv(output_dir / "snapshot/ohlcv.csv")
    assert saved_ohlcv["date"].max() <= "2025-05-25"
    saved_signal = json.loads((output_dir / "signal.json").read_text())
    saved_evaluation = json.loads((output_dir / "evaluation.json").read_text())
    assert saved_signal["time_horizon_label"] == "6-12 months"
    assert saved_signal["planned_entry_price"] == 4400.0
    assert saved_evaluation["outcome"] == "EXPIRED"
    assert saved_evaluation["actual_entry_price"] == 4500.0
    assert saved_evaluation["planned_entry_price"] == 4400.0
    assert saved_evaluation["entry_policy"] == "T1_OPEN"
    assert "OPENAI_API_KEY" not in (output_dir / "config.json").read_text()
    audit = json.loads((output_dir / "leakage_audit.json").read_text())
    assert audit["status"] == "PASSED"
    assert audit["model_snapshot_only"] is True
    assert audit["forward_execution_data_persisted"] is True
    assert audit["ohlcv_cutoff_passed"] is True
    assert audit["execution_policy"] == "T1_OPEN"
    assert audit["selected_interval"] is None
    assert audit["signal_timestamp"] == "2025-05-25T23:59:59+00:00"
    assert json.loads((output_dir / "execution_data.json").read_text()) == {}
    assert audit["actual_entry_price"] == 4500.0
    assert audit["planned_entry_price"] == 4400.0


def test_write_result_bundle_persists_execution_interval_metadata(tmp_path):
    output_dir = write_result_bundle(
        root=tmp_path,
        ticker="TSM",
        trade_date="2025-05-25",
        snapshot_data={},
        agent_report="",
        signal={},
        evaluation={},
        execution_data={
            "selected_interval": "1h",
            "attempts": [
                {"interval": "5m", "status": "error", "error": "unavailable"},
                {"interval": "1h", "status": "success", "error": None},
            ],
            "selected_frame": [{"timestamp": "2025-05-26T13:00:00+00:00"}],
        },
        config={},
    )

    assert json.loads((output_dir / "execution_data.json").read_text()) == {
        "attempts": [
            {"error": "unavailable", "interval": "5m", "status": "error"},
            {"error": None, "interval": "1h", "status": "success"},
        ],
        "selected_frame": [{"timestamp": "2025-05-26T13:00:00+00:00"}],
        "selected_interval": "1h",
    }
    audit = json.loads((output_dir / "leakage_audit.json").read_text())
    assert audit["selected_interval"] == "1h"


def test_write_result_bundle_versions_existing_bundle(tmp_path):
    kwargs = {
        "root": tmp_path,
        "ticker": "TSM",
        "trade_date": "2025-05-25",
        "snapshot_data": {},
        "agent_report": "",
        "signal": {},
        "evaluation": {},
        "config": {},
    }
    assert write_result_bundle(**kwargs).name == "v1.0"
    assert write_result_bundle(**kwargs).name == "v2.0"


def test_write_result_bundle_allocates_unique_versions_for_concurrent_writers(tmp_path):
    kwargs = {
        "root": tmp_path,
        "ticker": "TSM",
        "trade_date": "2025-05-25",
        "snapshot_data": {},
        "agent_report": "",
        "signal": {},
        "evaluation": {},
        "config": {},
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        output_dirs = list(executor.map(lambda _: write_result_bundle(**kwargs), range(2)))

    assert {output_dir.name for output_dir in output_dirs} == {"v1.0", "v2.0"}


def test_write_result_bundle_does_not_publish_failed_serialization(tmp_path, monkeypatch):
    def fail_write_json(path, value):
        if path.name == "evaluation.json":
            raise TypeError("not serializable")
        path.write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(evaluate_results, "_write_json", fail_write_json)
    with pytest.raises(TypeError, match="not serializable"):
        write_result_bundle(
            root=tmp_path,
            ticker="TSM",
            trade_date="2025-05-25",
            snapshot_data={},
            agent_report="",
            signal={},
            evaluation={},
            config={},
        )

    assert not list((tmp_path / "TSM" / "2025-05-25").glob("v1.0"))


def test_sanitize_config_redacts_url_credentials():
    sanitized = sanitize_config(
        {
            "api_url": "https://user:password@example.test/v1?api_key=query-secret&safe=1",
            "nested": ["https://example.test/?token=another-secret", "plain"],
        }
    )

    assert sanitized == {
        "api_url": "https://[REDACTED]:[REDACTED]@example.test/v1?api_key=%5BREDACTED%5D&safe=1",
        "nested": ["https://example.test/?token=%5BREDACTED%5D", "plain"],
    }


def test_write_result_bundle_rejects_future_snapshot_data(tmp_path):
    with pytest.raises(ValueError, match="after trade_date"):
        write_result_bundle(
            root=tmp_path,
            ticker="TSM",
            trade_date="2025-05-25",
            snapshot_data={
                "ohlcv": [{"date": "2025-05-26", "open": 1, "high": 1, "low": 1, "close": 1}]
            },
            agent_report="",
            signal={},
            evaluation={},
            config={},
        )


@pytest.mark.parametrize("source", ["news", "fundamentals", "sentiment", "broker_activity"])
def test_write_result_bundle_rejects_future_non_ohlcv_data(tmp_path, source):
    with pytest.raises(ValueError, match="after trade_date"):
        write_result_bundle(
            root=tmp_path,
            ticker="TSM",
            trade_date="2025-05-25",
            snapshot_data={
                source: [{"date": "2025-05-26", "published_at": "2025-05-26", "available_date": "2025-05-26"}]
            },
            agent_report="",
            signal={},
            evaluation={},
            config={},
        )


def test_write_result_bundle_rejects_undated_non_ohlcv_data(tmp_path):
    with pytest.raises(ValueError, match="no usable publication date"):
        write_result_bundle(
            root=tmp_path,
            ticker="TSM",
            trade_date="2025-05-25",
            snapshot_data={"news": [{"title": "undated"}]},
            agent_report="",
            signal={},
            evaluation={},
            config={},
        )


def test_write_result_bundle_redacts_secret_values_and_invalid_dates(tmp_path):
    with pytest.raises(ValueError, match="invalid trade date"):
        write_result_bundle(
            root=tmp_path,
            ticker="TSM",
            trade_date="2025-99-99",
            snapshot_data={},
            agent_report="",
            signal={},
            evaluation={},
            config={},
        )

    output_dir = write_result_bundle(
        root=tmp_path,
        ticker="TSM",
        trade_date="2025-05-25",
        snapshot_data={},
        agent_report="",
        signal={},
        evaluation={},
        config={"headers": {"Authorization": "Bearer sk-secret-value-123456"}},
    )
    text = (output_dir / "config.json").read_text()
    assert "sk-secret-value-123456" not in text
    assert "Bearer" not in text


def test_write_result_bundle_rejects_path_traversal_ticker(tmp_path):
    with pytest.raises(ValueError):
        write_result_bundle(
            root=tmp_path,
            ticker="../TSM",
            trade_date="2025-05-25",
            snapshot_data={},
            agent_report="",
            signal={},
            evaluation={},
            config={},
        )


def test_safe_ticker_component_rejects_dot_path_components():
    with pytest.raises(ValueError):
        safe_ticker_component("..")
