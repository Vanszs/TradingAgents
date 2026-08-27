import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from cli.main import app, build_headless_selections, save_report_to_disk
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating, SignalContract

runner = CliRunner()


def test_build_headless_selections():
    selections = build_headless_selections(
        ticker="BBRI.JK",
        analysis_date="2025-01-15",
        provider="openai",
        research_depth=3,
        language="Indonesian",
    )
    assert selections["ticker"] == "BBRI.JK"
    assert selections["analysis_date"] == "2025-01-15"
    assert selections["llm_provider"] == "openai"
    assert selections["research_depth"] == 3
    assert selections["output_language"] == "Indonesian"
    assert len(selections["analysts"]) >= 1


def test_save_report_to_disk_persists_signal_and_sanitized_config(tmp_path: Path):
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="Solid value buy.",
        investment_thesis="Growth in rural lending.",
        stop_loss=4100.0,
        take_profit=4800.0,
        time_horizon_days=63,
        time_horizon="1-3 months",
        next_review_date="2025-01-25",
        confidence=0.85,
    )
    signal = SignalContract(
        ticker="BBRI.JK",
        signal_date="2025-01-15",
        rating=decision.rating,
        action="BUY",
        time_horizon_days=63,
        time_horizon_label="1-3 months",
        planned_entry_price=4300.0,
        take_profit=4800.0,
        stop_loss=4100.0,
    )

    final_state = {
        "market_report": "# Market Report\nBullish momentum",
        "signal_contract": signal,
        "risk_debate_state": {"judge_decision": "Approve trade"},
    }

    raw_config = {
        "OPENAI_API_KEY": "sk-secret-key-do-not-leak",
        "backend_url": "https://user:password@example.com/v1?token=query-secret",
        "max_debate_rounds": 3,
    }

    save_path = tmp_path / "report_out"
    report_file = save_report_to_disk(final_state, "BBRI.JK", save_path, config=raw_config)

    assert report_file.exists()
    assert (save_path / "1_analysts" / "market.md").exists()
    assert (save_path / "signal.json").exists()
    assert (save_path / "config.json").exists()

    signal_json = json.loads((save_path / "signal.json").read_text(encoding="utf-8"))
    assert signal_json["action"] == "BUY"
    assert signal_json["planned_entry_price"] == 4300.0

    config_json_text = (save_path / "config.json").read_text(encoding="utf-8")
    assert "sk-secret-key-do-not-leak" not in config_json_text
    assert "password" not in config_json_text
    assert "[REDACTED]" in config_json_text


def test_analyze_cli_headless_execution(tmp_path: Path):
    mock_final_state = {
        "market_report": "Market looks fine",
        "signal_contract": SignalContract(
            ticker="BBRI.JK",
            signal_date="2025-01-15",
            rating=PortfolioRating.HOLD,
            action="HOLD",
            wns_recheck_date="2025-01-20",
            time_horizon_days=21,
            time_horizon_label="1 month",
        ),
    }

    out_dir = tmp_path / "headless_run"

    with patch("cli.main.run_analysis", return_value=mock_final_state) as mock_run:
        result = runner.invoke(
            app,
            [
                "analyze",
                "--ticker", "BBRI.JK",
                "--date", "2025-01-15",
                "--provider", "openai",
                "--output-dir", str(out_dir),
                "--headless",
            ],
        )
        assert result.exit_code == 0
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        assert kwargs["headless"] is True
        assert kwargs["output_dir"] == out_dir
        assert kwargs["selections"]["ticker"] == "BBRI.JK"
        assert kwargs["selections"]["analysis_date"] == "2025-01-15"
