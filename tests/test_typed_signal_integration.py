import pytest

from cli.commands.evaluate import signal_from_final_state
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    SignalContract,
    portfolio_decision_to_signal_contract,
)
from tradingagents.graph.propagation import Propagator


def test_portfolio_decision_requires_agent_selected_horizon():
    with pytest.raises(Exception, match="time_horizon_days"):
        PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Wait.",
            investment_thesis="No clear edge.",
            wns_recheck_date="2026-01-20",
        )


def test_portfolio_decision_preserves_typed_trader_entry_as_planned_price():
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="Enter.",
        investment_thesis="Upside.",
        take_profit=195.0,
        stop_loss=177.5,
        time_horizon_days=63,
        next_review_date="2026-01-20",
    )

    signal = portfolio_decision_to_signal_contract(
        decision,
        "TSM",
        "2026-01-10",
        planned_entry_price=183.75,
    )

    assert signal.planned_entry_price == 183.75
    assert signal.entry_mode.value == "T1_LIMIT"


def test_portfolio_decision_converts_to_buy_signal_without_prose_inference():
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="The prose says hold, but typed rating is actionable.",
        investment_thesis="Typed fields are authoritative.",
        take_profit=125.0,
        stop_loss=95.0,
        time_horizon_days=15,
        confidence=0.8,
        next_review_date="2026-01-20",
    )

    signal = portfolio_decision_to_signal_contract(decision, "NVDA", "2026-01-10")

    assert isinstance(signal, SignalContract)
    assert signal.action == "BUY"
    assert signal.rating is PortfolioRating.BUY
    assert signal.take_profit == 125.0
    assert signal.stop_loss == 95.0
    assert signal.time_horizon_days == 15
    assert signal.confidence == 0.8


def test_legacy_sell_decision_normalizes_to_wns():
    decision = PortfolioDecision(
        rating=PortfolioRating.SELL,
        executive_summary="Wait for a safer setup.",
        investment_thesis="Risk dominates.",
        time_horizon_days=20,
        wns_recheck_date="2026-01-20",
    )

    signal = portfolio_decision_to_signal_contract(decision, "NVDA", "2026-01-10")
    assert signal.action == "WNS"


def test_hold_signal_can_omit_execution_levels():
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="Wait.",
        investment_thesis="No clear edge.",
        time_horizon_days=20,
        wns_recheck_date="2026-01-20",
    )

    signal = portfolio_decision_to_signal_contract(decision, "NVDA", "2026-01-10")

    assert signal.action == "WNS"
    assert signal.take_profit is None
    assert signal.stop_loss is None


def test_portfolio_manager_returns_typed_signal_and_legacy_markdown():
    from unittest.mock import MagicMock

    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="Enter.",
        investment_thesis="Upside.",
        take_profit=125.0,
        stop_loss=95.0,
        time_horizon_days=20,
        next_review_date="2026-01-20",
    )
    structured = MagicMock()
    structured.invoke.return_value = decision
    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    state = {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-10",
        "past_context": "",
        "risk_debate_state": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
        "investment_plan": "",
        "trader_investment_plan": "",
    }

    result = create_portfolio_manager(llm)(state)

    assert result["signal_contract"].action == "BUY"
    assert "**Rating**: Buy" in result["final_trade_decision"]


def test_cli_rejects_typed_signal_without_agent_horizon():
    signal_data = {
        "ticker": "NVDA",
        "signal_date": "2026-01-10",
        "rating": "WNS",
        "action": "WNS",
        "wns_recheck_date": "2026-01-20",
    }
    with pytest.raises(Exception, match="time_horizon_days"):
        signal_from_final_state(
            {"signal_contract": signal_data},
            "NVDA",
            "2026-01-10",
        )


def test_cli_uses_typed_signal_before_markdown_fallback():
    signal = SignalContract(
        ticker="NVDA",
        signal_date="2026-01-10",
        rating=PortfolioRating.WNS,
        action="WNS",
        wns_recheck_date="2026-01-20",
        time_horizon_days=20,
    )

    result = signal_from_final_state(
        {"signal_contract": signal, "final_trade_decision": "**Rating**: Buy"},
        "NVDA",
        "2026-01-10",
    )

    assert result is signal


def test_cli_rejects_typed_signal_identity_mismatch():
    signal = SignalContract(
        ticker="NVDA",
        signal_date="2026-01-10",
        rating=PortfolioRating.WNS,
        action="WNS",
        wns_recheck_date="2026-01-20",
        time_horizon_days=20,
    )

    with pytest.raises(ValueError, match="ticker/date mismatch"):
        signal_from_final_state(
            {"signal_contract": signal},
            "TSM",
            "2026-01-10",
        )


def test_signal_from_final_state_requires_typed_signal_contract():
    with pytest.raises(ValueError, match="missing signal_contract"):
        signal_from_final_state(
            {"final_trade_decision": "**Rating**: Hold\n**Time Horizon**: unavailable"},
            "NVDA",
            "2026-01-10",
        )


def test_initial_state_has_empty_signal_contract_slot():
    state = Propagator().create_initial_state("NVDA", "2026-01-10")

    assert state["signal_contract"] is None


def test_signal_contract_accepts_legacy_price_target_field():
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="Enter.",
        investment_thesis="Upside.",
        price_target=125.0,
        stop_loss=95.0,
        time_horizon_days=20,
        next_review_date="2026-01-20",
    )

    signal = portfolio_decision_to_signal_contract(decision, "NVDA", "2026-01-10")

    assert signal.take_profit == 125.0


def test_signal_contract_rejects_actionable_missing_levels_without_reading_thesis():
    with pytest.raises(ValueError, match="BUY decision requires stop_loss and take_profit"):
        PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="No target here.",
            investment_thesis="take profit 999 and stop loss 1 are only prose.",
            time_horizon_days=20,
            next_review_date="2026-01-20",
        )


def test_strict_wns_without_date_or_price_raises_validation_error():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="WNS .* must explicitly specify"):
        PortfolioDecision(
            rating=PortfolioRating.WNS,
            executive_summary="Watching the market closely.",
            investment_thesis="Waiting for technical setup.",
            time_horizon_days=10,
        )


def test_limit_entry_price_preservation_in_signal_contract():
    from tradingagents.agents.schemas import EntryMode

    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="Accumulate on dips.",
        investment_thesis="Solid support level.",
        planned_entry_price=215.32,
        take_profit=250.0,
        stop_loss=200.0,
        time_horizon_days=30,
        next_review_date="2026-02-01",
    )

    signal = portfolio_decision_to_signal_contract(decision, "AAPL", "2026-01-15")

    assert signal.planned_entry_price == 215.32
    assert signal.entry_mode == EntryMode.T1_LIMIT


def test_single_shot_sell_evaluates_to_no_order():
    import pandas as pd

    from cli.commands.evaluate import _run_forward_evaluation
    from tradingagents.backtesting.horizon_evaluator import EvaluationOutcome

    signal = SignalContract(
        ticker="AAPL",
        signal_date="2026-01-15",
        rating=PortfolioRating.SELL,
        action="SELL",
        stop_loss=210.0,
        take_profit=190.0,
        wns_recheck_date="2026-01-20",
        time_horizon_days=10,
    )

    df = pd.DataFrame({
        "date": ["2026-01-15", "2026-01-16", "2026-01-17"],
        "open": [200.0, 195.0, 185.0],
        "high": [205.0, 198.0, 190.0],
        "low": [198.0, 190.0, 180.0],
        "close": [202.0, 192.0, 188.0],
    })

    result, _ = _run_forward_evaluation(
        ticker="AAPL",
        trade_date="2026-01-15",
        signal=signal,
        effective_horizon=10,
        ohlcv_df=df,
    )

    assert result.outcome == EvaluationOutcome.NO_ORDER
    assert result.side == "FLAT"
    assert result.realized_return_pct == 0.0