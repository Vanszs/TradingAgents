"""Tests for Quant Expectancy and Code Judo Blueprints (Rules 1-4).

Verifies:
1. Strict Execution Hierarchy: PM cannot override Trader's WNS into a BUY.
2. Breakout Targets with Fib Extensions: Structural levels calculate expansion targets (1.272x, 1.618x) and ATR targets.
3. Entry Mode Flexibility: Extreme oversold reversals use Buy Market (T1_OPEN) or tight limits (0.5x ATR) to prevent NO_FILL.
4. Falling Knife WNS Discipline: Sub-200 SMA breakdowns strictly maintain WNS, producing NO_ORDER and avoiding stop-loss hits.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import (
    EntryMode,
    EvaluationOutcome,
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
    portfolio_decision_to_signal_contract,
    render_pm_decision,
)
from tradingagents.backtesting.horizon_evaluator import HorizonEvaluator
from tradingagents.dataflows.structural_levels import (
    compute_structural_levels,
    get_market_structural_summary,
)


@pytest.mark.unit
class TestRule1StrictExecutionHierarchy:
    """Rule 1: Portfolio Manager CANNOT override a Trader's WNS into a BUY."""

    def test_pm_override_attempt_is_downgraded_to_wns_typed(self):
        # LLM returns BUY despite Trader proposing WNS
        pm_decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Attempting dip buy after panic selloff.",
            investment_thesis="Long-term growth is intact.",
            stop_loss=300.0,
            take_profit=380.0,
            time_horizon_days=20,
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = pm_decision
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        pm_node = create_portfolio_manager(mock_llm)

        trader_prop = TraderProposal(
            action=TraderAction.WNS,
            reasoning="Sub-200 SMA breakdown without structural support. Falling knife.",
            wns_recheck_date="2026-02-03",
            wns_trigger_price=295.0,
        )

        state = {
            "company_of_interest": "MSFT",
            "asset_type": "stock",
            "trade_date": "2026-01-13",
            "investment_plan": "**Recommendation**: WNS",
            "trader_investment_plan": "**Action**: WNS\nFINAL TRANSACTION PROPOSAL: **WNS**",
            "trader_proposal": trader_prop,
            "risk_debate_state": {
                "history": "",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "latest_speaker": "Aggressive",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 1,
            },
        }

        result = pm_node(state)

        # Invariant check: Rating MUST be downgraded to WNS
        assert "**Rating**: WNS" in result["final_trade_decision"]
        contract = result["signal_contract"]
        assert contract is not None
        assert contract.action == "WNS"
        assert contract.rating == PortfolioRating.WNS
        assert contract.wns_recheck_date == "2026-02-03"
        assert contract.wns_trigger_price == 295.0

    def test_pm_accepts_trader_buy(self):
        pm_decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Momentum breakout confirmed above 50 SMA.",
            investment_thesis="Strong catalyst with asymmetric R:R.",
            stop_loss=320.0,
            take_profit=370.0,
            time_horizon_days=20,
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = pm_decision
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        pm_node = create_portfolio_manager(mock_llm)

        trader_prop = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Reclaiming 50 SMA on volume. Target Fib 1.272x extension.",
            stop_loss=320.0,
            take_profit=370.0,
        )

        state = {
            "company_of_interest": "MSFT",
            "asset_type": "stock",
            "trade_date": "2023-10-30",
            "investment_plan": "**Recommendation**: Buy",
            "trader_investment_plan": "**Action**: Buy\nFINAL TRANSACTION PROPOSAL: **BUY**",
            "trader_proposal": trader_prop,
            "risk_debate_state": {
                "history": "",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "latest_speaker": "Aggressive",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 1,
            },
        }

        result = pm_node(state)
        assert "**Rating**: Buy" in result["final_trade_decision"]
        contract = result["signal_contract"]
        assert contract is not None
        assert contract.action == "BUY"
        assert contract.rating == PortfolioRating.BUY
        assert contract.stop_loss == 320.0
        assert contract.take_profit == 370.0

    def test_pm_can_downgrade_trader_buy_to_wns(self):
        pm_decision = PortfolioDecision(
            rating=PortfolioRating.WNS,
            executive_summary="Vetoing Buy to WNS due to macroeconomic tail risk.",
            investment_thesis="Earnings risk and fed policy uncertainty.",
            wns_recheck_date="2026-02-15",
            time_horizon_days=20,
        )

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = pm_decision
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        pm_node = create_portfolio_manager(mock_llm)

        trader_prop = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Setup looks constructive.",
            stop_loss=320.0,
            take_profit=370.0,
        )

        state = {
            "company_of_interest": "MSFT",
            "asset_type": "stock",
            "trade_date": "2026-01-13",
            "investment_plan": "**Recommendation**: Buy",
            "trader_investment_plan": "**Action**: Buy",
            "trader_proposal": trader_prop,
            "risk_debate_state": {
                "history": "",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "latest_speaker": "Conservative",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 1,
            },
        }

        result = pm_node(state)
        assert "**Rating**: WNS" in result["final_trade_decision"]
        contract = result["signal_contract"]
        assert contract.action == "WNS"


@pytest.mark.unit
class TestRule2BreakoutTargetsFibExtensions:
    """Rule 2: Fibonacci Extension Targets (1.272x, 1.618x) and ATR Target Channels."""

    def test_fibonacci_extensions_and_atr_targets_computed(self):
        dates = pd.date_range("2023-08-01", periods=70, freq="B")
        df = pd.DataFrame({
            "Open": [300.0] * 70,
            "High": [305.0] * 70,
            "Low": [295.0] * 70,
            "Close": [300.0] * 70,
        }, index=dates)
        # 60D Low at day 10 = $280, 60D High at day 40 = $330 (range = $50)
        df.iloc[-60, df.columns.get_loc("Low")] = 280.0
        df.iloc[-30, df.columns.get_loc("High")] = 330.0
        df.iloc[-1, df.columns.get_loc("Close")] = 325.0

        levels = compute_structural_levels(df, "2023-10-30")
        assert levels["60d_swing_high"] == 330.0
        assert levels["60d_swing_low"] == 280.0
        # fib_range = 50.0
        # 1.272x ext = 330 + 0.272 * 50 = 343.6
        # 1.618x ext = 330 + 0.618 * 50 = 360.9
        assert levels["fib_ext_1272"] == 343.6
        assert levels["fib_ext_1618"] == 360.9
        assert "atr_target_2x" in levels
        assert "atr_target_3x" in levels
        assert "atr_half" in levels
        assert levels["atr_half"] > 0

    def test_structural_summary_contains_forward_expansion_and_buffer(self):
        dates = pd.date_range("2023-08-01", periods=70, freq="B")
        df = pd.DataFrame({
            "Open": [300.0] * 70,
            "High": [305.0] * 70,
            "Low": [295.0] * 70,
            "Close": [300.0] * 70,
        }, index=dates)
        df.iloc[-60, df.columns.get_loc("Low")] = 280.0
        df.iloc[-30, df.columns.get_loc("High")] = 330.0

        levels = compute_structural_levels(df, "2023-10-30")
        assert levels["fib_ext_1272"] == 343.6
        assert levels["fib_ext_1618"] == 360.9


@pytest.mark.unit
class TestRule3EntryModeFlexibility:
    """Rule 3: Buy Market (T1_OPEN) or tight Limit (0.5x ATR) for Oversold Reversals."""

    def test_oversold_market_entry_fills_on_gap_up(self):
        signal_date = "2025-04-08"
        # Day after signal (2025-04-09) opens at $356.00 and rallies to $370.00
        future_df = pd.DataFrame({
            "date": ["2025-04-08", "2025-04-09", "2025-04-10"],
            "open": [354.56, 356.00, 365.00],
            "high": [355.00, 370.00, 375.00],
            "low": [350.00, 355.50, 362.00],
            "close": [354.56, 368.00, 372.00],
        })

        # Market Entry (T1_OPEN): entry_policy="T1_OPEN", planned_entry_price=None
        res_market = HorizonEvaluator.evaluate(
            ticker="MSFT",
            signal_date=signal_date,
            side="BUY",
            take_profit=380.0,
            stop_loss=340.0,
            time_horizon_days=2,
            ohlcv_df=future_df,
            planned_entry_price=None,
            entry_policy="T1_OPEN",
        )
        # Should fill at first_open (356.00)
        assert res_market.outcome != EvaluationOutcome.NO_FILL
        assert res_market.actual_entry_price == 356.00

        # Contrast with Deep Limit at $345 (where price never touched $345)
        res_deep_limit = HorizonEvaluator.evaluate(
            ticker="MSFT",
            signal_date=signal_date,
            side="BUY",
            take_profit=380.0,
            stop_loss=340.0,
            time_horizon_days=2,
            ohlcv_df=future_df,
            planned_entry_price=345.0,
            actual_entry_price=345.0,
            entry_policy="T1_LIMIT",
        )
        assert res_deep_limit.outcome == EvaluationOutcome.NO_FILL
        assert res_deep_limit.actual_entry_price is None

    def test_tight_limit_within_bar_range_fills(self):
        signal_date = "2025-04-08"
        # Bar range low is 355.50, high is 370.00. Tight limit at 355.80 (within 0.5x ATR) fills!
        future_df = pd.DataFrame({
            "date": ["2025-04-08", "2025-04-09"],
            "open": [354.56, 358.00],
            "high": [355.00, 370.00],
            "low": [350.00, 355.50],
            "close": [354.56, 368.00],
        })

        res_tight_limit = HorizonEvaluator.evaluate(
            ticker="MSFT",
            signal_date=signal_date,
            side="BUY",
            take_profit=380.0,
            stop_loss=340.0,
            time_horizon_days=1,
            ohlcv_df=future_df,
            planned_entry_price=356.00,
            actual_entry_price=356.00,
            entry_policy="T1_LIMIT",
        )
        assert res_tight_limit.outcome != EvaluationOutcome.NO_FILL
        assert res_tight_limit.actual_entry_price == 356.00


@pytest.mark.unit
class TestRule4FallingKnifeWNSDiscipline:
    """Rule 4: Breakdown below 200 SMA / major support stays WNS, generating NO_ORDER."""

    def test_falling_knife_wns_evaluates_to_no_order_zero_loss(self):
        signal_date = "2026-01-13"
        # Stock drops -14% over subsequent days
        future_df = pd.DataFrame({
            "date": ["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"],
            "open": [320.0, 315.0, 305.0, 290.0],
            "high": [322.0, 316.0, 306.0, 292.0],
            "low": [318.0, 304.0, 288.0, 275.0],
            "close": [319.0, 305.0, 289.0, 275.0],
        })

        # When WNS is disciplined, side="WNS" maps to NO_ORDER with 0% realized loss
        res_wns = HorizonEvaluator.evaluate(
            ticker="MSFT",
            signal_date=signal_date,
            side="WNS",
            take_profit=None,
            stop_loss=None,
            time_horizon_days=20,
            ohlcv_df=future_df,
        )
        assert res_wns.outcome == EvaluationOutcome.NO_ORDER
        assert res_wns.realized_return_pct == 0.0
        assert res_wns.actual_holding_days == 0


@pytest.mark.unit
class TestMultiHorizonDailyStructurePromptAlignment:
    """Task: Unify all timeframe representations to Institutional Multi-Horizon Daily Structure."""

    def test_market_analyst_prompt_uses_multi_horizon(self):
        from langchain_core.runnables import RunnableLambda

        from tradingagents.agents.analysts.market_analyst import create_market_analyst

        captured = []
        mock_res = MagicMock()
        mock_res.tool_calls = []
        mock_res.content = "Market Analysis"

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = RunnableLambda(lambda p: captured.append(p) or mock_res)

        node = create_market_analyst(mock_llm)
        state = {
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "trade_date": "2026-01-13",
            "messages": [],
        }
        node(state)
        assert len(captured) == 1
        sys_text = captured[0].messages[0].content
        assert "Multi-Horizon Daily Structure (Macro 52W/200 SMA -> Intermediate 60D/50 SMA -> Tactical 20D/20 EMA/ATR)" in sys_text
        assert "Horizon (Macro / Intermediate / Tactical)" in sys_text
        assert "1D Macro Trend confirmed by 1H Micro Structure" not in sys_text
        assert "Timeframe (1D/1H)" not in sys_text

    def test_neutral_debator_prompt_uses_multi_horizon(self):
        from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
        mock_llm = MagicMock()
        mock_res = MagicMock()
        mock_res.content = "Neutral Argument"
        mock_llm.invoke.return_value = mock_res

        node = create_neutral_debator(mock_llm)
        state = {
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "trade_date": "2026-01-13",
            "trader_investment_plan": "WNS",
            "risk_debate_state": {},
        }
        node(state)
        prompt_text = mock_llm.invoke.call_args[0][0]
        assert "Multi-Horizon Daily Structure (Macro 52W/200 SMA -> Intermediate 60D/50 SMA -> Tactical 20D/20 EMA/ATR)" in prompt_text
        assert "1D Macro + 1H Micro" not in prompt_text

    def test_bull_researcher_prompt_uses_tactical_support(self):
        from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
        mock_llm = MagicMock()
        mock_res = MagicMock()
        mock_res.content = "Bull Thesis"
        mock_llm.invoke.return_value = mock_res

        node = create_bull_researcher(mock_llm)
        state = {
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "trade_date": "2026-01-13",
            "investment_debate_state": {},
        }
        node(state)
        prompt_text = mock_llm.invoke.call_args[0][0]
        assert "tactical support" in prompt_text
        assert "1H micro support" not in prompt_text

    def test_structural_summary_multi_horizon_sections(self):
        dates = pd.date_range("2023-08-01", periods=70, freq="B")
        df = pd.DataFrame({
            "Open": [300.0] * 70,
            "High": [305.0] * 70,
            "Low": [295.0] * 70,
            "Close": [300.0] * 70,
        }, index=dates)
        from unittest.mock import patch
        with patch("tradingagents.dataflows.structural_levels.load_ohlcv", return_value=df):
            summary = get_market_structural_summary("AAPL", "2023-10-30")
            assert "1. **Macro Horizon" in summary
            assert "2. **Intermediate Horizon" in summary
            assert "3. **Tactical Horizon" in summary
            assert "Intraday 1H data not available" not in summary
