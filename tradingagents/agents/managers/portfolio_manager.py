"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import logging

import pandas as pd

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    portfolio_decision_to_signal_contract,
    render_pm_decision,
)

logger = logging.getLogger(__name__)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        trade_date = state.get("trade_date", "")
        instrument_context = build_instrument_context(
            state["company_of_interest"],
            state.get("asset_type", "stock"),
            trade_date=trade_date,
        )

        risk_debate_state = state.get("risk_debate_state", {})
        history = risk_debate_state.get("history", "")
        research_plan = state.get("investment_plan", "")
        trader_plan = state.get("trader_investment_plan", "")
        trader_proposal = state.get("trader_proposal")
        planned_entry_price = (
            trader_proposal.entry_price if trader_proposal is not None else None
        )

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Senior Portfolio Manager & Risk Gatekeeper, evaluate the risk debate and trader proposal to deliver the final allocation decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter/expand long exposure (Asymmetric R:R >= 2:1)
- **Overweight**: Constructive accumulation; scaled tranche execution
- **Hold**: Neutral prior; wait for confirmed stabilization or favorable R:R
- **Underweight**: Distribution/derisking; trim long inventory
- **Sell**: Complete exit of long exposure; capital preservation

**Context Dossier:**
- Current Trade Date: {trade_date}
- Research Manager Synthesis: **{research_plan}**
- Trader Execution Proposal: **{trader_plan}**
{lessons_line}
### Risk Debate Deliberation
{history if history else 'No risk debate recorded.'}

---

Be decisive and ground your decision in empirical risk asymmetry and structural invalidation levels.{get_language_instruction()}"""

        typed_decision = None
        if structured_llm is not None:
            try:
                typed_decision = structured_llm.invoke(prompt)
                final_trade_decision = render_pm_decision(typed_decision)
            except Exception:
                final_trade_decision = invoke_structured_or_freetext(
                    None, llm, prompt, render_pm_decision, "Portfolio Manager"
                )
        else:
            final_trade_decision = invoke_structured_or_freetext(
                None, llm, prompt, render_pm_decision, "Portfolio Manager"
            )

        signal_contract = None
        if typed_decision is not None and trade_date:
            # Deterministic Python date calculation if missing
            if not typed_decision.next_review_date:
                days_delta = 7 if typed_decision.rating in (PortfolioRating.BUY, PortfolioRating.SELL) else 21
                try:
                    typed_decision.next_review_date = (pd.to_datetime(trade_date) + pd.Timedelta(days=days_delta)).strftime("%Y-%m-%d")
                except Exception:
                    typed_decision.next_review_date = trade_date

            try:
                signal_contract = portfolio_decision_to_signal_contract(
                    typed_decision,
                    state["company_of_interest"],
                    trade_date,
                    planned_entry_price=planned_entry_price,
                )
            except Exception as exc:
                logger.error("Failed to build SignalContract for %s on %s: %s", state.get("company_of_interest"), trade_date, exc)
                signal_contract = None

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "signal_contract": signal_contract,
        }

    return portfolio_manager_node
