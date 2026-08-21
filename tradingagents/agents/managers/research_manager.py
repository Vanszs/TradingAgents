"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        trade_date = state.get("trade_date", "")
        instrument_context = build_instrument_context(
            state["company_of_interest"],
            state.get("asset_type", "stock"),
            trade_date=trade_date,
        )
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        company_name = state["company_of_interest"]
        prompt = f"""You are the Lead Research Manager synthesizing the dialectical debate for `{company_name}` under a strict **Spot Long-Only (BUY vs WNS)** mandate.

{instrument_context}

**Decision Scale & Mandate (Spot Long-Only)**:
- **Buy**: Strong conviction in asymmetric long upside. Formulate high-level strategic directional consensus and catalyst timeline.
- **Overweight**: Constructive view; accumulation warranted.
- **Hold**: Neutral prior; wait for confirmed stabilization.
- **WNS (Wait and See)**: The default prior whenever entry placement or timing cannot be committed immediately. You MUST provide at least one explicit re-evaluation term:
  1. Temporal Gate: "check after date X" (e.g., post-earnings release, macro catalyst, CPI).
  2. Structural Price Gate: "check again after touch price level Y" (e.g., pullback to 200 SMA demand zone $YYY).
- **Underweight**: Cautious view; trim exposure/distribution.
- **Sell**: Complete liquidation / capital preservation exit of existing long inventory to cash.

### Debate History
{history if history else 'No debate history available.'}

Deliver a decisive, evidence-based judgment in valid JSON matching the ResearchPlan schema.""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
