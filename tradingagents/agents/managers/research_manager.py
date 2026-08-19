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

        prompt = f"""As the Senior Research Manager, synthesize the debate between the Bull and Bear analysts against quantitative levels to formulate the directional investment plan.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; asymmetric upside
- **Overweight**: Constructive view; accumulation warranted
- **Hold**: Balanced view, unclear catalysts, or elevated regime uncertainty (Default prior)
- **Underweight**: Cautious view; trim exposure/distribution
- **Sell**: Strong conviction in the bear thesis; asymmetric downside

---

**Debate History:**
{history if history else 'No debate recorded.'}""" + get_language_instruction()

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
