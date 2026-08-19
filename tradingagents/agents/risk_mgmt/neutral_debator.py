from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state.get("risk_debate_state", {})
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        trader_decision = state.get("trader_investment_plan", "N/A")
        asset_type = state.get("asset_type", "stock")
        company_name = state["company_of_interest"]
        trade_date = state.get("trade_date", "")

        instrument_context = build_instrument_context(
            company_name, asset_type, trade_date=trade_date
        )

        prompt = f"""You are the Quantitative Risk Arbiter (The Portfolio Strategist) evaluating `{company_name}`.
Mandate: Arbitrate objectively between aggressive upside and conservative risk using mathematical expectancy (R:R >= 2:1) and market regime alignment.

{instrument_context}

### Trader Proposal Under Review
{trader_decision}

### Risk Debate History
{history if history else 'Opening round: Evaluate market regime, expectancy, and execution staging.'}

Balance the debate: verify 1D Macro + 1H Micro trend alignment, check invalidation geometry, and propose balanced staged limit tranches.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument if history else argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument if neutral_history else argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "judge_decision": risk_debate_state.get("judge_decision", ""),
            "count": risk_debate_state.get("count", 0) + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
