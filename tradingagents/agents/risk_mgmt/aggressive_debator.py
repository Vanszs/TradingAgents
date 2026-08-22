from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state.get("risk_debate_state", {})
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        trader_decision = state.get("trader_investment_plan", "N/A")
        asset_type = state.get("asset_type", "stock")
        company_name = state["company_of_interest"]
        trade_date = state.get("trade_date", "")

        instrument_context = build_instrument_context(
            company_name, asset_type, trade_date=trade_date
        )

        prompt = f"""You are the Aggressive Risk Specialist (The Alpha Desk) evaluating `{company_name}`.
Mandate: Champion the bull case and maximize upside capture on the proposed trade. Refute conservative caution by highlighting growth catalysts, momentum continuation, and the opportunity cost of under-allocation.

{instrument_context}

### Trader Proposal Under Review
{trader_decision}

### Risk Debate History
{history if history else 'Opening round: Present the aggressive case for maximizing upside capture.'}

Deliver an assertive, data-backed defense of the trade, advocate for ambitious targets, and counter downside fears.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument if history else argument,
            "aggressive_history": aggressive_history + "\n" + argument if aggressive_history else argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "judge_decision": risk_debate_state.get("judge_decision", ""),
            "count": risk_debate_state.get("count", 0) + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
