from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state.get("risk_debate_state", {})
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        trader_decision = state.get("trader_investment_plan", "N/A")
        asset_type = state.get("asset_type", "stock")
        company_name = state["company_of_interest"]
        trade_date = state.get("trade_date", "")

        instrument_context = build_instrument_context(
            company_name, asset_type, trade_date=trade_date
        )

        prompt = f"""You are the Chief Risk Officer (The Capital Preservation Desk) evaluating `{company_name}`.
Mandate: Champion the bear case and protect firm capital. Stress-test the trade for drawdown vulnerability, adverse gap shocks, overvaluation, and hidden structural weaknesses.

{instrument_context}

### Trader Proposal Under Review
{trader_decision}

### Risk Debate History
{history if history else 'Opening round: Stress-test the proposal against downside risks and tail-risk.'}

**Risk Assessment Policy**:
- **Falling Knife / Sub-200 SMA Breakdown Veto**: Strictly enforce WNS whenever price breaks below the 200 SMA or major multi-month support floors without structural base or divergence, or if solvency is deteriorating. Zero dip-buying in freefall.
- **Strict Execution Hierarchy**: If the Trader proposed WNS, validate the prudence of the Trader's gate and recommend ratifying WNS to preserve capital.
- **Asymmetric Oversold Mean-Reversion**: If the asset is testing a major macro/52W support floor with intact business fundamentals, tight stop-loss (Risk <= 5%), and R:R >= 2.5:1, do NOT issue a blanket WNS veto out of vague downtrend fear. Instead, advocate for defensive position sizing (e.g. 5-10% equity) and strict stop-loss adherence.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument if history else argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument if conservative_history else argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "judge_decision": risk_debate_state.get("judge_decision", ""),
            "count": risk_debate_state.get("count", 0) + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
