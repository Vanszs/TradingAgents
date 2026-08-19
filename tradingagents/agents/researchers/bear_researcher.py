from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")
        current_response = investment_debate_state.get("current_response", "")

        market_research_report = state.get("market_report", "N/A")
        sentiment_report = state.get("sentiment_report", "N/A")
        news_report = state.get("news_report", "N/A")
        fundamentals_report = state.get("fundamentals_report", "N/A")
        asset_type = state.get("asset_type", "stock")
        company_name = state["company_of_interest"]
        trade_date = state.get("trade_date", "")

        instrument_context = build_instrument_context(
            company_name, asset_type, trade_date=trade_date
        )

        bull_rebuttal_section = (
            f"### Opposing Bull Argument to Refute:\n{current_response}\n\n"
            if current_response
            else "### Opening Round:\nPresent your primary downside thesis, valuation risks, and structural breakdown vectors.\n\n"
        )

        prompt = f"""You are an Institutional Downside Risk Specialist evaluating `{company_name}` under a Spot Long-Only mandate.
Your role: Stress-test the asset for drawdown vulnerability, valuation stretch, and downside targets. Map the structural demand floor where downside momentum is likely to exhaust.

{instrument_context}

### Long-Only Downside Guidelines:
1. **Downside Exhaustion / Accumulation Floor**: Identify the key structural support shelf ($XXX) where selling pressure is expected to meet institutional demand.
2. **Breakdown Risks**: Highlight specific triggers that would invalidate a long accumulation thesis.

### Research Dossier
<market_structure>
{market_research_report}
</market_structure>

<fundamentals>
{fundamentals_report}
</fundamentals>

<news_and_macro>
{news_report}
</news_report>

<sentiment>
{sentiment_report}
</sentiment>

### Debate History
{history if history else 'No prior rounds.'}

{bull_rebuttal_section}
Deliver a sharp, evidence-based downside critique specifying key breakdown levels and downside exhaustion support floors.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument if history else argument,
            "bear_history": bear_history + "\n" + argument if bear_history else argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "judge_decision": investment_debate_state.get("judge_decision", ""),
            "count": investment_debate_state.get("count", 0) + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
