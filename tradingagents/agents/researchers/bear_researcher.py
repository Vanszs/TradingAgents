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

        prompt = f"""You are an Institutional Downside Risk Specialist evaluating `{company_name}`.
Your role: Rigorously stress-test the asset for distribution risk, severe drawdown vulnerability, valuation stretch, thesis invalidation vectors, and capital preservation exit triggers.

{instrument_context}

### Downside Risk & Thesis Invalidation Guidelines:
1. **Distribution & Breakdown Vectors**: Identify specific technical, fundamental, or macroeconomic triggers that would break market structure and force immediate risk-off exits.
2. **Downside Vulnerability & Capital Preservation**: Highlight asymmetric downside risks, valuation headwinds, and adverse scenarios where exposure should be reduced or avoided entirely.

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
Deliver a sharp, evidence-based downside critique specifying key breakdown levels and capital preservation exit triggers.""" + get_language_instruction()

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
