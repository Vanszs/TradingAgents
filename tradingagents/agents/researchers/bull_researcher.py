from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")
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

        bear_rebuttal_section = (
            f"### Opposing Bear Argument to Address:\n{current_response}\n\n"
            if current_response
            else "### Opening Round:\nPresent your primary upside thesis, growth drivers, and catalyst timeline.\n\n"
        )

        prompt = f"""You are an Institutional Long-Only Bull Analyst evaluating `{company_name}`.
Your role: Build a high-expectancy upside thesis grounded in Spot Equity Long-Only accumulation and market structure.

{instrument_context}

### Long-Only Execution Guidelines:
1. **Spot Long Mandate**: Focus on long accumulation (no shorting).
2. **Limit Accumulation on Dips**: If price is pulling back, identify the optimal **Limit Accumulation Zone ($XXX)** anchored near structural support (Fibonacci levels, 20D/60D swing floors, or 1H micro support) with R:R >= 2:1.
3. **Structural Invalidation**: Define hard protective invalidation below confirmed support.

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

{bear_rebuttal_section}
Deliver a sharp, evidence-based argument specifying planned accumulation zone, invalidation price floor, take profit targets, and catalyst timeline.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument if history else argument,
            "bull_history": bull_history + "\n" + argument if bull_history else argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "judge_decision": investment_debate_state.get("judge_decision", ""),
            "count": investment_debate_state.get("count", 0) + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
