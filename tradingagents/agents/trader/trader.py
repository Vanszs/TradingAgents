"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        trade_date = state.get("trade_date", "")
        instrument_context = build_instrument_context(company_name, asset_type, trade_date=trade_date)
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Senior Execution Trader converting the Research Manager's plan into a high-expectancy transaction under a Spot Equity Long-Only mandate. "
                    "Ground price levels (entry, stop loss, take profit) in the **Multi-Timeframe Structure (1D Macro + 1H Micro)**.\n\n"
                    "**Execution & Entry Discipline**:\n"
                    "- **Immediate Market Entry (Buy)**: Use if price is currently sitting directly at confirmed support with bullish momentum and R:R >= 2:1.\n"
                    "- **Conditional Limit Accumulation (Hold with Limit Levels)**: If current price is extended or pulling back towards support (e.g. current 390, demand floor 350-355), set action to `Hold`, specify `entry_price` at the Limit Accumulation floor (e.g. 355), set `stop_loss` below structural support (e.g. 340), and `take_profit` at target (e.g. 420) ensuring R:R >= 2:1.\n"
                    "- **Multi-Timeframe Confluence**: Use 1D swing floors and Fibonacci levels for macro invalidation (Stop Loss), and use 1H 24-bar swing levels and 1H EMA 20/50 alignment for precise entry timing.\n"
                    "- **Risk-to-Reward (R:R)**: Ensure (take_profit - entry_price) / (entry_price - stop_loss) >= 2.0.\n"
                    "- Provide a specific recommendation to buy, sell, or hold, anchored in empirical market structure."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"### Target Instrument & Market Structure\n{instrument_context}\n\n"
                    f"### Research Manager Investment Plan\n{investment_plan}\n\n"
                    f"### Execution Assignment\n"
                    f"Evaluate execution feasibility for `{company_name}`. Provide concrete action, reasoning, entry price, stop loss, take profit, and sizing guidance."
                ),
            },
        ]

        typed_proposal = None
        if structured_llm is not None:
            try:
                typed_proposal = structured_llm.invoke(messages)
                trader_plan = render_trader_proposal(typed_proposal)
            except Exception:
                trader_plan = invoke_structured_or_freetext(
                    None, llm, messages, render_trader_proposal, "Trader"
                )
        else:
            trader_plan = invoke_structured_or_freetext(
                None, llm, messages, render_trader_proposal, "Trader"
            )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "trader_proposal": typed_proposal,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
