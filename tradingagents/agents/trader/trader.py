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
        instrument_context = build_instrument_context(company_name, asset_type)
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent converting the Research Manager's plan into a concrete transaction. "
                    "Provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the research plan provided below.\n\n"
                    "IMPORTANT for price fields (entry_price, stop_loss, take_profit):\n"
                    "- Always provide specific numerical values when possible.\n"
                    "- NEVER output the string 'None' — either provide a number or omit the field entirely.\n"
                    "- If you cannot determine a price, leave the field empty (null), do not write 'None'."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The Research Manager provided this investment plan for {company_name}. "
                    f"{instrument_context} Convert it into an executable transaction proposal.\n\n"
                    f"Research Manager Investment Plan: {investment_plan}\n\n"
                    f"Provide the action, reasoning, price levels, and sizing guidance."
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
