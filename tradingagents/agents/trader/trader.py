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
                    f"You are the Senior Execution Trader translating the Research Plan for `{company_name}` into an actionable execution contract under a strict **Spot Long-Only** mandate.\n\n"
                    f"{instrument_context}\n\n"
                    "**Execution Taxonomy & Structural Expectancy Guidelines**:\n"
                    "1. **Buy Market**: Immediate market order at open if price is confirmed at major support with bullish momentum and natural structural asymmetry.\n"
                    "2. **Buy Limit**: Staged limit accumulation order at concrete structural demand floor Y (20D/60D Swing Low, Fib retracement, or dynamic support) with protective stop loss strictly below invalidation support.\n"
                    "3. **Structural Expectancy (Anti-Gaming)**: Anchor Take Profit at realistic structural resistance (Swing High, Chandelier Exit, or Fib extension). Anchor Stop Loss at key structural support. Do NOT invent unrealistic high targets to artificially force R:R; if natural structural R:R is unfavorable, you MUST choose **WNS (Wait and See)**.\n"
                    "4. **WNS (Wait and See)**: Zero capital allocated today. Specify `wns_recheck_date` (catalyst date YYYY-MM-DD) and/or `wns_trigger_price` (pullback demand zone level).\n"
                    "5. **Dynamic Horizon**: Calibrate `max_holding_days` (1–63 trading days) based on target distance relative to daily ATR.\n"
                    "6. **Sell**: Liquidate existing long inventory to 100% cash.\n\n"
                    "Deliver your proposal strictly matching the TraderProposal schema."
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
