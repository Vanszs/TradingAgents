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
                    "1. **Buy Market (Standard Directional Entry - Momentum Breakout & High Consolidation)**: Default action for all momentum breakouts reclaiming dynamic levels (20 EMA, 50 SMA, 200 SMA) or consolidating in an uptrend below resistance. Action MUST be `Buy Market` (fill at T+1 Open, entry_price=None). "
                    "**Crucial Breakout Target Rule (Anti-Resistance Paralysis)**: When price is consolidating within 5% below a 52-Week High, 60D High, or major resistance in a macro uptrend (above rising 50/200 SMA with RSI 50–74), do NOT set Take Profit at the 52W High itself (which creates an artificially compressed sub-1:1 R:R). You MUST anchor Take Profit to **Fibonacci Extensions (1.272x or 1.618x)** or **+3x ATR Expansion Channel** to capture the impending breakout continuation, ensuring convex R:R >= 2.5:1.\n"
                    "2. **Buy Market / Tight Limit (Bullish Trend Pullback & Volatility-Aware Asymmetry)**: When an asset in a macro uptrend (above 200 SMA) pulls back to structural support (Fib 50%/61.8%, 20D swing low, or 50 SMA) with RSI in the 30–45 range:\n"
                    "   - **Entry**: Execute `Buy Market` (T+1 Open) or tight limit within 0.5x ATR.\n"
                    "   - **Volatility-Aware Stop Loss**: Place Stop Loss below the structural floor with a mandatory **1.0x to 1.5x ATR buffer** (e.g. SL = Support - 1.0x ATR) to prevent fatal noise whipsaws during market opens.\n"
                    "   - **Expansion Target (Anti-Capping)**: To maintain a robust R:R >= 2.5:1 with the wider volatility stop, do NOT cap Take Profit at minor immediate resistance. Anchor Take Profit to **major multi-stage expansion targets (Fib Extensions 1.272x/1.618x, 60D High, or +3x ATR channel)**.\n"
                    "3. **Buy Market (Capitulation Bottom Reversal)**: When an asset below 200 SMA tests a Major 52-Week Support Base with Extreme Oversold RSI < 28 AND Kronos confirms a BULLISH trajectory, execute `Buy Market` (T+1 Open) with tight Stop Loss below the 52W floor (Risk <= 5%, R:R >= 2.5:1).\n"
                    "4. **WNS (Wait and See - Strict Invalidation & Falling Knife Gate)**: Zero capital allocated today. You MUST choose WNS when:\n"
                    "   - The asset is in a True Falling Knife breakdown below the 200 SMA or major support WITHOUT a structural base, capitulation volume, or bullish divergence (e.g. RSI 35–55 with active distribution). Zero dip-buying in freefall.\n"
                    "   - The asset is at extreme overbought resistance (e.g. ATH with RSI > 75) where natural R:R < 1.5:1.\n"
                    "   - Natural structural R:R < 1.8:1 or market trend is choppy / indeterminate.\n"
                    "   Always specify `wns_recheck_date` and `wns_trigger_price` when issuing WNS.\n"
                    "5. **Static exits only**: Every BUY must provide fixed stop loss and take profit. Zero trailing or moving stops post-entry.\n\n"
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
