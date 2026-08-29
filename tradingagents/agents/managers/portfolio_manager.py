"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from tradingagents.agents.schemas import (
    EntryMode,
    PortfolioDecision,
    PortfolioRating,
    SignalContract,
    TraderAction,
    portfolio_decision_to_signal_contract,
    render_pm_decision,
)

logger = logging.getLogger(__name__)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
    invoke_structured_with_recovery,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        trade_date = state.get("trade_date", "")
        instrument_context = build_instrument_context(
            state["company_of_interest"],
            state.get("asset_type", "stock"),
            trade_date=trade_date,
        )

        risk_debate_state = state.get("risk_debate_state", {})
        history = risk_debate_state.get("history", "")
        research_plan = state.get("investment_plan", "")
        trader_plan = state.get("trader_investment_plan", "")
        trader_proposal = state.get("trader_proposal")

        # Identify if Trader proposed WNS (Rule 1: Strict Execution Hierarchy)
        trader_is_wns = False
        if trader_proposal is not None:
            if getattr(trader_proposal, "action", None) == TraderAction.WNS:
                trader_is_wns = True
        elif trader_plan:
            if re.search(r"FINAL TRANSACTION PROPOSAL:\s*\*\*WNS\*\*|\*\*Action\*\*:\s*WNS|\bAction:\s*WNS\b", str(trader_plan), re.IGNORECASE):
                trader_is_wns = True

        # Planned entry price and mode resolution
        planned_entry_price = None
        resolved_entry_mode = EntryMode.T1_OPEN
        if trader_proposal is not None:
            if getattr(trader_proposal, "action", None) == TraderAction.BUY_LIMIT and trader_proposal.entry_price is not None:
                planned_entry_price = trader_proposal.entry_price
                resolved_entry_mode = EntryMode.T1_LIMIT
            elif getattr(trader_proposal, "action", None) in (TraderAction.BUY_MARKET, TraderAction.BUY):
                resolved_entry_mode = EntryMode.T1_OPEN
        elif trader_plan and re.search(r"FINAL TRANSACTION PROPOSAL:\s*\*\*BUY LIMIT\*\*|\*\*Action\*\*:\s*Buy Limit|\bAction:\s*Buy Limit\b", str(trader_plan), re.IGNORECASE):
            resolved_entry_mode = EntryMode.T1_LIMIT

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        company_name = state["company_of_interest"]
        prompt = f"""You are the Chief Investment Officer making the final capital allocation decision for `{company_name}` under a strict **Spot Long-Only (BUY vs WNS)** mandate.

{instrument_context}

{lessons_line}
### Research Plan
{research_plan}

### Trader Proposal
{trader_plan}

### Risk Committee Debate
{history if history else 'No risk debate history.'}

**Strict Execution Hierarchy Governance**:
1. **Trader WNS Invariant (Absolute Rule)**: If the Trader Proposal is **WNS**, the Portfolio Manager **CANNOT** override or upgrade it into a BUY. You MUST issue **WNS**. Capital cannot be deployed when the execution trader identifies an unviable entry, falling knife breakdown, or lack of mathematical edge.
2. **Trader BUY Evaluation**: If the Trader Proposal is **BUY** (Market or Limit), the Portfolio Manager may either:
   - **Accept BUY**: Authorize execution if the risk committee confirms favorable mathematical expectancy (R:R >= 2:1) and structural asymmetry.
   - **Downgrade to WNS**: Veto/downgrade to WNS if risk is excessive, solvency is deteriorating, or tail risk is unacceptable.

**Capital Allocation Governance**:
- **BUY (Market or Limit)**: Authorize ONLY when Trader proposed BUY and there is positive mathematical expectancy:
  1. **Momentum Breakout & Consolidation in Uptrend**: Reclaiming key dynamic levels or consolidating in an uptrend with targets anchored to Fibonacci Extensions (1.272x, 1.618x) or +3x ATR channels and R:R >= 2:1. For assets with < 180 bars (new IPOs), evaluate based on 20D/60D swing base rather than penalizing lack of 200 SMA. Do not veto breakout setups just because price is near a prior high if structural R:R to Fib extension is positive.
  2. **Bullish Trend Pullback / Dip in Uptrend**: Asset in an uptrend pulling back to support (Fib 50%/61.8%, 20D swing low) with RSI 30–48 and R:R >= 2:1.
  3. **Capitulation Reversal**: Extreme oversold at 52W support (RSI < 28 or positive Kronos forecast) and R:R >= 2.5:1 with tight stop loss.
  *(If Trader proposal sizing exceeds risk guidelines, adjust/scale down position size to 5–10%, do NOT veto mathematically valid trades to WNS solely due to sizing).*
- **WNS (Wait and See - Strict Invalidation & Falling Knife Gate)**: You MUST issue WNS when:
  1. Trader proposed WNS (Mandatory Rule 1 Invariant).
  2. Asset is a True Falling Knife breakdown below 200 SMA / major support with RSI 35–55 without a structural base.
  3. Overbought exhaustion at resistance where R:R < 1.8:1.
  4. Natural structural R:R < 1.8:1 or market trend is choppy / indeterminate.
  When issuing WNS, you MUST define `wns_recheck_date` (catalyst date YYYY-MM-DD) and/or `wns_trigger_price` (structural level).

Output your decision strictly matching the PortfolioDecision schema.{get_language_instruction()}"""

        typed_decision = None
        if structured_llm is not None:
            try:
                typed_decision = structured_llm.invoke(prompt)
                final_trade_decision = render_pm_decision(typed_decision)
            except Exception:
                typed_decision, final_trade_decision = invoke_structured_with_recovery(
                    None, llm, prompt, PortfolioDecision, render_pm_decision, "Portfolio Manager"
                )
        else:
            typed_decision, final_trade_decision = invoke_structured_with_recovery(
                None, llm, prompt, PortfolioDecision, render_pm_decision, "Portfolio Manager"
            )

        # Rule 1 Invariant Enforcement: Downgrade to WNS if Trader proposed WNS
        if trader_is_wns:
            if typed_decision is not None and typed_decision.rating == PortfolioRating.BUY:
                logger.warning(
                    "Strict Execution Hierarchy Invariant Triggered: Trader proposed WNS for %s on %s. "
                    "Portfolio Manager cannot override Trader WNS to BUY. Downgrading PM decision to WNS.",
                    company_name, trade_date,
                )
                typed_decision.rating = PortfolioRating.WNS
                typed_decision.stop_loss = None
                typed_decision.take_profit = None
                typed_decision.price_target = None
                typed_decision.planned_entry_price = None
                if trader_proposal is not None:
                    if not typed_decision.wns_recheck_date and trader_proposal.wns_recheck_date:
                        typed_decision.wns_recheck_date = trader_proposal.wns_recheck_date
                    if typed_decision.wns_trigger_price is None and trader_proposal.wns_trigger_price is not None:
                        typed_decision.wns_trigger_price = trader_proposal.wns_trigger_price
                    if typed_decision.wns_condition_type is None and trader_proposal.wns_condition_type is not None:
                        typed_decision.wns_condition_type = trader_proposal.wns_condition_type
                if not typed_decision.wns_recheck_date and typed_decision.wns_trigger_price is None:
                    try:
                        typed_decision.wns_recheck_date = (pd.to_datetime(trade_date) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
                    except Exception:
                        typed_decision.wns_recheck_date = trade_date
                final_trade_decision = render_pm_decision(typed_decision)
            elif typed_decision is None and final_trade_decision:
                final_trade_decision = re.sub(
                    r"(\b\*\*Rating\*\*:\s*|\bRating:\s*)(Buy|Overweight)",
                    r"\1WNS",
                    final_trade_decision,
                    flags=re.IGNORECASE,
                )

        signal_contract = None
        if typed_decision is not None and trade_date:
            if typed_decision.rating == PortfolioRating.BUY:
                typed_decision.entry_mode = resolved_entry_mode
                if resolved_entry_mode == EntryMode.T1_OPEN:
                    typed_decision.planned_entry_price = None

            # Deterministic Python date calculation if missing
            if not typed_decision.next_review_date:
                days_delta = 7 if typed_decision.rating == PortfolioRating.BUY else 21
                try:
                    typed_decision.next_review_date = (pd.to_datetime(trade_date) + pd.Timedelta(days=days_delta)).strftime("%Y-%m-%d")
                except Exception:
                    typed_decision.next_review_date = trade_date

            try:
                signal_contract = portfolio_decision_to_signal_contract(
                    typed_decision,
                    state["company_of_interest"],
                    trade_date,
                    planned_entry_price=planned_entry_price,
                )
            except Exception as exc:
                logger.error("Failed to build SignalContract for %s on %s: %s. Building fallback WNS contract.", state.get("company_of_interest"), trade_date, exc)
                recheck = (pd.to_datetime(trade_date) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
                signal_contract = SignalContract(
                    ticker=state["company_of_interest"],
                    signal_date=trade_date,
                    rating=PortfolioRating.WNS,
                    action="WNS",
                    time_horizon_days=20,
                    confidence=0.7,
                    wns_recheck_date=recheck,
                    thesis_summary="Automated fallback WNS contract due to contract conversion recovery.",
                )
        elif trade_date:
            # Failsafe fallback contract when typed_decision was None
            recheck = (pd.to_datetime(trade_date) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            signal_contract = SignalContract(
                ticker=state["company_of_interest"],
                signal_date=trade_date,
                rating=PortfolioRating.WNS,
                action="WNS",
                time_horizon_days=20,
                confidence=0.7,
                wns_recheck_date=recheck,
                thesis_summary="Fallback WNS signal contract from text report recovery.",
            )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state.get("history", ""),
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state.get("count", 0),
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "signal_contract": signal_contract,
        }

    return portfolio_manager_node
