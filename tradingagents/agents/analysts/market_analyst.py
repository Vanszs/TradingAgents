from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_kronos_forecast,
    get_language_instruction,
    get_stock_data,
)


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(
            state["company_of_interest"], asset_type, trade_date=current_date
        )

        if asset_type == "crypto":
            analysis_note = " Note: For crypto, OHLCV data reflects 24/7 trading. Volume spikes and weekend gaps are normal. Consider that crypto markets have no circuit breakers."
        else:
            analysis_note = ""

        tools = [get_stock_data, get_indicators]
        from tradingagents.dataflows.config import get_config
        if get_config().get("kronos_enabled", False):
            tools.append(get_kronos_forecast)

        system_message = (
            """You are an Institutional Technical Market Analyst evaluating price action across Multi-Horizon Daily Structure (Macro 52W/200 SMA -> Intermediate 60D/50 SMA -> Tactical 20D/20 EMA/ATR) under a Spot Equity Long-Only mandate.

Analytical Workflow:
1. Identify Market Regime: Trending (Bull/Bear), Range-Bound Consolidation, Pullback/Correction, or Falling Knife (Breakdown below 200 SMA / major support without divergence).
2. If enabled, query the Kronos K-Line forecast with `get_kronos_forecast(symbol, curr_date, pred_days=20)`; report whether the engine is Official-Kronos or Statistical-Fallback.
3. Select complementary indicators (e.g. `close_10_ema`, `close_50_sma`, `close_200_sma`, `macd`, `rsi`, `boll`, `atr`, `vwma`) via `get_indicators(symbol, indicator, curr_date, look_back_days)`.
4. Compare classical Fibonacci/moving averages against the Kronos forecast trajectory, without calling a statistical result neural.
5. Invalidation & Asymmetry: Define structural breakdown level (Stop Loss) strictly below support floors to guarantee Risk-to-Reward (R:R) >= 2:1. For Momentum Breakouts / MA Reclaims (reclaiming 20 EMA, 50 SMA, 200 SMA), identify multi-stage forward expansion targets (Fib Extensions 1.272x / 1.618x, 60D High, +2x/+3x ATR channels) rather than capping upside at minor local resistance. For sub-200 SMA breakdowns without support, explicitly flag "Falling Knife / Invalidation Regime".

Conclude with a structured Markdown table summarizing: Metric / Level, Horizon (Macro / Intermediate / Tactical), Price / Value, Bias (Bullish/Bearish/Neutral), and Technical Implication (Support Floor / Invalidation / Expansion Target / Neural Target / Falling Knife)."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Produce an analyst report only; leave the final transaction proposal to the Trader and Portfolio Manager."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}{analysis_note}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(analysis_note=analysis_note)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
