from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
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

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are an Institutional Technical Market Analyst evaluating price action across a Multi-Timeframe hierarchy (1D Macro Trend confirmed by 1H Micro Structure) under a Spot Equity Long-Only mandate.

Analytical Workflow:
1. Identify Market Regime: Trending (Bull/Bear), Range-Bound Consolidation, or Pullback/Correction.
2. Select up to 8 complementary indicators (e.g. `close_10_ema`, `close_50_sma`, `close_200_sma`, `macd`, `rsi`, `boll`, `atr`, `vwma`) via `get_indicators(symbol, indicator, curr_date, look_back_days)`.
3. Use `get_stock_data(symbol, start_date, end_date)` if raw OHLCV bar analysis is required.
4. Quantify Support Floors & Accumulation Zones: Locate key swing lows (20D/60D), Fibonacci retracements (50%/61.8%), and intraday 1H micro swing floors where conditional Limit Buy orders can be staged.
5. Invalidation & Asymmetry: Define the structural breakdown level (Stop Loss) strictly below support floors to guarantee Risk-to-Reward (R:R) >= 2:1.

Conclude with a structured Markdown table summarizing: Metric / Level, Timeframe (1D/1H), Price / Value, Bias (Bullish/Bearish/Neutral), and Technical Implication (Support Floor / Invalidation)."""
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
