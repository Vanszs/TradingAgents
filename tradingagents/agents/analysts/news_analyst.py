from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_exchange_filing_context,
    build_instrument_context,
    get_global_news,
    get_insider_transactions,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.web_search_tools import get_web_search


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(
            ticker, asset_type, trade_date=current_date
        )
        filing_context = build_exchange_filing_context(ticker, asset_type)

        from tradingagents.dataflows.config import get_config

        tools = [get_news, get_global_news]
        if asset_type != "crypto":
            tools.append(get_insider_transactions)
        web_search_guidance = ""
        if not get_config().get("backtest_mode", False):
            tools.append(get_web_search)
            web_search_guidance = " Use get_web_search(query) for real-time catalysts and breaking developments."

        system_message = (
            f"You are an institutional news and catalyst analyst evaluating `{ticker}` for Spot Equity Long-Only accumulation. "
            f"Use get_news(ticker, start_date, end_date) for {asset_label}-specific developments, get_insider_transactions(ticker) for insider accumulation, "
            f"and get_global_news(curr_date, look_back_days, limit) for macroeconomic context.{web_search_guidance}\n\n"
            "Catalyst & Valuation Framework:\n"
            "1. Classify catalysts: Structural Growth, Transitory Panic/Overreaction, Regulatory Clearance, or Fundamental Deterioration.\n"
            "2. Assess market pricing status: Fresh vs Priced-In vs Sentiment Divergence.\n"
            "3. If price is pulling back on non-fatal noise, identify catalyst-backed Limit Accumulation opportunity near key structural support.\n"
            "Append a structured Markdown table summarizing: Catalyst Event, Date/Source, Impact (Bullish/Bearish), Pricing Status, and Accumulation Implication."
            + f"\n\n{filing_context}"
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
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
