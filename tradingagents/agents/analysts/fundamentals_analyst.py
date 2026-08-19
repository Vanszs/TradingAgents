from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_exchange_filing_context,
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_language_instruction,
)
from tradingagents.agents.utils.web_search_tools import get_web_search


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(ticker, asset_type)
        filing_context = build_exchange_filing_context(ticker, asset_type)

        from tradingagents.dataflows.config import get_config

        tools = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]
        web_search_guidance = ""
        if not get_config().get("backtest_mode", False):
            tools.append(get_web_search)
            web_search_guidance = " Use `get_web_search(query)` for forward guidance, capital raises, or material regulatory filings not in statements."

        system_message = (
            f"You are a fundamental equity analyst evaluating corporate solvency, cash-flow quality, and valuation for `{ticker}`. "
            "Analyze multi-year quarterly and annual financial statements. Focus on revenue growth, operating margin resilience, free cash flow yield, debt maturity, and capital returns. "
            "Tools: `get_fundamentals(ticker, curr_date)`, `get_balance_sheet(ticker, freq, curr_date)`, `get_cashflow(ticker, freq, curr_date)`, and `get_income_statement(ticker, freq, curr_date)`. "
            f"{web_search_guidance}"
            " Append a structured Markdown table at the end summarizing key metrics, valuation multiples, and balance sheet health."
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
