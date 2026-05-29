"""Crypto Fundamentals Analyst — analyzes tokenomics, dev activity, and market sentiment.

Replaces the stock fundamentals analyst for crypto assets (asset_type='crypto').
Uses CoinGecko, GitHub (via CoinGecko dev data), and Alternative.me Fear & Greed.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.crypto_fundamental_tools import (
    get_crypto_dev_activity,
    get_crypto_market_sentiment,
    get_crypto_network_metrics,
    get_crypto_onchain_news,
    get_crypto_tokenomics,
)
from tradingagents.agents.utils.web_search_tools import get_web_search


def create_crypto_fundamentals_analyst(llm):
    """Create a crypto fundamentals analyst node for the trading graph.

    Analyzes: tokenomics, supply dynamics, developer activity, network metrics,
    and market sentiment (Fear & Greed, BTC dominance).
    """
    tools = [
        get_crypto_tokenomics,
        get_crypto_dev_activity,
        get_crypto_network_metrics,
        get_crypto_market_sentiment,
        get_crypto_onchain_news,
        get_web_search,
    ]

    system_message = (
        "You are a crypto fundamentals researcher analyzing a digital asset. "
        "Unlike traditional company fundamentals, crypto fundamentals focus on:\n"
        "1. **Tokenomics**: supply dynamics (circulating vs max supply), inflation/deflation, "
        "market cap, fully diluted valuation, price performance vs ATH\n"
        "2. **Developer Activity**: GitHub commits (last 4 weeks), contributors, stars/forks, "
        "project momentum and health\n"
        "3. **Network Metrics**: 30-day price trend, volume analysis, market position\n"
        "4. **Market Sentiment**: Fear & Greed Index (0-100), BTC dominance trends, "
        "social community size\n"
        "5. **On-Chain Metrics & News**: Etherscan data (supply, burns, token contracts) "
        "and RSS news sentiment for EVM-based assets\n\n"
        "Use the available tools: `get_crypto_tokenomics`, `get_crypto_dev_activity`, "
        "`get_crypto_network_metrics`, `get_crypto_market_sentiment`, `get_crypto_onchain_news`. "
        "Use `get_web_search(query)` for the latest real-time context not covered by the other "
        "tools — exchange listings, regulatory news, protocol upgrades, hacks, or partnerships.\n\n"
        "Apply this evidence-based grading rubric:\n"
        "- **Tokenomics**: Is supply deflationary (capped max supply)? "
        "What % is circulating? High supply ratio = less future dilution risk.\n"
        "- **Dev Activity**: >50 commits/4w = very active; 20-50 = active; "
        "5-20 = moderate; <5 = concerning (possible abandonment).\n"
        "- **Sentiment**: Fear & Greed <25 = extreme fear (contrarian buy signal possible); "
        ">75 = extreme greed (caution, possible top).\n"
        "- **BTC Dominance**: Rising dominance = altcoin headwinds; "
        "falling dominance = altcoin season potential.\n"
        "- **On-Chain Supply**: Compare on-chain circulating supply vs CoinGecko reported supply. "
        "Large discrepancy may indicate locked/burned tokens. "
        "For EVM tokens: contract address confirms legitimacy.\n\n"
        "Write a comprehensive report covering all four areas. "
        "Append a Markdown table at the end summarizing key metrics with grades. "
        "Provide specific, actionable insights to help traders make informed decisions."
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
                " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}"
                "For your reference, the current date is {current_date}. {instrument_context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))

    def crypto_fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker, asset_type="crypto")

        filled_prompt = prompt.partial(
            current_date=current_date,
            instrument_context=instrument_context,
        )
        chain = filled_prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = result.content or ""
        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return crypto_fundamentals_analyst_node
