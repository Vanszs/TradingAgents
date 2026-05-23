"""LangGraph tool wrapper for SearXNG web search."""
from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.tools import tool

from tradingagents.dataflows.searxng import search


@tool
def get_web_search(
    query: Annotated[str, (
        "Specific search query. ALWAYS include the ticker/coin name. "
        "Good examples: 'NVDA earnings Q1 2026 revenue', "
        "'Bitcoin ETF approval SEC news 2026', "
        "'CARV token tokenomics supply schedule', "
        "'Ethereum Pectra upgrade impact'. "
        "Bad examples: 'latest news', 'market update' (too generic)."
    )],
    category: Annotated[
        Literal["news", "general"],
        "Search category: 'news' for recent articles/headlines, 'general' for analysis/docs/whitepapers"
    ] = "news",
) -> str:
    """Search the web for latest news and analysis using SearXNG.

    Use this tool when you need:
    - Recent news not covered by get_news() or get_global_news()
    - Real-time price analysis, analyst opinions, or community sentiment
    - Regulatory news, partnership announcements, or protocol upgrades
    - Any information that may be more recent than your training data

    Always include the asset name/ticker in the query for relevant results.
    """
    return search(query, category=category)
