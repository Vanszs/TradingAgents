"""LangGraph tool wrapper for Exa Time-Travel & SearXNG search."""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.exa_search import ExaTimeTravelSearch

_exa_client = ExaTimeTravelSearch()


@tool
def get_web_search(
    query: Annotated[
        str,
        (
            "Specific search query. ALWAYS include the ticker/coin name. "
            "Good examples: 'NVDA earnings Q1 2026 revenue', "
            "'BBRI dividen interim 2024 jadwal', "
            "'Bitcoin ETF approval SEC news'. "
            "Bad examples: 'latest news', 'market update' (too generic)."
        ),
    ],
    trade_date: Annotated[
        Optional[str],
        "Trading date context (YYYY-MM-DD). If omitted, inferred from state/runtime config.",
    ] = None,
    category: Annotated[
        Literal["news", "general"],
        "Search category: 'news' for recent articles/headlines, 'general' for analysis/docs/whitepapers",
    ] = "news",
) -> str:
    """Search the web for news, company announcements, and financial analysis.

    Uses Exa.ai time-travel search to guarantee zero lookahead leakage when a
    historical trade_date is provided. Automatically caches responses locally.
    """
    active_date = trade_date or get_config().get("trade_date")
    return _exa_client.search(
        query=query,
        trade_date=active_date,
        num_results=5,
        category=category,
    )
