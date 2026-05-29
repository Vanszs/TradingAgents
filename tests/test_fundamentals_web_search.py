"""Verify the fundamentals analysts (stock + crypto) have web-search access.

``get_web_search`` should be bound to both fundamentals analysts and present
in the fundamentals ToolNode for both asset types, so the LLM can optionally
fetch real-time context not covered by the structured fundamental tools.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph


def _fundamentals_tool_names(asset_type: str) -> list[str]:
    """Build only the tool nodes (no LLM init / network) for the given asset."""
    g = object.__new__(TradingAgentsGraph)
    g.asset_type = asset_type
    nodes = g._create_tool_nodes()
    return list(nodes["fundamentals"].tools_by_name.keys())


@pytest.mark.unit
def test_stock_fundamentals_toolnode_has_web_search():
    names = _fundamentals_tool_names("stock")
    assert "get_web_search" in names
    # Structured fundamental tools still present (not replaced).
    assert "get_fundamentals" in names


@pytest.mark.unit
def test_crypto_fundamentals_toolnode_has_web_search():
    names = _fundamentals_tool_names("crypto")
    assert "get_web_search" in names
    assert "get_crypto_tokenomics" in names


@pytest.mark.unit
def test_stock_fundamentals_analyst_binds_web_search():
    from langchain_core.runnables import RunnableLambda

    captured = {}

    def _bind(tools):
        captured["tools"] = tools
        return RunnableLambda(lambda _pv: MagicMock(tool_calls=[]))

    llm = MagicMock()
    llm.bind_tools.side_effect = _bind

    from tradingagents.agents.analysts.fundamentals_analyst import (
        create_fundamentals_analyst,
    )

    node = create_fundamentals_analyst(llm)
    node({
        "trade_date": "2026-05-29",
        "company_of_interest": "AAPL",
        "messages": [],
    })
    assert "get_web_search" in [t.name for t in captured["tools"]]


@pytest.mark.unit
def test_crypto_fundamentals_analyst_binds_web_search():
    from langchain_core.runnables import RunnableLambda

    captured = {}

    def _bind(tools):
        captured["tools"] = tools
        return RunnableLambda(lambda _pv: MagicMock(content="report"))

    llm = MagicMock()
    llm.bind_tools.side_effect = _bind

    from tradingagents.agents.analysts.crypto_fundamentals_analyst import (
        create_crypto_fundamentals_analyst,
    )

    node = create_crypto_fundamentals_analyst(llm)
    node({
        "trade_date": "2026-05-29",
        "company_of_interest": "BTC-USD",
        "messages": [],
    })
    assert "get_web_search" in [t.name for t in captured["tools"]]
