"""End-to-end smoke tests for the TradingAgents pipeline.

These tests mock the LLM and data-fetching layers so they run fast,
require no API keys, and make no HTTP calls.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSerializable

from tradingagents.default_config import DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

class _MockLLM(RunnableSerializable):
    """Mock LLM that is a proper Runnable, satisfies bind_tools and invoke."""

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input, config=None, **kwargs):
        return AIMessage(
            content="Mock analyst report. Rating: Hold",
            tool_calls=[],
            id="mock-msg-id",
        )

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("mock does not support structured output")


def _mock_create_llm_client(*args, **kwargs):
    client = MagicMock()
    client.get_llm.return_value = _MockLLM()
    return client


# Patch targets — must patch where the name is looked up
_PATCH_LLM = "tradingagents.graph.trading_graph.create_llm_client"
_PATCH_MEMORY_PENDING = "tradingagents.agents.utils.memory.TradingMemoryLog.get_pending_entries"
_PATCH_MEMORY_CONTEXT = "tradingagents.agents.utils.memory.TradingMemoryLog.get_past_context"
_PATCH_MEMORY_STORE = "tradingagents.agents.utils.memory.TradingMemoryLog.store_decision"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def patched_graph(tmp_path):
    """Yield a TradingAgentsGraph with mocked LLM and no real data calls."""
    with patch(_PATCH_LLM, side_effect=_mock_create_llm_client), \
         patch("tradingagents.graph.trading_graph.yf"), \
         patch(_PATCH_MEMORY_PENDING, return_value=[]), \
         patch(_PATCH_MEMORY_CONTEXT, return_value=""), \
         patch(_PATCH_MEMORY_STORE):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1
        config["checkpoint_enabled"] = False
        config["results_dir"] = str(tmp_path / "results")
        config["data_cache_dir"] = str(tmp_path / "cache")
        yield TradingAgentsGraph(debug=False, config=config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

VALID_DECISIONS = {"Buy", "Overweight", "Hold", "Underweight", "Sell"}


@pytest.mark.smoke
def test_e2e_stock_pipeline_returns_decision(patched_graph):
    """Full NVDA pipeline returns a valid decision with all state fields populated."""
    final_state, decision = patched_graph.propagate("NVDA", "2026-01-15", asset_type="stock")

    assert decision in VALID_DECISIONS
    for key in ("market_report", "sentiment_report", "news_report", "fundamentals_report"):
        assert key in final_state
    assert "final_trade_decision" in final_state
    assert final_state["final_trade_decision"] != ""


@pytest.mark.smoke
def test_e2e_crypto_pipeline_returns_decision(patched_graph):
    """Full BTC-USD pipeline returns a decision; fundamentals_report is NOT empty."""
    final_state, decision = patched_graph.propagate("BTC-USD", "2026-01-15", asset_type="crypto")

    assert decision in VALID_DECISIONS
    # After Phase 0, crypto fundamentals should NOT be skipped
    assert final_state["fundamentals_report"] != ""


@pytest.mark.smoke
def test_e2e_stock_pipeline_uses_yfinance_fundamentals(patched_graph):
    """Regression: stock tool_nodes['fundamentals'] contains stock fundamental tools."""
    tool_node = patched_graph.tool_nodes["fundamentals"]
    tool_names = set(tool_node.tools_by_name.keys())
    expected = {"get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"}
    assert expected.issubset(tool_names)


@pytest.mark.smoke
def test_e2e_crypto_uses_crypto_fundamentals_tools():
    """Auto-skips until crypto_fundamental_tools module exists (Phase 1)."""
    pytest.importorskip("tradingagents.agents.utils.crypto_fundamental_tools")
    from tradingagents.agents.utils import crypto_fundamental_tools
    assert hasattr(crypto_fundamental_tools, "get_crypto_fundamentals")


@pytest.mark.smoke
def test_graph_topology_identical_for_stock_and_crypto(tmp_path):
    """Both asset types produce the same graph node names."""
    with patch(_PATCH_LLM, side_effect=_mock_create_llm_client), \
         patch("tradingagents.graph.trading_graph.yf"), \
         patch(_PATCH_MEMORY_PENDING, return_value=[]), \
         patch(_PATCH_MEMORY_CONTEXT, return_value=""), \
         patch(_PATCH_MEMORY_STORE):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1
        config["checkpoint_enabled"] = False
        config["results_dir"] = str(tmp_path / "results")
        config["data_cache_dir"] = str(tmp_path / "cache")

        graph_stock = TradingAgentsGraph(debug=False, config=config)
        graph_crypto = TradingAgentsGraph(debug=False, config=config)

    stock_nodes = set(graph_stock.graph.nodes.keys())
    crypto_nodes = set(graph_crypto.graph.nodes.keys())
    assert stock_nodes == crypto_nodes
    assert "Portfolio Manager" in stock_nodes
    assert "Trader" in stock_nodes
