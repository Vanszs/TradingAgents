"""End-to-end LangGraph pipeline tests for the BluesMind provider.

Verifies the full TradingAgentsGraph execution path — from config loading
through LangGraph node traversal to final decision — when configured with
provider="bluesmind" and model="moonshotai/kimi-k2.6".

All LLM calls and data-fetching are mocked so no real API keys or network
access are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSerializable

from tradingagents.default_config import DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Shared mock LLM — same pattern as test_e2e_smoke.py
# ---------------------------------------------------------------------------

class _MockLLM(RunnableSerializable):
    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input, config=None, **kwargs):
        return AIMessage(
            content="BluesMind mock report. Rating: Hold",
            tool_calls=[],
            id="bm-mock-id",
        )

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("mock does not support structured output")


def _mock_create_llm_client(*args, **kwargs):
    client = MagicMock()
    client.get_llm.return_value = _MockLLM()
    return client


_PATCH_LLM     = "tradingagents.graph.trading_graph.create_llm_client"
_PATCH_YF      = "tradingagents.graph.trading_graph.yf"
_PATCH_PENDING = "tradingagents.agents.utils.memory.TradingMemoryLog.get_pending_entries"
_PATCH_CONTEXT = "tradingagents.agents.utils.memory.TradingMemoryLog.get_past_context"
_PATCH_STORE   = "tradingagents.agents.utils.memory.TradingMemoryLog.store_decision"

VALID_DECISIONS = {"Buy", "Overweight", "Hold", "Underweight", "Sell"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bluesmind_config(tmp_path):
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"]          = "bluesmind"
    config["deep_think_llm"]        = "moonshotai/kimi-k2.6"
    config["quick_think_llm"]       = "moonshotai/kimi-k2.6"
    config["backend_url"]           = None          # use provider default
    config["max_debate_rounds"]     = 1
    config["max_risk_discuss_rounds"] = 1
    config["checkpoint_enabled"]    = False
    config["results_dir"]           = str(tmp_path / "results")
    config["data_cache_dir"]        = str(tmp_path / "cache")
    return config


@pytest.fixture()
def bluesmind_graph(bluesmind_config, monkeypatch):
    monkeypatch.setenv("BLUESMIND_API_KEY", "sk-e2e-test-key")
    with patch(_PATCH_LLM, side_effect=_mock_create_llm_client), \
         patch(_PATCH_YF), \
         patch(_PATCH_PENDING, return_value=[]), \
         patch(_PATCH_CONTEXT, return_value=""), \
         patch(_PATCH_STORE):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        yield TradingAgentsGraph(debug=False, config=bluesmind_config)


# ---------------------------------------------------------------------------
# 1. Config layer — bluesmind config is accepted without error
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestBluesmindConfig:
    def test_config_provider_is_bluesmind(self, bluesmind_config):
        assert bluesmind_config["llm_provider"] == "bluesmind"

    def test_config_model_is_kimi(self, bluesmind_config):
        assert bluesmind_config["deep_think_llm"] == "moonshotai/kimi-k2.6"
        assert bluesmind_config["quick_think_llm"] == "moonshotai/kimi-k2.6"

    def test_graph_instantiates_with_bluesmind(self, bluesmind_graph):
        assert bluesmind_graph is not None

    def test_graph_has_deep_and_quick_llm(self, bluesmind_graph):
        assert bluesmind_graph.deep_thinking_llm is not None
        assert bluesmind_graph.quick_thinking_llm is not None

    def test_llm_client_called_with_bluesmind_provider(self, bluesmind_config, monkeypatch):
        """create_llm_client must be called with provider='bluesmind'."""
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test")
        calls = []
        def capturing_factory(provider, model, base_url=None, **kw):
            calls.append({"provider": provider, "model": model})
            return _mock_create_llm_client()
        with patch(_PATCH_LLM, side_effect=capturing_factory), \
             patch(_PATCH_YF), \
             patch(_PATCH_PENDING, return_value=[]), \
             patch(_PATCH_CONTEXT, return_value=""), \
             patch(_PATCH_STORE):
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            TradingAgentsGraph(debug=False, config=bluesmind_config)
        assert all(c["provider"] == "bluesmind" for c in calls)
        assert all(c["model"] == "moonshotai/kimi-k2.6" for c in calls)


# ---------------------------------------------------------------------------
# 2. LangGraph topology — graph structure is correct
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestBluesmindGraphTopology:
    def test_graph_has_required_nodes(self, bluesmind_graph):
        nodes = set(bluesmind_graph.graph.nodes.keys())
        for required in ("Trader", "Portfolio Manager", "Research Manager",
                         "Bull Researcher", "Bear Researcher"):
            assert required in nodes, f"missing node: {required}"

    def test_graph_has_analyst_tool_nodes(self, bluesmind_graph):
        assert "market" in bluesmind_graph.tool_nodes
        assert "fundamentals" in bluesmind_graph.tool_nodes
        assert "news" in bluesmind_graph.tool_nodes
        assert "social" in bluesmind_graph.tool_nodes

    def test_graph_topology_matches_openai_baseline(self, bluesmind_config, tmp_path, monkeypatch):
        """bluesmind graph must have identical node set as openai graph."""
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

        openai_config = DEFAULT_CONFIG.copy()
        openai_config.update({
            "llm_provider": "openai",
            "deep_think_llm": "gpt-5.4",
            "quick_think_llm": "gpt-5.4-mini",
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "checkpoint_enabled": False,
            "results_dir": str(tmp_path / "openai_results"),
            "data_cache_dir": str(tmp_path / "openai_cache"),
        })

        with patch(_PATCH_LLM, side_effect=_mock_create_llm_client), \
             patch(_PATCH_YF), \
             patch(_PATCH_PENDING, return_value=[]), \
             patch(_PATCH_CONTEXT, return_value=""), \
             patch(_PATCH_STORE):
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            g_openai = TradingAgentsGraph(debug=False, config=openai_config)
            g_bluesmind = TradingAgentsGraph(debug=False, config=bluesmind_config)

        assert set(g_openai.graph.nodes.keys()) == set(g_bluesmind.graph.nodes.keys())


# ---------------------------------------------------------------------------
# 3. Full pipeline propagation — stock
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestBluesmindStockPipeline:
    def test_propagate_returns_valid_decision(self, bluesmind_graph):
        _, decision = bluesmind_graph.propagate("NVDA", "2026-01-15", asset_type="stock")
        assert decision in VALID_DECISIONS

    def test_propagate_populates_all_analyst_reports(self, bluesmind_graph):
        final_state, _ = bluesmind_graph.propagate("NVDA", "2026-01-15", asset_type="stock")
        for key in ("market_report", "sentiment_report", "news_report", "fundamentals_report"):
            assert key in final_state, f"missing state key: {key}"

    def test_propagate_final_decision_not_empty(self, bluesmind_graph):
        final_state, decision = bluesmind_graph.propagate("NVDA", "2026-01-15", asset_type="stock")
        assert final_state["final_trade_decision"] != ""
        assert decision != ""

    def test_propagate_returns_tuple(self, bluesmind_graph):
        result = bluesmind_graph.propagate("NVDA", "2026-01-15", asset_type="stock")
        assert isinstance(result, tuple) and len(result) == 2

    def test_propagate_non_us_ticker(self, bluesmind_graph):
        """Non-US ticker (TSM) must also complete without error."""
        _, decision = bluesmind_graph.propagate("TSM", "2026-01-15", asset_type="stock")
        assert decision in VALID_DECISIONS


# ---------------------------------------------------------------------------
# 4. Full pipeline propagation — crypto
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestBluesmindCryptoPipeline:
    def test_crypto_propagate_returns_valid_decision(self, bluesmind_graph):
        _, decision = bluesmind_graph.propagate("BTC-USD", "2026-01-15", asset_type="crypto")
        assert decision in VALID_DECISIONS

    def test_crypto_fundamentals_report_not_empty(self, bluesmind_graph):
        final_state, _ = bluesmind_graph.propagate("BTC-USD", "2026-01-15", asset_type="crypto")
        assert final_state["fundamentals_report"] != ""


# ---------------------------------------------------------------------------
# 5. Debate rounds — config is respected
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestBluesmindDebateRounds:
    def test_single_debate_round_completes(self, bluesmind_graph):
        """max_debate_rounds=1 must complete without hanging."""
        _, decision = bluesmind_graph.propagate("AAPL", "2026-01-15", asset_type="stock")
        assert decision in VALID_DECISIONS

    def test_two_debate_rounds_complete(self, bluesmind_config, tmp_path, monkeypatch):
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test")
        config = bluesmind_config.copy()
        config["max_debate_rounds"] = 2
        config["max_risk_discuss_rounds"] = 2
        with patch(_PATCH_LLM, side_effect=_mock_create_llm_client), \
             patch(_PATCH_YF), \
             patch(_PATCH_PENDING, return_value=[]), \
             patch(_PATCH_CONTEXT, return_value=""), \
             patch(_PATCH_STORE):
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            graph = TradingAgentsGraph(debug=False, config=config)
        _, decision = graph.propagate("AAPL", "2026-01-15", asset_type="stock")
        assert decision in VALID_DECISIONS


# ---------------------------------------------------------------------------
# 6. Missing API key — must fail fast with clear error
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBluesmindMissingKey:
    def test_missing_key_raises_on_get_llm(self, monkeypatch):
        monkeypatch.delenv("BLUESMIND_API_KEY", raising=False)
        from tradingagents.llm_clients.factory import create_llm_client
        client = create_llm_client("bluesmind", "moonshotai/kimi-k2.6")
        with pytest.raises(ValueError, match="BLUESMIND_API_KEY"):
            client.get_llm()

    def test_missing_key_error_message_names_env_var(self, monkeypatch):
        monkeypatch.delenv("BLUESMIND_API_KEY", raising=False)
        from tradingagents.llm_clients.factory import create_llm_client
        client = create_llm_client("bluesmind", "moonshotai/kimi-k2.6")
        try:
            client.get_llm()
        except ValueError as exc:
            assert "BLUESMIND_API_KEY" in str(exc)
        else:
            pytest.fail("Expected ValueError not raised")


# ---------------------------------------------------------------------------
# 7. Env-var override — TRADINGAGENTS_LLM_PROVIDER=bluesmind
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBluesmindEnvOverride:
    def test_env_override_sets_provider(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "bluesmind")
        monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "moonshotai/kimi-k2.6")
        monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "moonshotai/kimi-k2.6")
        import importlib

        import tradingagents.default_config as dc
        config = importlib.reload(dc).DEFAULT_CONFIG
        assert config["llm_provider"] == "bluesmind"
        assert config["deep_think_llm"] == "moonshotai/kimi-k2.6"
        assert config["quick_think_llm"] == "moonshotai/kimi-k2.6"
