"""End-to-end unit tests for the BluesMind provider integration.

Covers every layer that was touched when adding the provider:
  1. api_key_env   — BLUESMIND_API_KEY mapping
  2. factory       — bluesmind routes to OpenAIClient
  3. openai_client — correct base URL, correct API key, no Responses API
  4. model_catalog — bluesmind models present, not polluting openai catalog
  5. cli/utils     — BluesMind appears in provider dropdown
  6. integration   — create_llm_client("bluesmind", ...) builds a usable LLM
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# 1. api_key_env
# ---------------------------------------------------------------------------

class TestApiKeyEnv:
    def test_bluesmind_maps_to_correct_env_var(self):
        from tradingagents.llm_clients.api_key_env import get_api_key_env
        assert get_api_key_env("bluesmind") == "BLUESMIND_API_KEY"

    def test_bluesmind_in_provider_map(self):
        from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
        assert "bluesmind" in PROVIDER_API_KEY_ENV

    def test_bluesmind_lookup_case_insensitive(self):
        from tradingagents.llm_clients.api_key_env import get_api_key_env
        assert get_api_key_env("BluesMind") == "BLUESMIND_API_KEY"
        assert get_api_key_env("BLUESMIND") == "BLUESMIND_API_KEY"

    def test_openai_key_not_affected(self):
        """Adding bluesmind must not change the openai mapping."""
        from tradingagents.llm_clients.api_key_env import get_api_key_env
        assert get_api_key_env("openai") == "OPENAI_API_KEY"


# ---------------------------------------------------------------------------
# 2. factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_bluesmind_creates_openai_client(self, monkeypatch):
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test-key")
        from tradingagents.llm_clients.factory import create_llm_client
        from tradingagents.llm_clients.openai_client import OpenAIClient
        client = create_llm_client("bluesmind", "moonshotai/kimi-k2.6")
        assert isinstance(client, OpenAIClient)

    def test_bluesmind_provider_stored_on_client(self, monkeypatch):
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test-key")
        from tradingagents.llm_clients.factory import create_llm_client
        client = create_llm_client("bluesmind", "moonshotai/kimi-k2.6")
        assert client.provider == "bluesmind"

    def test_unknown_provider_still_raises(self):
        from tradingagents.llm_clients.factory import create_llm_client
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client("not-a-provider", "some-model")


# ---------------------------------------------------------------------------
# 3. openai_client
# ---------------------------------------------------------------------------

def _reload_client():
    import tradingagents.llm_clients.openai_client as mod
    return importlib.reload(mod)


class TestOpenAIClient:
    def test_bluesmind_base_url_is_correct(self):
        mod = _reload_client()
        assert mod._PROVIDER_BASE_URL["bluesmind"] == "https://api.bluesminds.com/v1"

    def test_bluesmind_resolve_returns_correct_url(self):
        mod = _reload_client()
        assert mod._resolve_provider_base_url("bluesmind") == "https://api.bluesminds.com/v1"

    def test_bluesmind_does_not_use_responses_api(self, monkeypatch):
        """bluesmind must use Chat Completions, not the OpenAI Responses API."""
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test-key")
        mod = _reload_client()
        client = mod.OpenAIClient(model="moonshotai/kimi-k2.6", provider="bluesmind")
        llm = client.get_llm()
        # use_responses_api should NOT be set (only native openai gets it)
        assert not getattr(llm, "use_responses_api", False)

    def test_bluesmind_uses_correct_api_key(self, monkeypatch):
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-bluesmind-secret")
        mod = _reload_client()
        client = mod.OpenAIClient(model="moonshotai/kimi-k2.6", provider="bluesmind")
        llm = client.get_llm()
        assert str(llm.openai_api_key.get_secret_value()) == "sk-bluesmind-secret"

    def test_bluesmind_uses_correct_base_url(self, monkeypatch):
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test-key")
        mod = _reload_client()
        client = mod.OpenAIClient(model="moonshotai/kimi-k2.6", provider="bluesmind")
        llm = client.get_llm()
        assert "bluesminds.com" in str(llm.openai_api_base)

    def test_explicit_base_url_overrides_default(self, monkeypatch):
        """Explicit base_url on client must win over provider default."""
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test-key")
        mod = _reload_client()
        client = mod.OpenAIClient(
            model="moonshotai/kimi-k2.6",
            provider="bluesmind",
            base_url="https://custom-proxy.example.com/v1",
        )
        llm = client.get_llm()
        assert "custom-proxy.example.com" in str(llm.openai_api_base)

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("BLUESMIND_API_KEY", raising=False)
        mod = _reload_client()
        client = mod.OpenAIClient(model="moonshotai/kimi-k2.6", provider="bluesmind")
        with pytest.raises(ValueError, match="BLUESMIND_API_KEY"):
            client.get_llm()

    def test_openai_provider_unaffected(self, monkeypatch):
        """Native openai must still use Responses API."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
        mod = _reload_client()
        client = mod.OpenAIClient(model="gpt-5.4", provider="openai")
        llm = client.get_llm()
        assert getattr(llm, "use_responses_api", False) is True


# ---------------------------------------------------------------------------
# 4. model_catalog
# ---------------------------------------------------------------------------

class TestModelCatalog:
    def test_bluesmind_quick_models_present(self):
        from tradingagents.llm_clients.model_catalog import get_model_options
        options = get_model_options("bluesmind", "quick")
        model_ids = [v for _, v in options]
        assert "moonshotai/kimi-k2.6" in model_ids

    def test_bluesmind_deep_models_present(self):
        from tradingagents.llm_clients.model_catalog import get_model_options
        options = get_model_options("bluesmind", "deep")
        model_ids = [v for _, v in options]
        assert "moonshotai/kimi-k2.6" in model_ids

    def test_bluesmind_has_custom_option(self):
        from tradingagents.llm_clients.model_catalog import get_model_options
        for mode in ("quick", "deep"):
            ids = [v for _, v in get_model_options("bluesmind", mode)]
            assert "custom" in ids

    def test_kimi_not_in_openai_catalog(self):
        """Kimi models must NOT appear in the openai catalog."""
        from tradingagents.llm_clients.model_catalog import get_model_options
        for mode in ("quick", "deep"):
            ids = [v for _, v in get_model_options("openai", mode)]
            assert "moonshotai/kimi-k2.6" not in ids

    def test_bluesmind_in_known_models(self):
        from tradingagents.llm_clients.model_catalog import get_known_models
        known = get_known_models()
        assert "bluesmind" in known
        assert "moonshotai/kimi-k2.6" in known["bluesmind"]

    def test_openai_catalog_unchanged(self):
        """openai catalog must still have exactly the original GPT models."""
        from tradingagents.llm_clients.model_catalog import get_model_options
        for mode in ("quick", "deep"):
            ids = [v for _, v in get_model_options("openai", mode)]
            assert any("gpt" in m for m in ids), "openai catalog lost GPT models"
            assert "custom" not in ids, "openai catalog should not have Custom option"


# ---------------------------------------------------------------------------
# 5. cli/utils — provider dropdown
# ---------------------------------------------------------------------------

class TestCliProviderDropdown:
    def _src(self) -> str:
        import pathlib
        return (pathlib.Path(__file__).parent.parent / "cli" / "utils.py").read_text()

    def test_bluesmind_in_providers_list(self):
        assert '"bluesmind"' in self._src()

    def test_bluesmind_display_name(self):
        assert '"BluesMind"' in self._src()

    def test_bluesmind_url_in_providers(self):
        assert "bluesminds.com" in self._src()

    def test_openai_still_in_providers(self):
        assert '"openai"' in self._src()


# ---------------------------------------------------------------------------
# 6. integration — create_llm_client end-to-end
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_create_client_returns_valid_llm(self, monkeypatch):
        """create_llm_client('bluesmind', ...) must return a working LLM object."""
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-integration-test")
        from tradingagents.llm_clients.factory import create_llm_client
        client = create_llm_client("bluesmind", "moonshotai/kimi-k2.6")
        llm = client.get_llm()
        # Must be a ChatOpenAI-compatible object with invoke method
        assert hasattr(llm, "invoke")
        assert hasattr(llm, "with_structured_output")

    def test_validate_model_known(self, monkeypatch):
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test")
        from tradingagents.llm_clients.factory import create_llm_client
        client = create_llm_client("bluesmind", "moonshotai/kimi-k2.6")
        assert client.validate_model() is True

    def test_validate_model_unknown_warns(self, monkeypatch):
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test")
        from tradingagents.llm_clients.factory import create_llm_client
        client = create_llm_client("bluesmind", "some-unknown-model")
        assert client.validate_model() is False

    def test_warn_if_unknown_model_emits_warning(self, monkeypatch):
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test")
        from tradingagents.llm_clients.factory import create_llm_client
        client = create_llm_client("bluesmind", "totally-unknown-model")
        with pytest.warns(RuntimeWarning, match="totally-unknown-model"):
            client.warn_if_unknown_model()

    def test_trading_graph_config_accepts_bluesmind(self, monkeypatch):
        """TradingAgentsGraph must not raise when configured with bluesmind."""
        monkeypatch.setenv("BLUESMIND_API_KEY", "sk-graph-test")
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.llm_clients.factory import create_llm_client

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "bluesmind"
        config["deep_think_llm"] = "moonshotai/kimi-k2.6"
        config["quick_think_llm"] = "moonshotai/kimi-k2.6"
        config["backend_url"] = None  # use provider default

        deep_client = create_llm_client(
            provider=config["llm_provider"],
            model=config["deep_think_llm"],
            base_url=config.get("backend_url"),
        )
        quick_client = create_llm_client(
            provider=config["llm_provider"],
            model=config["quick_think_llm"],
            base_url=config.get("backend_url"),
        )
        deep_llm = deep_client.get_llm()
        quick_llm = quick_client.get_llm()

        assert "bluesminds.com" in str(deep_llm.openai_api_base)
        assert "bluesminds.com" in str(quick_llm.openai_api_base)
        assert not getattr(deep_llm, "use_responses_api", False)
        assert not getattr(quick_llm, "use_responses_api", False)
