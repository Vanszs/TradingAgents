"""
Unit tests for config_resolver.py — env-based provider/model resolution.
"""
import unittest

from tradingagents.backtesting.config_resolver import ConfigError, resolve_agent_config
from tradingagents.backtesting.position import BacktestConfig, AgentConfig


def _env(**kwargs):
    """Build an env dict with API keys for known providers."""
    base = {
        "OPENAI_API_KEY": "sk-test",
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "GOOGLE_API_KEY": "sk-google-test",
    }
    base.update(kwargs)
    return base


class TestResolveAgentConfig(unittest.TestCase):

    def test_yaml_empty_env_set(self):
        """YAML empty, env set -> uses env."""
        agent = resolve_agent_config(
            {"provider": "", "model": ""},
            env=_env(TRADINGAGENTS_LLM_PROVIDER="openai", TRADINGAGENTS_LLM_MODEL="gpt-4o"),
        )
        self.assertEqual(agent.provider, "openai")
        self.assertEqual(agent.model, "gpt-4o")

    def test_yaml_set_env_set(self):
        """YAML set, env set -> YAML wins."""
        agent = resolve_agent_config(
            {"provider": "anthropic", "model": "claude-3"},
            env=_env(TRADINGAGENTS_LLM_PROVIDER="openai", TRADINGAGENTS_LLM_MODEL="gpt-4o"),
        )
        self.assertEqual(agent.provider, "anthropic")
        self.assertEqual(agent.model, "claude-3")

    def test_yaml_set_env_empty(self):
        """YAML set, env empty -> uses YAML."""
        agent = resolve_agent_config(
            {"provider": "google", "model": "gemini-1.5-pro"},
            env=_env(),
        )
        self.assertEqual(agent.provider, "google")
        self.assertEqual(agent.model, "gemini-1.5-pro")

    def test_both_empty_raises(self):
        """YAML empty, env empty -> ConfigError."""
        with self.assertRaises(ConfigError):
            resolve_agent_config({"provider": "", "model": ""}, env={})

    def test_missing_model_raises(self):
        """Provider set, model missing -> ConfigError."""
        with self.assertRaises(ConfigError):
            resolve_agent_config(
                {"provider": "openai", "model": ""},
                env=_env(TRADINGAGENTS_LLM_PROVIDER="openai"),
            )

    def test_missing_api_key_raises(self):
        """Known provider, missing API key -> ConfigError."""
        with self.assertRaises(ConfigError):
            resolve_agent_config(
                {"provider": "openai", "model": "gpt-4o"},
                env={},
            )

    def test_ollama_no_key_required(self):
        """Ollama provider, no key -> no error."""
        agent = resolve_agent_config(
            {"provider": "ollama", "model": "llama3"},
            env={},
        )
        self.assertEqual(agent.provider, "ollama")
        self.assertEqual(agent.model, "llama3")

    def test_model_fallback_to_deep_think(self):
        """TRADINGAGENTS_DEEP_THINK_LLM fallback when TRADINGAGENTS_LLM_MODEL is empty."""
        agent = resolve_agent_config(
            {"provider": "", "model": ""},
            env=_env(TRADINGAGENTS_LLM_PROVIDER="openai", TRADINGAGENTS_DEEP_THINK_LLM="gpt-5.4"),
        )
        self.assertEqual(agent.model, "gpt-5.4")

    def test_temperature_and_other_fields(self):
        """Other agent fields are passed through."""
        agent = resolve_agent_config(
            {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.5,
                "max_thesis_chars": 1000,
                "deterministic_seed": 123,
            },
            env=_env(),
        )
        self.assertAlmostEqual(agent.temperature, 0.5)
        self.assertEqual(agent.max_thesis_chars, 1000)
        self.assertEqual(agent.deterministic_seed, 123)


if __name__ == "__main__":
    unittest.main()
