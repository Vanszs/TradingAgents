"""
Environment-based agent config resolver for the backtester.

Follows the repo's existing env-var override pattern:
- TRADINGAGENTS_LLM_PROVIDER → provider
- TRADINGAGENTS_LLM_MODEL → model (backtest-specific; separate from deep/quick)

Resolution order:
1. YAML value (if non-empty) → used.
2. Env var (if non-empty) → used.
3. Both empty → ConfigError raised.
"""
from __future__ import annotations

import os
from typing import Any

from .position import AgentConfig, BacktestConfig


class ConfigError(Exception):
    """Raised when required configuration is missing."""


def resolve_agent_config(yaml_agent: dict[str, Any], env: dict[str, str] | None = None) -> AgentConfig:
    """
    Resolve agent configuration from YAML dict + env vars.

    Parameters
    ----------
    yaml_agent : dict
        The ``agent`` section from the YAML config. Keys like ``provider``,
        ``model``, ``temperature`` etc. Empty-string values are treated as
        "not set".
    env : dict, optional
        Environment variable mapping. Defaults to ``os.environ``.

    Returns
    -------
    AgentConfig
        Fully resolved agent config with provider and model populated.

    Raises
    ------
    ConfigError
        If neither YAML nor env provides a provider or model.
    """
    if env is None:
        env = dict(os.environ)

    # Provider: YAML first, then env
    provider = yaml_agent.get("provider", "").strip()
    if not provider:
        provider = env.get("TRADINGAGENTS_LLM_PROVIDER", "").strip()

    # Model: YAML first, then env (TRADINGAGENTS_LLM_MODEL, fallback to DEEP_THINK)
    model = yaml_agent.get("model", "").strip()
    if not model:
        model = env.get("TRADINGAGENTS_LLM_MODEL", "").strip()
    if not model:
        model = env.get("TRADINGAGENTS_DEEP_THINK_LLM", "").strip()

    if not provider:
        raise ConfigError(
            "Agent provider is required. Set agent.provider in YAML "
            "or export TRADINGAGENTS_LLM_PROVIDER."
        )
    if not model:
        raise ConfigError(
            "Agent model is required. Set agent.model in YAML "
            "or export TRADINGAGENTS_LLM_MODEL."
        )

    # Validate API key for known providers
    _validate_api_key(provider, env)

    return AgentConfig(
        provider=provider,
        model=model,
        temperature=float(yaml_agent.get("temperature", 0.2)),
        memory_enabled=bool(yaml_agent.get("memory_enabled", False)),
        web_search_enabled=bool(yaml_agent.get("web_search_enabled", False)),
        news_provider=yaml_agent.get("news_provider", "snapshot"),
        max_thesis_chars=int(yaml_agent.get("max_thesis_chars", 2000)),
        deterministic_seed=int(yaml_agent.get("deterministic_seed", 42)),
        backtest_mode=bool(yaml_agent.get("backtest_mode", True)),
        run_frequency=yaml_agent.get("run_frequency", "daily"),
        report_language=yaml_agent.get("report_language", "English"),
        cache_reports=bool(yaml_agent.get("cache_reports", True)),
    )


# Provider → env var for API key (subset of llm_clients/api_key_env.py)
_PROVIDER_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "bluesmind": "BLUESMIND_API_KEY",
    "sumopod": "SUMOPOD_API_KEY",
    "ollama": None,
}


def _validate_api_key(provider: str, env: dict[str, str]) -> None:
    """Assert that the API key env var is set for known providers."""
    key_env = _PROVIDER_KEY_ENV.get(provider.lower())
    if key_env is None:
        return  # unknown provider or keyless (ollama)
    if not env.get(key_env, "").strip():
        raise ConfigError(
            f"API key env var {key_env} is required for provider={provider!r}. "
            f"Export {key_env} or add it to your .env file."
        )


def validate_backtest_config(config: BacktestConfig) -> None:
    """
    Validate a BacktestConfigV2 with env-var resolution for agent fields.

    Calls ``resolve_agent_config`` internally; raises ConfigError on failure.
    """
    yaml_agent = {
        "provider": config.agent.provider,
        "model": config.agent.model,
        "temperature": config.agent.temperature,
        "memory_enabled": config.agent.memory_enabled,
        "web_search_enabled": config.agent.web_search_enabled,
        "news_provider": config.agent.news_provider,
        "max_thesis_chars": config.agent.max_thesis_chars,
        "deterministic_seed": config.agent.deterministic_seed,
        "backtest_mode": config.agent.backtest_mode,
        "run_frequency": config.agent.run_frequency,
        "report_language": config.agent.report_language,
        "cache_reports": config.agent.cache_reports,
    }
    config.agent = resolve_agent_config(yaml_agent)
    config.validate()
