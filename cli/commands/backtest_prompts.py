"""
Interactive prompts and config builders for the backtest command.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import questionary
import typer

from cli.utils import (
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_glm_region,
    ask_minimax_region,
    ask_openai_reasoning_effort,
    ask_qwen_region,
    confirm_ollama_endpoint,
    console,
    ensure_api_key,
    select_deep_thinking_agent,
    select_llm_provider,
    select_shallow_thinking_agent,
)
from tradingagents.backtesting.data_window import ALLOWED_LOOKBACKS
from tradingagents.backtesting.decision_schema import AgentConfig

ALLOWED_LOOKBACK_CHOICES = [v for v in ALLOWED_LOOKBACKS if v is not None]


def _prompt_for_missing_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    """
    Interactively prompt the user for the **always-required** runtime
    inputs (``ticker``, ``start_date``, ``end_date``) and for any
    optional field still missing from the yaml.

    The yaml values for ticker / start_date / end_date are shown as
    defaults — pressing Enter keeps them, but the user is always
    prompted. This guarantees the CLI never silently runs a backtest
    against a yaml-defined ticker when the user wanted to test a
    different one.

    Returns the merged config (prompt values win for the three
    always-required fields; yaml values are kept for everything else).
    """
    # ---- Ticker (always prompted) ----
    default_ticker = (raw_config.get("ticker") or "").strip().upper()
    while True:
        ticker = questionary.text(
            "Ticker symbol (e.g. BUMI.JK, AAPL):",
            default=default_ticker,
            validate=lambda v: v.strip() != "" or "Ticker is required.",
        ).ask()
        if not ticker:
            raise typer.Exit(1)
        raw_config["ticker"] = ticker.strip().upper()
        break

    # ---- Start date (always prompted) ----
    default_start = str(raw_config.get("start_date") or "")
    while True:
        date_str = questionary.text(
            "Start date (YYYY-MM-DD):",
            default=default_start,
            validate=lambda v: (
                v.strip() != ""
                and _is_valid_date(v.strip())
                or "Use YYYY-MM-DD format."
            ),
        ).ask()
        if not date_str:
            raise typer.Exit(1)
        if _is_valid_date(date_str.strip()):
            raw_config["start_date"] = date_str.strip()
            break

    # ---- End date (always prompted) ----
    default_end = str(raw_config.get("end_date") or "")
    while True:
        date_str = questionary.text(
            "End date (YYYY-MM-DD):",
            default=default_end,
            validate=lambda v: (
                v.strip() != ""
                and _is_valid_date(v.strip())
                or "Use YYYY-MM-DD format."
            ),
        ).ask()
        if not date_str:
            raise typer.Exit(1)
        if _is_valid_date(date_str.strip()):
            end = date_str.strip()
            if raw_config["start_date"] > end:
                console.print(
                    "[red]End date must be on or after start date.[/red]"
                )
                continue
            raw_config["end_date"] = end
            break

    # ---- Optional: initial_cash (prompt only if missing) ----
    if "initial_cash" not in raw_config or raw_config["initial_cash"] is None:
        default_cash = str(raw_config.get("initial_cash") or 100000000)
        cash = questionary.text(
            "Initial cash (IDR):",
            default=default_cash,
            validate=lambda v: v.strip().isdigit() and int(v) > 0
            or "Enter a positive integer.",
        ).ask()
        raw_config["initial_cash"] = int(cash.strip()) if cash else 100_000_000

    return raw_config


def _is_valid_date(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _select_lookback(default: Optional[int]) -> Optional[int]:
    """
    Interactive prompt for lookback window.

    Returns the chosen lookback (one of ALLOWED_LOOKBACK_CHOICES), or None
    to use the yaml's value / full history.
    """
    default_label = (
        f"current yaml value: {default}"
        if default is not None
        else "no window (full history)"
    )
    choices = [
        questionary.Choice(
            f"{v} trading days  [dim](keep current setting)[/dim]"
            if v == default
            else f"{v} trading days",
            value=v,
        )
        for v in ALLOWED_LOOKBACK_CHOICES
    ]
    choices.append(
        questionary.Choice(
            "Use yaml value  [dim]({})[/dim]".format(default_label),
            value=None,
        )
    )
    return questionary.select(
        "Lookback window (single period per run):",
        choices=choices,
    ).ask()


def _build_llm_agent_config(agent_overrides: Optional[dict[str, Any]] = None) -> AgentConfig:
    """
    Reuse the LLM provider / model selectors from cli/utils.py to build an
    AgentConfig for the daily TradingAgentsRunner. Also exports the
    resolved provider/model/backend into os.environ so the runner's
    deepcopy(DEFAULT_CONFIG) picks up the right LLM.
    """
    console.rule(
        "[bold cyan]LLM Configuration[/bold cyan]", align="left"
    )
    llm_provider, backend_url = select_llm_provider()

    if llm_provider == "qwen":
        llm_provider, backend_url = ask_qwen_region()
    elif llm_provider == "minimax":
        llm_provider, backend_url = ask_minimax_region()
    elif llm_provider == "glm":
        llm_provider, backend_url = ask_glm_region()

    if llm_provider == "ollama":
        confirm_ollama_endpoint(backend_url)

    ensure_api_key(llm_provider)

    shallow = select_shallow_thinking_agent(llm_provider)
    deep = select_deep_thinking_agent(llm_provider)

    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    if llm_provider == "google":
        thinking_level = ask_gemini_thinking_config()
    elif llm_provider == "openai":
        reasoning_effort = ask_openai_reasoning_effort()
    elif llm_provider == "anthropic":
        anthropic_effort = ask_anthropic_effort()

    # Propagate LLM choice into the process env so TradingAgentsRunner
    # (which deepcopies DEFAULT_CONFIG) resolves the right provider/model.
    import os
    os.environ["TRADINGAGENTS_LLM_PROVIDER"] = llm_provider.lower()
    os.environ["TRADINGAGENTS_QUICK_THINK_LLM"] = shallow
    os.environ["TRADINGAGENTS_DEEP_THINK_LLM"] = deep
    if backend_url:
        os.environ["TRADINGAGENTS_LLM_BACKEND_URL"] = backend_url
    if thinking_level:
        os.environ["TRADINGAGENTS_GOOGLE_THINKING_LEVEL"] = thinking_level
    if reasoning_effort:
        os.environ["TRADINGAGENTS_OPENAI_REASONING_EFFORT"] = reasoning_effort
    if anthropic_effort:
        os.environ["TRADINGAGENTS_ANTHROPIC_EFFORT"] = anthropic_effort

    # Reload DEFAULT_CONFIG so it picks up the env vars we just set.
    # TradingAgentsRunner does deepcopy(DEFAULT_CONFIG) on every call;
    # it must see the updated provider/model, not the import-time values.
    import importlib

    import tradingagents.default_config
    importlib.reload(tradingagents.default_config)
    # Also reload anything else that captured DEFAULT_CONFIG at import time.
    from tradingagents.backtesting import agent_runner
    importlib.reload(agent_runner)

    # Minimal AgentConfig for TradingAgentsRunner — the LLM fields are
    # resolved through env vars above; explicit strategy overrides stay typed.
    return AgentConfig(**(agent_overrides or {}))
