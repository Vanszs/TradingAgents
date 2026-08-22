"""
CLI command: tradingagents backtest

Runs the walk-forward backtest with 5-tier decision mapping, trigger-based
execution, and lookback window. Single period per invocation.

Usage:
    tradingagents backtest
    tradingagents backtest --config path/to/config.yaml
    tradingagents backtest --lookback 60
    tradingagents backtest --dry-run
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import questionary
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

load_dotenv()

logger = logging.getLogger(__name__)

from cli.utils import console
from tradingagents.backtesting import (
    BacktestEngine,
    TradingAgentsRunner,
)
from tradingagents.backtesting.data_window import ALLOWED_LOOKBACKS
from tradingagents.backtesting.decision_schema import AgentConfig
from tradingagents.backtesting.trigger_evaluator import TriggerConfig

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


def _build_llm_agent_config() -> AgentConfig:
    """
    Reuse the LLM provider / model selectors from cli/main.py to build an
    AgentConfig for the daily TradingAgentsRunner. Also exports the
    resolved provider/model/backend into os.environ so the runner's
    deepcopy(DEFAULT_CONFIG) picks up the right LLM.
    """
    from cli.main import (
        ask_anthropic_effort,
        ask_gemini_thinking_config,
        ask_glm_region,
        ask_minimax_region,
        ask_openai_reasoning_effort,
        ask_qwen_region,
        confirm_ollama_endpoint,
        ensure_api_key,
        select_deep_thinking_agent,
        select_llm_provider,
        select_shallow_thinking_agent,
    )

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
    # resolved through env vars above; AgentConfig only carries the
    # safety-guard flags (memory_enabled, backtest_mode, ...).
    return AgentConfig()


# ---------------------------------------------------------------------------
# Live UI helpers
# ---------------------------------------------------------------------------

PHASE_ICONS = {
    "start": "[bold green]▶[/bold green]",
    "day_start": "[bold cyan]📅[/bold cyan]",
    "day_complete": "[bold green]✓[/bold green]",
    "execute": "[bold yellow]⚡[/bold yellow]",
    "risk": "[bold red]🛡[/bold red]",
    "snapshot": "[bold blue]📊[/bold blue]",
    "agent": "[bold magenta]🤖[/bold magenta]",
    "parse": "[bold cyan]📝[/bold cyan]",
    "trigger": "[bold green]🚦[/bold green]",
    "orders": "[bold yellow]📤[/bold yellow]",
    "wait": "[bold yellow]⏳[/bold yellow]",
    "done": "[bold green]🏁[/bold green]",
    "complete": "[bold green]✅[/bold green]",
    "error": "[bold red]❌[/bold red]",
}

PHASE_LABELS = {
    "start": "Backtest started",
    "day_start": "Processing day",
    "day_complete": "Day complete",
    "execute": "Executing orders at open",
    "risk": "Checking risk events",
    "snapshot": "Creating snapshot (no leakage)",
    "agent": "Running Daily Review Agent (LLM call — may take a while)",
    "agent_node": "LLM node running",
    "parse": "Parsing decision",
    "trigger": "Trigger-Based Execution Agent",
    "orders": "Generating orders for next day",
    "wait": "Still running (LLM call in flight...)",
    "done": "Backtest done — finalizing",
    "complete": "Backtest complete",
    "error": "Error",
}


class _BacktestUI:
    """Lightweight state holder for the live progress display."""

    def __init__(self, n_days: int, ticker: str, lookback: Optional[int]):
        self.n_days = n_days
        self.ticker = ticker
        self.lookback = lookback
        self.log_messages: list[tuple[str, str, str]] = []
        self.current_day = 0
        self.current_phase = ""
        self.current_date = ""
        # Per-node LLM timing state. Set by ``record_node_done`` when
        # an ``agent_node`` event fires; cleared on each new day.
        self.current_node: str = ""
        self.current_node_started_at: Optional[float] = None
        self.node_durations: list[tuple[str, float]] = []
        self.status = "running"
        self.start_time: Optional[float] = None
        self.trigger_triggered = 0
        self.trigger_skipped = 0
        self.trigger_total = 0
        self._wait_logged = False
        self.trade_log: list[dict[str, str]] = []

    def elapsed(self) -> str:
        if self.start_time is None:
            return "00:00"
        e = int(time.time() - self.start_time)
        return f"{e // 60:02d}:{e % 60:02d}"

    def add_log(self, icon: str, phase: str, detail: str = "") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append((ts, icon, f"{phase} {detail}".strip()))

    def record_node_start(self, name: str) -> None:
        """Mark a new LLM node as the active one (start clock)."""
        self.current_node = name
        self.current_node_started_at = time.time()

    def record_node_done(self, name: str, duration: float) -> None:
        """Record that node ``name`` completed after ``duration``s."""
        self.current_node = ""
        self.current_node_started_at = None
        self.node_durations.append((name, duration))
        # Cap history to avoid unbounded growth in long backtests.
        if len(self.node_durations) > 200:
            self.node_durations = self.node_durations[-200:]
        # Visual feedback: dim log entry so the main "agent finished"
        # line below stays the most prominent. Use a smaller icon.
        self.add_log("•", f"{name}", f"{duration:.1f}s")

    def clear_day_state(self) -> None:
        """Reset per-day UI state (called on day_start)."""
        self.current_node = ""
        self.current_node_started_at = None
        self.node_durations = []

    def active_node_label(self) -> str:
        """Render the currently-running LLM node for the progress
        panel. Returns an empty string when no node is active."""
        if not self.current_node or self.current_node_started_at is None:
            return ""
        elapsed = time.time() - self.current_node_started_at
        return f"  🔄 {self.current_node} ({elapsed:.1f}s)"


def _render_header(layout: Layout, ui: _BacktestUI) -> None:
    lookback_str = (
        f"{ui.lookback} trading days" if ui.lookback is not None
        else "no window (full history)"
    )
    layout["header"].update(
        Panel(
            f"[bold green]TradingAgents Walk-Forward Backtest[/bold green]\n"
            f"[dim]Ticker: {ui.ticker} | Lookback: {lookback_str} | "
            f"Days: {ui.n_days}[/dim]",
            border_style="green",
            padding=(0, 2),
        )
    )


def _render_progress(layout: Layout, ui: _BacktestUI) -> None:
    pct = (ui.current_day / ui.n_days) if ui.n_days else 0
    bar_len = 30
    filled = int(pct * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    icon = PHASE_ICONS.get(ui.current_phase, "⏳")
    label = PHASE_LABELS.get(ui.current_phase, ui.current_phase)

    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Line1", ratio=1)

    line1 = Text()
    line1.append("Day ", style="bold cyan")
    line1.append(f"{ui.current_day}/{ui.n_days}", style="bold cyan")
    line1.append(f"  [{bar}] {pct * 100:5.1f}%", style="green")
    table.add_row(line1)

    line2 = Text()
    line2.append(f"{icon} ", style="yellow")
    line2.append(f"{label}", style="yellow")
    if ui.current_date:
        line2.append(f"   trade_date={ui.current_date}", style="dim")
    line2.append(f"   ⏱ elapsed={ui.elapsed()}", style="dim")
    if ui.trigger_total:
        line2.append(
            f"   🚦 triggered={ui.trigger_triggered}/"
            f"{ui.trigger_total}",
            style="green",
        )
    table.add_row(line2)

    # Active-node indicator: shows the currently-running LLM node
    # (e.g. "🔄 Market Analyst (4.2s)"). Empty when no node is
    # running, so most rows just stay blank.
    active = ui.active_node_label()
    if active:
        line3 = Text()
        line3.append(active, style="bold magenta")
        table.add_row(line3)

    layout["progress"].update(
        Panel(table, border_style="cyan", padding=(0, 1))
    )


def _render_log(layout: Layout, ui: _BacktestUI) -> None:
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Time", style="cyan", width=10)
    table.add_column("", width=2, justify="center")
    table.add_column("Event", style="white", ratio=1)

    for ts, icon, event in ui.log_messages[-15:]:
        table.add_row(ts, icon, event)

    layout["log"].update(
        Panel(table, title="Activity Log", border_style="blue", padding=(0, 1))
    )


def _render_trading(layout: Layout, ui: _BacktestUI) -> None:
    table = Table(
        show_header=True,
        header_style="bold yellow",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Date", style="cyan", width=12)
    table.add_column("Action", style="white", width=16)
    table.add_column("Price", style="green", width=14, justify="right")
    table.add_column("Qty", style="white", width=8, justify="right")
    table.add_column("PnL", style="white", width=14, justify="right")

    for entry in ui.trade_log[-15:]:
        table.add_row(
            entry["date"],
            entry["action"],
            entry["price"],
            entry["qty"],
            entry["pnl"],
        )

    layout["trading"].update(
        Panel(table, title="Trading Activity", border_style="yellow", padding=(0, 1))
    )


def _render_footer(
    layout: Layout, ui: _BacktestUI, summary: Optional[dict]
) -> None:
    if ui.status == "complete" and summary:
        body = Text()
        body.append("✅ Backtest Complete\n\n", style="bold green")
        body.append(
            f"Final equity:     {summary.get('final_equity', 0):,.0f} IDR\n",
            style="bold",
        )
        pnl_pct = summary.get("total_return_pct", 0)
        color = "green" if pnl_pct >= 0 else "red"
        body.append(
            f"Total return:     {pnl_pct:+.2f}%\n",
            style=f"bold {color}",
        )
        body.append(
            f"Trades:           {summary.get('number_of_trades', 0)}\n"
        )
        body.append(
            f"Trigger hit rate: "
            f"{summary.get('trigger_hit_rate', 0) * 100:5.1f}% "
            f"({summary.get('trigger_triggered', 0)}/"
            f"{summary.get('trigger_total_decisions', 0)})\n"
        )
        if summary.get("lookback_days") is not None:
            body.append(
                f"Lookback:          {summary.get('lookback_days')} trading days\n"
            )
    else:
        body = Text("Backtest running...", style="dim")
    layout["footer"].update(
        Panel(body, border_style="grey50", padding=(0, 1))
    )


def _render_all(
    layout: Layout, ui: _BacktestUI, summary: Optional[dict]
) -> None:
    _render_header(layout, ui)
    _render_progress(layout, ui)
    _render_trading(layout, ui)
    _render_log(layout, ui)
    _render_footer(layout, ui, summary)


def _make_progress_callback(ui: _BacktestUI):
    """
    WalkForwardRunner does call back now — see ``_on_progress`` in
    :func:`backtest`. This factory is kept for backward compatibility with
    any callers that may still use it.
    """
    def _callback(phase, trade_date, day_idx, total_days, **extra):
        ui.n_days = total_days
        ui.current_day = day_idx
        ui.current_date = trade_date or "(running...)"
        if phase in PHASE_LABELS:
            ui._wait_logged = False
            ui.current_phase = phase
            ui.add_log(
                PHASE_ICONS.get(phase, "⏳"),
                PHASE_LABELS.get(phase, phase),
                f"day {day_idx}/{total_days}",
            )
    return _callback


# ---------------------------------------------------------------------------
# Final summary rendering
# ---------------------------------------------------------------------------


def _print_results_table(
    summary: dict, leakage: dict, output_paths: dict
) -> None:
    table = Table(
        title="Backtest Summary",
        show_header=True,
        header_style="bold cyan",
        border_style="green",
        padding=(0, 2),
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    def _fmt(val, fmt=".2f", default="—"):
        return f"{val:{fmt}}" if val is not None else default

    rows = [
        ("Ticker", summary.get("ticker", "?")),
        ("Period", f"{summary.get('start_date', '?')} → {summary.get('end_date', '?')}"),
        ("Lookback (days)", str(summary.get("lookback_days", "—"))),
        ("Decision mapping", summary.get("decision_mapping_mode", "?")),
        ("Initial cash", f"{_fmt(summary.get('initial_cash'), ',.0f', '0')} IDR"),
        ("Final equity", f"{_fmt(summary.get('final_equity'), ',.0f', '0')} IDR"),
        ("Total return", f"{_fmt(summary.get('total_return_pct'), '+.2f', '0.00')}%" if summary.get('total_return_pct') is not None else "—"),
        ("Max drawdown", f"{_fmt(summary.get('max_drawdown_pct'), '.2f', '0.00')}%" if summary.get('max_drawdown_pct') is not None else "—"),
        ("Sharpe ratio", _fmt(summary.get("sharpe_ratio"))),
        ("Total trades", str(summary.get("number_of_trades") or 0)),
        ("Long win rate", f"{_fmt(summary.get('long_win_rate'), '.2f', '0.00')}%"),
        ("Short win rate", f"{_fmt(summary.get('short_win_rate'), '.2f', '0.00')}%"),
        ("Trigger hit rate", f"{(summary.get('trigger_hit_rate') or 0.0) * 100:5.1f}%"),
        ("Trigger triggered", f"{summary.get('trigger_triggered') or 0}/{summary.get('trigger_total_decisions') or 0}"),
        ("Avg hold (long)", f"{_fmt(summary.get('avg_holding_period_long_days'), '.1f', '0.0')} d"),
        ("Avg hold (short)", f"{_fmt(summary.get('avg_holding_period_short_days'), '.1f', '0.0')} d"),
        ("Margin calls", str(summary.get("margin_calls_count") or 0)),
        ("Liquidations", str(summary.get("liquidations_count") or 0)),
    ]
    for k, v in rows:
        table.add_row(k, v)

    console.print(table)

    if leakage:
        status = leakage.get("status", "UNKNOWN")
        status_style = "green" if status == "PASSED" else "red"
        console.print(
            f"\n[bold {status_style}]Leakage audit: {status}[/bold {status_style}]"
        )
        for name, result in leakage.get("checks", {}).items():
            icon = "✅" if result == "PASSED" else "❌"
            console.print(f"  {icon} {name}: {result}")

    if output_paths:
        console.print("\n[bold]Output files:[/bold]")
        for name, path in output_paths.items():
            exists = Path(path).exists()
            mark = "✅" if exists else "❌"
            console.print(f"  {mark} {name}: {path}")

    # Rating distribution
    rating_dist = summary.get("rating_distribution", {})
    if rating_dist:
        console.print("\n[bold]Rating distribution:[/bold]")
        for rating, count in sorted(rating_dist.items(), key=lambda x: -x[1]):
            console.print(f"  {rating}: {count}")

    # Trigger reason counts
    reason_counts = summary.get("trigger_reason_counts", {})
    if reason_counts:
        console.print("\n[bold]Trigger reasons fired:[/bold]")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            console.print(f"  {reason}: {count}")


def _print_dry_run_plan(
    config: dict[str, Any], lookback: Optional[int]
) -> None:
    console.rule(
        "[bold cyan]Dry-Run Plan[/bold cyan]", align="left"
    )
    decision_mapping = config.get("decision_mapping", {}) or {}
    trigger = config.get("trigger", {}) or {}
    margin = config.get("margin", {}) or {}
    execution = config.get("execution", {}) or {}

    plan = Table(
        show_header=False,
        border_style="cyan",
        padding=(0, 1),
        expand=True,
    )
    plan.add_column("Field", style="cyan", no_wrap=True)
    plan.add_column("Value", style="white")

    # Resolve effective lookback: CLI override wins, else yaml value, else "none".
    effective_lookback = (
        lookback if lookback is not None else config.get("lookback_days")
    )
    effective_lookback_str = (
        str(effective_lookback) if effective_lookback is not None else "(none — full history)"
    )

    plan.add_row("Ticker", str(config.get("ticker", "?")))
    plan.add_row(
        "Period",
        f"{config.get('start_date', '?')} → {config.get('end_date', '?')}",
    )
    plan.add_row("Initial cash", f"{config.get('initial_cash', 0):,.0f} IDR")
    plan.add_row(
        "Lookback (CLI override)",
        str(lookback) if lookback is not None else "(not set)",
    )
    plan.add_row(
        "Lookback (yaml)",
        str(config.get("lookback_days", "(none)")),
    )
    plan.add_row("Lookback (effective)", effective_lookback_str)
    plan.add_row(
        "Decision mapping mode",
        str(decision_mapping.get("mode", "strict_5tier")),
    )
    plan.add_row(
        "allow_explicit_reverse",
        str(decision_mapping.get("allow_explicit_reverse", False)),
    )
    plan.add_row(
        "allow_short_on_underweight",
        str(decision_mapping.get("allow_short_on_underweight", False)),
    )
    plan.add_row(
        "allow_short_on_sell",
        str(decision_mapping.get("allow_short_on_sell", True)),
    )
    plan.add_row(
        "base_allocation_pct",
        f"{decision_mapping.get('base_allocation_pct', 0.20):.2%}",
    )
    plan.add_row("--- Risk ---", "")
    plan.add_row(
        "Max leverage",
        f"{margin.get('max_leverage', 2.0):.1f}x",
    )
    plan.add_row(
        "Initial margin",
        f"{margin.get('initial_margin_pct', 0.50):.0%}",
    )
    plan.add_row(
        "Maintenance margin",
        f"{margin.get('maintenance_margin_pct', 0.35):.0%}",
    )
    plan.add_row(
        "Auto-liquidate on breach",
        str(margin.get("auto_liquidate_on_breach", True)),
    )
    plan.add_row("--- Execution ---", "")
    plan.add_row(
        "Buy fee",
        f"{execution.get('buy_fee', 0.0015):.4%}",
    )
    plan.add_row(
        "Sell fee",
        f"{execution.get('sell_fee', 0.0019):.4%}",
    )
    plan.add_row(
        "Slippage",
        f"{execution.get('slippage', 0.001):.4%}",
    )
    plan.add_row("--- Trigger ---", "")
    plan.add_row(
        "Trigger enabled",
        str(trigger.get("enabled", True)),
    )
    plan.add_row(
        "On TP/SL hit",
        str(trigger.get("trigger_on_tp_sl_hit", True)),
    )
    plan.add_row(
        "On setup invalid",
        str(trigger.get("trigger_on_setup_invalid", True)),
    )
    plan.add_row(
        "On R:R deteriorated",
        str(trigger.get("trigger_on_rr_deteriorated", True)),
    )
    plan.add_row(
        "R:R drop threshold",
        f"{trigger.get('rr_deterioration_pct', 0.30):.0%}",
    )
    plan.add_row(
        "Use real R:R when available",
        str(trigger.get("use_real_rr_when_available", True)),
    )
    plan.add_row(
        "On strong exit signal",
        str(trigger.get("trigger_on_strong_exit_signal", True)),
    )
    plan.add_row(
        "On better candidate",
        str(trigger.get("trigger_on_better_candidate", True)),
    )
    plan.add_row(
        "On entry condition change",
        str(trigger.get("trigger_on_entry_condition_changed", True)),
    )
    console.print(plan)

    # Resolved-config dump: show what the engine will actually consume
    # after CLI overrides are applied. Helps users verify that
    # `tradingagents backtest --lookback 60` is using 60, not the yaml
    # value.
    console.print()
    resolved_lookback = (
        lookback if lookback is not None else config.get("lookback_days")
    )
    resolved = {
        "ticker": config.get("ticker"),
        "start_date": config.get("start_date"),
        "end_date": config.get("end_date"),
        "initial_cash": config.get("initial_cash"),
        "lookback_days": resolved_lookback,
        "decision_mapping": {
            "mode": decision_mapping.get("mode", "strict_5tier"),
            "allow_explicit_reverse": decision_mapping.get(
                "allow_explicit_reverse", False
            ),
            "allow_short_on_underweight": decision_mapping.get(
                "allow_short_on_underweight", False
            ),
            "allow_short_on_sell": decision_mapping.get(
                "allow_short_on_sell", True
            ),
            "base_allocation_pct": decision_mapping.get(
                "base_allocation_pct", 0.20
            ),
        },
        "margin": {
            "max_leverage": margin.get("max_leverage", 2.0),
            "initial_margin_pct": margin.get("initial_margin_pct", 0.50),
            "maintenance_margin_pct": margin.get("maintenance_margin_pct", 0.35),
        },
        "trigger": {
            "enabled": trigger.get("enabled", True),
            "rr_deterioration_pct": trigger.get("rr_deterioration_pct", 0.30),
            "use_real_rr_when_available": trigger.get(
                "use_real_rr_when_available", True
            ),
        },
    }
    console.rule(
        "[bold cyan]Resolved config (after CLI overrides)[/bold cyan]",
        align="left",
    )
    import json

    console.print_json(json.dumps(resolved, indent=2, default=str))

    # Conceptual explainer — users keep mixing up the two date / window
    # settings. Surface the distinction clearly so they can verify the
    # command line does what they expect.
    _print_window_explainer(config.get("start_date"), config.get("end_date"), resolved_lookback)

    console.print(
        "\n[dim]No execution performed. Pass --no-dry-run to run.[/dim]"
    )


def _print_window_explainer(
    start_date: Optional[str],
    end_date: Optional[str],
    lookback: Optional[int],
) -> None:
    """
    Print a short conceptual block that distinguishes the two time
    settings the user might be confused about:

    * ``start_date`` / ``end_date`` — the **outer window** of the
      backtest. The runner will loop once per trading day in this
      range. Trading days are taken from the OHLCV file.
    * ``lookback_days`` — the **inner window** the agent sees when
      making each decision. Each snapshot the agent receives contains
      the last ``lookback_days`` trading days of OHLCV (and a
      time-bounded slice of news / sentiment / broker activity). It
      is independent of the outer window.
    """
    console.rule(
        "[bold cyan]start_date / end_date vs lookback_days[/bold cyan]",
        align="left",
    )
    console.print(
        "  [bold]start_date, end_date[/bold] — outer window of the "
        "backtest.\n"
        "      The runner loops once per trading day found in the OHLCV "
        "file inside this range.\n"
        f"      You asked for: [cyan]{start_date} → {end_date}[/cyan]"
    )
    console.print()
    if lookback is None:
        lookback_str = "full history (no window)"
    else:
        lookback_str = f"{lookback} trading days"
    console.print(
        "  [bold]lookback_days[/bold] — inner window the agent sees "
        "per decision.\n"
        "      Each daily snapshot the agent receives contains only the "
        "last N trading days of OHLCV (plus a time-bounded slice of "
        "news / sentiment / broker activity). Independent of the outer "
        "window.\n"
        f"      You asked for: [cyan]{lookback_str}[/cyan]"
    )
    console.print()
    console.print(
        "  [dim]Example: 3-month backtest with lookback=20 → the agent "
        "runs ~60 times (3 months × 5/7 trading days), and each time it "
        "sees only the last 20 trading days of price history.[/dim]"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def backtest(
    config_path: Path = Path("backtest.yaml"),
    lookback: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    """
    Run walk-forward backtest.

    Two time settings govern a backtest run; they are independent:

    * ``start_date`` / ``end_date`` — the **outer window**. The runner
      loops once per trading day found in the OHLCV file inside this
      range. Always prompted at runtime.
    * ``lookback_days`` — the **inner window** the agent sees per
      decision. Each daily snapshot contains the last N trading days
      of OHLCV (and time-bounded news / sentiment / broker activity).
      Independent of the outer window; controls how much *history* the
      agent looks at, not how many days the backtest runs for.

    Example: 3-month backtest with lookback=60 means the agent runs
    ~60 times (≈ 3 months × 5/7 trading days) and each time sees the
    last 60 trading days of price history.

    If the OHLCV file is missing or out of date, the CLI will offer
    to download it from yfinance automatically.

    LLM-call footprint
    ------------------
    Each trading day fires 9 LLM calls in order: Market Analyst, Bull
    Researcher, Bear Researcher, Research Manager, Trader, Aggressive
    Analyst, Conservative Analyst, Neutral Analyst, Portfolio Manager.
    With a slow reasoning model (e.g. sumopod) each call can take
    5-15s — a 96-day backtest can run for 3-4 hours. The CLI logs
    per-node durations so you can see which step is taking time.

    Parameters
    ----------
    config_path : Path
        Path to the backtest yaml config (default: backtest.yaml).
    lookback : int | None
        Optional override for the lookback window. Must be one of
        ALLOWED_LOOKBACKS (5, 10, 20, 40, 60, 80, 100, 120, 240).
    dry_run : bool
        If True, validate config and print plan, do not execute.
    """
    console.rule(
        "[bold green]TradingAgents Walk-Forward Backtest[/bold green]",
        align="left",
    )

    # 1) Load yaml
    if not config_path.exists():
        console.print(
            f"[red]Config file not found:[/red] {config_path.resolve()}"
        )
        raise typer.Exit(1)

    import yaml
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "backtest" in raw:
        raw = raw["backtest"]

    # 2) Interactive prompt for any missing required fields
    raw = _prompt_for_missing_config(raw)

    # 3) Validate lookback if provided
    if lookback is not None and lookback not in ALLOWED_LOOKBACKS:
        console.print(
            f"[red]Invalid --lookback {lookback}. "
            f"Allowed: {ALLOWED_LOOKBACKS}[/red]"
        )
        raise typer.Exit(1)

    # 4) Use YAML lookback_days if --lookback not provided via CLI
    if lookback is None:
        lookback = raw.get("lookback_days", 240)

    # 5) Dry-run: print plan and exit
    if dry_run:
        _print_dry_run_plan(raw, lookback)
        return

    # 6) Build LLM agent config
    proceed = questionary.confirm(
        "Start backtest?", default=True
    ).ask()
    if not proceed:
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    agent_config = _build_llm_agent_config()
    agent_runner = TradingAgentsRunner(
        reports_root=raw.get("output", {}).get("reports_root", "reports"),
        agent_config=agent_config,
    )

    # 7) Build the engine with CLI overrides.
    # Use ``from_dict`` (not ``from_yaml``) so the prompted ticker /
    # start_date / end_date / initial_cash that the user just typed are
    # actually used. ``from_yaml`` re-reads the file from disk and would
    # silently drop the prompt values.
    engine = BacktestEngine.from_dict(
        raw,
        agent_runner=agent_runner,
        lookback_days=lookback,
    )

    # 7b) Ensure OHLCV data exists. If the file is missing or stale
    # (doesn't cover end_date), ask the user whether to auto-fetch from
    # yfinance. This avoids the cryptic
    # ``FileNotFoundError: OHLCV file for DEWA.JK not found`` error when
    # running against a fresh ticker.
    ok, msg = engine.ensure_ohlcv(auto_fetch=False)
    if not ok:
        console.print(f"\n[yellow]⚠️  {msg}[/yellow]")
        fetch = questionary.confirm(
            f"Auto-fetch OHLCV for {engine.config.ticker} from yfinance?",
            default=True,
        ).ask()
        if not fetch:
            console.print(
                "[red]Cannot run backtest without OHLCV data.[/red] "
                f"Place a CSV at {Path(engine.config.data.data_root) / engine.config.ticker / 'ohlcv.csv'} "
                f"or re-run with auto-fetch enabled."
            )
            raise typer.Exit(1)
        try:
            with console.status(
                f"[bold cyan]Fetching {engine.config.ticker} OHLCV "
                f"({engine.config.start_date} → {engine.config.end_date}) "
                f"from yfinance...[/bold cyan]"
            ):
                ok, msg = engine.ensure_ohlcv(auto_fetch=True)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Fetch failed:[/red] {exc}")
            raise typer.Exit(1)
        if not ok:
            console.print(f"[red]Could not fetch OHLCV:[/red] {msg}")
            raise typer.Exit(1)
        console.print(f"[green]✅[/green] {msg}")
    else:
        console.print(f"[dim]{msg}[/dim]")

    cfg = engine.config
    # Count actual market days from the OHLCV file so the progress bar is
    # accurate from the first render (the rough ``int(days * 5/7)`` would
    # show ``Day 0/1`` for short or single-day windows).
    n_days = _count_market_days(engine, cfg.start_date, cfg.end_date)
    ui = _BacktestUI(
        n_days=n_days,
        ticker=cfg.ticker,
        lookback=cfg.lookback_days,
    )
    ui.add_log("⏳", f"Loading config: {config_path}")
    ui.add_log("📅", f"Period: {cfg.start_date} → {cfg.end_date}")
    ui.add_log(
        "📈",
        f"Trading days: {n_days} (counted from OHLCV market dates)",
    )
    ui.add_log(
        "💰",
        f"Cash: {cfg.initial_cash:,.0f} IDR | "
        f"Lev: {cfg.margin.max_leverage:.1f}x",
    )
    ui.add_log("🚦", f"Lookback: {cfg.lookback_days} trading days")
    ui.add_log("🧭", f"Decision mode: {cfg.decision_mapping.mode}")
    if n_days == 1:
        ui.add_log(
            "⚠️",
            "Only 1 trading day in range — check start_date/end_date.",
        )

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=12),
    )
    layout["body"].split_column(
        Layout(name="progress", size=6),
        Layout(name="panels", ratio=1),
    )
    layout["panels"].split_row(
        Layout(name="log", ratio=1),
        Layout(name="trading", ratio=1),
    )

    summary: Optional[dict] = None
    error: Optional[BaseException] = None
    last_activity: list[float] = [time.time()]

    # Real progress callback — the runner fires this on every phase boundary.
    # No more fake phase cycling: the UI only reflects what the runner
    # actually did, and the elapsed-time spinner only ticks when the runner
    # is silent for >3 seconds (e.g. waiting for the LLM).
    def _on_progress(
        phase: str,
        trade_date: Optional[str],
        day_idx: int,
        total_days: int,
        **extra: Any,
    ) -> None:
        last_activity[0] = time.time()
        if total_days:
            ui.n_days = total_days
        if day_idx:
            ui.current_day = day_idx
        if trade_date:
            ui.current_date = trade_date
        # Reset per-day UI state (active node + node timings) at the
        # start of each trading day so the panel reflects the new
        # day, not leftovers from the previous one.
        if phase == "day_start":
            ui.clear_day_state()
        # Per-node LLM timing events. ``BacktestAgentCallback`` fires
        # ``phase="agent_node"`` once per graph node, with ``node``
        # and ``duration`` in ``extra``. These are sub-phase events
        # of the surrounding ``agent`` phase, so we render them as
        # dim log lines and update the active-node indicator
        # without flipping ``current_phase``.
        if phase == "agent_node":
            node_name = extra.get("node", "?")
            duration = extra.get("duration", 0.0)
            err = extra.get("error")
            if err:
                ui.add_log(
                    "❌",
                    f"{node_name}",
                    f"{duration:.1f}s [error: {err[:60]}]",
                )
            else:
                ui.record_node_done(node_name, duration)
            return
        # Handle executed trades — log them to the trading panel
        if phase == "execute":
            trades = extra.get("trades", [])
            action_map = {
                "SELL_TO_OPEN":   "Short(sell) entry",
                "BUY_TO_OPEN":    "Long(buy) entry",
                "BUY_TO_CLOSE":   "Cover(buy) close",
                "SELL_TO_CLOSE":  "Close(sell) close",
                "BUY_TO_ADD":     "Add Long(buy)",
                "SELL_TO_ADD":    "Add Short(sell)",
                "BUY_TO_REDUCE":  "Reduce Long(buy)",
                "SELL_TO_REDUCE": "Reduce Short(sell)",
            }
            for t in trades:
                ot = t.get("order_type", "")
                action = action_map.get(ot, ot)
                is_close = "CLOSE" in ot or "REDUCE" in ot
                realized = t.get("realized_pnl", 0.0)
                if is_close and realized != 0.0:
                    pnl_str = f"{realized:+,.0f}"
                elif is_close:
                    pnl_str = "—"
                else:
                    pnl_str = "-"
                ui.trade_log.append({
                    "date": t.get("date", trade_date or ""),
                    "action": action,
                    "price": f"{t.get('price', 0):,.0f}",
                    "qty": str(t.get("quantity", 0)),
                    "pnl": pnl_str,
                })
            ui.add_log(
                PHASE_ICONS.get(phase, "⚡"),
                f"Executed {len(trades)} order(s)",
                f"day {day_idx}/{total_days}",
            )
            return
        if phase and phase in PHASE_LABELS:
            ui._wait_logged = False
            ui.current_phase = phase
            # Build a useful detail string.
            extra_str = ""
            if phase == "trigger":
                trig = extra.get("triggered", False)
                reasons = extra.get("reasons", [])
                extra_str = (
                    f"triggered={trig}"
                    + (f" reasons={','.join(reasons)}" if reasons else "")
                )
            elif phase in ("agent", "parse", "risk", "day_start", "day_complete"):
                extra_str = f"day {day_idx}/{total_days}"
            ui.add_log(
                PHASE_ICONS.get(phase, "⏳"),
                PHASE_LABELS.get(phase, phase),
                extra_str,
            )

    # Wire the real callback into the engine before starting the runner.
    engine.progress_callback = _on_progress

    try:
        with Live(layout, refresh_per_second=4, console=console) as live:
            import threading

            def _run() -> None:
                nonlocal summary, error
                try:
                    ui.start_time = time.time()
                    logger.info("[BACKTEST] Starting engine.run() in background thread")
                    summary = engine.run()
                    logger.info(f"[BACKTEST] engine.run() completed successfully in {time.time() - ui.start_time:.3f}s")
                except BaseException as exc:  # noqa: BLE001
                    error = exc
                    logger.error(f"[BACKTEST] engine.run() failed: {type(exc).__name__}: {exc}")

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            # Watchdog: if the runner thread goes silent for >3 seconds
            # (e.g. waiting on a slow LLM), tick the heartbeat so the
            # user knows the process is still alive. The phase is set to
            # "wait" so it shows up in the progress panel itself.
            def _watchdog() -> None:
                last_heartbeat = 0.0
                stuck_start = None
                last_log_time = 0.0
                
                while True:
                    if summary is not None or error is not None:
                        return
                    if thread.is_alive():
                        time.sleep(0.5)
                        idle = time.time() - last_activity[0]
                        
                        # Detect stuck state (idle > 30 seconds)
                        if idle > 30.0 and stuck_start is None:
                            stuck_start = time.time()
                            logger.warning(f"[WATCHDOG] Thread idle for {idle:.1f}s - possible stuck state")
                        
                        # Log stuck warning every 60 seconds
                        if stuck_start and (time.time() - stuck_start) > 60:
                            stuck_duration = time.time() - stuck_start
                            logger.error(f"[WATCHDOG] Thread stuck for {stuck_duration:.0f}s!")
                            stuck_start = time.time()  # Reset to log again
                        
                        if idle > 3.0 and (time.time() - last_heartbeat) > 5.0:
                            ui.current_phase = "wait"
                            if not ui._wait_logged:
                                ui.add_log(
                                    PHASE_ICONS["wait"],
                                    PHASE_LABELS["wait"],
                                    f"idle {int(idle)}s — day {ui.current_day}/{ui.n_days}",
                                )
                                ui._wait_logged = True
                            last_heartbeat = time.time()

            wd = threading.Thread(target=_watchdog, daemon=True)
            wd.start()

            # Render the layout at 4Hz. The progress callback is the only
            # thing that updates the UI state.
            while thread.is_alive() or (summary is None and error is None):
                _render_all(layout, ui, summary)
                live.refresh()
                time.sleep(0.25)

            thread.join()
            if error:
                raise error

            ui.status = "complete"
            ui.current_phase = "complete"
            ui.current_day = ui.n_days
            inner_summary = summary.get("summary", summary) if summary else None
            if inner_summary:
                ui.trigger_triggered = inner_summary.get("trigger_triggered", 0)
                ui.trigger_skipped = inner_summary.get("trigger_skipped", 0)
                ui.trigger_total = inner_summary.get("trigger_total_decisions", 0)
            ui.add_log("✅", f"Backtest complete: {ui.n_days} days")
            _render_all(layout, ui, inner_summary)
            live.refresh()
    except KeyboardInterrupt:
        console.print("\n[yellow]Backtest cancelled by user.[/yellow]")
        raise typer.Exit(130)
    except Exception as exc:  # noqa: BLE001
        ui.status = "error"
        ui.add_log("❌", f"Error: {exc}")
        console.print(f"\n[red]Backtest failed:[/red] {exc}")
        console.print(traceback.format_exc())
        raise typer.Exit(1)

    # 8) Print final summary
    console.print()
    console.rule("[bold green]Backtest Results[/bold green]", align="left")
    if summary is None:
        console.print("[red]No summary produced.[/red]")
        return
    inner = summary.get("summary", summary) if summary else None
    _print_results_table(
        summary=inner,
        leakage=summary.get("leakage_audit", {}) or {},
        output_paths=summary.get("output_paths", {}) or {},
    )


def _estimate_trading_days(start_date: str, end_date: str) -> int:
    """
    Rough estimate of trading days between two dates (assumes ~5/7 days).
    Used only for the progress UI; the runner controls the real loop.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days
        return max(1, int(total_days * 5 / 7))
    except (ValueError, TypeError):
        return 1


def _count_market_days(engine: Any, start_date: str, end_date: str) -> int:
    """
    Count the actual trading days in the OHLCV file that fall inside
    ``[start_date, end_date]``. This is far more accurate than the rough
    calendar-day estimate and matches what the runner will actually loop
    over (the runner also uses the snapshot provider's market dates).

    Falls back to the rough estimate if the snapshot provider cannot be
    reached (e.g. ticker has no OHLCV file yet).
    """
    try:
        provider = getattr(engine, "snapshot_provider", None)
        if provider is None:
            return _estimate_trading_days(start_date, end_date)
        all_dates = provider.get_market_dates(engine.config.ticker)
        in_range = [d for d in all_dates if start_date <= d <= end_date]
        return max(1, len(in_range))
    except Exception:  # noqa: BLE001
        return _estimate_trading_days(start_date, end_date)


if __name__ == "__main__":
    backtest()
