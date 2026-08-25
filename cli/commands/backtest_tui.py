"""
Live TUI for the backtest command: phase constants, the ``_BacktestUI``
state holder, and the per-panel render helpers driven at 4 Hz.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
        final_eq = summary.get("final_equity")
        final_eq_val = 0.0 if final_eq is None else float(final_eq)
        body.append(
            f"Final equity:     {final_eq_val:,.0f} IDR\n",
            style="bold",
        )
        pnl_pct = summary.get("total_return_pct")
        pnl_pct_val = 0.0 if pnl_pct is None else float(pnl_pct)
        color = "green" if pnl_pct_val >= 0 else "red"
        body.append(
            f"Total return:     {pnl_pct_val:+.2f}%\n",
            style=f"bold {color}",
        )
        body.append(
            f"Trades:           {summary.get('number_of_trades') or 0}\n"
        )
        hit_rate = summary.get("trigger_hit_rate")
        hit_rate_val = 0.0 if hit_rate is None else float(hit_rate)
        body.append(
            f"Trigger hit rate: "
            f"{hit_rate_val * 100:5.1f}% "
            f"({summary.get('trigger_triggered') or 0}/"
            f"{summary.get('trigger_total_decisions') or 0})\n"
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
