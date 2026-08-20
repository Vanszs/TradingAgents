"""Rich Live display for the single-shot signal evaluator."""
from __future__ import annotations

import datetime
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.progress_contract import (
    ALL_TEAMS,
    ANALYST_AGENT_NAMES,
    ANALYST_ORDER,
    ANALYST_REPORT_MAP,
    REPORT_SECTIONS,
    classify_message_type,
    format_tool_args,
    short_agent_label,
)
from cli.stats_handler import StatsCallbackHandler
from cli.utils import create_cli_layout


@dataclass
class SingleShotTUI:
    ticker: str
    trade_date: str
    console: Console = field(default_factory=Console)
    stats_handler: Optional[StatsCallbackHandler] = None
    statuses: dict[str, str] = field(default_factory=dict)
    messages: deque[tuple[str, str, str]] = field(default_factory=lambda: deque(maxlen=100))
    report_sections: dict[str, str] = field(default_factory=dict)
    current_report: str = ""
    started_at: float = field(default_factory=time.time)
    _live: Optional[Live] = field(default=None, init=False, repr=False)
    _processed_message_ids: set[str] = field(default_factory=set, init=False, repr=False)

    STAGES = (
        "Market Data",
        "Analyst Team",
        "Research Team",
        "Trader",
        "Risk Team",
        "Portfolio Manager",
        "Forward Evaluator",
        "Result Bundle",
    )
    TEAMS = {
        **ALL_TEAMS,
        "Evaluation": ["Forward Evaluator", "Result Bundle"],
    }
    REPORT_TITLES = {
        "market_report": "Market Analysis",
        "sentiment_report": "Social Sentiment",
        "news_report": "News Analysis",
        "fundamentals_report": "Fundamentals Analysis",
        "investment_plan": "Research Team Decision",
        "trader_investment_plan": "Trading Team Plan",
        "final_trade_decision": "Portfolio Management Decision",
    }

    def __post_init__(self) -> None:
        agents = [agent for team in ALL_TEAMS.values() for agent in team]
        self.statuses = {name: "pending" for name in (*agents, *self.STAGES)}
        self.layout = create_cli_layout()

    def __enter__(self) -> "SingleShotTUI":
        self._live = Live(
            self.layout,
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.__enter__()
        self.refresh()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_value, traceback)
            self._live = None

    def start_phase(self, phase: str) -> None:
        self._set_status(phase, "in_progress")
        self.message("System", f"{phase} started")
        self.refresh()

    def complete_phase(self, phase: str) -> None:
        self._set_status(phase, "completed")
        self.message("System", f"{phase} completed")
        self.refresh()

    def fail_phase(self, phase: str, error: object) -> None:
        self._set_status(phase, "error")
        self.message("Error", f"{phase}: {error}")
        self.refresh()

    def complete_graph(self) -> None:
        for team in ALL_TEAMS.values():
            for agent in team:
                if self.statuses[agent] != "error":
                    self.statuses[agent] = "completed"
        for phase in self.STAGES[1:6]:
            if self.statuses[phase] != "error":
                self.statuses[phase] = "completed"
        self.refresh()

    def consume_graph_chunk(self, chunk: dict[str, Any]) -> None:
        """Consume one LangGraph values chunk without affecting graph execution."""
        try:
            self._consume_graph_chunk(chunk)
        except Exception as exc:
            self.message("Error", f"Progress update: {exc}")
        self.refresh()

    def _consume_graph_chunk(self, chunk: dict[str, Any]) -> None:
        self._consume_messages(chunk.get("messages", []))
        self._update_reports(chunk)
        self._update_agent_statuses(chunk)

    def _consume_messages(self, messages: list[Any]) -> None:
        for message in messages:
            message_id = getattr(message, "id", None)
            if message_id is not None:
                message_id = str(message_id)
                if message_id in self._processed_message_ids:
                    continue
                self._processed_message_ids.add(message_id)
            kind, content = classify_message_type(message)
            if content and content.strip():
                self.message(kind, content)
            for tool_call in getattr(message, "tool_calls", []) or []:
                if isinstance(tool_call, dict):
                    self.add_tool_call(tool_call.get("name", "tool"), tool_call.get("args", {}))
                else:
                    self.add_tool_call(tool_call.name, tool_call.args)

    def _update_reports(self, chunk: dict[str, Any]) -> None:
        for section in REPORT_SECTIONS:
            content = chunk.get(section)
            if content:
                self.report_sections[section] = str(content)
                self.current_report = (
                    f"### {self.REPORT_TITLES[section]}\n{self.report_sections[section]}"
                )

    def _update_agent_statuses(self, chunk: dict[str, Any]) -> None:
        selected = [key for key in ANALYST_ORDER if key in ANALYST_AGENT_NAMES]
        found_active = False
        all_analysts_done = True
        for key in selected:
            agent = ANALYST_AGENT_NAMES[key]
            report = self.report_sections.get(ANALYST_REPORT_MAP[key])
            if report:
                self.statuses[agent] = "completed"
            elif not found_active:
                self.statuses[agent] = "in_progress"
                found_active = True
                all_analysts_done = False
            else:
                self.statuses[agent] = "pending"
                all_analysts_done = False

        if all_analysts_done:
            debate = chunk.get("investment_debate_state") or {}
            judge_done = bool(debate.get("judge_decision"))
            for agent in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                self.statuses[agent] = "completed" if judge_done else "in_progress"
            self.statuses["Research Team"] = "completed" if judge_done else "in_progress"

            if judge_done and not chunk.get("trader_investment_plan"):
                self.statuses["Trader"] = "in_progress"

        if chunk.get("trader_investment_plan"):
            self.statuses["Trader"] = "completed"
            risk = chunk.get("risk_debate_state") or {}
            risk_done = bool(risk.get("judge_decision") or chunk.get("signal_contract"))

            for key, agent in (
                ("aggressive_history", "Aggressive Analyst"),
                ("conservative_history", "Conservative Analyst"),
                ("neutral_history", "Neutral Analyst"),
            ):
                self.statuses[agent] = "completed" if (risk.get(key) or risk_done) else "in_progress"
            self.statuses["Risk Team"] = "completed" if risk_done else "in_progress"

            if risk_done:
                self.statuses["Portfolio Manager"] = "completed"
            else:
                self.statuses["Portfolio Manager"] = "in_progress"

    def _set_status(self, name: str, status: str) -> None:
        if name not in self.statuses:
            raise KeyError(f"unknown progress item: {name}")
        self.statuses[name] = status

    def message(self, kind: str, content: object) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, kind, str(content)))

    def add_tool_call(self, tool_name: str, args: object) -> None:
        self.message("Tool", f"{tool_name}: {format_tool_args(args)}")

    def set_report(self, title: str, content: object) -> None:
        self.current_report = f"## {title}\n\n{content}"
        self.refresh()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.render())

    def render(self) -> Layout:
        self.layout["header"].update(
            Panel(
                f"[bold green]TradingAgents Single-Shot Evaluator[/bold green]\n"
                f"[dim]Ticker:[/dim] [yellow]{self.ticker}[/yellow]  "
                f"[dim]T0:[/dim] [yellow]{self.trade_date}[/yellow]  "
                "[dim]Horizon:[/dim] [yellow]Agent-selected[/yellow]",
                title="TradingAgents CLI",
                border_style="green",
                padding=(0, 2),
            )
        )

        icons = {"completed": ("✓", "green"), "in_progress": ("⟳", "cyan"), "error": ("✗", "red"), "pending": ("·", "dim")}
        progress = Text()
        for team, agents in self.TEAMS.items():
            progress.append(f" {team}\n", style="bold cyan")
            for agent in agents:
                status = self.statuses[agent]
                icon, color = icons[status]
                progress.append(f"  {icon} ", style=color)
                progress.append(f"{short_agent_label(agent)}\n", style=color if status != "pending" else "dim")
        self.layout["progress"].update(Panel(progress, title="Progress", border_style="cyan", padding=(0, 1)))

        messages = Table(
            show_header=True,
            header_style="bold magenta",
            show_footer=False,
            expand=True,
            box=box.MINIMAL,
            show_lines=True,
            padding=(0, 1),
        )
        messages.add_column("Time", style="cyan", width=8, justify="center")
        messages.add_column("Type", style="green", width=10, justify="center")
        messages.add_column("Content", style="white", no_wrap=False, ratio=1)
        for timestamp, kind, content in list(reversed(self.messages))[:12]:
            messages.add_row(timestamp, kind, Text(content[:200], overflow="fold"))
        self.layout["messages"].update(Panel(messages, title="Messages & Tools", border_style="blue", padding=(1, 2)))

        report = Markdown(self.current_report) if self.current_report else "[italic]Waiting for analysis report...[/italic]"
        self.layout["analysis"].update(Panel(report, title="Current Report", border_style="green", padding=(1, 2)))

        agent_statuses = [self.statuses[agent] for team in ALL_TEAMS.values() for agent in team]
        agents_completed = sum(status == "completed" for status in agent_statuses)
        stats_parts = [f"Agents: {agents_completed}/{len(agent_statuses)}"]
        if self.stats_handler:
            stats = self.stats_handler.get_stats()
            stats_parts.extend([f"LLM: {stats['llm_calls']}", f"Tools: {stats['tool_calls']}" ])
            tokens = f"Tokens: {stats['tokens_in']}↑ {stats['tokens_out']}↓" if stats["tokens_in"] or stats["tokens_out"] else "Tokens: --"
            stats_parts.append(tokens)
        reports_completed = sum(bool(content) for content in self.report_sections.values())
        stats_parts.append(f"Reports: {reports_completed}/{len(REPORT_SECTIONS)}")
        elapsed = int(time.time() - self.started_at)
        stats_parts.append(f"⏱ {elapsed // 60:02d}:{elapsed % 60:02d}")
        footer = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        footer.add_column("Stats", justify="center")
        footer.add_row(" | ".join(stats_parts))
        self.layout["footer"].update(Panel(footer, border_style="grey50"))
        return self.layout


def render_signal_summary(signal, horizon_days: int) -> str:
    return "\n".join(
        (
            f"- **Agent Rating:** {signal.rating.value}",
            f"- **Action:** {signal.action}",
            f"- **Planned Entry:** {signal.planned_entry_price if signal.planned_entry_price is not None else 'None'}",
            f"- **Take Profit:** {signal.take_profit if signal.take_profit is not None else 'None'}",
            f"- **Stop Loss:** {signal.stop_loss if signal.stop_loss is not None else 'None'}",
            f"- **Entry Mode:** {signal.entry_mode.value if signal.entry_mode is not None else 'None'}",
            f"- **Reference Price:** {signal.reference_price_at_signal if signal.reference_price_at_signal is not None else 'None'}",
            f"- **Horizon:** {horizon_days} trading days",
        )
    )


def render_evaluation_summary(result, label: str) -> str:
    entry_str = f"{result.entry_date} @ {result.actual_entry_price:,.2f}" if (result.entry_date and result.actual_entry_price is not None) else "None"
    exit_str = f"{result.exit_date} @ {result.exit_price:,.2f}" if (result.exit_date and result.exit_price is not None) else "None"
    lines = [
        f"### Horizon Realization: {label}",
        f"- **Actual Entry ({result.entry_policy}):** {entry_str}",
        f"- **Fill Status:** {getattr(result.outcome, 'value', result.outcome)}",
        f"- **Planned Entry:** {result.planned_entry_price if result.planned_entry_price is not None else 'None'}",
        f"- **Exit:** {exit_str}",
        f"- **Holding Period:** {result.actual_holding_days} / {result.planned_time_horizon_days} trading days",
        f"- **Realized Return:** {result.realized_return_pct:+.2f}%",
        f"- **Max Runup (MFE):** +{result.max_favorable_excursion_pct:.2f}%",
        f"- **Max Drawdown (MAE):** {result.max_adverse_excursion_pct:.2f}%",
    ]
    if getattr(result, "planned_rr_ratio", None) is not None:
        lines.append(f"- **Planned Risk:Reward:** 1:{result.planned_rr_ratio:.2f}")
    if getattr(result, "realized_rr_ratio", None) is not None:
        lines.append(f"- **Realized Risk:Reward:** 1:{result.realized_rr_ratio:.2f}")
    return "\n".join(lines)
