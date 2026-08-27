import datetime
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import questionary
import typer
from dotenv import load_dotenv

# Load .env before anything else so TRADINGAGENTS_* env vars are available
load_dotenv()

import time

from rich.console import Console
from rich.live import Live

from cli.announcements import display_announcements, fetch_announcements
from cli.display import (
    update_analyst_statuses,
    update_display,
    update_research_team_status,
)
from cli.message_buffer import MessageBuffer, message_buffer
from cli.progress_contract import (
    ANALYST_ORDER as PROGRESS_ANALYST_ORDER,
)
from cli.progress_contract import (
    CANONICAL_ANALYST_ORDER,
    classify_message_type,
)
from cli.report_io import display_complete_report, save_report_to_disk
from cli.selections import build_headless_selections, get_user_selections
from cli.stats_handler import StatsCallbackHandler
from cli.utils import (
    create_cli_layout,
    get_analysis_date,
    get_ticker,
)
from tradingagents.backtesting.snapshot_provider import SnapshotDataProvider
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


def run_analysis(
    checkpoint: bool = False,
    selections: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    headless: bool = False,
    kronos: bool = False,
):
    # First get user selections (interactive or headless)
    if selections is None:
        selections = get_user_selections()

    # Historical dates run in a fail-closed point-in-time sandbox.
    analysis_date = str(selections["analysis_date"])
    try:
        parsed_analysis_date = datetime.datetime.strptime(analysis_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter("analysis_date must use YYYY-MM-DD") from exc
    if parsed_analysis_date > datetime.date.today():
        raise typer.BadParameter("analysis_date cannot be in the future")

    config = DEFAULT_CONFIG.copy()
    config["trade_date"] = analysis_date
    config["curr_date"] = analysis_date
    config["kronos_enabled"] = bool(kronos or selections.get("kronos_enabled", False))
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")
    config["checkpoint_enabled"] = checkpoint

    is_historical = parsed_analysis_date < datetime.date.today()
    config.update({
        "point_in_time_mode": is_historical,
        "backtest_mode": is_historical,
        "memory_enabled": not is_historical,
        "web_search_enabled": not is_historical,
        "snapshot_data": None,
    })
    if is_historical:
        config["data_vendors"] = {
            "core_stock_apis": "snapshot",
            "technical_indicators": "snapshot",
            "fundamental_data": "snapshot",
            "news_data": "snapshot",
        }
        config["tool_vendors"] = {"get_kronos_forecast": "snapshot"}
        provider = SnapshotDataProvider(
            data_root=config.get("data_root", "data"),
            snapshot_root=config.get("snapshot_root", "snapshots"),
            fetch_from_api=False,
            api_cache_dir=config.get("api_cache_dir", "api_cache"),
        )
        try:
            snapshot = provider.create_snapshot(selections["ticker"], analysis_date)
        except FileNotFoundError as exc:
            console.print(
                f"[red]No local snapshot data for {selections['ticker']}: {exc}[/red]\n"
                f"[yellow]Historical analysis is PIT-fail-closed (no live fetch). "
                f"Provision data first:[/yellow] "
                f"[cyan]python scripts/download_ohlcv_data.py {selections['ticker']}[/cyan]"
            )
            raise typer.Exit(1) from exc
        config["snapshot_data"] = {
            "ohlcv": snapshot.ohlcv,
            "news": snapshot.news,
            "fundamentals": snapshot.fundamentals,
            "sentiment": snapshot.sentiment,
            "broker_activity": snapshot.broker_activity,
            "spec": snapshot.spec,
        }

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (supports enum or raw string items)
    selected_set = {
        analyst.value if hasattr(analyst, "value") else str(analyst)
        for analyst in selections.get("analysts", [])
    }
    selected_analyst_keys = [a for a in CANONICAL_ANALYST_ORDER if a in selected_set]
    analyst_execution_plan = build_analyst_execution_plan(
        selected_analyst_keys,
        concurrency_limit=config["analyst_concurrency_limit"],
    )
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
        asset_type=selections["asset_type"],
    )

    # Create result directory
    base_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    v = 1
    while (base_dir / f"v{v}.0").exists():
        v += 1
    results_dir = base_dir / f"v{v}.0"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    # Initialize message buffer with selected analysts and disk targets
    message_buffer.init_for_analysis(
        selected_analyst_keys,
        log_file=log_file,
        report_dir=report_dir,
    )

    # Track start time for elapsed display
    start_time = time.time()

    # Now start the display layout
    layout = create_cli_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        display_analysts = [
            analyst.value if hasattr(analyst, "value") else str(analyst)
            for analyst in selections.get("analysts", [])
        ]
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(display_analysts)}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Update agent status to in_progress for the first analyst
        if selected_analyst_keys:
            first_analyst = get_initial_analyst_node(analyst_execution_plan)
            message_buffer.update_agent_status(first_analyst, "in_progress")
            analyst_wall_time_tracker.mark_started(selected_analyst_keys[0])
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Initialize state and get graph args with callbacks
        past_context = ""
        if hasattr(graph, "memory_log") and graph.memory_log:
            try:
                past_context = graph.memory_log.get_past_context(
                    selections["ticker"], as_of=str(selections["analysis_date"])
                )
            except Exception as e:
                logger.debug(f"Failed to retrieve past context: {e}")

        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"],
            selections["analysis_date"],
            asset_type=selections["asset_type"],
            past_context=past_context,
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(
            callbacks=[stats_handler],
            max_concurrency=config["analyst_concurrency_limit"],
        )

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process all messages in chunk, deduplicating by message ID
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in message_buffer._processed_message_ids:
                        continue
                    message_buffer._processed_message_ids.add(msg_id)

                msg_type, content = classify_message_type(message)
                if content and content.strip():
                    message_buffer.add_message(msg_type, content)

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(
                message_buffer,
                chunk,
                wall_time_tracker=analyst_wall_time_tracker,
            )

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge:
                    if message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                        )
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

            trace.append(chunk)

        # Streamed chunks are per-node deltas, not full state. Merge them
        # so every report field populated across the run is present.
        final_state = {}
        for chunk in trace:
            final_state.update(chunk)

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        # Persist decision to memory log for future reflection context
        if (
            hasattr(graph, "memory_log")
            and graph.memory_log
            and final_state.get("final_trade_decision")
        ):
            try:
                graph.memory_log.store_decision(
                    ticker=selections["ticker"],
                    trade_date=str(selections["analysis_date"]),
                    final_trade_decision=final_state["final_trade_decision"],
                )
            except Exception as e:
                logger.debug(f"Failed to store decision in memory log: {e}")

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis reporting
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")
    console.print(f"[dim]{analyst_wall_time_tracker.format_summary()}[/dim]")

    # If output_dir is given explicitly or in headless mode, write without prompting
    if output_dir is not None:
        save_path = output_dir
        report_file = save_report_to_disk(final_state, selections["ticker"], save_path, config=config)
        console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
        console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        return final_state

    if headless:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        report_file = save_report_to_disk(final_state, selections["ticker"], save_path, config=config)
        console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
        console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        return final_state

    # Interactive prompts
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path, config=config)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)

    return final_state


@app.command()
def analyze(
    ticker: Optional[str] = typer.Option(
        None,
        "--ticker",
        "-t",
        help="Stock ticker symbol to analyze (e.g. BBRI.JK, NVDA, AAPL).",
    ),
    trade_date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Target analysis date (YYYY-MM-DD). Defaults to today.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider override (e.g. openai, anthropic, google, bluesmind, ollama).",
    ),
    research_depth: Optional[int] = typer.Option(
        None,
        "--depth",
        help="Research debate depth rounds (1=Shallow, 3=Medium, 5=Deep).",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--lang",
        "-l",
        help="Output report language (e.g. English, Indonesian, Chinese).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Custom directory path to save reports, signal.json, and config.json.",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Run non-interactively without terminal prompts (ideal for cron / automated scripts).",
    ),
    checkpoint: bool = typer.Option(
        False,
        "--checkpoint",
        help="Enable checkpoint/resume: save state after each node so a crashed run can resume.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    kronos: bool = typer.Option(
        False,
        "--kronos/--no-kronos",
        help="Enable Kronos neural forecast dataflow for Technical Analyst.",
    ),
):
    """
    Run full multi-agent financial analysis.

    Can be run interactively or headlessly using CLI flags (--ticker, --date, --headless).
    """
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")

    # Build selections: if ticker is provided or headless flag is set, run non-interactively
    is_headless = headless or (ticker is not None)
    if is_headless:
        target_ticker = ticker or "SPY"
        selections = build_headless_selections(
            ticker=target_ticker,
            analysis_date=trade_date,
            provider=provider,
            research_depth=research_depth,
            language=language,
        )
    else:
        selections = None

    run_analysis(
        checkpoint=checkpoint,
        selections=selections,
        output_dir=output_dir,
        headless=is_headless,
        kronos=kronos,
    )


@app.command(name="backtest")
def backtest_cmd(
    config: Path = typer.Option(
        "backtest.yaml",
        "--config",
        "-c",
        help="Path to the backtest yaml config (default: backtest.yaml).",
    ),
    lookback: Optional[int] = typer.Option(
        None,
        "--lookback",
        help=(
            "Override lookback window in trading days. "
            "Allowed: 5, 10, 20, 40, 60, 80, 100, 120, 240. "
            "Single period per run."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and print plan, do not execute the backtest.",
    ),
):
    """
    Run walk-forward backtest.

    Spot Long-Only BUY/WNS decision mapping with trigger-based execution.
    Uses snapshot data with strict cutoffs to avoid leakage. Single period
    per invocation.
    """
    from cli.commands.backtest import backtest

    backtest(config_path=config, lookback=lookback, dry_run=dry_run)


@app.command(name="evaluate-signal")
def evaluate_signal_cli(
    ticker: Optional[str] = typer.Option(
        None,
        "--ticker",
        "-t",
        help="Stock ticker symbol (prompted when omitted).",
    ),
    trade_date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Target prediction date T0 (prompted when omitted).",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM Provider override.",
    ),
    kronos: bool = typer.Option(
        False,
        "--kronos/--no-kronos",
        help="Enable Kronos neural forecast dataflow.",
    ),
):
    """
    Run Single-Shot Agentic Prediction at T0 and Evaluate over Horizon H days.
    """
    from cli.commands.evaluate import evaluate_signal_cmd

    if ticker is None:
        ticker = get_ticker()
    if trade_date is None:
        trade_date = get_analysis_date()

    evaluate_signal_cmd(
        ticker=ticker,
        trade_date=trade_date,
        llm_provider=provider,
        kronos_enabled=kronos,
    )


@app.callback(invoke_without_command=True)
def main_menu(ctx: typer.Context):
    """Interactive top-level selector when no subcommand is provided."""
    if ctx.invoked_subcommand is not None:
        return

    display_announcements(console, fetch_announcements())

    choices = [
        questionary.Choice("📊 Analisis Saham Hari Ini (Live Interactive Analysis)", value="live"),
        questionary.Choice("🎯 Single-Shot Evaluator (Uji Prediksi Tanggal Tertentu & Hitung ROI Sinyal AI)", value="evaluate"),
        questionary.Choice("📈 Walk-Forward Backtest (Simulasi Portofolio Multi-Bulan & Akun Margin)", value="backtest"),
        questionary.Choice("❌ Keluar", value="exit"),
    ]

    selected = questionary.select(
        "Pilih Mode TradingAgents yang Ingin Dijalankan:",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:green bold"),
            ("highlighted", "fg:cyan bold"),
            ("pointer", "fg:yellow bold"),
        ]),
    ).ask()

    if selected == "live":
        run_analysis(checkpoint=False)
    elif selected == "evaluate":
        evaluate_signal_cli(ticker=None, trade_date=None, provider=None)
    elif selected == "backtest":
        backtest_cmd(config=Path("backtest.yaml"), lookback=None, dry_run=False)
    else:
        console.print("[dim]Keluar.[/dim]")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
