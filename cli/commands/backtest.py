"""
CLI command: tradingagents backtest

Runs the walk-forward backtest with Spot Long-Only BUY/WNS decision mapping, trigger-based
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
from rich.layout import Layout
from rich.live import Live

load_dotenv()

logger = logging.getLogger(__name__)

from cli.commands.backtest_prompts import (
    ALLOWED_LOOKBACK_CHOICES,
    _build_llm_agent_config,
    _is_valid_date,
    _prompt_for_missing_config,
    _select_lookback,
)
from cli.commands.backtest_report import (
    _print_dry_run_plan,
    _print_results_table,
    _print_window_explainer,
)
from cli.commands.backtest_tui import (
    PHASE_ICONS,
    PHASE_LABELS,
    _BacktestUI,
    _render_all,
    _render_footer,
    _render_header,
    _render_log,
    _render_progress,
    _render_trading,
)
from cli.utils import console
from tradingagents.backtesting import (
    BacktestEngine,
    TradingAgentsRunner,
)
from tradingagents.backtesting.data_window import ALLOWED_LOOKBACKS
from tradingagents.backtesting.decision_schema import AgentConfig
from tradingagents.backtesting.trigger_evaluator import TriggerConfig

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

    agent_config = _build_llm_agent_config(raw.get("agent", {}))
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
                "BUY_TO_OPEN":   "Buy entry",
                "SELL_TO_CLOSE": "Static exit",
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
