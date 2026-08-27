"""
Static (non-live) output for the backtest command: results table,
dry-run plan, and the window explainer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from rich.table import Table

from cli.utils import console


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
        ("Win rate", f"{_fmt(summary.get('win_rate'), '.2f', '0.00')}%"),
        ("Trigger hit rate", f"{(summary.get('trigger_hit_rate') or 0.0) * 100:5.1f}%"),
        ("Trigger triggered", f"{summary.get('trigger_triggered') or 0}/{summary.get('trigger_total_decisions') or 0}"),
        ("Avg hold", f"{_fmt(summary.get('avg_holding_period_days'), '.1f', '0.0')} d"),
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
        str(decision_mapping.get("mode", "spot_long_only")),
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
        str(decision_mapping.get("allow_short_on_sell", False)),
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
            "mode": decision_mapping.get("mode", "spot_long_only"),
            "allow_explicit_reverse": decision_mapping.get(
                "allow_explicit_reverse", False
            ),
            "allow_short_on_underweight": decision_mapping.get(
                "allow_short_on_underweight", False
            ),
            "allow_short_on_sell": decision_mapping.get(
                "allow_short_on_sell", False
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
