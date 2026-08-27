"""
CLI command: tradingagents evaluate-signal

Executes a single-shot prediction at target date T0 and evaluates the resulting
signal over horizon H trading days using the Forward Horizon Evaluator.
"""
from __future__ import annotations

import datetime
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from cli.commands.evaluate_results import write_result_bundle
from cli.commands.evaluate_tui import (
    SingleShotTUI,
    render_evaluation_summary,
    render_signal_summary,
)
from cli.stats_handler import StatsCallbackHandler
from cli.utils import detect_asset_type
from tradingagents.agents.schemas import SignalContract
from tradingagents.backtesting.horizon_evaluator import (
    EvaluationOutcome,
    EvaluationResult,
    HorizonEvaluator,
)
from tradingagents.backtesting.snapshot_provider import SnapshotDataProvider
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

console = Console()


def signal_from_final_state(final_state, ticker: str, trade_date: str) -> SignalContract:
    """Extract and validate the PM's typed SignalContract."""
    typed_signal = final_state.get("signal_contract")
    if typed_signal is None:
        raise ValueError("final_state missing signal_contract from Portfolio Manager")
    signal = (
        typed_signal
        if isinstance(typed_signal, SignalContract)
        else SignalContract.model_validate(typed_signal)
    )
    if signal.ticker != ticker or signal.signal_date != trade_date:
        raise ValueError(
            f"typed signal ticker/date mismatch: expected {ticker}/{trade_date}, "
            f"got {signal.ticker}/{signal.signal_date}"
        )
    return signal


def evaluate_signal_cmd(
    ticker: str = typer.Option(
        ...,
        "--ticker",
        "-t",
        help="Stock ticker symbol (e.g. BBRI.JK, AAPL, NVDA, BTC-USD).",
    ),
    trade_date: str = typer.Option(
        ...,
        "--date",
        "-d",
        help="Target prediction date T0 (YYYY-MM-DD).",
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM Provider override (e.g. openai, anthropic, google, bluesmind).",
    ),
    kronos_enabled: bool = typer.Option(
        False,
        "--kronos/--no-kronos",
        help="Enable the Kronos forecasting tool in the market analyst graph.",
    ),
):
    """
    Run Single-Shot Agentic Prediction at T0 and Evaluate over Horizon H days.
    """
    try:
        parsed_date = datetime.datetime.strptime(trade_date, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter("date must use YYYY-MM-DD format") from exc
    if parsed_date > datetime.date.today():
        raise typer.BadParameter("date cannot be in the future")

    stats_handler = StatsCallbackHandler()
    with SingleShotTUI(
        ticker=ticker,
        trade_date=trade_date,
        console=console,
        stats_handler=stats_handler,
    ) as tui:
        return _evaluate_signal_with_tui(
            ticker=ticker,
            trade_date=trade_date,
            llm_provider=llm_provider,
            kronos_enabled=kronos_enabled,
            tui=tui,
            stats_handler=stats_handler,
        )


def _run_forward_evaluation(
    *,
    ticker: str,
    trade_date: str,
    signal: SignalContract,
    effective_horizon: int,
    ohlcv_df: pd.DataFrame,
):
    entry_policy = "ASSUMED_AI_ENTRY" if signal.planned_entry_price is not None else "T1_OPEN"

    if signal.action == "BUY":
        eval_side = "LONG"
    else:
        # Spot Long-Only: SELL from FLAT inventory or WNS evaluates as NO_ORDER (0% return)
        eval_side = "FLAT"

    if eval_side == "FLAT":
        result = EvaluationResult(
            ticker=ticker,
            signal_date=trade_date,
            entry_date=None,
            actual_entry_price=None,
            exit_date=None,
            exit_price=0.0,
            outcome=EvaluationOutcome.NO_ORDER,
            side="FLAT",
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            planned_time_horizon_days=effective_horizon,
            actual_holding_days=0,
            realized_return_pct=0.0,
            max_favorable_excursion_pct=0.0,
            max_adverse_excursion_pct=0.0,
            planned_entry_price=signal.planned_entry_price,
            entry_policy=entry_policy,
            reference_price_at_signal=signal.reference_price_at_signal,
        )
        return result, {}

    actual_entry_price = signal.planned_entry_price
    actual_entry_timestamp = signal.signal_timestamp if actual_entry_price is not None else None

    result = HorizonEvaluator.evaluate(
        ticker=ticker,
        signal_date=trade_date,
        side=eval_side,
        take_profit=signal.take_profit,
        stop_loss=signal.stop_loss,
        time_horizon_days=effective_horizon,
        ohlcv_df=ohlcv_df,
        planned_entry_price=signal.planned_entry_price,
        actual_entry_price=actual_entry_price,
        actual_entry_date=actual_entry_timestamp,
        entry_policy=entry_policy,
        signal_timestamp=signal.signal_timestamp,
        entry_timestamp=actual_entry_timestamp,
        reference_price_at_signal=signal.reference_price_at_signal,
        max_holding_days=getattr(signal, "max_holding_days", None),
    )

    return result, {}


def _evaluate_signal_with_tui(
    *,
    ticker: str,
    trade_date: str,
    llm_provider: Optional[str],
    kronos_enabled: bool = False,
    tui: SingleShotTUI,
    stats_handler: StatsCallbackHandler,
):
    # 1. Fetch OHLCV history up to forward horizon
    tui.start_phase("Market Data")
    try:
        provider = SnapshotDataProvider(
            data_root=DEFAULT_CONFIG.get("data_root", "data"),
            snapshot_root=DEFAULT_CONFIG.get("snapshot_root", "snapshots"),
            fetch_from_api=True,  # auto-provision from yfinance when local data missing
            api_cache_dir=DEFAULT_CONFIG.get("api_cache_dir", "api_cache"),
        )
        ohlcv_df = provider.load_full_ohlcv(ticker)
    except Exception as e:
        console.print(f"[red]Market Data failed for {ticker}: {e}[/red]")
        tui.fail_phase("Market Data", e)
        raise typer.Exit(1)

    if ohlcv_df.empty:
        tui.fail_phase("Market Data", f"No market data available for {ticker}")
        raise typer.Exit(1)

    tui.complete_phase("Market Data")

    # 2. Run Single-Shot Agentic Graph at T0 (Strict Point-In-Time Sandbox)
    config_override = dict(DEFAULT_CONFIG)
    config_override["trade_date"] = trade_date
    config_override["curr_date"] = trade_date
    config_override["point_in_time_mode"] = True
    config_override["backtest_mode"] = True
    config_override["memory_enabled"] = False
    config_override["kronos_enabled"] = kronos_enabled
    hist_cut = ohlcv_df[ohlcv_df["date"].astype(str) <= trade_date].copy()
    if hist_cut.empty:
        console.print(
            f"[red]No OHLCV bars for {ticker} at or before {trade_date}. "
            f"Check the ticker or pick a later date.[/red]"
        )
        tui.fail_phase("Market Data", f"No bars <= {trade_date} for {ticker}")
        raise typer.Exit(1)
    config_override["snapshot_data"] = {
        "ohlcv": hist_cut.to_dict(orient="records"),
        "news": [],
        "fundamentals": [],
        "sentiment": [],
        "broker_activity": [],
    }
    if llm_provider:
        config_override["llm_provider"] = llm_provider

    asset_type = detect_asset_type(ticker).value

    tui.start_phase("Analyst Team")
    try:
        graph = TradingAgentsGraph(
            config=config_override,
            asset_type=asset_type,
            callbacks=[stats_handler],
            debug=False,
        )
        final_state, _ = graph.propagate(
            company_name=ticker,
            trade_date=trade_date,
            on_chunk=tui.consume_graph_chunk,
        )
    except Exception as e:
        tui.fail_phase("Analyst Team", e)
        raise typer.Exit(1)

    tui.complete_graph()

    # 3. Prefer the typed contract; retain markdown parsing for legacy states.
    try:
        signal = signal_from_final_state(
            final_state,
            ticker=ticker,
            trade_date=trade_date,
        )
        effective_horizon = signal.time_horizon_days
        if not 1 <= effective_horizon <= 252:
            raise typer.BadParameter("effective horizon must be between 1 and 252 trading days")
    except Exception as e:
        tui.fail_phase("Portfolio Manager", e)
        raise
    if signal.action not in ("HOLD", "WNS"):
        if signal.planned_entry_price is None:
            reference_cutoff = pd.Timestamp(f"{trade_date}T23:59:59+00:00")
            daily_bars = ohlcv_df[ohlcv_df["date"].astype(str) <= trade_date]
            if daily_bars.empty:
                tui.fail_phase("Market Data", "no reference market data available at signal date")
                raise typer.Exit(1)
            ref_close = float(daily_bars.iloc[-1]["close"])
            signal = signal.model_copy(update={"planned_entry_price": ref_close})
        reference_cutoff = pd.Timestamp(f"{trade_date}T23:59:59+00:00")
        daily_bars = ohlcv_df[ohlcv_df["date"].astype(str) <= trade_date]
        if daily_bars.empty:
            tui.fail_phase("Market Data", "no reference market data available at signal date")
            raise typer.Exit(1)
        reference_price = float(daily_bars.iloc[-1]["close"])

        signal = signal.model_copy(
            update={
                "signal_timestamp": reference_cutoff.isoformat(),
                "reference_price_at_signal": reference_price,
            }
        )

    tui.message("Agent", "Typed signal received")
    tui.set_report("Agent Signal Summary", render_signal_summary(signal, effective_horizon))

    # 4. Run Forward Horizon Evaluator
    tui.start_phase("Forward Evaluator")

    try:
        result, execution_data = _run_forward_evaluation(
            ticker=ticker,
            trade_date=trade_date,
            signal=signal,
            effective_horizon=effective_horizon,
            ohlcv_df=ohlcv_df,
        )
    except Exception as e:
        tui.fail_phase("Forward Evaluator", e)
        raise
    tui.complete_phase("Forward Evaluator")

    # 5. Render Evaluation Results
    label = result.outcome.value.replace("_", " ")
    tui.message("Evaluator", label)
    tui.set_report("Forward Horizon Evaluation", render_evaluation_summary(result, label))

    tui.start_phase("Result Bundle")
    try:
        output_dir = write_result_bundle(
            root="result_backtest",
            ticker=ticker,
            trade_date=trade_date,
            snapshot_data=config_override["snapshot_data"],
            agent_report="\n\n".join(
                f"## {key}\n{final_state.get(key, '')}"
                for key in (
                    "market_report",
                    "sentiment_report",
                    "news_report",
                    "fundamentals_report",
                    "investment_plan",
                    "trader_investment_plan",
                    "final_trade_decision",
                )
                if final_state.get(key)
            ),
            signal=signal,
            evaluation=result,
            execution_data=execution_data,
            config=config_override,
        )
    except Exception as e:
        tui.fail_phase("Result Bundle", e)
        raise
    tui.complete_phase("Result Bundle")
    tui.message("System", f"Result bundle saved to {output_dir}")
    tui.set_report(
        "Completed",
        f"{render_signal_summary(signal, effective_horizon)}\n\n"
        f"{render_evaluation_summary(result, label)}\n\n"
        f"**Bundle:** `{output_dir}`",
    )
    return result
