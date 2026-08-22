"""
Smoke test for the per-node LLM timing callback.

Runs a 3-day backtest with a stub agent that fires
``agent_node`` events through the engine's progress callback. The
events are routed to ``_BacktestUI.record_node_done`` so we can
verify the log lines render correctly.

This is offline (no real LLM calls) but exercises the full plumbing:
``BacktestEngine.run`` → ``WalkForwardBacktestRunner`` →
``progress_callback(phase="agent_node", ...)`` → ``_BacktestUI``.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.backtesting import TradingAgentsRunner
from tradingagents.backtesting.engine import BacktestEngine


class _FakeTimedAgent(TradingAgentsRunner):
    """Stub agent that returns Hold and synthesises per-node
    ``agent_node`` events through the engine's progress callback."""

    def __init__(self, progress_callback):
        super().__init__(reports_root="reports")
        self._progress = progress_callback

    def run(self, ticker, trade_date, snapshot, portfolio=None, **kwargs):
        day_idx = kwargs.get("day_idx", 0)
        n_days = kwargs.get("n_days", 0)
        # Simulate the 9-node graph with realistic per-node durations.
        for node, dur in [
            ("Market Analyst", 0.20),
            ("Bull Researcher", 0.30),
            ("Bear Researcher", 0.30),
            ("Research Manager", 0.25),
            ("Trader", 0.20),
            ("Aggressive Analyst", 0.15),
            ("Conservative Analyst", 0.15),
            ("Neutral Analyst", 0.15),
            ("Portfolio Manager", 0.20),
        ]:
            if self._progress is not None:
                self._progress(
                    "agent_node",
                    str(trade_date),
                    day_idx,
                    n_days,
                    node=node,
                    duration=dur,
                )
        return (
            "## Trading Decision\n"
            "**Rating**: Hold\n"
            "**Confidence**: 0.5\n"
            "**Action**: Hold current position.\n"
        )


def main() -> None:
    import yaml

    from cli.commands.backtest import _BacktestUI

    config_path = Path("backtest.yaml")
    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}
    if "backtest" in raw:
        raw = raw["backtest"]
    raw["end_date"] = "2026-05-27"  # 3-day smoke

    captured = {"events": []}

    def progress(phase, trade_date, day_idx, n_days, **extra):
        captured["events"].append((phase, trade_date, day_idx, n_days, extra))

    engine = BacktestEngine.from_dict(
        raw,
        agent_runner=_FakeTimedAgent(progress),
        lookback_days=60,
    )
    engine.progress_callback = progress

    print(f"Smoke test: {engine.config.ticker} "
          f"{engine.config.start_date} → {engine.config.end_date}, "
          f"lookback={engine.config.lookback_days}")

    # Build a UI and route the events through it so we can verify the
    # per-node log lines render correctly.
    ui = _BacktestUI(n_days=3, ticker=engine.config.ticker, lookback=60)
    ui.start_time = time.time()

    def routed(phase, trade_date, day_idx, n_days, **extra):
        progress(phase, trade_date, day_idx, n_days, **extra)
        if phase == "agent_node":
            ui.record_node_done(extra["node"], extra["duration"])
        elif phase == "day_start":
            ui.clear_day_state()

    engine.progress_callback = routed
    # Re-attach to the agent runner (the engine forwards it at run
    # time, but we want the per-node events routed too).
    engine.agent_runner._progress = routed

    result = engine.run()
    summary = result.get("summary", {})

    # Per-day node durations check.
    node_events = [e for e in captured["events"] if e[0] == "agent_node"]
    print(f"\nCaptured {len(node_events)} agent_node events "
          f"({len(node_events)//9} days × 9 nodes)")
    if node_events:
        print(f"  First node: {node_events[0][4]['node']} "
              f"({node_events[0][4]['duration']:.2f}s)")
        print(f"  Last node:  {node_events[-1][4]['node']} "
              f"({node_events[-1][4]['duration']:.2f}s)")
    assert len(node_events) == 27, f"expected 27 events, got {len(node_events)}"

    # UI log lines check.
    print(f"\nUI log_messages ({len(ui.log_messages)} entries):")
    for ts, icon, msg in ui.log_messages[-15:]:
        print(f"  {ts} {icon} {msg}")

    # Per-node lines should be present.
    rendered = " | ".join(m[2] for m in ui.log_messages)
    for node in [
        "Market Analyst", "Bull Researcher", "Bear Researcher",
        "Research Manager", "Trader", "Aggressive Analyst",
        "Conservative Analyst", "Neutral Analyst", "Portfolio Manager",
    ]:
        assert node in rendered, f"missing log line for {node}"

    print(f"\nFinal equity: {summary.get('final_equity', 0):,.0f} IDR")
    print(f"Total return: {summary.get('total_return_pct', 0):+.2f}%")
    print(f"Leakage: {result.get('leakage_audit', {}).get('status', '?')}")
    print("\nSmoke test passed: per-node timing works end-to-end.")


if __name__ == "__main__":
    main()
