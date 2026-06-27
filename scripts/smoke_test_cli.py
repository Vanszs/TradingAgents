"""
Offline smoke test for the new CLI command.

Bypasses interactive prompts by passing all values through kwargs.
Uses a Hold-stub agent to avoid LLM calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.backtesting import TradingAgentsRunner
from tradingagents.backtesting.decision_schema import ParsedDecision


class _HoldRunner(TradingAgentsRunner):
    """Stub agent that always returns a Hold rating."""

    def run(self, ticker, trade_date, snapshot, portfolio=None, **kwargs):  # type: ignore[override]
        return (
            "## Trading Decision\n"
            "**Rating**: Hold\n"
            "**Confidence**: 0.5\n"
            "**Action**: Hold current position.\n"
        )


def main() -> None:
    from cli.commands.backtest import _BacktestUI
    import yaml

    # Load config
    config_path = Path("backtest.yaml")
    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}
    if "backtest" in raw:
        raw = raw["backtest"]

    # The CLI now always prompts for ticker / start_date / end_date, so
    # the offline smoke test skips the prompt layer and supplies the
    # values directly through the engine.
    raw["end_date"] = "2026-05-30"  # 5-day smoke

    # Build engine
    from tradingagents.backtesting.engine import BacktestEngine
    engine = BacktestEngine.from_dict(
        raw,
        agent_runner=_HoldRunner(),
        lookback_days=60,
    )

    print(f"Smoke test: {engine.config.ticker} "
          f"{engine.config.start_date} → {engine.config.end_date}, "
          f"lookback={engine.config.lookback_days}")

    result = engine.run()
    summary = result.get("summary", {})
    print(f"  Final equity: {summary.get('final_equity', 0):,.0f} IDR")
    print(f"  Total return: {summary.get('total_return_pct', 0):+.2f}%")
    print(f"  Trigger hit rate: {summary.get('trigger_hit_rate', 0) * 100:.1f}% "
          f"({summary.get('trigger_triggered', 0)}/{summary.get('trigger_total_decisions', 0)})")
    print(f"  Leakage: {result.get('leakage_audit', {}).get('status', '?')}")
    print(f"  Rating dist: {summary.get('rating_distribution', {})}")


if __name__ == "__main__":
    main()
