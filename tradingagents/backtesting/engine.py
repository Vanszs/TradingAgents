"""
BacktestEngine entry point for the stock margin backtester.
"""
from __future__ import annotations

from typing import Any, Optional

import yaml

from .agent_runner import TradingAgentsRunner
from .decision_schema import (
    AgentConfig,
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    LeakageGuardConfig,
    MarginConfig,
    OutputConfig,
)
from .position import DecisionMappingConfig
from .trigger_evaluator import TriggerConfig
from .walk_forward_runner import WalkForwardBacktestRunner


class BacktestEngine:
    """
    Entry point for running a stock margin backtest.

    Usage::

        engine = BacktestEngine.from_yaml("backtest.yaml")
        result = engine.run()

    The ``lookback_days`` and ``trigger`` overrides can be supplied as
    constructor kwargs to run a single-period test without editing the yaml.
    """

    def __init__(
        self,
        config: BacktestConfig,
        agent_runner: TradingAgentsRunner | None = None,
        lookback_days: Optional[int] = None,
        trigger: Optional[TriggerConfig] = None,
        progress_callback: Optional[Any] = None,
    ):
        self.config = config
        self.agent_runner = agent_runner
        self.progress_callback = progress_callback
        self._snapshot_provider: Any = None
        # Apply CLI-level overrides on top of the yaml-loaded config.
        if lookback_days is not None:
            self.config.lookback_days = lookback_days
        if trigger is not None:
            self.config.trigger = trigger

    def run(self) -> dict[str, Any]:
        runner = WalkForwardBacktestRunner(
            config=self.config,
            agent_runner=self.agent_runner,
            progress_callback=self.progress_callback,
        )
        return runner.run()

    def ensure_ohlcv(self, auto_fetch: bool = False) -> "tuple[bool, str]":
        """
        Check that the OHLCV file for the configured ticker exists and
        covers the backtest's date range. Returns ``(ok, message)``.

        If ``auto_fetch=True`` and the file is missing or stale, attempts
        to download it from yfinance using the CLI's
        :func:`cli.data_fetch.ensure_ohlcv` helper. The download is
        performed lazily so importing the engine does not require
        yfinance/pandas at import time.

        The auto-fetch is opt-in: production backtests should
        pre-download data and disable network access during the run.
        """
        from pathlib import Path

        from .snapshot_provider import SnapshotDataProvider

        data_root = self.config.data.data_root
        ticker = self.config.ticker
        csv_path = Path(data_root) / ticker / "ohlcv.csv"

        # First pass: provider's own check.
        try:
            provider = SnapshotDataProvider(
                data_root=data_root,
                snapshot_root=self.config.data.snapshot_root,
                lookback_days=self.config.lookback_days,
            )
            provider.load_full_ohlcv(ticker)
            # Also verify the file covers end_date.
            import pandas as pd

            df = pd.read_csv(csv_path)
            if "date" in df.columns and not df.empty:
                last = str(df["date"].max())[:10]
                if last >= str(self.config.end_date):
                    return True, f"OHLCV ready: {csv_path}"
            # Falls through to fetch if coverage is short.
        except FileNotFoundError:
            pass
        except ValueError as exc:
            # SnapshotDataProvider rejects lookback not in ALLOWED_LOOKBACKS.
            return False, f"Invalid config: {exc}"

        if not auto_fetch:
            return False, (
                f"OHLCV file not found or out of date for {ticker} at "
                f"{csv_path}. Re-run with auto_fetch=True to download."
            )

        # Auto-fetch path. Imported lazily so this module is usable
        # without yfinance installed.
        from cli.data_fetch import ensure_ohlcv as _cli_ensure_ohlcv

        try:
            path, fetched = _cli_ensure_ohlcv(
                ticker=ticker,
                start_date=str(self.config.start_date),
                end_date=str(self.config.end_date),
                output_root=data_root,
                lookback_days=self.config.lookback_days,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"OHLCV auto-fetch failed: {exc}"

        if fetched:
            return True, f"OHLCV downloaded to {path}"
        return True, f"OHLCV already present at {path}"

    @property
    def snapshot_provider(self) -> "SnapshotDataProvider":
        """
        Lazily build a snapshot provider for read-only inspection (e.g. the
        CLI uses this to count market days for the progress bar). The
        provider is cheap (it just opens an OHLCV file) and is the same
        one the runner will use.
        """
        if self._snapshot_provider is None:
            from .snapshot_provider import SnapshotDataProvider

            self._snapshot_provider = SnapshotDataProvider(
                data_root=self.config.data.data_root,
                snapshot_root=self.config.data.snapshot_root,
                lookback_days=self.config.lookback_days,
            )
        return self._snapshot_provider

    @classmethod
    def from_yaml(
        cls,
        path: str,
        agent_runner: TradingAgentsRunner | None = None,
        lookback_days: Optional[int] = None,
        trigger: Optional[TriggerConfig] = None,
        progress_callback: Optional[Any] = None,
    ) -> "BacktestEngine":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if "backtest" in raw:
            raw = raw["backtest"]

        config = cls._build_config(raw)
        return cls(
            config=config,
            agent_runner=agent_runner,
            lookback_days=lookback_days,
            trigger=trigger,
            progress_callback=progress_callback,
        )

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        agent_runner: TradingAgentsRunner | None = None,
        lookback_days: Optional[int] = None,
        trigger: Optional[TriggerConfig] = None,
        progress_callback: Optional[Any] = None,
    ) -> "BacktestEngine":
        """
        Build an engine from a dict that may have been mutated by the CLI
        prompt layer (e.g. the user typed a different ticker at the prompt).

        This is the **correct** entry point when the CLI accepts input
        interactively — ``from_yaml`` re-reads the file from disk, which
        would silently discard any prompt-time overrides.
        """
        if "backtest" in raw:
            raw = raw["backtest"]
        config = cls._build_config(raw)
        return cls(
            config=config,
            agent_runner=agent_runner,
            lookback_days=lookback_days,
            trigger=trigger,
            progress_callback=progress_callback,
        )

    @staticmethod
    def _build_config(raw: dict[str, Any]) -> BacktestConfig:
        execution = ExecutionConfig(**raw.get("execution", {}))
        data = DataConfig(**raw.get("data", {}))
        agent = AgentConfig(**raw.get("agent", {}))
        leakage_guard = LeakageGuardConfig(**raw.get("leakage_guard", {}))
        output = OutputConfig(**raw.get("output", {}))
        margin = MarginConfig(**raw.get("margin", {}))

        decision_mapping = DecisionMappingConfig(
            **raw.get("decision_mapping", {})
        )
        trigger_raw = raw.get("trigger")
        trigger = TriggerConfig(**trigger_raw) if trigger_raw else None

        nested_keys = {
            "execution",
            "data",
            "agent",
            "leakage_guard",
            "output",
            "margin",
            "decision_mapping",
            "trigger",
        }
        top_level = {k: v for k, v in raw.items() if k not in nested_keys}

        return BacktestConfig(
            **top_level,
            execution=execution,
            data=data,
            agent=agent,
            leakage_guard=leakage_guard,
            output=output,
            margin=margin,
            decision_mapping=decision_mapping,
            trigger=trigger,
        )
