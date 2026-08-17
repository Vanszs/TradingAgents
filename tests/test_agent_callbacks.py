"""
Tests for the per-node LLM timing callback
(``BacktestAgentCallback``).

These tests cover:

* The callback fires one ``agent_node`` event per top-level graph node
  with correct ``node`` and ``duration`` fields.
* Nested chains (LLM, tool, retriever) are filtered out via
  ``parent_run_id`` and the ``KNOWN_NODES`` allowlist.
* ``on_chain_error`` surfaces the error to the progress callback
  along with the duration up to the failure point.
* The runner plumbs the progress callback into the agent runner, so
  end-to-end ``WalkForwardBacktestRunner.run`` triggers
  ``agent_node`` events when the agent_runner emits callback events.
* The CLI's ``_BacktestUI`` correctly records and renders per-node
  timings.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.backtesting.agent_callbacks import KNOWN_NODES, BacktestAgentCallback
from tradingagents.backtesting.agent_runner import TradingAgentsRunner
from tradingagents.backtesting.engine import BacktestEngine

# ---------------------------------------------------------------------------
# Capture helper
# ---------------------------------------------------------------------------


class _Capture:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, int, int, dict]] = []

    def __call__(
        self,
        phase: str,
        trade_date: str,
        day_idx: int,
        n_days: int,
        **extra: Any,
    ) -> None:
        self.events.append((phase, trade_date, day_idx, n_days, extra))


# ---------------------------------------------------------------------------
# Direct callback tests
# ---------------------------------------------------------------------------


class TestCallbackFires:
    def test_known_top_level_node_emits_event(self):
        cap = _Capture()
        h = BacktestAgentCallback(cap, trade_date="2026-05-25", day_idx=1, n_days=5)
        rid = uuid.uuid4()
        h.on_chain_start(
            {"name": "x"}, {}, run_id=rid, parent_run_id=None,
            metadata={"langgraph_node": "Market Analyst"},
        )
        time.sleep(0.01)
        h.on_chain_end({}, run_id=rid, parent_run_id=None)
        assert len(cap.events) == 1
        phase, td, di, n, extra = cap.events[0]
        assert phase == "agent_node"
        assert td == "2026-05-25"
        assert di == 1
        assert n == 5
        assert extra["node"] == "Market Analyst"
        assert extra["duration"] >= 0.01
        assert "started_at" in extra

    def test_all_known_nodes_recognized(self):
        cap = _Capture()
        h = BacktestAgentCallback(cap, trade_date="2026-05-25", day_idx=1, n_days=5)
        for name in KNOWN_NODES:
            rid = uuid.uuid4()
            h.on_chain_start(
                {"name": "x"}, {}, run_id=rid, parent_run_id=None,
                metadata={"langgraph_node": name},
            )
            h.on_chain_end({}, run_id=rid, parent_run_id=None)
        emitted_nodes = {e[4]["node"] for e in cap.events}
        assert emitted_nodes == KNOWN_NODES

    def test_nested_chain_with_parent_is_ignored(self):
        cap = _Capture()
        h = BacktestAgentCallback(cap)
        rid = uuid.uuid4()
        parent = uuid.uuid4()
        h.on_chain_start(
            {"name": "ChatOpenAI"}, {}, run_id=rid, parent_run_id=parent,
            metadata={"langgraph_node": "Market Analyst"},
        )
        h.on_chain_end({}, run_id=rid, parent_run_id=parent)
        assert cap.events == []

    def test_nested_langgraph_metadata_with_parent_is_ignored(self):
        cap = _Capture()
        h = BacktestAgentCallback(cap)
        rid = uuid.uuid4()
        parent = uuid.uuid4()
        metadata = {
            "langgraph_node": "Market Analyst",
            "langgraph_step": 1,
            "langgraph_path": ("__pregel_pull", "Market Analyst"),
        }
        h.on_chain_start({"name": "x"}, {}, run_id=rid, parent_run_id=parent, metadata=metadata)
        h.on_chain_end({}, run_id=rid, parent_run_id=parent)
        assert cap.events == []

    def test_unknown_node_name_is_ignored(self):
        cap = _Capture()
        h = BacktestAgentCallback(cap)
        rid = uuid.uuid4()
        h.on_chain_start(
            {"name": "x"}, {}, run_id=rid, parent_run_id=None,
            metadata={"langgraph_node": "Tool Node"},
        )
        h.on_chain_end({}, run_id=rid, parent_run_id=None)
        assert cap.events == []

    def test_missing_metadata_uses_tag_fallback(self):
        cap = _Capture()
        h = BacktestAgentCallback(cap)
        rid = uuid.uuid4()
        h.on_chain_start(
            {"name": "x"}, {}, run_id=rid, parent_run_id=None,
            tags=["langgraph:node:Portfolio Manager"],
        )
        h.on_chain_end({}, run_id=rid, parent_run_id=None)
        assert len(cap.events) == 1
        assert cap.events[0][4]["node"] == "Portfolio Manager"

    def test_missing_metadata_uses_serialized_name_fallback(self):
        cap = _Capture()
        h = BacktestAgentCallback(cap)
        rid = uuid.uuid4()
        h.on_chain_start(
            {"name": "Trader"}, {}, run_id=rid, parent_run_id=None,
        )
        h.on_chain_end({}, run_id=rid, parent_run_id=None)
        # "Trader" is in KNOWN_NODES so it should be accepted.
        assert len(cap.events) == 1
        assert cap.events[0][4]["node"] == "Trader"

    def test_error_path_includes_error_message(self):
        cap = _Capture()
        h = BacktestAgentCallback(cap)
        rid = uuid.uuid4()
        h.on_chain_start(
            {"name": "x"}, {}, run_id=rid, parent_run_id=None,
            metadata={"langgraph_node": "Trader"},
        )
        h.on_chain_error(ValueError("boom"), run_id=rid, parent_run_id=None)
        assert len(cap.events) == 1
        extra = cap.events[0][4]
        assert extra["node"] == "Trader"
        assert "ValueError" in extra["error"]
        assert "boom" in extra["error"]

    def test_set_day_clears_pending(self):
        cap = _Capture()
        h = BacktestAgentCallback(cap, trade_date="2026-05-25", day_idx=1, n_days=2)
        # Stale entry from a hung node on the previous day.
        rid = uuid.uuid4()
        h.on_chain_start(
            {"name": "x"}, {}, run_id=rid, parent_run_id=None,
            metadata={"langgraph_node": "Market Analyst"},
        )
        # New day starts — pending should be wiped.
        h.set_day("2026-05-26", 2, 2)
        # An end for the old run_id must not fire (it was wiped).
        h.on_chain_end({}, run_id=rid, parent_run_id=None)
        assert cap.events == []
        # New day's start fires correctly with the new context.
        rid2 = uuid.uuid4()
        h.on_chain_start(
            {"name": "x"}, {}, run_id=rid2, parent_run_id=None,
            metadata={"langgraph_node": "Market Analyst"},
        )
        h.on_chain_end({}, run_id=rid2, parent_run_id=None)
        assert len(cap.events) == 1
        assert cap.events[0][1] == "2026-05-26"
        assert cap.events[0][2] == 2

    def test_buggy_progress_callback_does_not_propagate(self):
        def bad_cb(*a, **k):
            raise RuntimeError("ui bug")

        h = BacktestAgentCallback(bad_cb)
        rid = uuid.uuid4()
        h.on_chain_start(
            {"name": "x"}, {}, run_id=rid, parent_run_id=None,
            metadata={"langgraph_node": "Market Analyst"},
        )
        # Must not raise.
        h.on_chain_end({}, run_id=rid, parent_run_id=None)


# ---------------------------------------------------------------------------
# End-to-end: callback reaches the runner
# ---------------------------------------------------------------------------


class TestRunnerPlumbing:
    """Verify that ``WalkForwardBacktestRunner`` propagates the
    progress callback down to the agent runner so the
    ``BacktestAgentCallback`` is actually attached."""

    def test_runner_forwards_progress_callback_to_agent_runner(self):
        from tradingagents.backtesting.walk_forward_runner import (
            WalkForwardBacktestRunner,
        )

        received: list[Any] = []

        class _FakeAgent(TradingAgentsRunner):
            def __init__(self):
                super().__init__(reports_root="reports")
                self.saved_progress = None

            def run(self, *args, **kwargs):
                return "## Trading Decision\n**Rating**: Hold\n"

        # Build via from_dict so we exercise the real engine path.
        import tempfile

        import yaml

        with open("backtest.yaml") as f:
            raw = yaml.safe_load(f).get("backtest", {})
        # Use the same 5-day slice the smoke test uses.
        raw["start_date"] = "2026-05-25"
        raw["end_date"] = "2026-05-29"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as t:
            yaml.safe_dump({"backtest": raw}, t)
            cfg_path = t.name

        agent = _FakeAgent()
        engine = BacktestEngine.from_yaml(
            cfg_path, agent_runner=agent, lookback_days=60
        )

        def progress(*args, **kwargs):
            received.append((args, kwargs))

        engine.progress_callback = progress
        engine.run()

        # The runner must have set the agent runner's progress
        # callback so any LLM call (real or stubbed) fires it.
        assert agent._progress is progress, (
            "WalkForwardBacktestRunner should forward progress_callback "
            "to its agent_runner so BacktestAgentCallback can attach."
        )


# ---------------------------------------------------------------------------
# CLI UI: per-node log lines & active label
# ---------------------------------------------------------------------------


class TestUiNodeTracking:
    def test_record_node_done_appends_to_log(self):
        from cli.commands.backtest import _BacktestUI

        ui = _BacktestUI(n_days=10, ticker="TEST", lookback=60)
        ui.start_time = time.time()
        ui.record_node_done("Market Analyst", 9.2)
        ui.record_node_done("Bull Researcher", 11.4)
        # node_durations captured for later use
        assert ui.node_durations == [
            ("Market Analyst", 9.2),
            ("Bull Researcher", 11.4),
        ]
        # log contains both
        rendered = " | ".join(m[2] for m in ui.log_messages)
        assert "Market Analyst" in rendered
        assert "9.2s" in rendered
        assert "Bull Researcher" in rendered

    def test_active_label_during_running_node(self):
        from cli.commands.backtest import _BacktestUI

        ui = _BacktestUI(n_days=10, ticker="TEST", lookback=60)
        ui.start_time = time.time()
        assert ui.active_node_label() == ""
        ui.record_node_start("Market Analyst")
        time.sleep(0.05)
        label = ui.active_node_label()
        assert "Market Analyst" in label
        assert "0.0" in label or "0.1" in label
        ui.record_node_done("Market Analyst", 0.5)
        assert ui.active_node_label() == ""

    def test_clear_day_state_resets(self):
        from cli.commands.backtest import _BacktestUI

        ui = _BacktestUI(n_days=10, ticker="TEST", lookback=60)
        ui.start_time = time.time()
        ui.record_node_done("Market Analyst", 9.2)
        assert len(ui.node_durations) == 1
        ui.clear_day_state()
        assert ui.node_durations == []
        assert ui.current_node == ""
        assert ui.current_node_started_at is None

    def test_node_durations_capped(self):
        from cli.commands.backtest import _BacktestUI

        ui = _BacktestUI(n_days=1000, ticker="TEST", lookback=60)
        for i in range(250):
            ui.record_node_done(f"Node {i}", 0.1)
        # Capped to 200 most-recent.
        assert len(ui.node_durations) == 200
        assert ui.node_durations[0][0] == "Node 50"
        assert ui.node_durations[-1][0] == "Node 249"
