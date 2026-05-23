"""Unit tests for checkpoint behavior with bluesmind provider.

Verifies:
1. Checkpoint DB dibuat saat checkpoint_enabled=True
2. Checkpoint rows tersimpan saat crash mid-pipeline
3. Checkpoint rows dihapus setelah run sukses (by design)
4. Resume bekerja dari checkpoint yang tersimpan
5. checkpoint_enabled=False tidak membuat DB
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSerializable

from tradingagents.default_config import DEFAULT_CONFIG

_PATCH_LLM     = "tradingagents.graph.trading_graph.create_llm_client"
_PATCH_YF      = "tradingagents.graph.trading_graph.yf"
_PATCH_PENDING = "tradingagents.agents.utils.memory.TradingMemoryLog.get_pending_entries"
_PATCH_CONTEXT = "tradingagents.agents.utils.memory.TradingMemoryLog.get_past_context"
_PATCH_STORE   = "tradingagents.agents.utils.memory.TradingMemoryLog.store_decision"


class _MockLLM(RunnableSerializable):
    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input, config=None, **kwargs):
        return AIMessage(content="mock", tool_calls=[], id="x")

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("mock")


class _CrashingLLM(RunnableSerializable):
    """LLM yang crash pada invoke ke-N."""
    call_count: int = 0
    crash_on: int = 3

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input, config=None, **kwargs):
        self.call_count += 1
        if self.call_count >= self.crash_on:
            raise RuntimeError("Simulated crash")
        return AIMessage(content="mock", tool_calls=[], id="x")

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("mock")


def _make_factory(llm):
    def factory(*a, **kw):
        c = MagicMock()
        c.get_llm.return_value = llm
        return c
    return factory


def _make_config(tmp_path, checkpoint_enabled: bool):
    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": "bluesmind",
        "deep_think_llm": "moonshotai/kimi-k2.6",
        "quick_think_llm": "moonshotai/kimi-k2.6",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "checkpoint_enabled": checkpoint_enabled,
        "results_dir": str(tmp_path / "results"),
        "data_cache_dir": str(tmp_path / "cache"),
    })
    return config


def _db_path(tmp_path, ticker="NVDA"):
    return tmp_path / "cache" / "checkpoints" / f"{ticker}.db"


def _checkpoint_rows(db_path) -> int:
    if not db_path.exists():
        return -1
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _set_key(monkeypatch):
    monkeypatch.setenv("BLUESMIND_API_KEY", "sk-test")


# ---------------------------------------------------------------------------
# 1. checkpoint_enabled=True — DB dibuat
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_checkpoint_db_created_when_enabled(tmp_path):
    config = _make_config(tmp_path, checkpoint_enabled=True)
    with patch(_PATCH_LLM, side_effect=_make_factory(_MockLLM())), \
         patch(_PATCH_YF), \
         patch(_PATCH_PENDING, return_value=[]), \
         patch(_PATCH_CONTEXT, return_value=""), \
         patch(_PATCH_STORE):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        g = TradingAgentsGraph(debug=False, config=config)
        g.propagate("NVDA", "2026-01-15")

    assert _db_path(tmp_path).exists(), "Checkpoint DB harus dibuat"


# ---------------------------------------------------------------------------
# 2. Run sukses — checkpoint rows dihapus (by design)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_checkpoint_cleared_after_successful_run(tmp_path):
    config = _make_config(tmp_path, checkpoint_enabled=True)
    with patch(_PATCH_LLM, side_effect=_make_factory(_MockLLM())), \
         patch(_PATCH_YF), \
         patch(_PATCH_PENDING, return_value=[]), \
         patch(_PATCH_CONTEXT, return_value=""), \
         patch(_PATCH_STORE):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        g = TradingAgentsGraph(debug=False, config=config)
        g.propagate("NVDA", "2026-01-15")

    rows = _checkpoint_rows(_db_path(tmp_path))
    assert rows == 0, (
        f"Setelah run sukses checkpoint harus dihapus (rows={rows}). "
        "Ini by design — bukan bug."
    )


# ---------------------------------------------------------------------------
# 3. Crash mid-pipeline — checkpoint rows tersimpan untuk resume
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_checkpoint_saved_after_crash(tmp_path):
    crashing_llm = _CrashingLLM()
    config = _make_config(tmp_path, checkpoint_enabled=True)
    with patch(_PATCH_LLM, side_effect=_make_factory(crashing_llm)), \
         patch(_PATCH_YF), \
         patch(_PATCH_PENDING, return_value=[]), \
         patch(_PATCH_CONTEXT, return_value=""), \
         patch(_PATCH_STORE):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        g = TradingAgentsGraph(debug=False, config=config)
        with pytest.raises(Exception):
            g.propagate("NVDA", "2026-01-15")

    rows = _checkpoint_rows(_db_path(tmp_path))
    assert rows > 0, (
        f"Setelah crash, checkpoint rows harus tersimpan untuk resume (rows={rows})"
    )


# ---------------------------------------------------------------------------
# 4. checkpoint_enabled=False — tidak ada DB
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_checkpoint_db_when_disabled(tmp_path):
    config = _make_config(tmp_path, checkpoint_enabled=False)
    with patch(_PATCH_LLM, side_effect=_make_factory(_MockLLM())), \
         patch(_PATCH_YF), \
         patch(_PATCH_PENDING, return_value=[]), \
         patch(_PATCH_CONTEXT, return_value=""), \
         patch(_PATCH_STORE):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        g = TradingAgentsGraph(debug=False, config=config)
        g.propagate("NVDA", "2026-01-15")

    assert not _db_path(tmp_path).exists(), "Tidak boleh ada DB kalau checkpoint disabled"


# ---------------------------------------------------------------------------
# 5. Resume — run kedua setelah crash melanjutkan dari checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_resume_after_crash(tmp_path):
    """Run kedua setelah crash harus detect checkpoint dan resume."""
    # Run 1: crash
    crashing_llm = _CrashingLLM()
    config = _make_config(tmp_path, checkpoint_enabled=True)
    with patch(_PATCH_LLM, side_effect=_make_factory(crashing_llm)), \
         patch(_PATCH_YF), \
         patch(_PATCH_PENDING, return_value=[]), \
         patch(_PATCH_CONTEXT, return_value=""), \
         patch(_PATCH_STORE):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        g = TradingAgentsGraph(debug=False, config=config)
        with pytest.raises(Exception):
            g.propagate("NVDA", "2026-01-15")

    assert _checkpoint_rows(_db_path(tmp_path)) > 0, "Checkpoint harus ada setelah crash"

    # Run 2: sukses — harus detect checkpoint
    from tradingagents.graph.checkpointer import checkpoint_step
    step = checkpoint_step(str(tmp_path / "cache"), "NVDA", "2026-01-15")
    assert step is not None, "checkpoint_step() harus return step number, bukan None"

    with patch(_PATCH_LLM, side_effect=_make_factory(_MockLLM())), \
         patch(_PATCH_YF), \
         patch(_PATCH_PENDING, return_value=[]), \
         patch(_PATCH_CONTEXT, return_value=""), \
         patch(_PATCH_STORE):
        g2 = TradingAgentsGraph(debug=False, config=config)
        _, decision = g2.propagate("NVDA", "2026-01-15")

    assert decision in {"Buy", "Overweight", "Hold", "Underweight", "Sell"}
    # Setelah sukses, checkpoint harus dihapus
    assert _checkpoint_rows(_db_path(tmp_path)) == 0
