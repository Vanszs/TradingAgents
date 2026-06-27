"""
LangChain callback handler that surfaces per-node LLM timing to the
backtest progress callback.

The TradingAgents graph runs ~9 LLM calls per trading day (1 analyst +
researchers + research manager + trader + risk debators + portfolio
manager). With slow reasoning models, a single day can take 60-150s
and the user has no visibility into which node is responsible.

This callback fires ``on_chain_start`` / ``on_chain_end`` for every
langgraph node and reports the duration to the runner's
``progress_callback`` so the CLI can render per-node timing in the
progress panel.

Filtering
---------
LangChain fires ``on_chain_*`` for every nested chain too (the LLM
inside a node is itself a chain). To avoid double-counting, we
restrict emission to:

1. Top-level events (``parent_run_id is None``), AND
2. Events whose ``langgraph_node`` metadata key is set, AND
3. Events whose node name is in :data:`KNOWN_NODES`.

This guarantees one duration log per node, no chatter from internal
LLM/tool chains.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


# Node names that appear in the TradingAgents stock graph when
# ``selected_analysts=["market"]``. We restrict emission to this
# allowlist so nested chains (LLM, tool, retriever) don't pollute the
# log.
KNOWN_NODES: frozenset[str] = frozenset(
    {
        "Market Analyst",
        "Bull Researcher",
        "Bear Researcher",
        "Research Manager",
        "Trader",
        "Aggressive Analyst",
        "Conservative Analyst",
        "Neutral Analyst",
        "Portfolio Manager",
    }
)


def _noop_progress(**kwargs: Any) -> None:
    """Default no-op progress callback. Replaced by the runner."""
    return None


class BacktestAgentCallback(BaseCallbackHandler):
    """
    Times each top-level graph node and reports the duration to a
    user-supplied ``progress_callback``.

    Parameters
    ----------
    progress_callback : callable, optional
        ``progress_callback(phase, trade_date, day_idx, n_days, **extra)``
        where ``phase="agent_node"`` and ``extra`` carries
        ``{"node": str, "duration": float, "started_at": float}``.
        Defaults to a no-op so the callback is safe to construct
        unconditionally.
    trade_date : str, optional
        The current trading date. Used in the progress payload so the
        CLI can group node timings under the right day.
    day_idx : int, optional
        1-based day index within the run.
    n_days : int, optional
        Total number of trading days in the run.
    """

    def __init__(
        self,
        progress_callback: Optional[Any] = None,
        *,
        trade_date: str = "",
        day_idx: int = 0,
        n_days: int = 0,
    ) -> None:
        super().__init__()
        self._progress = progress_callback or _noop_progress
        self.trade_date = trade_date
        self.day_idx = day_idx
        self.n_days = n_days
        # run_id -> (node_name, monotonic_start). LangChain run_ids
        # are UUIDs; we key the dict by their string form.
        self._pending: dict[UUID, tuple[str, float]] = {}
        # Lock so on_chain_start/end are safe under any thread
        # interleaving the LLM client might use.
        self._lock = threading.Lock()

    def set_day(self, trade_date: str, day_idx: int, n_days: int) -> None:
        """Update the day context. Call once per trading day before
        the agent runs."""
        self.trade_date = trade_date
        self.day_idx = day_idx
        self.n_days = n_days
        # Discard any pending entries from the previous day; a hung
        # start should not leak into the next day's timing.
        with self._lock:
            self._pending.clear()

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Only top-level graph nodes have parent_run_id == None. Inner
        # chains (LLM, tool, retriever) have a parent.
        if parent_run_id is not None:
            return
        node_name = self._extract_node_name(serialized, tags, metadata)
        if node_name not in KNOWN_NODES:
            return
        with self._lock:
            self._pending[run_id] = (node_name, time.monotonic())
        logger.info(f"[AGENT_CALLBACK] Node START: {node_name} | day={self.trade_date}")

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            pending = self._pending.pop(run_id, None)
        if pending is None:
            return
        node_name, started_at = pending
        duration = time.monotonic() - started_at
        logger.info(f"[AGENT_CALLBACK] Node END: {node_name} | duration={duration:.3f}s | day={self.trade_date}")
        self._emit(node_name=node_name, duration=duration, started_at=started_at)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        # Surface the error in the log so the user can see which node
        # blew up (and how long it took before it did).
        with self._lock:
            pending = self._pending.pop(run_id, None)
        if pending is None:
            return
        node_name, started_at = pending
        duration = time.monotonic() - started_at
        logger.error(f"[AGENT_CALLBACK] Node ERROR: {node_name} | duration={duration:.3f}s | error={type(error).__name__}: {error} | day={self.trade_date}")
        self._emit(
            node_name=node_name,
            duration=duration,
            started_at=started_at,
            error=f"{type(error).__name__}: {error}",
        )

    def _emit(
        self,
        *,
        node_name: str,
        duration: float,
        started_at: float,
        error: Optional[str] = None,
    ) -> None:
        extra: dict[str, Any] = {
            "node": node_name,
            "duration": duration,
            "started_at": started_at,
        }
        if error is not None:
            extra["error"] = error
        try:
            self._progress(
                "agent_node",
                self.trade_date,
                self.day_idx,
                self.n_days,
                **extra,
            )
        except Exception:  # noqa: BLE001
            # A buggy progress callback must never poison the
            # agent run.
            pass

    @staticmethod
    def _extract_node_name(
        serialized: dict[str, Any],
        tags: Optional[list[str]],
        metadata: Optional[dict[str, Any]],
    ) -> str:
        # LangGraph populates ``metadata["langgraph_node"]`` with the
        # node name (e.g. "Market Analyst"). That's the most reliable
        # source.
        if metadata:
            node = metadata.get("langgraph_node")
            if node:
                return str(node)
        # Fallback: langgraph sometimes sets ``tags=["langgraph:node:<name>"]``
        if tags:
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("langgraph:node:"):
                    return tag.split(":", 2)[2]
        # Last resort: the chain's own ``name`` field.
        if serialized:
            name = serialized.get("name")
            if name:
                return str(name)
        return ""
