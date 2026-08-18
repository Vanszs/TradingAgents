# TradingAgents/graph/trading_graph.py

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_news,
    get_stock_data,
    get_verified_market_snapshot,
)
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.agents.utils.web_search_tools import get_web_search
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
        asset_type: str = "stock",
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
            asset_type: "stock" or "crypto"
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []
        self.asset_type = asset_type

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        self.memory_log = (
            TradingMemoryLog(self.config)
            if self.config.get("memory_enabled", True)
            else TradingMemoryLog()
        )

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            analyst_concurrency_limit=self.config.get("analyst_concurrency_limit", 1),
            asset_type=self.asset_type,
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        if self.asset_type == "crypto":
            try:
                from tradingagents.agents.utils.crypto_fundamental_tools import (
                    get_crypto_dev_activity,
                    get_crypto_market_sentiment,
                    get_crypto_network_metrics,
                    get_crypto_onchain_news,
                    get_crypto_tokenomics,
                )
                fundamentals_tools = [
                    get_crypto_tokenomics, get_crypto_dev_activity,
                    get_crypto_network_metrics, get_crypto_market_sentiment,
                    get_crypto_onchain_news,
                ]
            except ImportError:
                # Phase 1 not yet implemented — fall back to stock tools
                fundamentals_tools = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]
        else:
            fundamentals_tools = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]

        # Both stock and crypto fundamentals analysts can fall back to web
        # search for real-time context not covered by structured tools.
        fundamentals_tools = [*fundamentals_tools, get_web_search]

        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                    # Ground-truth verified market data snapshot
                    get_verified_market_snapshot,
                ]
            ),
            "social": ToolNode(
                [
                    # Sentiment analyst pre-fetches data directly (no tool-calling).
                    # This node exists for graph topology consistency but is never invoked.
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    get_news,
                    get_global_news,
                    get_web_search,
                    # get_insider_transactions only for stock — conditionally included
                    *([] if self.asset_type == "crypto" else [get_insider_transactions]),
                ]
            ),
            "fundamentals": ToolNode(fundamentals_tools),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()
        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark
        return benchmark_map.get("", "SPY")

    @staticmethod
    def _cutoff_price_history(data: pd.DataFrame, cutoff: str) -> pd.DataFrame:
        """Keep only price rows observable by the cutoff date."""
        if data is None or data.empty:
            return data

        frame = data.copy()
        if "Date" in frame.columns:
            dates = pd.to_datetime(frame["Date"], errors="coerce", utc=True)
            dates = dates.dt.tz_localize(None).dt.normalize()
        else:
            dates = pd.DatetimeIndex(
                pd.to_datetime(frame.index, errors="coerce", utc=True)
            ).tz_localize(None).normalize()
        return frame.loc[dates <= pd.Timestamp(cutoff)].copy()

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5,
        benchmark: str = "SPY", as_of: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Fetch returns using only prices observable by ``as_of`` when set.

        ``benchmark`` is the index used as the alpha baseline (resolved by the
        caller via ``_resolve_benchmark``). Returns ``(raw_return, alpha_return,
        actual_holding_days)`` or ``(None, None, None)`` if price data is
        unavailable (too recent, delisted, or network error).
        """
        try:
            import yfinance as yf

            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            stock = yf.Ticker(ticker).history(start=trade_date, end=end_str)
            bench = yf.Ticker(benchmark).history(start=trade_date, end=end_str)

            if as_of is not None:
                stock = self._cutoff_price_history(stock, as_of)
                bench = self._cutoff_price_history(bench, as_of)
            if len(stock) < 2 or len(bench) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def _resolve_pending_entries(
        self,
        ticker: str,
        as_of: Optional[str] = None,
    ) -> None:
        """Resolve pending log entries for ticker at the start of a new run."""
        from tradingagents.dataflows.config import is_point_in_time_mode

        pending = [
            e for e in self.memory_log.get_pending_entries()
            if e["ticker"] == ticker and (as_of is None or not is_point_in_time_mode() or e["date"] <= str(as_of))
        ]
        if not pending:
            return

        benchmark = self._resolve_benchmark(ticker)
        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(
                ticker,
                entry["date"],
                benchmark=benchmark,
                as_of=as_of if is_point_in_time_mode() else None,
            )
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
                benchmark_name=benchmark,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def propagate(self, company_name, trade_date, asset_type: str = None, *, on_chunk=None):
        """Run the trading agents graph for a company on a specific date.

        ``asset_type`` selects between the stock pipeline (default) and the
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.

        If ``asset_type`` is not passed, defaults to ``self.asset_type`` set
        at construction time — so callers only need to specify it once.
        """
        if asset_type is None:
            asset_type = self.asset_type
        self.ticker = company_name

        # Ensure active runtime config receives trade_date context for tools.
        set_config({"trade_date": str(trade_date), "curr_date": str(trade_date)})

        # Resolve pending memory-log entries up to this trade date in PIT runs.
        self._resolve_pending_entries(company_name, as_of=str(trade_date))

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(company_name, trade_date, asset_type=asset_type, on_chunk=on_chunk)
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def _run_graph(self, company_name, trade_date, asset_type: str = "stock", on_chunk=None):
        """Execute the graph and write the resulting state to disk and memory log."""
        graph_start_time = time.time()
        logger.info(f"[GRAPH] Starting _run_graph for {company_name} on {trade_date}")

        # Initialize state — inject memory log context for PM.
        step_start = time.time()
        past_context = self.memory_log.get_past_context(company_name, as_of=str(trade_date))
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, asset_type=asset_type, past_context=past_context
        )
        args = self.propagator.get_graph_args(
            max_concurrency=self.config.get("analyst_concurrency_limit", 1),
        )
        logger.info(f"[GRAPH] Initial state created in {time.time() - step_start:.3f}s")
        logger.info(f"[GRAPH] State keys: {list(init_agent_state.keys())}")

        # Inject thread_id so same ticker+date resumes, different date starts fresh.
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        step_start = time.time()
        stream_updates = self.debug or on_chunk is not None
        if stream_updates:
            logger.info("[GRAPH] Running in debug/stream mode...")
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if on_chunk is not None:
                    try:
                        on_chunk(chunk)
                    except Exception:
                        logger.exception("[GRAPH] Progress callback failed")
                if self.debug and chunk.get("messages"):
                    chunk["messages"][-1].pretty_print()
                trace.append(chunk)
            # Streamed chunks are per-node deltas. Merge them so the returned
            # state matches what graph.invoke() yields in the non-debug path.
            final_state = {}
            for chunk in trace:
                final_state.update(chunk)
        else:
            logger.info("[GRAPH] Invoking LangGraph...")
            final_state = self.graph.invoke(init_agent_state, **args)
        
        graph_duration = time.time() - step_start
        logger.info(f"[GRAPH] Graph execution completed in {graph_duration:.3f}s")

        # Store current state for reflection.
        self.curr_state = final_state

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run.
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )

        # Log summary of final state
        logger.info("[GRAPH] Final state summary:")
        logger.info(f"[GRAPH]   - market_report length: {len(final_state.get('market_report', ''))}")
        logger.info(f"[GRAPH]   - sentiment_report length: {len(final_state.get('sentiment_report', ''))}")
        logger.info(f"[GRAPH]   - news_report length: {len(final_state.get('news_report', ''))}")
        logger.info(f"[GRAPH]   - fundamentals_report length: {len(final_state.get('fundamentals_report', ''))}")
        logger.info(f"[GRAPH]   - final_trade_decision length: {len(final_state.get('final_trade_decision', ''))}")
        logger.info(f"[GRAPH]   - signal_contract present: {final_state.get('signal_contract') is not None}")

        total_duration = time.time() - graph_start_time
        logger.info(f"[GRAPH] Total _run_graph time: {total_duration:.3f}s")

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
            "signal_contract": (
                final_state["signal_contract"].model_dump(mode="json")
                if final_state.get("signal_contract") is not None
                else None
            ),
        }

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return parse_rating(full_signal)
