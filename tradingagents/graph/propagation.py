# TradingAgents/graph/propagation.py

import logging
from typing import Any, Dict, List, Optional

from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)

logger = logging.getLogger(__name__)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit
        logger.info(f"[PROPAGATOR] Initialized with max_recur_limit={max_recur_limit}")
        # List of callback handlers to be passed to graph.invoke so
        # langgraph fires ``on_chain_*`` / ``on_node_*`` events on
        # each graph node. Set by the backtest agent runner so per-
        # node LLM timings can be reported to the CLI progress UI.
        self.callbacks: Optional[List] = None

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph."""
        logger.info(f"[PROPAGATOR] Creating initial state for {company_name} on {trade_date}")
        logger.info(f"[PROPAGATOR] past_context length: {len(past_context)}")
        
        state = {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }
        
        logger.info(f"[PROPAGATOR] Initial state created with keys: {list(state.keys())}")
        return state

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        # Backtest progress path: the agent runner sets
        # ``self.callbacks`` once at construction. That list is the
        # authoritative source for per-node timing events. The
        # ``callbacks=`` arg here is the historical entry point and
        # is kept for backward compatibility.
        effective = callbacks if callbacks is not None else self.callbacks
        config = {"recursion_limit": self.max_recur_limit}
        if effective:
            config["callbacks"] = effective
        
        logger.info(f"[PROPAGATOR] Graph args: recursion_limit={self.max_recur_limit}, has_callbacks={effective is not None}")
        
        return {
            "stream_mode": "values",
            "config": config,
        }
