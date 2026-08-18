# TradingAgents/graph/setup.py

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic

DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}

RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        analyst_concurrency_limit: int = 1,
        asset_type: str = "stock",
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.analyst_concurrency_limit = analyst_concurrency_limit
        self.asset_type = asset_type

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(
            selected_analysts,
            concurrency_limit=self.analyst_concurrency_limit,
        )

        try:
            from tradingagents.agents.analysts.crypto_fundamentals_analyst import (
                create_crypto_fundamentals_analyst,
            )
            _crypto_analyst_available = True
        except ImportError:
            _crypto_analyst_available = False

        if self.asset_type == "crypto" and _crypto_analyst_available:
            fundamentals_factory = lambda: create_crypto_fundamentals_analyst(self.quick_thinking_llm)
        else:
            fundamentals_factory = lambda: create_fundamentals_analyst(self.quick_thinking_llm)

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": fundamentals_factory,
        }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        def invoke_node(node, state):
            if hasattr(node, "invoke"):
                return node.invoke(state)
            return node(state)

        def analyst_node(spec, node):
            def run(state):
                branch_state = dict(state)
                branch_state["messages"] = state[spec.message_key]
                result = invoke_node(node, branch_state)
                return {
                    "messages": result.get("messages", []),
                    spec.message_key: result.get("messages", []),
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "messages"
                    },
                }

            return run

        def tool_node(spec, node):
            def run(state):
                branch_state = dict(state)
                branch_state["messages"] = state[spec.message_key]
                result = invoke_node(node, branch_state)
                return {
                    "messages": result.get("messages", []),
                    spec.message_key: result.get("messages", []),
                }

            return run

        def analyst_route(spec, route):
            def run(state):
                branch_state = dict(state)
                branch_state["messages"] = state[spec.message_key]
                destination = route(branch_state)
                return spec.completion_node if destination == spec.clear_node else destination

            return run

        # Add analyst branches. Each branch owns its tool-loop message channel.
        for spec in plan.specs:
            workflow.add_node(
                spec.agent_node,
                analyst_node(spec, analyst_factories[spec.key]()),
            )
            workflow.add_node(spec.tool_node, tool_node(spec, self.tool_nodes[spec.key]))
            workflow.add_node(spec.completion_node, lambda _state: {})

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges: fan out all selected analysts, then wait for every branch.
        for spec in plan.specs:
            workflow.add_edge(START, spec.agent_node)
            workflow.add_conditional_edges(
                spec.agent_node,
                analyst_route(
                    spec,
                    getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                ),
                [spec.tool_node, spec.completion_node],
            )
            workflow.add_edge(spec.tool_node, spec.agent_node)

        workflow.add_edge([spec.completion_node for spec in plan.specs], "Bull Researcher")

        # Add remaining edges with complete path maps to prevent routing fall-throughs
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )

        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")

        for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            workflow.add_conditional_edges(
                risk_node,
                self.conditional_logic.should_continue_risk_analysis,
                RISK_ANALYSIS_PATH_MAP,
            )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
