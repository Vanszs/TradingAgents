from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import START

from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import GraphSetup

ANALYST_NODES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}


def _fake_factory(report_key):
    def factory(_llm):
        def analyst(state):
            return {
                "messages": [AIMessage(content=report_key)],
                report_key: report_key,
            }

        return analyst

    return factory


def _build_setup():
    factories = {
        "create_market_analyst": _fake_factory("market_report"),
        "create_sentiment_analyst": _fake_factory("sentiment_report"),
        "create_news_analyst": _fake_factory("news_report"),
        "create_fundamentals_analyst": _fake_factory("fundamentals_report"),
        "create_bull_researcher": lambda _llm: lambda _state: {},
        "create_bear_researcher": lambda _llm: lambda _state: {},
        "create_research_manager": lambda _llm: lambda _state: {},
        "create_trader": lambda _llm: lambda _state: {},
        "create_aggressive_debator": lambda _llm: lambda _state: {},
        "create_neutral_debator": lambda _llm: lambda _state: {},
        "create_conservative_debator": lambda _llm: lambda _state: {},
        "create_portfolio_manager": lambda _llm: lambda _state: {},
    }
    patches = [patch(f"tradingagents.graph.setup.{name}", factory) for name, factory in factories.items()]
    for patcher in patches:
        patcher.start()
    setup = GraphSetup(
        quick_thinking_llm=object(),
        deep_thinking_llm=object(),
        tool_nodes={key: lambda _state: {} for key in ANALYST_NODES},
        conditional_logic=ConditionalLogic(),
        analyst_concurrency_limit=2,
    )
    return setup, patches


def test_selected_analysts_fan_out_and_join_without_message_clear_nodes():
    setup, patches = _build_setup()
    try:
        workflow = setup.setup_graph(["market", "social", "news", "fundamentals"])
    finally:
        for patcher in patches:
            patcher.stop()

    assert {
        (START, agent_node) for agent_node in ANALYST_NODES.values()
    }.issubset(workflow.edges)
    assert not any(node.startswith("Msg Clear") for node in workflow.nodes)
    assert not any(
        source in ANALYST_NODES.values() and target in ANALYST_NODES.values()
        for source, target in workflow.edges
    )

    # The barrier must wait for all selected completion nodes, not selected order.
    plan = build_analyst_execution_plan(["market", "social", "news", "fundamentals"])
    expected_barrier = (
        tuple(spec.completion_node for spec in plan.specs),
        "Bull Researcher",
    )
    assert expected_barrier in workflow.waiting_edges


def test_selected_subset_only_fans_out_selected_branches():
    setup, patches = _build_setup()
    try:
        workflow = setup.setup_graph(["news", "market"])
    finally:
        for patcher in patches:
            patcher.stop()

    assert (START, "News Analyst") in workflow.edges
    assert (START, "Market Analyst") in workflow.edges
    assert (START, "Sentiment Analyst") not in workflow.edges
    assert (START, "Fundamentals Analyst") not in workflow.edges


def test_propagator_passes_langgraph_concurrency_bound():
    args = Propagator().get_graph_args(max_concurrency=3)

    assert args["config"]["max_concurrency"] == 3


def test_analyst_execution_plan_exposes_isolated_message_channels():
    from tradingagents.graph.analyst_execution import build_analyst_execution_plan

    plan = build_analyst_execution_plan(["market", "social", "news", "fundamentals"])

    assert [spec.message_key for spec in plan.specs] == [
        "market_messages",
        "social_messages",
        "news_messages",
        "fundamentals_messages",
    ]


def test_tracker_starts_all_unreported_analysts_in_parallel():
    from tradingagents.graph.analyst_execution import (
        AnalystWallTimeTracker,
        build_analyst_execution_plan,
        sync_analyst_tracker_from_chunk,
    )

    tracker = AnalystWallTimeTracker(build_analyst_execution_plan(["market", "news"]))
    sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
    sync_analyst_tracker_from_chunk(tracker, {"market_report": "done"}, now=13.0)
    sync_analyst_tracker_from_chunk(
        tracker,
        {"market_report": "done", "news_report": "done"},
        now=18.0,
    )

    assert tracker.get_wall_times() == {"market": 3.0, "news": 8.0}
