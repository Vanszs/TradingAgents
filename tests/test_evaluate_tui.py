from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from cli.commands.evaluate_tui import (
    SingleShotTUI,
    render_evaluation_summary,
    render_signal_summary,
)
from cli.progress_contract import ALL_TEAMS
from cli.stats_handler import StatsCallbackHandler


@dataclass
class _Signal:
    rating: object
    action: str
    take_profit: float
    stop_loss: float
    planned_entry_price: float
    entry_mode: object | None = None
    reference_price_at_signal: float | None = None


@dataclass
class _Evaluation:
    entry_date: str
    actual_entry_price: float
    planned_entry_price: float
    exit_date: str
    exit_price: float
    actual_holding_days: int
    planned_time_horizon_days: int
    realized_return_pct: float
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    planned_rr_ratio: float
    realized_rr_ratio: float
    entry_policy: str = "T1_OPEN"
    outcome: str = "HIT_TAKE_PROFIT"


def test_single_shot_tui_renders_planned_and_actual_entry_separately():
    signal = _Signal(type("Rating", (), {"value": "Buy"})(), "BUY", 195.0, 177.5, 183.75)
    evaluation = _Evaluation(
        "2025-05-27", 191.28, 183.75, "2025-05-27", 195.0, 1, 63,
        1.94, 2.17, -0.21, 0.27, 0.27,
    )

    signal_text = render_signal_summary(signal, 63)
    evaluation_text = render_evaluation_summary(evaluation, "HIT TAKE PROFIT")

    assert "**Planned Entry:** 183.75" in signal_text
    assert "**Actual Entry (T1_OPEN):** 2025-05-27 @ 191.28" in evaluation_text
    assert "**Planned Entry:** 183.75" in evaluation_text


def test_single_shot_tui_matches_main_cli_regions():
    tui = SingleShotTUI("TSM", "2025-05-25", console=Console(record=True))
    assert {child.name for child in tui.layout.children} == {"header", "main", "footer"}
    assert {child.name for child in tui.layout["main"].children} == {"upper", "analysis"}
    assert {child.name for child in tui.layout["upper"].children} == {"progress", "messages"}


def test_single_shot_tui_phase_statuses_and_report():
    tui = SingleShotTUI("TSM", "2025-05-25", console=Console(record=True))
    tui.start_phase("Market Data")
    assert tui.statuses["Market Data"] == "in_progress"
    tui.complete_phase("Market Data")
    tui.set_report("Signal", "BUY; horizon 20 trading days")
    rendered = tui.render()
    assert tui.statuses["Market Data"] == "completed"
    assert "Signal" in tui.current_report
    assert rendered["analysis"] is tui.layout["analysis"]
    assert any(kind == "System" for _, kind, _ in tui.messages)


def test_single_shot_tui_failure_is_not_completion():
    tui = SingleShotTUI("TSM", "2025-05-25", console=Console(record=True))
    tui.start_phase("Forward Evaluator")
    tui.fail_phase("Forward Evaluator", "bad data")
    assert tui.statuses["Forward Evaluator"] == "error"
    assert "bad data" in tui.messages[-1][2]


def test_single_shot_tui_tracks_graph_agents_and_reports():
    tui = SingleShotTUI("TSM", "2025-05-25", console=Console(record=True))
    tui.consume_graph_chunk({"market_report": "market", "messages": []})
    assert tui.statuses["Risk Team"] == "pending"
    assert tui.statuses["Market Analyst"] == "completed"
    assert tui.statuses["Sentiment Analyst"] == "in_progress"

    tui.consume_graph_chunk({
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "investment_debate_state": {"judge_decision": "research done"},
        "trader_investment_plan": "trade plan",
        "messages": [],
    })
    assert tui.statuses["Research Manager"] == "completed"
    assert tui.statuses["Trader"] == "completed"
    assert "Trading Team Plan" in tui.current_report


def test_single_shot_tui_renders_main_cli_columns_and_real_stats():
    stats = StatsCallbackHandler()
    stats.llm_calls = 2
    stats.tool_calls = 3
    stats.tokens_in = 100
    stats.tokens_out = 50
    tui = SingleShotTUI("TSM", "2025-05-25", console=Console(record=True), stats_handler=stats)
    tui.message("Agent", "hello")
    tui.add_tool_call("get_news", {"ticker": "TSM"})
    rendered = tui.render()
    messages = rendered["messages"].renderable.renderable
    assert isinstance(messages, Table)
    assert [column.header for column in messages.columns] == ["Time", "Type", "Content"]
    footer = rendered["footer"].renderable.renderable
    console = Console(record=True, width=120)
    console.print(footer)
    footer_text = console.export_text()
    assert "Agents:" in footer_text
    assert "LLM: 2" in footer_text
    assert "Tools: 3" in footer_text
    assert "Tokens: 100↑ 50↓" in footer_text


def test_single_shot_tui_uses_shared_team_contract():
    tui = SingleShotTUI("TSM", "2025-05-25", console=Console(record=True))
    for team, agents in ALL_TEAMS.items():
        assert tui.TEAMS[team] == agents
