from unittest.mock import MagicMock

from tradingagents.graph.trading_graph import TradingAgentsGraph


def test_propagate_streams_chunks_to_callback_and_merges_final_state(tmp_path):
    final_state = {
        "final_trade_decision": "Rating: Hold",
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-10",
        "market_report": "market",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "", "current_response": "", "judge_decision": ""},
        "trader_investment_plan": "",
        "risk_debate_state": {"aggressive_history": "", "conservative_history": "", "neutral_history": "", "history": "", "judge_decision": ""},
        "investment_plan": "",
    }
    graph = MagicMock()
    graph.debug = False
    graph.config = {"results_dir": str(tmp_path), "data_cache_dir": str(tmp_path)}
    graph.asset_type = "stock"
    graph.memory_log = MagicMock()
    graph.memory_log.get_past_context.return_value = ""
    graph.memory_log.store_decision.return_value = None
    graph.propagator.create_initial_state.return_value = {"messages": []}
    graph.propagator.get_graph_args.return_value = {"stream_mode": "values", "config": {}}
    graph.graph.stream.return_value = iter([
        {"market_report": "market"},
        final_state,
    ])
    graph.process_signal = lambda _: "Hold"
    graph.log_states_dict = {}
    graph._log_state = MagicMock()
    graph._resolve_pending_entries = MagicMock()
    graph._run_graph = lambda *args, **kwargs: TradingAgentsGraph._run_graph(graph, *args, **kwargs)

    chunks = []
    result = TradingAgentsGraph.propagate(graph, "NVDA", "2026-01-10", on_chunk=chunks.append)

    assert len(chunks) == 2
    assert result == (final_state, "Hold")
    assert graph.graph.invoke.call_count == 0


def test_progress_callback_failure_does_not_abort_graph(tmp_path):
    graph = MagicMock()
    graph.debug = False
    graph.config = {"results_dir": str(tmp_path), "data_cache_dir": str(tmp_path)}
    graph.asset_type = "stock"
    graph.memory_log = MagicMock()
    graph.memory_log.get_past_context.return_value = ""
    graph.propagator.create_initial_state.return_value = {"messages": []}
    graph.propagator.get_graph_args.return_value = {"stream_mode": "values", "config": {}}
    graph.graph.stream.return_value = iter([{"final_trade_decision": "Rating: Hold"}])
    graph.signal_processor = MagicMock()
    graph.signal_processor.process_signal.return_value = "Hold"
    graph.log_states_dict = {}
    graph._log_state = MagicMock()
    graph._resolve_pending_entries = MagicMock()
    graph._run_graph = lambda *args, **kwargs: TradingAgentsGraph._run_graph(graph, *args, **kwargs)

    TradingAgentsGraph.propagate(graph, "NVDA", "2026-01-10", on_chunk=lambda _: (_ for _ in ()).throw(RuntimeError("ui")))

    assert graph.memory_log.store_decision.called
