import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import tradingagents.dataflows.config as config_module
from tradingagents.backtesting.snapshot_provider import SnapshotDataProvider
from tradingagents.dataflows.alpha_vantage_fundamentals import (
    _filter_reports_by_date,
)
from tradingagents.dataflows.alpha_vantage_fundamentals import (
    get_fundamentals as get_alpha_fundamentals,
)
from tradingagents.dataflows.alpha_vantage_news import get_news as get_alpha_news
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.y_finance import get_fundamentals as get_yf_fundamentals
from tradingagents.dataflows.yfinance_news import (
    _extract_article_data,
    get_global_news_yfinance,
    get_news_yfinance,
)


@pytest.fixture
def pit_config():
    original = config_module._config
    config_module._config = {
        "point_in_time_mode": True,
        "backtest_mode": False,
        "data_vendors": {"news_data": "yfinance", "fundamental_data": "alpha_vantage"},
        "tool_vendors": {},
        "news_article_limit": 20,
        "global_news_article_limit": 10,
        "global_news_lookback_days": 7,
        "global_news_queries": ["macro"],
    }
    yield
    config_module._config = original


def test_pit_route_never_falls_back_to_live_vendor(pit_config):
    calls = []
    original_methods = __import__(
        "tradingagents.dataflows.interface", fromlist=["VENDOR_METHODS"]
    ).VENDOR_METHODS
    methods = dict(original_methods)
    methods["get_news"] = {
        "yfinance": lambda *args: calls.append("yfinance") or "live",
        "snapshot": lambda *args: calls.append("snapshot") or (_ for _ in ()).throw(ConnectionError()),
    }
    module = __import__("tradingagents.dataflows.interface", fromlist=["VENDOR_METHODS"])
    module.VENDOR_METHODS = methods
    try:
        with pytest.raises(RuntimeError, match="No available vendor"):
            route_to_vendor("get_news", "TEST", "2025-01-01", "2025-01-02")
    finally:
        module.VENDOR_METHODS = original_methods
    assert calls == ["snapshot"]


def test_alpha_reports_require_publication_date_in_pit(pit_config):
    reports = {
        "quarterlyReports": [
            {"fiscalDateEnding": "2024-12-31", "reportedDate": "2025-02-20", "x": 1},
            {"fiscalDateEnding": "2024-09-30", "x": 2},
            {"fiscalDateEnding": "2024-06-30", "reportedDate": "2025-03-20", "x": 3},
        ]
    }
    filtered = _filter_reports_by_date(reports, "2025-03-01", point_in_time=True)
    assert [row["x"] for row in filtered["quarterlyReports"]] == [1]


def test_alpha_overview_without_publication_date_is_unavailable_in_pit(pit_config):
    with patch(
        "tradingagents.dataflows.alpha_vantage_fundamentals._make_api_request",
        return_value={"FiscalYearEnd": "2024-12-31", "MarketCapitalization": "1"},
    ):
        result = get_alpha_fundamentals("TEST", "2025-03-01")
    assert "No fundamentally available" in result


def test_alpha_news_pit_normalizes_timezone_before_cutoff(pit_config):
    with patch(
        "tradingagents.dataflows.alpha_vantage_news._make_api_request",
        return_value={
            "feed": [
                {"title": "timezone valid", "time_published": "2025-01-01T23:30:00+01:00"},
            ]
        },
    ):
        result = get_alpha_news("TEST", "2025-01-01", "2025-01-01")
    assert result["feed"][0]["title"] == "timezone valid"


def test_alpha_news_pit_rejects_response_without_publication_data(pit_config):
    with patch(
        "tradingagents.dataflows.alpha_vantage_news._make_api_request",
        return_value={"feed": [{"title": "undated"}]},
    ):
        result = get_alpha_news("TEST", "2025-01-01", "2025-01-01")
    assert "No news" in result


def test_yahoo_statement_without_publication_date_is_unavailable_in_pit(pit_config):
    ticker = MagicMock()
    ticker.quarterly_balance_sheet = pd.DataFrame(
        {pd.Timestamp("2024-12-31"): [100]}, index=["Total Assets"]
    )
    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=ticker):
        from tradingagents.dataflows.y_finance import get_balance_sheet
        result = get_balance_sheet("TEST", curr_date="2025-03-01")
    assert "No balance sheet" in result


def test_yahoo_info_without_publication_date_is_unavailable_in_pit(pit_config):
    ticker = MagicMock()
    ticker.info = {"longName": "TEST", "marketCap": 100}
    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=ticker):
        result = get_yf_fundamentals("TEST", "2025-03-01")
    assert "No fundamentally available" in result


def test_yfinance_flat_article_extracts_publication_timestamp():
    data = _extract_article_data(
        {"title": "Headline", "providerPublishTime": 1735689600, "publisher": "Wire"}
    )
    assert data["pub_date"] is not None
    assert data["publisher"] == "Wire"


def test_yfinance_ticker_news_pit_excludes_missing_and_late_articles(pit_config):
    ticker = MagicMock()
    ticker.get_news.return_value = [
        {"title": "missing"},
        {"title": "late", "providerPublishTime": 1735776000},
        {"title": "valid", "providerPublishTime": 1735689600},
    ]
    with patch("tradingagents.dataflows.yfinance_news.yf.Ticker", return_value=ticker):
        result = get_news_yfinance("TEST", "2025-01-01", "2025-01-01")
    assert "valid" in result
    assert "missing" not in result
    assert "late" not in result


def test_yfinance_news_pit_normalizes_timezone_before_cutoff(pit_config):
    ticker = MagicMock()
    # 2025-01-01 23:30 UTC is 2025-01-02 in UTC+01:00; still valid for T0.
    ticker.get_news.return_value = [
        {"title": "timezone valid", "pubDate": "2025-01-01T23:30:00+01:00"},
    ]
    with patch("tradingagents.dataflows.yfinance_news.yf.Ticker", return_value=ticker):
        result = get_news_yfinance("TEST", "2025-01-01", "2025-01-01")
    assert "timezone valid" in result


def test_yfinance_global_news_pit_excludes_articles_outside_lookback(pit_config):
    search = MagicMock()
    search.news = [
        {"title": "old", "providerPublishTime": 1735689600},
        {"title": "valid", "providerPublishTime": 1735862400},
    ]
    with patch("tradingagents.dataflows.yfinance_news.yf.Search", return_value=search):
        result = get_global_news_yfinance("2025-01-03", look_back_days=1, limit=10)
    assert "valid" in result
    assert "old" not in result


def test_yfinance_global_news_pit_excludes_missing_and_late_articles(pit_config):
    search = MagicMock()
    search.news = [
        {"title": "missing"},
        {"title": "late", "providerPublishTime": 1735776000},
        {"title": "valid", "providerPublishTime": 1735689600},
    ]
    with patch("tradingagents.dataflows.yfinance_news.yf.Search", return_value=search):
        result = get_global_news_yfinance("2025-01-01", look_back_days=7, limit=10)
    assert "valid" in result
    assert "missing" not in result
    assert "late" not in result


def test_snapshot_api_info_records_have_no_historical_availability(tmp_path):
    provider = SnapshotDataProvider(
        data_root=str(tmp_path / "data"),
        api_cache_dir=str(tmp_path / "cache"),
        fetch_from_api=True,
    )
    ticker = MagicMock()
    ticker.info = {"longName": "TEST", "marketCap": 100}
    ticker.quarterly_balance_sheet = pd.DataFrame()
    ticker.quarterly_income_stmt = pd.DataFrame()
    ticker.quarterly_cashflow = pd.DataFrame()
    with patch("yfinance.Ticker", return_value=ticker):
        records = provider._fetch_and_cache_fundamentals("TEST")
    assert records == []
    cached = json.loads((tmp_path / "cache" / "TEST" / "fundamentals.json").read_text())
    assert cached == []


def test_existing_snapshot_fundamentals_remain_available(tmp_path):
    symbol_dir = tmp_path / "data" / "TEST"
    symbol_dir.mkdir(parents=True)
    (symbol_dir / "fundamentals.json").write_text(
        json.dumps([{"metric": "revenue", "value": 10, "available_date": "2025-01-01"}])
    )
    provider = SnapshotDataProvider(data_root=str(tmp_path / "data"), fundamental_buffer_days=0)
    assert provider.get_fundamentals("TEST", "2025-01-02")[0]["metric"] == "revenue"


def test_backtest_sentiment_does_not_fetch_live_data_without_snapshot(monkeypatch):
    import tradingagents.agents.analysts.sentiment_analyst as sentiment
    from tradingagents.dataflows import config as config_module

    original = config_module._config
    config_module._config = {"backtest_mode": True, "snapshot_data": {}}
    try:
        monkeypatch.setattr(sentiment, "get_fear_greed_index", lambda **_: (_ for _ in ()).throw(AssertionError("live F&G")))
        monkeypatch.setattr(sentiment, "get_news", MagicMock())
        monkeypatch.setattr(sentiment, "ExaTimeTravelSearch", MagicMock())
        sentiment.get_news.invoke.return_value = "live news must not be used"
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError()
        llm.invoke.return_value.content = "report"
        result = sentiment.create_sentiment_analyst(llm)({
            "company_of_interest": "AAPL",
            "trade_date": "2025-01-01",
            "asset_type": "stock",
            "messages": [],
        })
        assert result["sentiment_report"] == "report"
        sentiment.ExaTimeTravelSearch.assert_not_called()
    finally:
        config_module._config = original


def test_crypto_tools_fail_closed_in_pit_mode(pit_config, monkeypatch):
    from tradingagents.agents.utils.crypto_fundamental_tools import (
        get_crypto_dev_activity,
        get_crypto_market_sentiment,
        get_crypto_network_metrics,
        get_crypto_onchain_news,
        get_crypto_tokenomics,
    )
    monkeypatch.setattr("tradingagents.dataflows.coingecko.requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("live coingecko")))
    monkeypatch.setattr("tradingagents.dataflows.github_activity.requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("live github")))
    monkeypatch.setattr("tradingagents.dataflows.crypto_news.requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("live rss")))
    monkeypatch.setattr("tradingagents.dataflows.fear_greed.requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("live fear greed")))

    assert "Point-in-time historical data unavailable" in get_crypto_tokenomics.invoke({"ticker": "BTC-USD"})
    assert "Point-in-time historical data unavailable" in get_crypto_dev_activity.invoke({"ticker": "BTC-USD"})
    assert "Point-in-time historical data unavailable" in get_crypto_network_metrics.invoke({"ticker": "BTC-USD"})
    assert "Point-in-time historical data unavailable" in get_crypto_market_sentiment.invoke({"ticker": "BTC-USD"})
    assert "Point-in-time historical data unavailable" in get_crypto_onchain_news.invoke({"ticker": "BTC-USD"})


def test_crypto_market_sentiment_uses_snapshot_in_pit_mode(monkeypatch):
    import tradingagents.dataflows.config as config_module
    from tradingagents.agents.utils.crypto_fundamental_tools import get_crypto_market_sentiment

    original = config_module._config
    config_module._config = {
        "point_in_time_mode": True,
        "trade_date": "2025-01-01",
        "snapshot_data": {"sentiment": [{"source": "fear_greed", "score": 25}]},
    }
    try:
        with patch("tradingagents.agents.utils.crypto_fundamental_tools.get_fear_greed_index") as fear_greed:
            result = get_crypto_market_sentiment.invoke({"ticker": "BTC-USD"})
        fear_greed.assert_not_called()
        assert "fear_greed" in result
        assert "25" in result
    finally:
        config_module._config = original


def test_memory_log_get_past_context_as_of_filtering(tmp_path):
    from tradingagents.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.md")})
    log.store_decision("NVDA", "2026-01-05", "**Rating**: Buy\n\nOld good decision.")
    log.store_decision("NVDA", "2026-01-15", "**Rating**: Sell\n\nFuture decision.")
    log.update_with_outcome("NVDA", "2026-01-05", 0.1, 0.05, 5, "Old outcome.")
    log.update_with_outcome("NVDA", "2026-01-15", -0.05, -0.02, 5, "Future outcome.")

    ctx = log.get_past_context("NVDA", as_of="2026-01-10")
    assert "2026-01-05" in ctx
    assert "2026-01-15" not in ctx


def test_snapshot_news_and_insider_enforces_date_cutoff():
    import tradingagents.dataflows.config as cfg
    from tradingagents.dataflows.snapshot import (
        snapshot_get_global_news,
        snapshot_get_insider_transactions,
        snapshot_get_news,
    )

    orig = cfg._config
    cfg._config = {
        "trade_date": "2025-01-10",
        "snapshot_data": {
            "news": [
                {"title": "valid news", "published_at": "2025-01-05T10:00:00Z"},
                {"title": "future news", "published_at": "2025-01-12T10:00:00Z"},
            ],
            "fundamentals": [
                {"metric": "insider_purchase", "value": 1000, "available_date": "2025-01-05"},
                {"metric": "insider_sale", "value": 500, "available_date": "2025-01-15"},
            ]
        }
    }
    try:
        news_res = snapshot_get_news("AAPL", start_date="2025-01-01", end_date="2025-01-10")
        assert "valid news" in news_res
        assert "future news" not in news_res

        global_news = snapshot_get_global_news(curr_date="2025-01-10")
        assert "valid news" in global_news
        assert "future news" not in global_news

        insider_res = snapshot_get_insider_transactions("AAPL")
        assert "insider_purchase" in insider_res
        assert "insider_sale" not in insider_res
    finally:
        cfg._config = orig


def test_yfinance_news_exact_end_date_boundary_in_pit(pit_config):
    ticker = MagicMock()
    # 2025-01-01 is 1735689600
    # 2025-01-02 is 1735776000
    ticker.get_news.return_value = [
        {"title": "exact day", "providerPublishTime": 1735689600},
        {"title": "next day", "providerPublishTime": 1735776000},
    ]
    with patch("tradingagents.dataflows.yfinance_news.yf.Ticker", return_value=ticker):
        res = get_news_yfinance("TEST", "2025-01-01", "2025-01-01")
    assert "exact day" in res
    assert "next day" not in res


def test_backtest_graph_disables_persistent_memory(tmp_path):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = dict(DEFAULT_CONFIG)
    config.update({
        "memory_enabled": False,
        "memory_log_path": str(tmp_path / "must-not-be-created.md"),
        "results_dir": str(tmp_path / "results"),
        "data_cache_dir": str(tmp_path / "cache"),
    })
    with patch("tradingagents.graph.trading_graph.create_llm_client") as create_client:
        client = MagicMock()
        client.get_llm.return_value = MagicMock()
        create_client.return_value = client
        graph = TradingAgentsGraph(config=config)

    assert graph.memory_log.load_entries() == []
    assert not (tmp_path / "must-not-be-created.md").exists()


def test_pit_flag_is_set_by_backtest_runtime_config():
    from tradingagents.backtesting.agent_runner import TradingAgentsRunner
    from tradingagents.backtesting.decision_schema import AgentConfig

    config = TradingAgentsRunner(agent_config=AgentConfig())._safe_agent_runtime_config()
    assert config["point_in_time_mode"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))