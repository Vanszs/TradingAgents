from __future__ import annotations

import json

from tradingagents.dataflows.exa_search import ExaTimeTravelSearch


def test_search_category_changes_exa_request(monkeypatch, tmp_path):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    requests = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": []}

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("tradingagents.dataflows.exa_search.requests.post", post)
    searcher = ExaTimeTravelSearch(cache_dir=tmp_path)
    searcher.search("AAPL", trade_date="2025-01-01", category="news")
    searcher.search("AAPL", trade_date="2025-01-01", category="general")

    assert requests[0]["category"] == "news"
    assert "category" not in requests[1]


def test_historical_search_does_not_use_live_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(
        "tradingagents.dataflows.exa_search.searxng_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live fallback used")),
    )

    result = ExaTimeTravelSearch(cache_dir=tmp_path).search("AAPL", trade_date="2020-01-01")

    assert "Historical search unavailable" in result


def test_pit_search_without_trade_date_stays_fail_closed(monkeypatch, tmp_path):
    from tradingagents.dataflows import config as config_module

    original = config_module._config
    config_module._config = {"point_in_time_mode": True, "backtest_mode": False}
    monkeypatch.setattr(
        "tradingagents.dataflows.exa_search.searxng_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live fallback used")),
    )
    try:
        result = ExaTimeTravelSearch(cache_dir=tmp_path).search("AAPL")
    finally:
        config_module._config = original

    assert "Historical search unavailable" in result


def test_historical_results_are_post_filtered(tmp_path):
    searcher = ExaTimeTravelSearch(cache_dir=tmp_path)
    key = ""
    from tradingagents.dataflows.exa_search import _compute_cache_key, _resolve_end_published_date

    key = _compute_cache_key("AAPL", _resolve_end_published_date("2025-01-01"), 5)
    (tmp_path / f"{key}.json").write_text(
        json.dumps(
            {
                "results": [
                    {"title": "valid", "publishedDate": "2024-12-31T00:00:00Z"},
                    {"title": "future", "publishedDate": "2025-01-02T00:00:00Z"},
                    {"title": "unknown"},
                    {
                        "title": "updated late",
                        "publishedDate": "2024-12-31T00:00:00Z",
                        "updatedDate": "2025-01-02T00:00:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = searcher.search("AAPL", trade_date="2025-01-01")

    assert "valid" in result
    assert "future" not in result
    assert "unknown" not in result
    assert "updated late" not in result
