"""Phase 4: Comprehensive test suite for crypto fundamentals dataflows and tools."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

import tradingagents.dataflows.coingecko as coingecko_mod
import tradingagents.dataflows.crypto_id_map as id_map_mod
import tradingagents.dataflows.crypto_news as crypto_news_mod
import tradingagents.dataflows.defillama as defillama_mod
import tradingagents.dataflows.fear_greed as fear_greed_mod
import tradingagents.dataflows.github_activity as github_mod
import tradingagents.dataflows.onchain_metrics as onchain_mod

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch):
    """Clear all module-level caches before each test."""
    monkeypatch.setattr(coingecko_mod, "_CACHE", {})
    monkeypatch.setattr(id_map_mod, "_list_cache", {})
    monkeypatch.setattr(id_map_mod, "_list_cache_ts", 0.0)
    monkeypatch.setattr(github_mod, "_CACHE", {})
    monkeypatch.setattr(defillama_mod, "_CACHE", {})
    monkeypatch.setattr(onchain_mod, "_CACHE", {})
    monkeypatch.setattr(crypto_news_mod, "_CACHE", {})
    monkeypatch.setattr(fear_greed_mod, "_CACHE", None)


def _mock_response(json_data=None, status_code=200, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError()
    return resp


# ─── Class 1: TestCryptoIdMap ────────────────────────────────────────────────


class TestCryptoIdMap:
    @pytest.mark.unit
    def test_known_ticker_returns_id(self):
        from tradingagents.dataflows.crypto_id_map import ticker_to_coingecko_id

        assert ticker_to_coingecko_id("BTC-USD") == "bitcoin"

    @pytest.mark.unit
    def test_eth_ticker(self):
        from tradingagents.dataflows.crypto_id_map import ticker_to_coingecko_id

        assert ticker_to_coingecko_id("ETH-USD") == "ethereum"

    @pytest.mark.unit
    def test_unknown_ticker_returns_none(self, monkeypatch):
        from tradingagents.dataflows.crypto_id_map import ticker_to_coingecko_id

        monkeypatch.setattr(
            "tradingagents.dataflows.crypto_id_map.requests.get",
            lambda *a, **kw: _mock_response(json_data=[]),
        )
        assert ticker_to_coingecko_id("FAKECOIN-USD") is None

    @pytest.mark.unit
    def test_base_symbol_fallback(self):
        from tradingagents.dataflows.crypto_id_map import ticker_to_coingecko_id

        assert ticker_to_coingecko_id("BTC-USDC") == "bitcoin"


# ─── Class 2: TestCoinGecko ──────────────────────────────────────────────────


class TestCoinGecko:
    @pytest.mark.unit
    def test_get_tokenomics_returns_string(self, monkeypatch):
        from tradingagents.dataflows.coingecko import get_tokenomics

        fake_data = {
            "name": "Bitcoin",
            "symbol": "btc",
            "market_cap_rank": 1,
            "market_data": {
                "market_cap": {"usd": 1_000_000_000_000},
                "current_price": {"usd": 50000},
                "circulating_supply": 19_000_000,
                "total_supply": 21_000_000,
                "max_supply": 21_000_000,
                "fully_diluted_valuation": {"usd": 1_050_000_000_000},
                "total_volume": {"usd": 30_000_000_000},
                "price_change_percentage_24h": 2.5,
                "price_change_percentage_7d": -1.2,
                "price_change_percentage_30d": 10.0,
                "ath": {"usd": 69000},
                "ath_change_percentage": {"usd": -27.5},
            },
            "community_data": {},
            "description": {"en": "Bitcoin is a cryptocurrency."},
        }
        monkeypatch.setattr(
            "tradingagents.dataflows.coingecko.requests.get",
            lambda *a, **kw: _mock_response(json_data=fake_data),
        )
        result = get_tokenomics("BTC-USD")
        assert isinstance(result, str)
        assert "Market Cap" in result

    @pytest.mark.unit
    def test_get_tokenomics_unknown_ticker(self, monkeypatch):
        from tradingagents.dataflows.coingecko import get_tokenomics

        monkeypatch.setattr(
            "tradingagents.dataflows.crypto_id_map.requests.get",
            lambda *a, **kw: _mock_response(json_data=[]),
        )
        result = get_tokenomics("FAKECOIN-USD")
        assert "unavailable" in result.lower()

    @pytest.mark.unit
    def test_rate_limit_graceful_degradation(self, monkeypatch):
        from tradingagents.dataflows.coingecko import get_tokenomics

        monkeypatch.setattr(
            "tradingagents.dataflows.coingecko.requests.get",
            lambda *a, **kw: _mock_response(status_code=429),
        )
        result = get_tokenomics("BTC-USD")
        assert "unavailable" in result.lower() or "rate limit" in result.lower()

    @pytest.mark.unit
    def test_network_error_graceful_degradation(self, monkeypatch):
        from tradingagents.dataflows.coingecko import get_tokenomics

        def raise_conn_error(*a, **kw):
            raise requests.exceptions.ConnectionError("no network")

        monkeypatch.setattr(
            "tradingagents.dataflows.coingecko.requests.get", raise_conn_error
        )
        result = get_tokenomics("BTC-USD")
        assert "unavailable" in result.lower()

    @pytest.mark.unit
    def test_caching(self, monkeypatch):
        from tradingagents.dataflows.coingecko import get_tokenomics

        call_count = {"n": 0}
        fake_data = {
            "name": "Bitcoin",
            "symbol": "btc",
            "market_cap_rank": 1,
            "market_data": {
                "market_cap": {"usd": 1e12},
                "current_price": {"usd": 50000},
                "circulating_supply": 19e6,
                "total_supply": 21e6,
                "max_supply": 21e6,
                "fully_diluted_valuation": {"usd": 1.05e12},
                "total_volume": {"usd": 3e10},
                "price_change_percentage_24h": 2.5,
                "price_change_percentage_7d": -1.2,
                "price_change_percentage_30d": 10.0,
                "ath": {"usd": 69000},
                "ath_change_percentage": {"usd": -27.5},
            },
            "community_data": {},
            "description": {"en": ""},
        }

        def mock_get(*a, **kw):
            call_count["n"] += 1
            return _mock_response(json_data=fake_data)

        monkeypatch.setattr(
            "tradingagents.dataflows.coingecko.requests.get", mock_get
        )
        get_tokenomics("BTC-USD")
        get_tokenomics("BTC-USD")
        assert call_count["n"] == 1


# ─── Class 3: TestFearGreed ─────────────────────────────────────────────────


class TestFearGreed:
    @pytest.mark.unit
    def test_returns_formatted_string(self, monkeypatch):
        from tradingagents.dataflows.fear_greed import get_fear_greed_index

        fake_data = {
            "data": [
                {"value": "25", "value_classification": "Fear", "timestamp": "1700000000"},
            ]
        }
        monkeypatch.setattr(
            "tradingagents.dataflows.fear_greed.requests.get",
            lambda *a, **kw: _mock_response(json_data=fake_data),
        )
        result = get_fear_greed_index()
        assert "Fear" in result
        assert "25" in result

    @pytest.mark.unit
    def test_network_error_returns_unavailable(self, monkeypatch):
        from tradingagents.dataflows.fear_greed import get_fear_greed_index

        def raise_err(*a, **kw):
            raise requests.exceptions.ConnectionError("no network")

        monkeypatch.setattr(
            "tradingagents.dataflows.fear_greed.requests.get", raise_err
        )
        result = get_fear_greed_index()
        assert "unavailable" in result.lower()

    @pytest.mark.unit
    def test_caching(self, monkeypatch):
        from tradingagents.dataflows.fear_greed import get_fear_greed_index

        call_count = {"n": 0}
        fake_data = {
            "data": [
                {"value": "50", "value_classification": "Neutral", "timestamp": "1700000000"},
            ]
        }

        def mock_get(*a, **kw):
            call_count["n"] += 1
            return _mock_response(json_data=fake_data)

        monkeypatch.setattr(
            "tradingagents.dataflows.fear_greed.requests.get", mock_get
        )
        get_fear_greed_index()
        get_fear_greed_index()
        assert call_count["n"] == 1


# ─── Class 4: TestGithubActivity ────────────────────────────────────────────


class TestGithubActivity:
    @pytest.mark.unit
    def test_no_api_key_uses_unauthenticated(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        from tradingagents.dataflows.github_activity import _gh_headers

        headers = _gh_headers()
        assert "Authorization" not in headers

    @pytest.mark.unit
    def test_with_api_key_uses_auth_header(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test_token")
        from tradingagents.dataflows.github_activity import _gh_headers

        headers = _gh_headers()
        assert "Authorization" in headers
        assert "test_token" in headers["Authorization"]

    @pytest.mark.unit
    def test_repo_not_found_returns_unavailable(self, monkeypatch):
        from tradingagents.dataflows.github_activity import get_dev_activity

        # Mock CoinGecko _get to return data with no repos_url
        monkeypatch.setattr(
            "tradingagents.dataflows.coingecko._get",
            lambda *a, **kw: {"links": {"repos_url": {"github": []}}},
        )
        result = get_dev_activity("ETH-USD")
        assert "unavailable" in result.lower()

    @pytest.mark.unit
    def test_activity_grade_very_active(self, monkeypatch):
        from tradingagents.dataflows.github_activity import get_dev_activity

        # Mock _fetch_repo_url to return a repo
        monkeypatch.setattr(
            "tradingagents.dataflows.github_activity._fetch_repo_url",
            lambda coin_id: "https://github.com/ethereum/go-ethereum",
        )
        # Mock _fetch_github_stats
        monkeypatch.setattr(
            "tradingagents.dataflows.github_activity._fetch_github_stats",
            lambda url: {
                "stargazers_count": 40000,
                "forks_count": 15000,
                "open_issues_count": 300,
                "pushed_at": "2026-05-20T10:00:00Z",
                "subscribers_count": 2000,
            },
        )
        # Mock _fetch_commit_count_4w to return 60
        monkeypatch.setattr(
            "tradingagents.dataflows.github_activity._fetch_commit_count_4w",
            lambda owner, repo: 60,
        )
        result = get_dev_activity("ETH-USD")
        assert "Very Active" in result

    @pytest.mark.unit
    def test_activity_grade_low(self, monkeypatch):
        from tradingagents.dataflows.github_activity import get_dev_activity

        monkeypatch.setattr(
            "tradingagents.dataflows.github_activity._fetch_repo_url",
            lambda coin_id: "https://github.com/some/repo",
        )
        monkeypatch.setattr(
            "tradingagents.dataflows.github_activity._fetch_github_stats",
            lambda url: {
                "stargazers_count": 100,
                "forks_count": 10,
                "open_issues_count": 5,
                "pushed_at": "2026-01-01T00:00:00Z",
                "subscribers_count": 10,
            },
        )
        monkeypatch.setattr(
            "tradingagents.dataflows.github_activity._fetch_commit_count_4w",
            lambda owner, repo: 2,
        )
        result = get_dev_activity("ETH-USD")
        assert "Low Activity" in result

    @pytest.mark.unit
    def test_network_error_graceful(self, monkeypatch):
        from tradingagents.dataflows.github_activity import get_dev_activity

        def raise_err(*a, **kw):
            raise requests.exceptions.ConnectionError("no network")

        monkeypatch.setattr(
            "tradingagents.dataflows.github_activity._fetch_repo_url",
            lambda coin_id: "https://github.com/ethereum/go-ethereum",
        )
        monkeypatch.setattr(
            "tradingagents.dataflows.github_activity._fetch_github_stats",
            lambda url: None,
        )
        result = get_dev_activity("ETH-USD")
        assert isinstance(result, str)


# ─── Class 5: TestDeFiLlama ─────────────────────────────────────────────────


class TestDeFiLlama:
    @pytest.mark.unit
    def test_defi_token_returns_tvl(self, monkeypatch):
        from tradingagents.dataflows.defillama import get_tvl

        fake_data = {
            "currentChainTvls": {"Ethereum": 5_000_000_000, "Polygon": 500_000_000},
            "tvl": [{"totalLiquidityUSD": 4e9}] * 32,
            "category": "DEX",
            "chains": ["Ethereum", "Polygon"],
        }
        monkeypatch.setattr(
            "tradingagents.dataflows.defillama.requests.get",
            lambda *a, **kw: _mock_response(json_data=fake_data),
        )
        result = get_tvl("UNI-USD")
        assert "TVL" in result

    @pytest.mark.unit
    def test_non_defi_token_returns_not_applicable(self):
        from tradingagents.dataflows.defillama import get_tvl

        result = get_tvl("BTC-USD")
        assert "not applicable" in result.lower() or "n/a" in result.lower()

    @pytest.mark.unit
    def test_network_error_graceful(self, monkeypatch):
        from tradingagents.dataflows.defillama import get_tvl

        def raise_err(*a, **kw):
            raise requests.exceptions.ConnectionError("no network")

        monkeypatch.setattr(
            "tradingagents.dataflows.defillama.requests.get", raise_err
        )
        result = get_tvl("UNI-USD")
        assert isinstance(result, str)


# ─── Class 6: TestOnchainMetrics ─────────────────────────────────────────────


class TestOnchainMetrics:
    @pytest.mark.unit
    def test_no_api_key_returns_setup_message(self, monkeypatch):
        from tradingagents.dataflows.onchain_metrics import get_onchain_metrics

        monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
        result = get_onchain_metrics("ETH-USD")
        assert "ETHERSCAN_API_KEY" in result

    @pytest.mark.unit
    def test_unsupported_token_returns_message(self, monkeypatch):
        from tradingagents.dataflows.onchain_metrics import get_onchain_metrics

        monkeypatch.setenv("ETHERSCAN_API_KEY", "fake_key")
        result = get_onchain_metrics("FAKECOIN-USD")
        # New behavior: auto-discovery via CoinGecko; unknown coin returns informative message
        assert isinstance(result, str) and len(result) > 20

    @pytest.mark.unit
    def test_eth_native_metrics(self, monkeypatch):
        from tradingagents.dataflows.onchain_metrics import get_onchain_metrics

        monkeypatch.setenv("ETHERSCAN_API_KEY", "fake_key")
        fake_data = {
            "status": "1",
            "result": {
                "EthSupply": "120000000000000000000000000",
                "BurntFees": "3000000000000000000000000",
            },
        }
        monkeypatch.setattr(
            "tradingagents.dataflows.onchain_metrics.requests.get",
            lambda *a, **kw: _mock_response(json_data=fake_data),
        )
        result = get_onchain_metrics("ETH-USD")
        assert "ETH" in result


# ─── Class 7: TestCryptoNews ─────────────────────────────────────────────────


class TestCryptoNews:
    @pytest.mark.unit
    def test_no_api_key_returns_setup_message(self, monkeypatch):
        """RSS-based: no API key needed, always returns news or 'no coverage' message."""
        from tradingagents.dataflows.crypto_news import get_crypto_news

        # RSS-based, no key needed
        # Mock RSS feeds to return empty (simulates network unavailable)
        monkeypatch.setattr(
            "tradingagents.dataflows.crypto_news._fetch_feed",
            lambda url: [],
        )
        result = get_crypto_news("BTC-USD")
        assert isinstance(result, str) and len(result) > 20
        assert "no recent news" in result.lower() or "no api key" in result.lower() or "BTC" in result

    @pytest.mark.unit
    def test_returns_headlines(self, monkeypatch):
        """RSS-based: mock _fetch_feed to return fake articles."""
        import tradingagents.dataflows.crypto_news as cp
        from tradingagents.dataflows.crypto_news import get_crypto_news
        cp._CACHE.clear()

        fake_articles = [
            {"title": "Bitcoin hits new high", "desc": "BTC surges past $100k", "date": "Sat, 23 May 2026", "link": ""},
            {"title": "ETH upgrade coming", "desc": "ethereum upgrade details", "date": "Fri, 22 May 2026", "link": ""},
            {"title": "Market crash fears", "desc": "bitcoin market analysis", "date": "Fri, 22 May 2026", "link": ""},
        ]
        monkeypatch.setattr(
            "tradingagents.dataflows.crypto_news._fetch_feed",
            lambda url: fake_articles,
        )
        result = get_crypto_news("BTC-USD")
        assert "Bitcoin hits new high" in result
        assert "ETH upgrade coming" in result or "Market crash fears" in result

    @pytest.mark.unit
    def test_no_coverage_for_unknown_coin(self, monkeypatch):
        """Unknown meme coin with no news returns informative message."""
        import tradingagents.dataflows.crypto_news as cp
        from tradingagents.dataflows.crypto_news import get_crypto_news
        cp._CACHE.clear()

        monkeypatch.setattr(
            "tradingagents.dataflows.crypto_news._fetch_feed",
            lambda url: [],
        )
        result = get_crypto_news("FAKECOIN-USD")
        assert isinstance(result, str)
        assert "no recent news" in result.lower() or "no coverage" in result.lower() or "FAKECOIN" in result

    @pytest.mark.unit
    def test_network_error_graceful(self, monkeypatch):
        import tradingagents.dataflows.crypto_news as cp
        from tradingagents.dataflows.crypto_news import get_crypto_news
        cp._CACHE.clear()

        # Simulate all RSS feeds failing
        monkeypatch.setattr(
            "tradingagents.dataflows.crypto_news._fetch_feed",
            lambda url: [],
        )
        result = get_crypto_news("BTC-USD")
        assert isinstance(result, str) and len(result) > 10


# ─── Class 8: TestCryptoFundamentalTools ─────────────────────────────────────


class TestCryptoFundamentalTools:
    @pytest.mark.unit
    def test_all_tools_return_strings(self, monkeypatch):
        from tradingagents.agents.utils.crypto_fundamental_tools import (
            get_crypto_dev_activity,
            get_crypto_market_sentiment,
            get_crypto_network_metrics,
            get_crypto_onchain_news,
            get_crypto_tokenomics,
        )

        monkeypatch.setattr("tradingagents.dataflows.coingecko.get_tokenomics", lambda t: "tokenomics data")
        monkeypatch.setattr("tradingagents.dataflows.github_activity.get_dev_activity", lambda t: "dev data")
        monkeypatch.setattr("tradingagents.dataflows.coingecko.get_market_data", lambda t: "market data")
        monkeypatch.setattr("tradingagents.dataflows.defillama.get_tvl", lambda t: "tvl data")
        monkeypatch.setattr("tradingagents.dataflows.fear_greed.get_fear_greed_index", lambda: "fear greed")
        monkeypatch.setattr("tradingagents.dataflows.coingecko._get", lambda *a, **kw: None)
        monkeypatch.setattr("tradingagents.dataflows.onchain_metrics.get_onchain_metrics", lambda t: "onchain")
        monkeypatch.setattr("tradingagents.dataflows.crypto_news.get_crypto_news", lambda t, **kw: "news")

        tools = [
            get_crypto_tokenomics,
            get_crypto_dev_activity,
            get_crypto_network_metrics,
            get_crypto_market_sentiment,
            get_crypto_onchain_news,
        ]
        for tool in tools:
            result = tool.invoke({"ticker": "BTC-USD"})
            assert isinstance(result, str)

    @pytest.mark.unit
    def test_get_crypto_tokenomics_tool(self, monkeypatch):
        from tradingagents.agents.utils.crypto_fundamental_tools import get_crypto_tokenomics

        monkeypatch.setattr("tradingagents.dataflows.coingecko.get_tokenomics", lambda t: "tok data")
        result = get_crypto_tokenomics.invoke({"ticker": "BTC-USD"})
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_get_crypto_dev_activity_tool(self, monkeypatch):
        from tradingagents.agents.utils.crypto_fundamental_tools import get_crypto_dev_activity

        monkeypatch.setattr("tradingagents.dataflows.github_activity.get_dev_activity", lambda t: "dev info")
        result = get_crypto_dev_activity.invoke({"ticker": "ETH-USD"})
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_get_crypto_network_metrics_tool(self, monkeypatch):
        from tradingagents.agents.utils.crypto_fundamental_tools import get_crypto_network_metrics

        monkeypatch.setattr("tradingagents.dataflows.coingecko.get_market_data", lambda t: "price")
        monkeypatch.setattr("tradingagents.dataflows.defillama.get_tvl", lambda t: "tvl")
        result = get_crypto_network_metrics.invoke({"ticker": "BTC-USD"})
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_get_crypto_market_sentiment_tool(self, monkeypatch):
        from tradingagents.agents.utils.crypto_fundamental_tools import get_crypto_market_sentiment

        monkeypatch.setattr("tradingagents.dataflows.fear_greed.get_fear_greed_index", lambda: "sentiment")
        monkeypatch.setattr("tradingagents.dataflows.coingecko._get", lambda *a, **kw: None)
        result = get_crypto_market_sentiment.invoke({"ticker": "BTC-USD"})
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_get_crypto_onchain_news_tool(self, monkeypatch):
        from tradingagents.agents.utils.crypto_fundamental_tools import get_crypto_onchain_news

        monkeypatch.setattr("tradingagents.dataflows.onchain_metrics.get_onchain_metrics", lambda t: "onchain")
        monkeypatch.setattr("tradingagents.dataflows.crypto_news.get_crypto_news", lambda t, **kw: "news")
        result = get_crypto_onchain_news.invoke({"ticker": "ETH-USD"})
        assert isinstance(result, str)


# ─── Class 9: TestCryptoFundamentalsAnalyst ──────────────────────────────────


class TestCryptoFundamentalsAnalyst:
    @pytest.mark.unit
    def test_analyst_node_returns_fundamentals_report(self, monkeypatch):
        from langchain_core.messages import AIMessage

        from tradingagents.agents.analysts.crypto_fundamentals_analyst import (
            create_crypto_fundamentals_analyst,
        )

        ai_msg = AIMessage(content="This is a fundamentals report for BTC.")

        # Create a mock chain whose invoke returns our AIMessage
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ai_msg

        # Mock prompt so that any partial() chain and | operator returns our chain
        mock_prompt = MagicMock()
        mock_prompt.partial.return_value = mock_prompt
        mock_prompt.__or__ = lambda self, other: mock_chain

        monkeypatch.setattr(
            "tradingagents.agents.analysts.crypto_fundamentals_analyst.ChatPromptTemplate.from_messages",
            lambda *a, **kw: mock_prompt,
        )

        mock_llm = MagicMock()
        node = create_crypto_fundamentals_analyst(mock_llm)
        state = {
            "trade_date": "2026-05-20",
            "company_of_interest": "BTC-USD",
            "messages": [],
        }
        result = node(state)
        assert "fundamentals_report" in result
        assert isinstance(result["fundamentals_report"], str)

    @pytest.mark.unit
    def test_analyst_node_state_has_messages(self, monkeypatch):
        from langchain_core.messages import AIMessage

        from tradingagents.agents.analysts.crypto_fundamentals_analyst import (
            create_crypto_fundamentals_analyst,
        )

        ai_msg = AIMessage(content="Analysis complete.")

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ai_msg

        mock_prompt = MagicMock()
        mock_prompt.partial.return_value = mock_prompt
        mock_prompt.__or__ = lambda self, other: mock_chain

        monkeypatch.setattr(
            "tradingagents.agents.analysts.crypto_fundamentals_analyst.ChatPromptTemplate.from_messages",
            lambda *a, **kw: mock_prompt,
        )

        mock_llm = MagicMock()
        node = create_crypto_fundamentals_analyst(mock_llm)
        state = {
            "trade_date": "2026-05-20",
            "company_of_interest": "ETH-USD",
            "messages": [],
        }
        result = node(state)
        assert "messages" in result
        assert len(result["messages"]) > 0
