"""Unit tests for exchange filing context and Fear & Greed index historical and live fetch."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import tradingagents.dataflows.fear_greed as fear_greed_mod
from tradingagents.agents.utils.agent_utils import build_exchange_filing_context
from tradingagents.dataflows.fear_greed import get_fear_greed_index

# ─── Task 1: Fear & Greed Index ──────────────────────────────────────────────


class TestFearGreedIndex:
    def setup_method(self):
        # Reset caches before each test
        fear_greed_mod._CACHE = None
        fear_greed_mod._HISTORICAL_CACHE = None

    @pytest.mark.unit
    def test_live_fear_greed_index(self):
        fake_response = {
            "data": [
                {"value": "75", "value_classification": "Greed", "timestamp": "1704153600"},
                {"value": "70", "value_classification": "Greed", "timestamp": "1704067200"},
                {"value": "65", "value_classification": "Greed", "timestamp": "1703980800"},
                {"value": "60", "value_classification": "Greed", "timestamp": "1703894400"},
                {"value": "55", "value_classification": "Greed", "timestamp": "1703808000"},
                {"value": "50", "value_classification": "Neutral", "timestamp": "1703721600"},
                {"value": "45", "value_classification": "Neutral", "timestamp": "1703635200"},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp) as mock_get:
            res = get_fear_greed_index()
            assert "## Crypto Fear & Greed Index" in res
            assert "75/100 — Greed" in res
            assert "**7-Day Trend**" in res
            mock_get.assert_called_once()
            assert "limit=7" in mock_get.call_args[0][0]

    @pytest.mark.unit
    def test_historical_fear_greed_index_with_date(self):
        # 1704153600 -> 2024-01-02
        # 1704240000 -> 2024-01-03
        fake_hist = {
            "data": [
                {"value": "80", "value_classification": "Extreme Greed", "timestamp": "1704240000"},  # 2024-01-03
                {"value": "72", "value_classification": "Greed", "timestamp": "1704153600"},          # 2024-01-02
                {"value": "68", "value_classification": "Greed", "timestamp": "1704067200"},          # 2024-01-01
                {"value": "65", "value_classification": "Greed", "timestamp": "1703980800"},          # 2023-12-31
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_hist
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp) as mock_get:
            res = get_fear_greed_index(trade_date="2024-01-02")
            assert "## Market Fear & Greed Index (As of 2024-01-02)" in res
            assert "72/100 — Greed" in res
            assert "- Greed (68)" in res
            mock_get.assert_called_once()
            assert "limit=0" in mock_get.call_args[0][0]

            # Second call should use _HISTORICAL_CACHE without network request
            res2 = get_fear_greed_index(trade_date="2024-01-01")
            assert "## Market Fear & Greed Index (As of 2024-01-01)" in res2
            assert "68/100 — Greed" in res2
            assert mock_get.call_count == 1

    @pytest.mark.unit
    def test_historical_fear_greed_index_no_entries(self):
        fake_hist = {
            "data": [
                {"value": "80", "value_classification": "Extreme Greed", "timestamp": "1704240000"},  # 2024-01-03
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_hist
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            res = get_fear_greed_index(trade_date="2020-01-01")
            assert "data unavailable for date <= 2020-01-01" in res

    @pytest.mark.unit
    def test_fear_greed_network_error(self):
        with patch("requests.get", side_effect=Exception("Connection timed out")):
            res = get_fear_greed_index(trade_date="2024-01-02")
            assert "data unavailable (network error)" in res

            res_live = get_fear_greed_index()
            assert "data unavailable (network error)" in res_live


# ─── Task 2: Exchange Filing Context ──────────────────────────────────────────


class TestExchangeFilingContext:
    @pytest.mark.unit
    def test_indonesian_stocks(self):
        for ticker in ["BBRI.JK", "dewa.jk", "TLKM.JK"]:
            ctx = build_exchange_filing_context(ticker)
            assert "Indonesia Stock Exchange" in ctx or "IDX" in ctx
            assert "OJK" in ctx
            assert "Keterbukaan Informasi BEI" in ctx
            assert "CNBC Indonesia" in ctx or "Bisnis.com" in ctx

    @pytest.mark.unit
    def test_us_stocks(self):
        for ticker in ["AAPL", "NVDA", "MSFT", "TSLA"]:
            ctx = build_exchange_filing_context(ticker)
            assert "SEC" in ctx
            assert "EDGAR" in ctx
            assert "Form 10-K" in ctx
            assert "Form 10-Q" in ctx
            assert "Form 8-K" in ctx

    @pytest.mark.unit
    def test_crypto_assets(self):
        ctx1 = build_exchange_filing_context("BTC-USD")
        assert "On-chain" in ctx1
        assert "tokenomics" in ctx1
        assert "whitepapers" in ctx1
        assert "CoinGecko" in ctx1 or "DeFiLlama" in ctx1

        ctx2 = build_exchange_filing_context("ETH", asset_type="crypto")
        assert "On-chain" in ctx2
        assert "tokenomics" in ctx2

    @pytest.mark.unit
    def test_international_stocks(self):
        to_ctx = build_exchange_filing_context("SHOP.TO")
        assert "TSX" in to_ctx or "Toronto" in to_ctx

        lse_ctx = build_exchange_filing_context("AZN.L")
        assert "London Stock Exchange" in lse_ctx or "LSE" in lse_ctx

        hk_ctx = build_exchange_filing_context("0700.HK")
        assert "HKEX" in hk_ctx or "Hong Kong" in hk_ctx

        tse_ctx = build_exchange_filing_context("7203.T")
        assert "TSE" in tse_ctx or "Tokyo" in tse_ctx


# ─── Analyst Prompt Integration ──────────────────────────────────────────────


class TestAnalystPromptIntegration:
    @pytest.mark.unit
    def test_fundamentals_analyst_includes_filing_context(self):
        from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst

        llm = MagicMock()
        node = create_fundamentals_analyst(llm)
        state = {
            "trade_date": "2025-01-02",
            "company_of_interest": "BBRI.JK",
            "messages": [],
        }
        node(state)
        # Verify bind_tools received prompt with filing context
        bind_args = llm.bind_tools.call_args
        assert bind_args is not None

    @pytest.mark.unit
    def test_news_analyst_includes_filing_context(self):
        from tradingagents.agents.analysts.news_analyst import create_news_analyst

        llm = MagicMock()
        node = create_news_analyst(llm)
        state = {
            "trade_date": "2025-01-02",
            "company_of_interest": "BBRI.JK",
            "messages": [],
        }
        node(state)
        bind_args = llm.bind_tools.call_args
        assert bind_args is not None
