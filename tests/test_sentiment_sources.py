"""Unit tests for the expanded sentiment analyst: Bluesky + Mastodon fetchers
and verification that all six data sources are wired into the prompt."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest
from pydantic import ValidationError

import tradingagents.dataflows.bluesky as bluesky_mod
import tradingagents.dataflows.mastodon as mastodon_mod
from tradingagents.agents.analysts.sentiment_analyst import (
    _build_system_message,
    create_sentiment_analyst,
)
from tradingagents.agents.schemas import (
    SentimentBand,
    SentimentReport,
    render_sentiment_report,
)


def _mock_urlopen(payload):
    """Return a context-manager mock whose read() yields json-encoded payload."""
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return cm


# ─── Bluesky fetcher ─────────────────────────────────────────────────────────


class TestBlueskyFetcher:
    @pytest.mark.unit
    def test_parses_posts(self):
        payload = {"posts": [{
            "author": {"handle": "trader.bsky.social"},
            "record": {"createdAt": "2026-05-28T10:00:00Z", "text": "Loading up on $NVDA"},
            "likeCount": 12, "repostCount": 3, "replyCount": 1,
        }]}
        with patch.object(bluesky_mod, "urlopen", return_value=_mock_urlopen(payload)):
            out = bluesky_mod.fetch_bluesky_posts("$NVDA")
        assert "trader.bsky.social" in out
        assert "Loading up on $NVDA" in out
        assert "12♥" in out

    @pytest.mark.unit
    def test_empty_results_placeholder(self):
        with patch.object(bluesky_mod, "urlopen", return_value=_mock_urlopen({"posts": []})):
            out = bluesky_mod.fetch_bluesky_posts("$ZZZZ")
        assert out.startswith("<no Bluesky posts found")

    @pytest.mark.unit
    def test_http_error_degrades_gracefully(self):
        err = HTTPError("url", 403, "Forbidden", {}, None)
        with patch.object(bluesky_mod, "urlopen", side_effect=err):
            out = bluesky_mod.fetch_bluesky_posts("$NVDA")
        assert out == "<bluesky unavailable: HTTPError>"


# ─── Mastodon fetcher ────────────────────────────────────────────────────────


class TestMastodonFetcher:
    @pytest.mark.unit
    def test_parses_posts_and_strips_html(self):
        payload = [{
            "account": {"acct": "fintwit@mastodon.social"},
            "created_at": "2026-05-28T10:00:00Z",
            "content": "<p>Bullish on <b>NVDA</b></p>",
            "favourites_count": 5, "reblogs_count": 2, "replies_count": 1,
        }]
        with patch.object(mastodon_mod, "urlopen", return_value=_mock_urlopen(payload)):
            out = mastodon_mod.fetch_mastodon_posts("NVDA")
        assert "fintwit@mastodon.social" in out
        assert "Bullish on NVDA" in out  # HTML tags stripped
        assert "<p>" not in out

    @pytest.mark.unit
    def test_empty_results_placeholder(self):
        with patch.object(mastodon_mod, "urlopen", return_value=_mock_urlopen([])):
            out = mastodon_mod.fetch_mastodon_posts("ZZZZ")
        assert out.startswith("<no Mastodon posts found")

    @pytest.mark.unit
    def test_invalid_tag_placeholder(self):
        out = mastodon_mod.fetch_mastodon_posts("$$$")
        assert out.startswith("<no Mastodon posts: invalid tag")

    @pytest.mark.unit
    def test_http_error_degrades_gracefully(self):
        err = HTTPError("url", 500, "err", {}, None)
        with patch.object(mastodon_mod, "urlopen", side_effect=err):
            out = mastodon_mod.fetch_mastodon_posts("NVDA")
        assert out == "<mastodon unavailable: HTTPError>"


# ─── Prompt wiring ───────────────────────────────────────────────────────────


class TestPromptWiring:
    @pytest.mark.unit
    def test_all_six_blocks_present(self):
        msg = _build_system_message(
            ticker="NVDA", start_date="2026-05-22", end_date="2026-05-29",
            news_block="NEWS_X", stocktwits_block="ST_X", reddit_block="RD_X",
            bluesky_block="BSKY_X", mastodon_block="MASTO_X", fear_greed_block="FG_X",
        )
        for tag in ("news", "stocktwits", "reddit", "bluesky", "mastodon", "fear_greed"):
            assert f"<start_of_{tag}>" in msg and f"<end_of_{tag}>" in msg
        for data in ("NEWS_X", "ST_X", "RD_X", "BSKY_X", "MASTO_X", "FG_X"):
            assert data in msg

    @pytest.mark.unit
    def test_breakdown_lists_new_sources(self):
        msg = _build_system_message(
            ticker="NVDA", start_date="2026-05-22", end_date="2026-05-29",
            news_block="", stocktwits_block="", reddit_block="",
        )
        assert "Bluesky" in msg and "Mastodon" in msg and "Fear & Greed" in msg


# ─── Node integration ────────────────────────────────────────────────────────


class TestSentimentNode:
    @pytest.mark.unit
    def test_node_injects_all_fetchers_into_prompt(self):
        captured = {}
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.5,
            confidence="high",
            narrative="breakdown",
        )
        structured = MagicMock()
        structured.invoke.side_effect = lambda prompt: (
            captured.__setitem__("prompt", prompt) or report
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        with patch("tradingagents.agents.analysts.sentiment_analyst.get_news") as gn, \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_stocktwits_messages", return_value="ST_DATA"), \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_reddit_posts", return_value="RD_DATA"), \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_bluesky_posts", return_value="BSKY_DATA") as fb, \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_mastodon_posts", return_value="MASTO_DATA") as fm, \
             patch("tradingagents.agents.analysts.sentiment_analyst.get_fear_greed_index", return_value="FG_DATA") as fg:
            gn.invoke.return_value = "NEWS_DATA"

            node = create_sentiment_analyst(llm)
            result = node({
                "company_of_interest": "NVDA",
                "trade_date": "2026-05-29",
                "asset_type": "stock",
                "messages": [],
            })

        # Fetchers called with the expected ticker-derived args.
        fb.assert_called_once_with("$NVDA")
        fm.assert_called_once_with("NVDA")
        fg.assert_called_once()
        # Every source's data made it into the formatted prompt messages.
        prompt_text = " ".join(m.content for m in captured["prompt"])
        for data in ("NEWS_DATA", "ST_DATA", "RD_DATA", "BSKY_DATA", "MASTO_DATA", "FG_DATA"):
            assert data in prompt_text
        # Output is the rendered structured report (deterministic header).
        assert "**Overall Sentiment:** **Bullish** (Score: 7.5/10)" in result["sentiment_report"]

    @pytest.mark.unit
    def test_crypto_strips_suffix_for_new_sources(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL, overall_score=5.0,
            confidence="low", narrative="n",
        )
        structured = MagicMock()
        structured.invoke.return_value = report
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        with patch("tradingagents.agents.analysts.sentiment_analyst.get_news") as gn, \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_stocktwits_messages", return_value=""), \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_reddit_posts", return_value=""), \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_bluesky_posts", return_value="") as fb, \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_mastodon_posts", return_value="") as fm, \
             patch("tradingagents.agents.analysts.sentiment_analyst.get_fear_greed_index", return_value=""):
            gn.invoke.return_value = ""
            node = create_sentiment_analyst(llm)
            node({
                "company_of_interest": "BTC-USD",
                "trade_date": "2026-05-29",
                "asset_type": "crypto",
                "messages": [],
            })
        fb.assert_called_once_with("$BTC")
        fm.assert_called_once_with("BTC")

    @pytest.mark.unit
    def test_falls_back_to_freetext_when_structured_unavailable(self):
        # Provider without structured-output support: with_structured_output raises.
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError
        llm.invoke.return_value = MagicMock(content="PLAIN_TEXT_REPORT")

        with patch("tradingagents.agents.analysts.sentiment_analyst.get_news") as gn, \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_stocktwits_messages", return_value=""), \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_reddit_posts", return_value=""), \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_bluesky_posts", return_value=""), \
             patch("tradingagents.agents.analysts.sentiment_analyst.fetch_mastodon_posts", return_value=""), \
             patch("tradingagents.agents.analysts.sentiment_analyst.get_fear_greed_index", return_value=""):
            gn.invoke.return_value = ""
            node = create_sentiment_analyst(llm)
            result = node({
                "company_of_interest": "NVDA",
                "trade_date": "2026-05-29",
                "messages": [],
            })
        assert result["sentiment_report"] == "PLAIN_TEXT_REPORT"


# ─── Schema + render ─────────────────────────────────────────────────────────


class TestSentimentSchema:
    @pytest.mark.unit
    def test_render_contains_band_score_confidence(self):
        md = render_sentiment_report(SentimentReport(
            overall_band=SentimentBand.MILDLY_BEARISH, overall_score=4.0,
            confidence="medium", narrative="NARR",
        ))
        assert "**Overall Sentiment:** **Mildly Bearish** (Score: 4.0/10)" in md
        assert "**Confidence:** Medium" in md
        assert "NARR" in md

    @pytest.mark.unit
    def test_all_six_bands_render(self):
        for band in SentimentBand:
            md = render_sentiment_report(SentimentReport(
                overall_band=band, overall_score=5.0, confidence="low", narrative="n",
            ))
            assert band.value in md

    @pytest.mark.unit
    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                overall_band=SentimentBand.BULLISH, overall_score=11.0,
                confidence="high", narrative="n",
            )
