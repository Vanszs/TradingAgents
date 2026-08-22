"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches several complementary data sources
before the LLM is invoked and injects them into the prompt as structured
blocks:

  1. News headlines      — Yahoo Finance (institutional framing)
  2. StockTwits messages  — retail-trader posts indexed by cashtag, with
                            user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts         — r/wallstreetbets, r/stocks, r/investing
  4. Bluesky posts        — decentralized X/Twitter alternative (keyword)
  5. Mastodon posts       — federated public hashtag timeline
  6. Fear & Greed Index   — aggregate market-mood proxy (0-100)

The agent does not use tool-calling; the data is in the prompt from
turn 0. The LLM produces the sentiment report in a single invocation.

See: https://github.com/TauricResearch/TradingAgents/issues/557
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.bluesky import fetch_bluesky_posts
from tradingagents.dataflows.exa_search import ExaTimeTravelSearch
from tradingagents.dataflows.fear_greed import get_fear_greed_index
from tradingagents.dataflows.mastodon import fetch_mastodon_posts
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit + Bluesky + Mastodon + Fear &
    Greed data, injects them into the prompt as structured blocks, and
    produces a deterministic sentiment report via structured output (with a
    free-text fallback for providers that do not support it).

    In backtest mode, reads from snapshot data instead of live APIs.
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(
            ticker, asset_type=asset_type, trade_date=end_date
        )

        # Check backtest mode
        from tradingagents.dataflows.config import get_config, is_point_in_time_mode
        runtime_config = get_config()
        is_backtest = is_point_in_time_mode(runtime_config)

        if is_backtest:
            # Historical mode is snapshot-only and fails closed when absent.
            snapshot_data = runtime_config.get("snapshot_data", {})
            news_items = snapshot_data.get("news", [])
            sentiment_items = snapshot_data.get("sentiment", [])

            news_block = _format_snapshot_news_block(news_items)
            stocktwits_block = _format_snapshot_sentiment_block(sentiment_items, "stocktwits")
            reddit_block = _format_snapshot_sentiment_block(sentiment_items, "reddit")
            bluesky_block = _format_snapshot_sentiment_block(sentiment_items, "bluesky")
            mastodon_block = _format_snapshot_sentiment_block(sentiment_items, "mastodon")
            fear_greed_block = _format_snapshot_sentiment_block(sentiment_items, "fear_greed")
            if not fear_greed_block:
                fear_greed_block = "Historical Fear & Greed data unavailable in snapshot."
            web_search_block = "Historical web-search data unavailable in snapshot."

            if asset_type == "crypto":
                community_context = (
                    "Focus on crypto-specific communities. "
                    "Snapshot-only historical data; missing sources are unavailable."
                )
            else:
                community_context = (
                    "Focus on stock-specific communities. "
                    "Snapshot-only historical data; missing sources are unavailable."
                )
        else:
            # Point-in-Time Live / Time-Travel mode:
            # 1. Historical or live Fear & Greed index (clamped <= end_date)
            fear_greed_block = get_fear_greed_index(trade_date=end_date)

            # 2. News headlines
            news_block = get_news.invoke(
                {"ticker": ticker, "start_date": start_date, "end_date": end_date}
            )

            # 3. Targeted Time-Travel Web Search for retail sentiment
            exa_searcher = ExaTimeTravelSearch()
            if ticker.endswith(".JK") or ticker.endswith(".jk"):
                query = f"diskusi sentimen ritel saham {ticker} forum investasi komunitas Stockbit X"
                community_context = (
                    "Focus on Indonesian retail investment communities (Stockbit stream, Twitter/X #saham, forum lokal)."
                )
            elif asset_type == "crypto":
                query = f"{ticker} crypto community sentiment retail mood discussions"
                community_context = (
                    "Focus on crypto communities (Reddit r/CryptoCurrency, crypto Twitter/X, Telegram sentiment)."
                )
            else:
                query = f"{ticker} retail investor sentiment discussion forum Reddit StockTwits"
                community_context = (
                    "Focus on stock communities: StockTwits streams, Reddit (r/wallstreetbets, r/stocks), and financial Twitter."
                )

            web_search_block = exa_searcher.search(query=query, trade_date=end_date, num_results=4)
            stocktwits_block = fetch_stocktwits_messages(ticker)
            reddit_block = fetch_reddit_posts(ticker)
            bluesky_query = ticker if asset_type == "crypto" else f"${ticker}"
            bluesky_block = fetch_bluesky_posts(bluesky_query)
            mastodon_block = fetch_mastodon_posts(ticker)

            if asset_type == "crypto":
                community_context = (
                    "Focus on crypto-specific communities: Reddit (r/CryptoCurrency, r/Bitcoin, r/ethereum, "
                    "r/CryptoMarkets), Twitter/X crypto hashtags, and Telegram sentiment. "
                    "Note: StockTwits data may still be available for some crypto tickers."
                )
            else:
                community_context = (
                    "Focus on stock-specific communities: StockTwits cashtag streams, "
                    "Reddit (r/wallstreetbets, r/stocks, r/investing), and financial Twitter."
                )

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            bluesky_block=bluesky_block,
            mastodon_block=mastodon_block,
            fear_greed_block=fear_greed_block,
            web_search_block=web_search_block,
            community_context=community_context,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Produce an analyst report only; leave the final transaction proposal to the Trader and Portfolio Manager."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so both the
        # structured and free-text paths receive the same input. The data is
        # already in the prompt (no tool-calling); structured output only
        # shapes the result into a deterministic header + narrative.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    bluesky_block: str = "",
    mastodon_block: str = "",
    fear_greed_block: str = "",
    web_search_block: str = "",
    community_context: str = "",
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on multiple complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion. Engagement signal via upvote score and comment count. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit_block}
<end_of_reddit>

### Bluesky posts — decentralized X/Twitter alternative (keyword search)
Fast-moving retail signal; much of "fintwit" now cross-posts here. Engagement via likes / reposts / replies.

<start_of_bluesky>
{bluesky_block}
<end_of_bluesky>

### Mastodon posts — federated network, public hashtag timeline
Smaller but less manipulated community signal. Engagement via favourites / boosts / replies.

<start_of_mastodon>
{mastodon_block}
<end_of_mastodon>

### Fear & Greed Index — aggregate market mood (0–100)
Macro risk-on/risk-off proxy. Extreme readings can be contrarian signals. (Crypto-derived index; also a useful broad sentiment gauge for equities.)

<start_of_fear_greed>
{fear_greed_block}
<end_of_fear_greed>

### Web search — Exa time-travel retail research
Historical web results, kept separate from direct social-source blocks.

<start_of_web_search>
{web_search_block}
<end_of_web_search>

## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this caveat explicitly. If the sources are silent on a given subreddit, say so.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish.
  Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish). 5 is neutral.
  Must be consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size across the six sources.
- **narrative**: Full source-by-source breakdown (news / StockTwits / Reddit / Bluesky / Mastodon / Fear & Greed / Exa web search)
  with specific evidence, cross-source divergences and alignments, dominant narrative themes, catalysts and risks,
  and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

## Community context

{community_context}

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Snapshot formatting helpers for backtest mode
# ---------------------------------------------------------------------------

def _format_snapshot_news_block(items: list) -> str:
    """Format snapshot news items into a readable block for the LLM."""
    if not items:
        return "No news data available in backtest snapshot."
    lines = []
    for item in items:
        pub = item.get("published_at", item.get("date", ""))
        title = item.get("title", "")
        summary = item.get("summary", item.get("description", ""))
        source = item.get("source", "")
        line = f"[{pub}] {source}: {title}" if source else f"[{pub}] {title}"
        lines.append(line)
        if summary:
            lines.append(f"  {summary}")
    return "\n\n".join(lines)


def _format_snapshot_sentiment_block(items: list, source_filter: str = "") -> str:
    """Format snapshot sentiment items filtered by source type."""
    if not items:
        return "Not available in backtest mode."

    # Filter by source if specified
    filtered = items
    if source_filter:
        filtered = [
            it for it in items
            if source_filter.lower() in it.get("source", "").lower()
            or source_filter.lower() in it.get("provider", "").lower()
        ]

    if not filtered:
        return f"No {source_filter} data available in backtest snapshot."

    lines = []
    for item in filtered[:30]:
        ts = item.get("timestamp", item.get("date", ""))
        score = item.get("score", item.get("sentiment_score", ""))
        source = item.get("source", item.get("provider", ""))
        label = item.get("label", item.get("sentiment_label", ""))
        text = item.get("text", item.get("headline", ""))
        parts = [f"[{ts}]"]
        if source:
            parts.append(f"{source}:")
        if label:
            parts.append(f"({label})")
        if score:
            parts.append(f"score={score}")
        if text:
            parts.append(f"— {text}")
        lines.append(" ".join(parts))
    return "\n".join(lines)
