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
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(ticker, asset_type=asset_type)

        # Pre-fetch every source. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.invoke(
            {"ticker": ticker, "start_date": start_date, "end_date": end_date}
        )

        # Convert crypto ticker format for StockTwits (BTC-USD -> BTC.X)
        if asset_type == "crypto":
            st_ticker = ticker.split("-")[0] + ".X"
        else:
            st_ticker = ticker
        stocktwits_block = fetch_stocktwits_messages(st_ticker, limit=30)

        if asset_type == "crypto":
            crypto_subs = ("CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "altcoin")
            reddit_block = fetch_reddit_posts(ticker.split("-")[0], subreddits=crypto_subs)
        else:
            reddit_block = fetch_reddit_posts(ticker)

        # Bluesky (X/Twitter alternative) + Mastodon: free, no-auth public
        # endpoints. Use the bare symbol/name as the search term/hashtag.
        base = ticker.split("-")[0] if asset_type == "crypto" else ticker
        bluesky_block = fetch_bluesky_posts(f"${base}")
        mastodon_block = fetch_mastodon_posts(base)
        # Fear & Greed Index: aggregate market mood (crypto index also serves
        # as a broad risk-on/risk-off proxy for equities).
        fear_greed_block = get_fear_greed_index()

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
            community_context=community_context,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
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
- **narrative**: Full source-by-source breakdown (news / StockTwits / Reddit / Bluesky / Mastodon / Fear & Greed)
  with specific evidence, cross-source divergences and alignments, dominant narrative themes, catalysts and risks,
  and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

## Community context

{community_context}

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
