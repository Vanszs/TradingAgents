"""yfinance-based news data fetching functions."""

from datetime import datetime, timezone
from typing import Optional

import yfinance as yf
from dateutil.relativedelta import relativedelta

from .config import get_config, is_point_in_time_mode
from .stockstats_utils import yf_retry


def _extract_article_data(article: dict) -> dict:
    """Normalize nested and flat yfinance article shapes."""
    content = article.get("content", article)
    provider = content.get("provider", {})
    publisher = (
        provider.get("displayName", "Unknown")
        if isinstance(provider, dict) and provider.get("displayName")
        else content.get("publisher", "Unknown")
    )
    url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
    link = url_obj.get("url", "") if isinstance(url_obj, dict) else content.get("link", "")
    raw_date = (
        content.get("pubDate")
        or content.get("published_at")
        or content.get("publishedAt")
        or content.get("providerPublishTime")
    )
    pub_date = None
    if raw_date:
        try:
            if isinstance(raw_date, (int, float)):
                pub_date = datetime.fromtimestamp(raw_date, tz=timezone.utc)
            else:
                pub_date = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                else:
                    pub_date = pub_date.astimezone(timezone.utc)
        except (ValueError, TypeError, OSError):
            pass
    return {
        "title": content.get("title", "No title"),
        "summary": content.get("summary", content.get("description", "")),
        "publisher": publisher,
        "link": link,
        "pub_date": pub_date,
    }


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve news for a specific stock ticker using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    article_limit = get_config()["news_article_limit"]
    try:
        stock = yf.Ticker(ticker)
        news = yf_retry(lambda: stock.get_news(count=article_limit))

        if not news:
            return f"No news found for {ticker}"

        # Parse date range for filtering
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, tzinfo=timezone.utc
        )
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )

        news_str = ""
        filtered_count = 0

        for article in news:
            data = _extract_article_data(article)

            if is_point_in_time_mode() and not data["pub_date"]:
                continue
            if data["pub_date"]:
                pub_date = data["pub_date"].astimezone(timezone.utc)
                upper_bound = end_dt if is_point_in_time_mode() else (end_dt + relativedelta(days=1))
                if not (start_dt <= pub_date <= upper_bound):
                    continue

            date_badge = f" [{pub_date.strftime('%Y-%m-%d %H:%M UTC')}]" if data["pub_date"] else ""
            news_str += f"###{date_badge} {data['title']} (source: {data['publisher']})\n"
            if data["summary"]:
                news_str += f"{data['summary'].strip()}\n\n"
            filtered_count += 1

        if filtered_count == 0:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


def get_global_news_yfinance(
    curr_date: str,
    look_back_days: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Retrieve global/macro economic news using yfinance Search.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back. ``None`` falls back to
            ``global_news_lookback_days`` from the active config.
        limit: Maximum number of articles to return. ``None`` falls back to
            ``global_news_article_limit`` from the active config.

    Returns:
        Formatted string containing global news articles
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]
    search_queries = config["global_news_queries"]

    all_news = []
    seen_titles = set()

    try:
        for query in search_queries:
            search = yf_retry(lambda q=query: yf.Search(
                query=q,
                news_count=limit,
                enable_fuzzy_query=True,
            ))

            if search.news:
                for article in search.news:
                    data = _extract_article_data(article)
                    title = data["title"]
                    if is_point_in_time_mode() and not data["pub_date"]:
                        continue

                    # Deduplicate by title
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_news.append(article)

            if len(all_news) >= limit:
                break

        if not all_news:
            return f"No global news found for {curr_date}"

        # Calculate date range
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )
        start_dt = (curr_dt - relativedelta(days=look_back_days)).replace(
            hour=0, minute=0, second=0
        )
        start_date = start_dt.strftime("%Y-%m-%d")

        news_str = ""
        for article in all_news[:limit]:
            data = _extract_article_data(article)
            if is_point_in_time_mode() and not data["pub_date"]:
                continue
            if data["pub_date"]:
                pub_date = data["pub_date"].astimezone(timezone.utc)
                upper_bound = curr_dt if is_point_in_time_mode() else (curr_dt + relativedelta(days=1))
                if pub_date < start_dt or pub_date > upper_bound:
                    continue
            title = data["title"]
            publisher = data["publisher"]
            link = data["link"]
            summary = data["summary"]

            news_str += f"### {title} (source: {publisher})\n"
            if summary:
                news_str += f"{summary}\n"
            if link:
                news_str += f"Link: {link}\n"
            news_str += "\n"

        return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching global news: {str(e)}"
