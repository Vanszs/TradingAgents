from datetime import datetime, timezone

from .alpha_vantage_common import _make_api_request, format_datetime_for_api
from .config import is_point_in_time_mode

_PUBLICATION_FIELDS = ("time_published", "published_at", "publishedAt", "publicationDate")


def _filter_pit_feed(result, start_date: str, end_date: str):
    if not is_point_in_time_mode() or not isinstance(result, dict):
        return result
    feed = result.get("feed")
    if not isinstance(feed, list):
        return f"No news available between {start_date} and {end_date}."
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    filtered = []
    for item in feed:
        raw = next((item.get(field) for field in _PUBLICATION_FIELDS if item.get(field)), None)
        if not raw:
            continue
        try:
            published = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            else:
                published = published.astimezone(timezone.utc)
        except ValueError:
            try:
                published = datetime.strptime(
                    str(raw)[:15], "%Y%m%dT%H%M%S"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        if start <= published <= end.replace(hour=23, minute=59, second=59):
            filtered.append(item)
    if not filtered:
        return f"No news available between {start_date} and {end_date}."
    result = dict(result)
    result["feed"] = filtered
    return result


def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }

    return _filter_pit_feed(
        _make_api_request("NEWS_SENTIMENT", params), start_date, end_date
    )

def get_global_news(curr_date, look_back_days: int = 7, limit: int = 50) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, and more.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    from datetime import datetime, timedelta

    from .config import get_config

    if look_back_days is None:
        look_back_days = get_config().get("global_news_lookback_days", 7)
    if limit is None:
        limit = get_config().get("global_news_article_limit", 50)

    # Calculate start date
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=int(look_back_days))
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }

    return _filter_pit_feed(
        _make_api_request("NEWS_SENTIMENT", params), start_date, curr_date
    )


def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)