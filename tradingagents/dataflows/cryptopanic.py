"""CryptoPanic news aggregator.

Requires CRYPTOPANIC_API_KEY. Degrades gracefully without key.
Returns recent crypto news headlines with sentiment.
"""
from __future__ import annotations
import logging
import os
import time
from typing import Optional
import requests

logger = logging.getLogger(__name__)
_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 600  # 10 minutes


def get_crypto_news(ticker: str, limit: int = 10) -> str:
    """Return recent news headlines for a crypto ticker."""
    api_key = os.environ.get('CRYPTOPANIC_API_KEY', '').strip()
    if not api_key:
        return (
            f'## Crypto News: {ticker}\n\n'
            'CRYPTOPANIC_API_KEY not configured. '
            'Register free at https://cryptopanic.com/developers/api/ to enable news sentiment analysis.'
        )

    # Extract base currency symbol
    currency = ticker.split('-')[0].upper()
    cache_key = f'{currency}:{limit}'
    now = time.time()
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return data

    try:
        resp = requests.get(
            'https://cryptopanic.com/api/v1/posts/',
            params={
                'auth_token': api_key,
                'currencies': currency,
                'public': 'true',
                'kind': 'news',
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get('results', [])[:limit]
        if not results:
            return f'## Crypto News: {ticker}\n\nNo recent news found.'

        lines = [f'## Crypto News: {ticker} (last {len(results)} articles)', '']
        for item in results:
            title = item.get('title', 'No title')
            votes = item.get('votes', {})
            positive = votes.get('positive', 0)
            negative = votes.get('negative', 0)
            sentiment = '🟢 Bullish' if positive > negative else ('🔴 Bearish' if negative > positive else '⚪ Neutral')
            source = item.get('source', {}).get('title', 'Unknown')
            lines.append(f'- [{sentiment}] {title} ({source})')

        result_str = '\n'.join(lines)
        _CACHE[cache_key] = (now, result_str)
        return result_str
    except Exception as exc:
        logger.warning('CryptoPanic request failed: %s', exc)
        return f'## Crypto News: {ticker}\n\nNews data unavailable (API error).'
