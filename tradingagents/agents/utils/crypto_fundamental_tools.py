"""LangGraph tool wrappers for crypto fundamental data.

These tools are used by the Crypto Fundamentals Analyst node.
Each tool returns a formatted string ready for LLM prompt injection.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.coingecko import get_tokenomics
from tradingagents.dataflows.fear_greed import get_fear_greed_index


@tool
def get_crypto_tokenomics(
    ticker: Annotated[str, "Crypto ticker symbol, e.g. BTC-USD, ETH-USDT"],
) -> str:
    """Get tokenomics data for a crypto asset: market cap, supply dynamics, price performance, community stats."""
    return get_tokenomics(ticker)


@tool
def get_crypto_dev_activity(
    ticker: Annotated[str, "Crypto ticker symbol, e.g. BTC-USD, ETH-USDT"],
) -> str:
    """Get developer activity: GitHub commits, contributors, stars, forks, last commit date."""
    from tradingagents.dataflows.github_activity import get_dev_activity
    return get_dev_activity(ticker)


@tool
def get_crypto_network_metrics(
    ticker: Annotated[str, "Crypto ticker symbol, e.g. BTC-USD, ETH-USDT"],
) -> str:
    """Get network metrics: 30-day price trend, volume, and TVL for DeFi protocols."""
    from tradingagents.dataflows.coingecko import get_market_data
    from tradingagents.dataflows.defillama import get_tvl
    price_data = get_market_data(ticker)
    tvl_data = get_tvl(ticker)
    return price_data + "\n\n" + tvl_data


@tool
def get_crypto_market_sentiment(
    ticker: Annotated[str, "Crypto ticker symbol, e.g. BTC-USD, ETH-USDT"],
) -> str:
    """Get market sentiment indicators: Fear & Greed Index, BTC dominance, social metrics."""
    fear_greed = get_fear_greed_index()
    # Get BTC dominance from CoinGecko global endpoint
    from tradingagents.dataflows.coingecko import _get
    global_data = _get("/global")
    dominance_section = ""
    if global_data:
        market_cap_pct = global_data.get("data", {}).get("market_cap_percentage", {})
        btc_dom = market_cap_pct.get("btc")
        eth_dom = market_cap_pct.get("eth")
        if btc_dom is not None:
            dominance_section = (
                f"\n\n## Market Dominance\n"
                f"- BTC Dominance: {btc_dom:.1f}%\n"
                f"- ETH Dominance: {eth_dom:.1f}%" if eth_dom else f"- BTC Dominance: {btc_dom:.1f}%"
            )
    return fear_greed + dominance_section


@tool
def get_crypto_onchain_news(
    ticker: Annotated[str, "Crypto ticker symbol, e.g. ETH-USD, LINK-USD"],
) -> str:
    """Get on-chain metrics (Etherscan) and recent news (RSS feeds) for a crypto asset."""
    from tradingagents.dataflows.crypto_news import get_crypto_news
    from tradingagents.dataflows.onchain_metrics import get_onchain_metrics
    onchain = get_onchain_metrics(ticker)
    news = get_crypto_news(ticker)
    return onchain + '\n\n' + news


# Convenience alias used by tests and as a single entry-point
get_crypto_fundamentals = get_crypto_tokenomics
