"""LangGraph tool wrapper for verified market snapshot."""
from __future__ import annotations

from typing import Annotated, Optional
from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.market_data_validator import format_verified_market_snapshot


@tool
def get_verified_market_snapshot(
    symbol: Annotated[str, "Ticker symbol of the company or asset to verify."],
    trade_date: Annotated[
        Optional[str],
        "Target trading date in YYYY-MM-DD format. If omitted, uses active trade date context.",
    ] = None,
) -> str:
    """Retrieve verified, authoritative OHLCV ground truth for a symbol at trade_date.

    Use this tool to verify exact open/high/low/close prices and avoid confabulating
    unsupported numbers in technical and market analysis reports.
    """
    active_date = trade_date or get_config().get("trade_date") or get_config().get("curr_date")
    if not active_date:
        return f"Error: No trade date context available for {symbol}."
    return format_verified_market_snapshot(symbol=symbol, trade_date=active_date)
