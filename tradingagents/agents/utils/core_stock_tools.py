from datetime import datetime
from typing import Annotated, Optional

from dateutil.relativedelta import relativedelta
from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[Optional[str], "Start date in yyyy-mm-dd format (defaults to 60 days before end_date)"] = None,
    end_date: Annotated[Optional[str], "End date in yyyy-mm-dd format (defaults to current trade date)"] = None,
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str, optional): Start date in yyyy-mm-dd format
        end_date (str, optional): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        dt_end = datetime.strptime(str(end_date).split("T")[0].split(" ")[0], "%Y-%m-%d")
        start_date = (dt_end - relativedelta(days=60)).strftime("%Y-%m-%d")
    return route_to_vendor("get_stock_data", symbol, start_date, end_date)
