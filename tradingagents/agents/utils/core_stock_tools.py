from datetime import datetime
from io import StringIO
from typing import Annotated, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.kronos import SUPPORTED_KRONOS_HORIZONS


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
        cfg = get_config()
        end_date = cfg.get("trade_date") or cfg.get("curr_date") or datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        dt_end = datetime.strptime(str(end_date).split("T")[0].split(" ")[0], "%Y-%m-%d")
        start_date = (dt_end - relativedelta(days=60)).strftime("%Y-%m-%d")
    return route_to_vendor("get_stock_data", symbol, start_date, end_date)


@tool
def get_kronos_forecast(
    symbol: Annotated[str, "ticker symbol of the company, e.g. NVDA, MSFT, BBRI.JK"],
    curr_date: Annotated[Optional[str], "Cutoff trade date in YYYY-MM-DD format (defaults to current trade date)"] = None,
    pred_days: Annotated[int, "Forward forecast horizon in trading bars (e.g. 5, 10, 20)"] = 20,
) -> str:
    """
    Retrieve a Kronos K-line forecast. The result identifies Official-Kronos versus Statistical-Fallback.
    Generates forward projected high, low, close, directional bias, and expected return %.
    """
    if pred_days not in SUPPORTED_KRONOS_HORIZONS:
        return (
            f"Kronos forecast error for {symbol}: pred_days must be one of "
            f"{SUPPORTED_KRONOS_HORIZONS}, got {pred_days!r}"
        )

    cfg = get_config()
    if curr_date is None:
        curr_date = cfg.get("trade_date") or cfg.get("curr_date") or datetime.now().strftime("%Y-%m-%d")

    # Fetch up to 512 bars lookback for Kronos receptive field
    try:
        dt_end = datetime.strptime(str(curr_date).split("T")[0].split(" ")[0], "%Y-%m-%d")
        dt_start = (dt_end - relativedelta(days=800)).strftime("%Y-%m-%d")
        csv_str = route_to_vendor("get_stock_data", symbol, dt_start, curr_date)
        if not csv_str or not isinstance(csv_str, str) or "close" not in csv_str.lower():
            return f"Error: Unable to retrieve historical price data for Kronos forecast on {symbol}."

        # Clean comment headers (#) before parsing CSV
        df = pd.read_csv(StringIO(csv_str), comment="#")
        df.columns = [c.lower().strip() for c in df.columns]

        # Normalize date column
        date_col = next((c for c in df.columns if c in ("date", "timestamp", "time")), df.columns[0])
        df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        df = df[df["date"] <= str(curr_date).split("T")[0].split(" ")[0]].sort_values("date")

        contract = route_to_vendor("get_kronos_forecast", symbol, df, curr_date, pred_days, cfg)
        return contract.raw_summary_markdown
    except Exception as exc:
        return f"Kronos forecast error for {symbol} on {curr_date}: {exc}"
