from datetime import datetime

from .alpha_vantage_common import _make_api_request
from .config import is_point_in_time_mode

_PUBLICATION_FIELDS = (
    "available_date",
    "availableDate",
    "reportedDate",
    "filingDate",
    "publicationDate",
    "publishedAt",
    "published_at",
)


def _publication_date(record: dict) -> str | None:
    for field in _PUBLICATION_FIELDS:
        value = record.get(field)
        if value:
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
            except ValueError:
                try:
                    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date().isoformat()
                except ValueError:
                    return None
    return None


def _filter_reports_by_date(result, curr_date: str, point_in_time: bool = False):
    """Filter reports by fiscal date, plus explicit publication date in PIT mode."""
    if not curr_date or not isinstance(result, dict):
        return result
    for key in ("annualReports", "quarterlyReports"):
        if key in result:
            filtered = []
            for record in result[key]:
                fiscal_date = record.get("fiscalDateEnding", "")
                if fiscal_date and fiscal_date > curr_date:
                    continue
                if point_in_time:
                    published = _publication_date(record)
                    if not published or published > curr_date:
                        continue
                filtered.append(record)
            result[key] = filtered
    return result


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """Retrieve overview data, excluding undated data in PIT mode."""
    result = _make_api_request("OVERVIEW", {"symbol": ticker})
    if not is_point_in_time_mode() or not curr_date:
        return result
    if not isinstance(result, dict):
        return f"No fundamentally available data for {ticker} on {curr_date}."
    published = _publication_date(result)
    if not published or published > curr_date:
        return f"No fundamentally available data for {ticker} on {curr_date}."
    return result


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date, is_point_in_time_mode())


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("CASH_FLOW", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date, is_point_in_time_mode())


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date, is_point_in_time_mode())

