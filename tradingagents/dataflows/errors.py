"""Standard vendor error types across all data providers."""
from __future__ import annotations

from typing import Optional


class VendorError(Exception):
    """Base class for all vendor-related data errors."""

    def __init__(self, vendor: str, message: str, symbol: Optional[str] = None):
        self.vendor = vendor
        self.symbol = symbol
        full_msg = f"[{vendor}] {message}"
        if symbol:
            full_msg = f"[{vendor}] {symbol}: {message}"
        super().__init__(full_msg)


class VendorNotConfiguredError(ValueError, VendorError):
    """Raised when a vendor is selected but its API key or config is missing."""

    def __init__(self, vendor: str, key_name: str):
        self.key_name = key_name
        msg = f"{key_name} environment variable is not set."
        super().__init__(vendor=vendor, message=msg)


class VendorRateLimitError(VendorError):
    """Raised when a vendor returns HTTP 429 or quota exhaustion."""

    def __init__(self, vendor: str, detail: str = "Rate limit exceeded"):
        super().__init__(vendor=vendor, message=detail)


class NoMarketDataError(VendorError):
    """Raised when a vendor returns zero rows or delisted response for an instrument."""

    def __init__(self, symbol: str, canonical: Optional[str] = None, detail: str = "no rows returned"):
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(vendor="market_data", message=msg, symbol=symbol)
