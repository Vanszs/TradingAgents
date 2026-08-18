"""Symbol normalization and market-data error types for vendor calls.

Yahoo Finance uses specific ticker conventions:
  - Metals: XAUUSD -> GC=F, XAGUSD -> SI=F
  - Energy: USOIL/WTI -> CL=F, BRENT -> BZ=F
  - Forex: EURUSD -> EURUSD=X, GBPJPY -> GBPJPY=X
  - Crypto: BTCUSD -> BTC-USD, ETHUSD -> ETH-USD
  - Index CFDs: SPX500 -> ^GSPC, NAS100 -> ^NDX
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class NoMarketDataError(Exception):
    """Raised when a vendor returns no rows/records for a symbol."""

    def __init__(self, symbol: str, canonical: str | None = None, detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


# ISO-4217 common retail forex codes
_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB", "IDR",
    }
)

_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK", "BNB"}
)

_ALIASES = {
    # Precious metals
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    # Energy
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    # Index CFDs -> Yahoo index symbols
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-\^=]+$")


def normalize_symbol(raw: str) -> str:
    """Map a user/broker symbol to its canonical Yahoo Finance symbol."""
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    s = s.rstrip("+")

    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif len(s) == 6 and s[:3] in _CRYPTO_BASES and s[3:] == "USD":
        canonical = f"{s[:3]}-USD"
    elif s[:-3] in _CRYPTO_BASES and s.endswith("USD") and "-" not in s:
        canonical = f"{s[:-3]}-USD"
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    elif "/" in s:
        parts = s.split("/")
        if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 3:
            if parts[0] in _FOREX_CURRENCIES and parts[1] in _FOREX_CURRENCIES:
                canonical = f"{parts[0]}{parts[1]}=X"
            elif parts[0] in _CRYPTO_BASES and parts[1] == "USD":
                canonical = f"{parts[0]}-USD"
            else:
                canonical = s
        else:
            canonical = s
    else:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to Yahoo symbol %r", raw, canonical)
    return canonical


def is_yahoo_safe(symbol: str) -> bool:
    """True when symbol only contains characters Yahoo symbols use."""
    return bool(symbol) and _YAHOO_SAFE.fullmatch(symbol) is not None
