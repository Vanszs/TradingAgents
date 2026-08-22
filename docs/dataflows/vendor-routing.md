# Data Vendor Routing & Fallbacks

TradingAgents integrates multi-vendor market data routing with fail-open resiliency.

---

## Supported Vendors

1. **Yahoo Finance (`yfinance`)**: Primary EOD OHLCV and 1H intraday price bars.
2. **Alpha Vantage**: Secondary technical indicators and fundamental financial statements.
3. **DeFiLlama**: Crypto protocol TVL, stablecoin market caps, and fee revenue.
4. **CoinGecko**: Crypto token market caps, trading volume, and price data.
5. **Reddit RSS / Bluesky / Mastodon**: Real-time social crowd sentiment and Fear & Greed indices.

---

## Routing & Fallback Architecture

```
User / Agent Request (e.g. get_stock_data, get_indicators)
   │
   ▼
Vendor Router (`tradingagents/dataflows/interface.py`)
   ├── 1. Primary Vendor Execution (e.g. yfinance)
   │     ├── If Successful ──> Return Normalized DataFrame / Series
   │     └── If 429 Rate Limit / HTTP Error ──> Trigger Secondary Vendor
   └── 2. Secondary Fallback Execution (e.g. Alpha Vantage)
         └── If All Fail ──> Return Structured Fallback / NoMarketDataError
```
