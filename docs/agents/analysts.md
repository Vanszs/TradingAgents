# Domain Analyst Team

The Analyst Team forms the first layer of the TradingAgents framework, executing in parallel via LangGraph fan-out.

---

## 1. Technical Market Analyst (`market_analyst.py`)
- **Domain**: Pure technical price action and market structure analysis.
- **Data Ingested**: 1D Macro (52W range, 60D/20D swing highs/lows, Fibonacci levels, ATR 14/20) and 1H Micro (24-bar swing, EMA 20/50 momentum).
- **Tools**: `get_stock_data`, `get_indicators` (RSI, MACD, Bollinger Bands, Moving Averages, VWMA).
- **Boundaries**: Strictly evaluates chart structure; leaves financial statement analysis and trade sizing to downstream desks.

---

## 2. Fundamentals Analyst (`fundamentals_analyst.py`)
- **Domain**: Corporate solvency, balance sheet health, valuation multiples, and earnings quality.
- **Data Ingested**: Quarterly and annual financial filings with filing-lag enforcement, real-time $T_0$ pricing context.
- **Tools**: `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`.
- **Boundaries**: Evaluates intrinsic corporate value; does not attempt chart timing or support/resistance calculations.

---

## 3. News & Macro Analyst (`news_analyst.py`)
- **Domain**: Corporate headlines, regulatory disclosures, earnings catalysts, insider transactions.
- **Tools**: `get_news`, `get_global_news`, `get_insider_transactions`.
- **Boundaries**: In backtest mode (`backtest_mode=True`), live web search tools are stripped dynamically to guarantee zero lookahead leakage.

---

## 4. Sentiment Analyst (`sentiment_analyst.py`)
- **Domain**: Crowd psychology, retail sentiment distributions, and market-wide Fear & Greed index.
- **Data Sources**: Multi-source social sentiment feeds (Reddit RSS, Mastodon, Bluesky, CNN Fear & Greed API).
