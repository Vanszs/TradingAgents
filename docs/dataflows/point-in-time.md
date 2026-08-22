# Point-in-Time Integrity & Anti-Leakage

In quantitative finance, lookahead bias (evaluating past decisions using future data) invalidates backtesting results. TradingAgents implements multi-layered causal enforcement.

---

## 1. Strict Timestamp Slicing

- **Daily Data (1D)**: All historical OHLCV bars, technical indicators, and price extrema are filtered strictly on or before `trade_date`:
  ```python
  history = work_df[work_df["date_str"] <= str(trade_date)[:10]].copy()
  ```
- **Intraday Data (1H)**: Cutoff timestamps use exact UTC slicing:
  ```python
  cutoff_ts = pd.Timestamp(trade_date, tz="UTC")
  history = work_df[work_df["ts_utc"] <= cutoff_ts].copy()
  ```

---

## 2. Point-in-Time Filing Lag Protection

Fundamental financial statements enforce publication lag buffers (default 45 days) to ensure corporate results are not visible to agents before their official SEC / regulatory release date.

---

## 3. Dynamic Tool Stripping in Backtesting

When `backtest_mode=True`, live un-dated web search tools (`get_web_search`) are dynamically stripped from prompt schemas, preventing the LLM from executing live internet queries about future events.
