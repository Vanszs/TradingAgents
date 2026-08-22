# Troubleshooting & FAQ

Common operational issues and how to resolve them quickly.

---

## 1. Missing API Keys Error

```text
ValueError: API key for provider 'openai' is not set.
```
- **Fix**: Ensure your `.env` file contains `OPENAI_API_KEY=sk-...` (or the corresponding key for your provider). 
- If running in headless mode, pass `--provider` with an active key or export the variable in your shell.

---

## 2. Rate Limit (HTTP 429) on Market Data

```text
yfinance.exceptions.YFRateLimitError: Too Many Requests
```
- **Fix**: TradingAgents includes exponential backoff retry logic. To reduce API pressure:
  1. Set `TRADINGAGENTS_ANALYST_CONCURRENCY=1` in `.env` to serialize data requests.
  2. Or configure a secondary provider in `DEFAULT_CONFIG["data_vendors"]` (e.g. `alpha_vantage`).

---

## 3. Indicator Returns "N/A" on Weekends

- **Fix**: The causal point-in-time fallback automatically looks up the latest valid trading session on or before the requested date. Ensure your local `data/` or `backtest_cache/` contains sufficient historical data.

---

## 4. Resetting Cache or Memory Logs

- Memory logs are stored at `~/.tradingagents/memory/trading_memory.md` (or the path set in `TRADINGAGENTS_MEMORY_LOG_PATH`).
- To clear past reflection memory and start fresh:
  ```bash
  rm -rf ~/.tradingagents/memory/trading_memory.md
  ```
