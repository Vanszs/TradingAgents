# Headless & Automation CLI Reference

TradingAgents provides scriptable CLI commands for automated pipelines, cron jobs, and CI/CD workflows.

---

## 1. Single Ticker Analysis (`analyze`)

```bash
# Basic run with default model (OpenAI GPT-5.4)
tradingagents analyze --ticker NVDA --headless

# Historical analysis date with Anthropic Claude
tradingagents analyze --ticker AAPL --date 2024-06-11 --provider anthropic --headless

# Localized Indonesian report output with custom output directory
tradingagents analyze --ticker BBRI.JK --lang Indonesian --output-dir ./my_reports --headless
```

### Options:
- `-t, --ticker TEXT`: Ticker symbol to analyze (e.g. `NVDA`, `AAPL`, `BBRI.JK`).
- `-d, --date TEXT`: Historical analysis date (`YYYY-MM-DD`). Defaults to today.
- `-p, --provider TEXT`: LLM provider (`openai`, `anthropic`, `google`, `deepseek`, `ollama`, `azure`).
- `-l, --lang TEXT`: Output report language (`English`, `Indonesian`, `Japanese`, `Chinese`, etc.).
- `--depth INTEGER`: Research debate depth rounds (`1` for Shallow, `3` for Medium, `5` for Deep).
- `--output-dir PATH`: Directory path to save generated reports.
- `--headless`: Run non-interactively without terminal prompts.

---

## 2. Forward Signal Evaluator (`evaluate-signal`)

```bash
# Evaluate a generated signal against subsequent daily candles
tradingagents evaluate-signal --ticker NVDA --date 2024-01-08
```

---

## 3. Walk-Forward Portfolio Backtest (`backtest`)

```bash
# Run multi-month backtesting simulation using YAML configuration
tradingagents backtest --config backtest.yaml --lookback 60
```
