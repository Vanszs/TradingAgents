# Quickstart

Start your first multi-agent market analysis in less than 2 minutes.

---

## 1. Interactive Terminal UI (TUI)

Launch the full interactive terminal workflow:

```bash
tradingagents
```

The system will display the welcome panel and guide you step-by-step:
1. **Ticker Selection**: Enter `NVDA`, `AAPL`, `MSFT`, `BBRI.JK`, etc.
2. **Analysis Date**: Pick today's date or a historical backtest date.
3. **Language**: Choose `English`, `Indonesian`, `Japanese`, `Chinese`, etc.
4. **Analyst Team**: Select active analyst branches (*Market, Fundamentals, News, Social*).
5. **Research Depth**: Set debate rounds ($1\text{--}3$).
6. **LLM Provider**: Choose your preferred model (OpenAI, Anthropic, Google, DeepSeek, Ollama).

---

## 2. Non-Interactive CLI Mode (Headless)

Run a fast, automated single-command analysis:

```bash
tradingagents analyze --ticker NVDA --provider openai --headless
```

To run on a historical date with localized report output:
```bash
tradingagents analyze --ticker MSFT --date 2026-04-15 --lang Indonesian --headless
```

---

## 3. Review Generated Artifacts

Reports and machine-readable signal contracts are saved automatically to `reports/`:
- `reports/{ticker}_{timestamp}/complete_report.md`: Full executive investment report.
- `reports/{ticker}_{timestamp}/signal.json`: Typed `SignalContract` payload for broker execution.
- Sub-team markdown notes under `1_analysts/`, `2_research/`, `3_trading/`, `4_risk/`, `5_portfolio/`.
