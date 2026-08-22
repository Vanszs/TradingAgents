# TradingAgents (Vanszs Edition)

<div align="center">
  <h3>Institutional-Grade Multi-Agent LLM Financial Trading & Backtesting Framework</h3>
  <p><em>Maintained and hardened by <strong>Vanszs</strong> | Hardened fork of <a href="https://github.com/TauricResearch/TradingAgents">Tauric Research's TradingAgents</a> (Apache-2.0 License)</em></p>
  <a href="https://github.com/Vanszs/TradingAgents/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/></a>
  <a href="https://github.com/Vanszs/TradingAgents"><img alt="Tests" src="https://img.shields.io/badge/Tests-751_passed-brightgreen.svg"/></a>
  <a href="https://github.com/Vanszs/TradingAgents/tree/main/docs"><img alt="GitBook Docs" src="https://img.shields.io/badge/Docs-GitBook_Ready-14C290?logo=gitbook"/></a>
</div>

---

## 🌟 What is TradingAgents (Vanszs Edition)?

**TradingAgents (Vanszs Edition)** is an institutional-grade, multi-agent financial trading and quantitative backtesting framework built on top of [LangGraph](https://github.com/langchain-ai/langgraph). It deploys specialized AI agents across a structured trading floor hierarchy: from domain-isolated market analysts and adversarial debaters to execution traders and a Chief Investment Officer (Portfolio Manager).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE TRADING FLOOR FLOW                            │
│                                                                             │
│  I. Analyst Team (Parallel Fan-Out)                                         │
│     ├── Technical Market Analyst (1D Macro / 1H Micro Structure)            │
│     ├── Fundamentals Analyst (Balance Sheet, Valuation, Financial History)  │
│     ├── News & Macro Analyst (Corporate Filings, Headlines, Sentiment)      │
│     └── Social Sentiment Analyst (Crowd Psychology, Fear & Greed Index)     │
│                                                                             │
│  II. Research & Debate Team (Adversarial Synthesis)                         │
│     ├── Bull Researcher (Upside Catalysts & Mean-Reversion Thesis)          │
│     ├── Bear Researcher (Downside Risks & Valuation Traps)                  │
│     └── Research Manager (Consensus Synthesis & Strategic Plan)             │
│                                                                             │
│  III. Execution Trader (Micro Tactical Pricing)                             │
│     └── Orders: Buy Market, Buy Limit, WNS, or Sell (SL < Entry < TP)       │
│                                                                             │
│  IV. Risk Management Committee (Stress Testing)                             │
│     └── Aggressive vs Conservative vs Neutral Risk Analysts                 │
│                                                                             │
│  V. Portfolio Management (CIO Capital Allocation)                           │
│     └── Final Verdict: BUY (Market/Limit) vs WNS (Wait and See)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Upgrades in Vanszs Edition

1. **Strict Spot Long-Only & Binary Mandate**: Zero short selling. Signals are strictly **`BUY`** (Market/Limit) or **`WNS`** (Wait and See with explicit catalyst date X or price trigger Y).
2. **Real Broker Execution Realism**: Backtests enforce static bracket orders ($TP$ and $SL$), limit-touch range verification (zero phantom fills), and time-in-force limits (`max_holding_days`).
3. **Forward Fibonacci Extensions & ATR Target Channels**: Injects forward expansion upside (1.272x, 1.618x) so momentum breakout setups are not penalized by backward-looking resistance walls.
4. **Anti-Gaming Prompt Architecture**: The Execution Trader anchors targets to structural resistance. If natural risk/reward is unfavorable, the AI selects **`WNS`** instead of inventing fantasy targets.
5. **Episodic Memory & Past Reflection**: Injects historical trade reflections (`past_context`) into the Chief Investment Officer's prompt, preventing repeated mistakes.
6. **Multi-Provider LLM Engine**: Native support for OpenAI (`/v1/responses`), Anthropic (effort control), Google Gemini (thinking levels), DeepSeek (multi-turn reasoning), Ollama (local self-hosted), Azure OpenAI, and BluesMind.

---

## 📦 Quick Installation

```bash
# 1. Clone repository
git clone https://github.com/Vanszs/TradingAgents.git
cd TradingAgents

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install package
pip install -e .
```

---

## ⚡ Quickstart

### 1. Interactive Terminal UI (TUI)
```bash
tradingagents
```

### 2. Automated Headless Analysis
```bash
tradingagents analyze --ticker NVDA --provider openai --headless
```

### 3. Forward Signal Evaluation
```bash
tradingagents evaluate-signal --ticker AAPL --date 2024-06-11
```

### 4. Multi-Period Backtesting
```bash
tradingagents backtest --config backtest.yaml --lookback 60
```

---

## 📚 Documentation (GitBook Ready)

Complete, institutional-grade documentation is available in the [`docs/`](docs/) directory:
- [Getting Started](docs/getting-started/quickstart.md)
- [System Architecture](docs/architecture/overview.md)
- [Multi-Agent Teams](docs/agents/analysts.md)
- [Dataflows & Structural Levels](docs/dataflows/structural-levels.md)
- [Quantitative Backtesting](docs/backtesting/broker-realism.md)
- [CLI Reference](docs/cli-reference/interactive-mode.md)
- [Developer & Extension Guide](docs/developer-guide/adding-new-agents.md)

---

## 🤝 Acknowledgements & Attribution

Special thanks and appreciation to **[Tauric Research](https://github.com/TauricResearch)** as the original creator of the initial TradingAgents architecture. 

This repository is distributed under the **[Apache License 2.0](LICENSE)**, honoring the open-source spirit while introducing advanced quantitative hardening and institutional execution mechanics.
