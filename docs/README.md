# TradingAgents

**TradingAgents** is an institutional-grade, multi-agent financial trading framework built on top of [LangGraph](https://github.com/langchain-ai/langgraph). It deploys specialized AI agents across a structured trading floor hierarchy: from domain-isolated market analysts and adversarial debaters to execution traders and a Chief Investment Officer (Portfolio Manager).

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

## Core Philosophy

1. **Spot Long-Only Mandate**: Tailored for spot equities and crypto assets (no naked shorting). Capital is strictly deployed into high-conviction **BUY** setups or preserved in **100% Cash via WNS**.
2. **Adversarial Gatekeeping**: The system assumes an asset is **not worth buying** until the Bull debate and Technical structure prove a statistically significant edge. Default is **`WNS` (Wait and See)**.
3. **Point-in-Time Causal Data**: Strict timestamp slicing guarantees zero future lookahead bias in historical evaluations.
4. **Real Broker Realism**: Backtests enforce static bracket order constraints ($TP$ and $SL$), limit-touch range verification (zero phantom fills), and time-in-force limits (`max_holding_days`).
