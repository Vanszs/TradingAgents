# Architecture Overview

TradingAgents decouples analytical fact-finding, thesis debate, micro tactical pricing, and portfolio capital allocation across an explicit LangGraph state graph.

---

## The 5-Layer Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA & INDICATOR LAYER                                                   │
│    • 1D Macro Trends, 1H Micro Structure, Fib Retracements & Extensions     │
│    • Causal ATR (14/20), Chandelier Exit (22, 3.0x ATR), Wilder Smoothing    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. DOMAIN ANALYST LAYER (Parallel Fan-Out)                                  │
│    • Market Analyst, Fundamentals Analyst, News Analyst, Sentiment Analyst │
│    • Zero state pollution: each branch writes to isolated message channels  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. RESEARCH & DEBATE LAYER (Adversarial Synthesis)                          │
│    • Bull Researcher: Catalysts, Breakouts, and Oversold Mean-Reversions   │
│    • Bear Researcher: Downside risks, Valuation Traps, Overhead Supply      │
│    • Research Manager: Debate judge emitting structured ResearchPlan       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. EXECUTION TRADER LAYER (Tactical Order Pricing)                          │
│    • Translates strategic view to concrete orders: Buy Market vs Buy Limit  │
│    • Structural Expectancy: Anchors TP at real resistance (Anti-Gaming R:R) │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. RISK COMMITTEE & CIO PORTFOLIO MANAGER                                   │
│    • Aggressive / Conservative / Neutral Risk Review                        │
│    • CIO final allocation: Strictly BUY vs WNS (SignalContract)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Patterns

1. **State Channel Isolation**: Each analyst operates on dedicated message lists (`market_messages`, `social_messages`, etc.) using LangGraph `add_messages` reducers, eliminating race conditions during parallel branch execution.
2. **Barrier Synchronization**: All parallel analyst branches join cleanly at `Bull Researcher` before sequential debate starts.
3. **Point-in-Time Memory Injection**: `past_context` loads historical trade reflections strictly as-of the signal date, enabling reflection learning without lookahead bias.
