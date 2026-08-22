# Spot Long-Only & Binary Mandate

TradingAgents is architected for strict **Spot Long-Only** asset allocation.

---

## 1. Binary Decision Output (BUY vs WNS)

Every analysis culminates in one of two actionable states:

```
┌──────────────────────────────────────────────┐
│            BINARY DECISION SCALE             │
├──────────────────────────────────────────────┤
│ 1. BUY (Market Buy / Limit Buy)              │
│    • High-conviction asymmetric upside.      │
│    • Enforces SL < Planned Entry < TP.       │
│    • Capital deployed: Allocates % equity.   │
├──────────────────────────────────────────────┤
│ 2. WNS (Wait and See / No Order)             │
│    • Market trend ambiguous or R:R < 1.8:1.  │
│    • Mandatory Catalyst Date X OR Price Y.   │
│    • Capital deployed: 100% Cash preserved.  │
└──────────────────────────────────────────────┘
```

---

## 2. Strict WNS Contract Invariant

A decision cannot simply declare "Hold" or "Wait" without actionable re-evaluation criteria. Every `WNS` decision **must** supply at least one trigger:
1. **Temporal Catalyst Gate (`wns_recheck_date`)**: e.g., `2026-04-28` (post-earnings release or macro central bank meeting).
2. **Structural Price Gate (`wns_trigger_price`)**: e.g., `$383.47` (support floor retest or resistance breakout level).

In backtesting, the runner holds **100% Cash** until the trigger price is touched or the catalyst date is reached.

---

## 3. Zero Short-Selling Guarantee

- Inverted short-selling order geometry (`TP < Entry < SL`) is strictly banned across all schemas and evaluators.
- A `SELL` signal generated while flat is mapped to `NO_ORDER` (0% return, zero capital at risk).
- A `SELL` signal generated while holding inventory triggers an orderly **Liquidation to 100% Cash**.
