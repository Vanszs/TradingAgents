# Execution Trader Desk

The Trader selects from three concrete execution tactics:

1. **Buy Market**: Used for **Momentum Breakouts** and **Confirmed Oversold Bounces**.
2. **Buy Limit**: Used for **Orderly Pullbacks** at concrete structural demand floors.
3. **WNS (Wait and See)**: Default when risk/reward is unfavorable, resistance is too close, or trend is ambiguous. Must supply `wns_recheck_date` or `wns_trigger_price`.

Static stop-loss, take-profit, time-stop, or horizon expiry closes an existing long; the agent never emits an exit order.

---

## 2. Structural Expectancy & Anti-Gaming Rules

- **Realistic Target Grounding**: Take Profit must anchor to real structural resistance (20D/60D Swing High, 200 SMA, or Fibonacci Extensions 1.272x / 1.618x for breakouts).
- **Anti-Gaming Constraint**: If the natural structural reward relative to risk yields $R:R < 1.8:1$, the Trader is strictly **forbidden from inventing higher fantasy targets**; the Trader **must select WNS**.
- **Holding Duration**: Sets `max_holding_days` (1–63 days) based on the planned trade horizon.
