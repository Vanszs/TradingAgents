# Execution Trader Desk

The Execution Trader (`tradingagents/agents/trader/trader.py`) translates the Research Manager's directional plan into a concrete, executable tactical order ticket (`TraderProposal`).

---

## 1. Execution Order Taxonomy

The Trader selects from four concrete execution tactics:

1. **Buy Market**: Used for **Momentum Breakouts** (reclaiming 20D high or moving averages) and **Confirmed Oversold Bounces** (immediate entry at open to capture momentum and avoid missing runaway gaps).
2. **Buy Limit**: Used for **Orderly Pullbacks** at concrete structural demand floors (Fibonacci retracements, 60D swing low, or dynamic EMA support).
3. **WNS (Wait and See)**: Default when risk/reward is unfavorable, resistance is too close, or trend is ambiguous. Must supply `wns_recheck_date` or `wns_trigger_price`.
4. **Sell**: Orderly liquidation of existing long inventory to 100% Cash.

---

## 2. Structural Expectancy & Anti-Gaming Rules

- **Realistic Target Grounding**: Take Profit must anchor to real structural resistance (20D/60D Swing High, Chandelier Exit, 200 SMA, or Fibonacci Extensions 1.272x / 1.618x for breakouts).
- **Anti-Gaming Constraint**: If the natural structural reward relative to risk yields $R:R < 1.8:1$, the Trader is strictly **forbidden from inventing higher fantasy targets**; the Trader **must select WNS**.
- **Dynamic Holding Duration**: Sets `max_holding_days` (1–63 days) calibrated to target distance divided by daily ATR.
