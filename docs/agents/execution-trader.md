# Execution Trader Desk

The Trader (`tradingagents/agents/trader/trader.py`) translates the Research Manager's strategic plan into concrete transaction pricing geometry under a strict **Spot Long-Only** mandate.

---

## 1. Execution Taxonomy & Tactics

1. **Buy Market (`T1_OPEN`)**:
   - Mandatory default for **Momentum Breakouts**, **IPO Base Breakouts**, **Dynamic Moving Average Reclaims (20 EMA, 50 SMA, 200 SMA)**, and **Consolidation below Resistance**.
   - Fills at $T+1$ Session Open without placing deep limit orders that risk `NO_FILL` missed alpha during fast momentum expansion.
2. **Buy Limit (`T1_LIMIT`)**:
   - Used exclusively for **Orderly Pullbacks** to structural demand floors (Fibonacci 50%/61.8%, 20D swing low, 50 SMA support) where price is mean-reverting.
   - Limit distance is tightly bounded within $\le 0.5\times \text{ATR}_{14}$ of the reference close.
3. **WNS (Wait and See)**:
   - Mandatory when trend is an unconfirmed Falling Knife (sub-200 SMA breakdown), resistance is too tight ($R:R < 1.8:1$), or regime is ambiguous. Must supply `wns_recheck_date` or `wns_trigger_price`.

---

## 2. Structural Expectancy & Risk Geometry

- **Target Grounding (Anti-Resistance Paralysis)**: When consolidating within 5% below a 52W High, 60D High, or IPO ceiling in an uptrend, Take Profit is anchored to **Fibonacci Extensions (1.272x / 1.618x)** or **+3x ATR Channels** to capture breakout continuation, maintaining convex $R:R \ge 2.5:1$.
- **Volatility-Aware Stop Loss**: Stop Loss is placed below the structural floor with a mandatory **$1.0\times$ to $1.5\times \text{ATR}_{14}$ buffer** to prevent market open noise stop-outs.
- **IPO & Short-History Invariant**: For assets with $< 180$ bars (new IPOs or recent listings), structure is evaluated strictly via 20D/60D swing range and volatility compression without penalizing the lack of a 200-day SMA.
- **Static Exits**: Every BUY order specifies fixed, entry-time stop loss and take profit targets. Zero trailing or ratcheting stops post-entry.
