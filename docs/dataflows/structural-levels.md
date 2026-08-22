# Structural Levels & Technical Indicators

TradingAgents pre-computes institutional-grade structural price levels (`tradingagents/dataflows/structural_levels.py`) and injects them directly into agent context.

---

## 1. 1D Macro Structural Indicators

- **52-Week & Multi-Month Swing Ranges**:
  - `52_week_high` / `52_week_low` (252 bars).
  - `60d_swing_high` / `60d_swing_low` (60-day range).
  - `20d_swing_high` / `20d_swing_low` (20-day swing shelf).
- **Directional Fibonacci Retracements**:
  - Automatically detects whether the 60D swing is a **Bullish Impulse** (Low $\to$ High) or **Bearish Leg** (High $\to$ Low).
  - Bullish: $\text{Fib}_{50\%} = H_{60} - 0.50 \times \text{range}$ (Pullback demand support).
  - Bearish: $\text{Fib}_{50\%} = L_{60} + 0.50 \times \text{range}$ (Counter-trend bounce resistance).
- **Fibonacci Extensions & Forward Target Channels**:
  - $\text{Fib Extension 1.272x} = H_{60} + 0.272 \times \text{range}$
  - $\text{Fib Extension 1.618x} = H_{60} + 0.618 \times \text{range}$
  - $\text{ATR Volatility Targets} = \text{Close} + 2.0 \times \text{ATR}_{14} \text{ and } \text{Close} + 3.0 \times \text{ATR}_{14}$
- **Causal Volatility Bands**:
  - $\text{ATR}_{14}$ & $\text{ATR}_{20}$ calculated via Wilder's Exponential Smoothing.
  - $\text{Chandelier Exit Long} = \max(\text{High}_{22}) - 3.0 \times \text{ATR}_{22}$.

---

## 2. 1H Micro Intraday Structure

- **24-Bar Swings**: Short-term demand and supply floors.
- **Intraday Momentum**: 1H $\text{EMA}_{20}$ and $\text{EMA}_{50}$ alignment (`BULLISH`, `BEARISH`, `NEUTRAL`).
