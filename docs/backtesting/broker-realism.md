# Real Broker Execution Realism

TradingAgents aligns backtesting assumptions with the physical execution constraints of real retail and institutional spot brokerages.

---

## 1. Pure Static Bracket Orders (Default)

In real spot trading, brokers provide standard static bracket orders:
- **Planned Entry Price** ($P_{\text{entry}}$)
- **Static Take Profit** ($TP$)
- **Static Stop Loss** ($SL$)

By default, `trailing_stop_pct = None` and `break_even_trigger_pct = None`. Backtest evaluation does **not** move or ratchet stop losses in-flight, preventing artificially inflated performance figures.

---

## 2. Order Fill Physics (Zero Phantom Fills)

- **Market Buy Orders**: Executed at next session's Open price ($P_{\text{open}}^{T+1}$).
- **Limit Buy Orders**: Staged at $P_{\text{limit}}$. Executed **only** if the bar range covers the price:
  $$\text{Low}_{T+1} \le P_{\text{limit}} \le \text{High}_{T+1}$$
  If the market gaps up or rallies away without touching the limit, the order remains unfilled (`NO_FILL`).

---

## 3. Cash Clamping & Capital Sizing

- Position sizing is strictly bounded by liquid free cash ($\text{Available Cash} \ge \text{Order Cost} + \text{Slippage} + \text{Commission}$).
- Over-allocation or negative cash balances are strictly rejected by `OrderGenerator`.
