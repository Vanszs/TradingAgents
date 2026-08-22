# Walk-Forward Backtesting Engine

The Walk-Forward Engine (`tradingagents/backtesting/walk_forward_runner.py`) simulates sequential multi-period portfolio execution across months or years of trading data.

---

## 1. Simulation Mechanics

```
For each trading date T in backtest date range:
  │
  ├── 1. Check Pending Triggers & Exits
  │     ├── Check Risk Engine: Did active positions hit SL / TP / Time-Stop?
  │     └── Check WNS Triggers: Did price touch wns_trigger_price or hit wns_recheck_date?
  │
  ├── 2. Determine Agent Execution
  │     ├── If Holding Active Position ──> Risk Management monitoring (Zero agent re-run)
  │     ├── If Flat & WNS Sleep Active ──> Skip agent until catalyst date or price trigger
  │     └── If Flat & Setup Awakened ────> Run LangGraph Multi-Agent Analysis at T0
  │
  ├── 3. Execute Orders at T+1
  │     └── Route generated orders to SimulatedBroker (Market Buy or Limit Buy)
  │
  └── 4. Mark-to-Market Portfolio Accounting
        └── Calculate Cash, Position Value, MTM Total Equity, and Margin Utilization
```

---

## 2. Institutional Performance Metrics (`metrics.py`)

- **Cumulative & Annualized Return (CAGR)**
- **Sharpe Ratio** (Risk-Free Rate adjusted)
- **Sortino Ratio** (Downside Semi-Deviation Root-Mean-Square across all $N$ days)
- **Calmar Ratio** ($\frac{\text{CAGR}}{\text{Max Drawdown}}$)
- **Win Rate & Profit Factor** ($\frac{\text{Gross Profit}}{\text{Gross Loss}}$)
- **FIFO Lot Reconciliation**: Tracks individual position tax lots and exact entry dates.
