# Chief Investment Officer (Portfolio Manager)

The Portfolio Manager (`tradingagents/agents/managers/portfolio_manager.py`) acts as the Chief Investment Officer (CIO), making the final capital allocation decision and generating the machine-readable `SignalContract`.

---

## 1. Responsibilities & Decision Hierarchy

1. **Synthesizes 4 Information Streams**:
   - Research Manager's strategic plan (`investment_plan`).
   - Execution Trader's proposed pricing geometry (`trader_proposal`).
   - Risk Committee debate transcript (`risk_debate_state`).
   - Episodic memory reflections (`past_context`).
2. **Issues Final Actionable Rating**:
   - Strictly binary: **BUY** (Market/Limit) vs **WNS** (Wait and See).
3. **Generates Signal Contract (`SignalContract`)**:
   - Serializes verified pricing geometry, allocation sizing, time horizon, and trigger conditions into `signal.json`.

---

## 3. Governance Invariants

1. **Trader WNS Invariant (Absolute Rule)**: If the Trader Proposal is **WNS**, the Portfolio Manager **CANNOT** override or upgrade it into a BUY. Capital cannot be deployed when execution identifies a falling knife breakdown or lack of mathematical edge.
2. **Decoupled Position Sizing & Signal Validity**: If the Risk Committee flags elevated asset volatility, the Portfolio Manager scales down position allocation (e.g. 5%–10% NAV) rather than vetoing a mathematically valid trade with $R:R \ge 2.0:1$ to WNS.
3. **Adaptive Sample-Length Handling**: For assets with $< 180$ bars (IPOs), the CIO assesses 20D/60D base structure rather than rejecting setups solely for lack of a 200 SMA.

```python
class SignalContract(BaseModel):
    ticker: str
    signal_date: str
    action: str              # "BUY" or "WNS"
    rating: PortfolioRating  # BUY or WNS
    planned_entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_mode: Optional[EntryMode]  # T1_OPEN, T1_LIMIT
    time_horizon_days: int
    max_holding_days: Optional[int]
    confidence: float
    thesis_summary: str
    
    # WNS Specific Triggers
    wns_condition_type: Optional[WNSConditionType]
    wns_recheck_date: Optional[str]
    wns_trigger_price: Optional[float]
```
