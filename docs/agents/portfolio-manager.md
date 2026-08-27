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

## 2. Pydantic Output Contract (`SignalContract`)

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
