# Comprehensive 16-Issue Architecture & Non-Hardcoded Remediation Plan

## 1. Executive Summary & Problem Formulation

This engineering specification details the exact remediation required to eliminate all **16 structural flaws, LLM reasoning ethics violations, and quantitative dataflow defects** identified during the multi-agent thermonuclear audit.

### 1.1 Core Principles
1. **Separation of Concerns & Role Integrity**: Every agent strictly performs its assigned jobdesk without prompt overreach or role drift.
2. **Anti-Prompt Gaming (Expectancy over Artificial Constraints)**: The Execution Trader and Portfolio Manager must not invent unrealistic price targets to satisfy hardcoded $R:R$ constraints. If structural resistance does not allow a favorable setup, the AI must choose **WNS (Wait and See)**.
3. **AI-Driven Dynamic Horizons with Broker Disipline**: The AI dynamically chooses `max_holding_days` (1–63 days) matching setup velocity; the Evaluator/Broker enforces this as a hard **Time-Stop** barrier (`HIT_TIME_STOP`) to prevent 46-day decay.
4. **Point-in-Time & Causal Dataflow Integrity**: Zero lookahead bias in intraday 1H data, chronological ascending time-series, and point-in-time fallback for non-trading days.
5. **Pure Spot Long-Only & Broker Realism**: Zero short-selling simulation, static bracket orders by default, and zero phantom limit fills.

---

## 2. Complete Inventory of the 16 Issues

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            THE 16 AUDIT FINDINGS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Dataflows & Causal Integrity                                       │
│   [Issue 1] Intraday 1H Lookahead Leakage (structural_levels.py:31-32)      │
│   [Issue 2] Time-Reversal Inversion Bias (y_finance.py:244-258)             │
│   [Issue 3] Weekend/Holiday Data Blackout -> Hallucination (stockstats:185) │
│   [Issue 4] Data Layer Policy Pollution (structural_levels.py:56-61)        │
│   [Issue 5] Short History Masquerading as 52-Week Range (structural:111)    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Analyst Nodes & Cognitive Grounding                                │
│   [Issue 6] Role Drift & Math Forcing on Market Analyst (market_analyst:37) │
│   [Issue 7] Ungrounded Value-Trap Bias & Price Amnesia (fundamentals:25)    │
│   [Issue 8] Attention Dilution from Swarm Boilerplate (market_analyst:48)   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 3 & 4: Debate, Trader & Anti-Gaming                                   │
│   [Issue 9] R:R Constraint Gaming -> Fantasy TP Targets (trader.py:37)      │
│   [Issue 10] Premature Hardcoded Numeric Anchors (trader.py:40)             │
│   [Issue 11] Role Drift on Research Manager (research_manager.py:36)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 5 & 6: Schemas & Decision Contracts                                   │
│   [Issue 12] Leaky WNS Payload & Validation Bypass (schemas.py:426)         │
│   [Issue 13] Residual Inverted Short Geometry (schemas:842, evaluate:106)   │
│   [Issue 14] Limit Entry Price Parameter Drop (schemas.py:822, 838)         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 7, 8 & 9: Graph Topology, Runner & Broker Realism                     │
│   [Issue 15] Static Horizon Decay Trap / 46-Day Hold (horizon:320)          │
│   [Issue 16] Walk-Forward Sleep Lockup on Flat Portfolio (walk_forward:542) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase-by-Phase Technical Specifications & Actionable Tasks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             EXECUTION PHASES                                │
│                                                                             │
│  Phase 1: Dataflow Point-in-Time & Causal Hardening (Issues 1-5)            │
│  Phase 2: Analyst Node Prompts & Role Boundary Restorations (Issues 6-8)    │
│  Phase 3: Debate & Execution Trader Anti-Gaming Refactor (Issues 9-11)      │
│  Phase 4: Schemas, WNS Sanitization & Limit Preservation (Issues 12-14)     │
│  Phase 5: Evaluator Time-Stop & Runner State Machine Hardening (Issues 15-16)│
│  Phase 6: Full Regression Verification & Test Suite Execution               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Dataflow Point-in-Time & Causal Hardening

#### Target Files
- `tradingagents/dataflows/structural_levels.py`
- `tradingagents/dataflows/y_finance.py`
- `tradingagents/dataflows/stockstats_utils.py`

#### Tasks
- [ ] **1.1 Fix Intraday 1H Lookahead Leakage (`structural_levels.py:31-34`) [Issue 1]**:
  - Do not hardcode `23:59:59` as the timestamp cutoff if an intraday timestamp is provided.
  ```python
  if len(str(trade_date)) > 10:
      cutoff_ts = pd.Timestamp(trade_date, tz="UTC")
  else:
      cutoff_ts = pd.Timestamp(f"{str(trade_date)[:10]} 23:59:59", tz="UTC")
  history = work_df[work_df["ts_utc"] <= cutoff_ts].sort_values("ts_utc").copy()
  ```
- [ ] **1.2 Standardize Chronological Ascending Time-Series (`y_finance.py:244-258`) [Issue 2]**:
  - Reverse the backward iteration in `get_stock_stats_indicators_window` to format date-value lines in strictly **ascending chronological order** ($t_{\text{oldest}} \to t_{\text{latest}}$).
  - Add explicit section header `# Time Series (Ascending Order: Past -> Present)`.
- [ ] **1.3 Point-in-Time Fallback for Non-Trading Days (`stockstats_utils.py:185-212`) [Issue 3]**:
  - When `curr_date` falls on a weekend or holiday, return the latest causal trading day's indicator value ($\le \text{curr\_date}$) instead of `"N/A: Not a trading day"`.
- [ ] **1.4 Remove Data Layer Policy Pollution (`structural_levels.py:56-61, 246-248`) [Issue 4]**:
  - Remove imperative trading rules (`"Rule: Trading proposals must ensure..."`) from data output summaries.
  - Return clean, objective mathematical metrics (1H EMAs, 1D SMAs, distance to 200 SMA, ATR) without subjective text bias.
- [ ] **1.5 Guard Small-Sample Lookback Extrema (`structural_levels.py:111-122`) [Issue 5]**:
  - If available history is $< 180$ bars, label range as `Available_History_High/Low ({n} bars)` instead of falsely labeling it `52_week_high/low`.

---

### Phase 2: Analyst Node Prompts & Role Boundary Restorations

#### Target Files
- `tradingagents/agents/analysts/market_analyst.py`
- `tradingagents/agents/analysts/fundamentals_analyst.py`
- `tradingagents/agents/analysts/news_analyst.py`
- `tradingagents/agents/analysts/sentiment_analyst.py`

#### Tasks
- [ ] **2.1 Restore Market Analyst Role Boundary (`market_analyst.py:30-55`) [Issue 6, 8]**:
  - Remove order ticketing commands (`"stage Limit Buy orders"`, `"guarantee R:R >= 2:1"`) from `market_analyst.py:37-38`.
  - Remove generic multi-turn swarm boilerplate (`"another assistant will help..."` in lines 48–51).
  - Focus prompt purely on multi-timeframe chart reading: support/resistance zones, trend alignment, volatility bands, and key momentum levels.
- [ ] **2.2 Ground Fundamentals Analyst with Real-Time Price & Regime Context (`fundamentals_analyst.py:25-35`) [Issue 7]**:
  - Inject current price ($T_0$ close) and macro moving average status into the user message context so the agent does not evaluate valuation in a vacuum or recommend buying falling knives.

---

### Phase 3: Debate & Execution Trader Anti-Gaming Refactor

#### Target Files
- `tradingagents/agents/trader/trader.py`
- `tradingagents/agents/managers/research_manager.py`
- `tradingagents/agents/managers/portfolio_manager.py`

#### Tasks
- [ ] **3.1 Eliminate R:R Constraint Gaming in Trader Prompt (`trader.py:30-60`) [Issue 9, 10]**:
  - Remove rigid mathematical coercion (`"must guarantee R:R >= 2:1"`).
  - Remove hardcoded example price numbers `(e.g., 350, 335, 400)`.
  - Replace with **Structural Expectancy Grounding**:
    > "Anchor Take Profit at realistic structural resistance (20D/60D Swing High, Fib retracement, or Chandelier exit). Anchor Stop Loss at structural support. If natural structural risk/reward is unfavorable ($R:R < 1.8:1$), do NOT invent higher price targets; you MUST choose **WNS (Wait and See)**."
  - Instruct Trader to calibrate `max_holding_days` (1–63 days) based on target distance relative to daily ATR.
- [ ] **3.2 Restore Research Manager Role Boundary (`research_manager.py:31-50`) [Issue 11]**:
  - In `research_manager.py:36`, replace `"Formulate clear entry, stop loss, and take profit targets"` with `"Formulate high-level strategic directional consensus and catalyst timeline"`.

---

### Phase 4: Schemas, WNS Sanitization & Limit Preservation

#### Target Files
- `tradingagents/agents/schemas.py`
- `cli/commands/evaluate.py`
- `tradingagents/backtesting/decision_schema.py`
- `tradingagents/backtesting/markdown_parser.py`

#### Tasks
- [ ] **4.1 Sanitize WNS Payload & Enforce Strict Validation (`schemas.py:426-440, 830-860`) [Issue 12]**:
  - In `portfolio_decision_to_signal_contract`: if `action in ("WNS", "HOLD", "NO_ORDER")`, strictly set `planned_entry_price = None`, `stop_loss = None`, `take_profit = None`, `entry_mode = EntryMode.T1_OPEN`.
  - In `PortfolioDecision._validate_portfolio_decision`: require that WNS decisions provide an explicit `wns_recheck_date` or `wns_trigger_price`.
- [ ] **4.2 Remove Residual Short-Selling Geometry (`schemas.py:842`, `evaluate.py:106`) [Issue 13]**:
  - Remove inverted short selling conditions (`take_profit < planned_entry < stop_loss`).
  - In `evaluate.py:106`, map non-BUY actions (`SELL`, `WNS`, `HOLD`) to `eval_side = "FLAT"` $\to$ `EvaluationOutcome.NO_ORDER` ($0.00\%$ return).
- [ ] **4.3 Preserve Limit Entry Price in Signal Contract Converter (`schemas.py:822, 838-847`) [Issue 14]**:
  - Resolve `raw_planned_entry = planned_entry_price if planned_entry_price is not None else getattr(decision, "planned_entry_price", None)`.
  - Set `entry_mode = EntryMode.T1_LIMIT` when $SL < P_{\text{entry}} < TP$.

---

### Phase 5: Evaluator Time-Stop & Runner State Machine Hardening

#### Target Files
- `tradingagents/backtesting/horizon_evaluator.py`
- `tradingagents/backtesting/walk_forward_runner.py`

#### Tasks
- [ ] **5.1 AI-Driven Time-Stop Horizon Enforcement (`horizon_evaluator.py:375-395, 690-710`) [Issue 15]**:
  - Read `max_holding_days` chosen dynamically by the AI agent (or `signal.max_holding_days`).
  - If position is still open on bar `max_holding_days` (and `max_holding_days < time_horizon_days`), execute **`HIT_TIME_STOP`** at `b_close`.
  - Prevents profitable trades (+15.5% MFE) from decaying over 46 days into Stop Loss.
- [ ] **5.2 Fix Walk-Forward Runner Sleep Lockup on Flat Portfolio (`walk_forward_runner.py:542-564`) [Issue 16]**:
  - Set `_next_reanalysis_date` **strictly** for `WNS` / `HOLD` decisions when portfolio is flat.
  - Reset `_next_reanalysis_date = None` immediately upon position exit/liquidation so the runner evaluates the next trading session.

---

### Phase 6: Full Regression Verification & Test Suite Execution

#### Target Files
- `tests/test_dynamic_exits.py`
- `tests/test_horizon_evaluator.py`
- `tests/test_typed_signal_integration.py`
- `tests/test_markdown_parser_v2.py`

#### Tasks
- [ ] **6.1 Add Test Cases Covering All 16 Issues**:
  - Intraday 1H timestamp filtering test.
  - Chronological time-series ascending order test.
  - Point-in-time non-trading day fallback test.
  - WNS payload sanitization test (all prices `None`).
  - AI-driven `HIT_TIME_STOP` horizon expiration test.
  - Walk-forward runner wake-up on position exit test.
- [ ] **6.2 Execute Full Test Suite**:
  - Run `pytest -v` across all test files.
  - Verify **747+ passed, 0 failed, 1 skipped**.

---

## 4. Execution Order for the Next AI Agent

1. **Step 1**: Implement Phase 1 Dataflow fixes (`structural_levels.py`, `y_finance.py`, `stockstats_utils.py`).
2. **Step 2**: Implement Phase 2 & 3 Prompt updates (`market_analyst.py`, `fundamentals_analyst.py`, `trader.py`, `research_manager.py`).
3. **Step 3**: Implement Phase 4 Schemas & Contracts (`schemas.py`, `evaluate.py`, `markdown_parser.py`).
4. **Step 4**: Implement Phase 5 Evaluator & Runner fixes (`horizon_evaluator.py`, `walk_forward_runner.py`).
5. **Step 5**: Run tests in Phase 6 (`pytest`) and verify 100% green suite.
