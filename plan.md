# Production Hardening & Architecture Master Plan (TradingAgents)

**Version:** 2.0.0 (Post-10-Subagent Thermonuclear & Ponytail Audit)  
**Roles:** Senior SWE, Senior Quant Trader, Senior AI Systems Engineer  
**Core Domain Strategy:** **Spot Equity Long-Only: Pure BUY (HAKA) & WNS (Wait and See / Staged Limit Accumulation)**  
**Safety Invariant:** **DO NOT delete or wipe `.understand-anything` or historical dataset snapshots.**

---

## 1. Domain Strategy & Execution Invariants

In our institutional spot long-only architecture, passive "blind hold" (holding without execution parameters) is forbidden:
1. **BUY (HAKA / Immediate Market Entry on $T+1$ Open)**:
   - Evaluated when price sits directly on confirmed structural demand/support with bullish momentum and $R:R \ge 2:1$.
2. **WNS (Wait and See / Limit Accumulation / `ASSUMED_AI_ENTRY`)**:
   - Evaluated when price is extended or pulling back towards support (e.g. current 390, demand floor 350–355).
   - **Mandatory Parameters**: Must output `planned_entry_price` (e.g. 355), `stop_loss` (e.g. 340), `take_profit` (e.g. 420), and `wns_recheck_date`.
   - The engine evaluates this as a limit order / price-touch trigger (`ASSUMED_AI_ENTRY`) on subsequent bars.
3. **SELL (Liquidation / Capital Preservation)**:
   - Complete exit of long inventory ($0\%$ allocation).

---

## 2. 10-Subagent Audit Matrix: Real Blockers vs False Positives

```
+-------------------------------------------------------------------------------------------------------------------------+
| VERDICT CLASSIFICATION (21 AUDIT FINDINGS)                                                                              |
+-------------------+-----------------------------------------------------------------------------------------------------+
| REAL BLOCKER (8)  | B1. Horizon Clamping >63d         B2. Margin Liquidation Limit Bug    B3. Planned Entry Lost in DSM     |
|                   | B4. .JK Benchmark Fallback to SPY B5. Limit Slippage Inversion        B6. ATR Morning Lookahead Leakage |
|                   | B7. yfinance Multi-Day Truncation B8. Borrow & Financing Unaccrued                                  |
+-------------------+-----------------------------------------------------------------------------------------------------+
| FALSE POSITIVE (5)| FP1. Layer Inversion (Lazy import, non-blocking)    FP2. Crypto Tools Bypass PIT (Fail-closed by design)|
|                   | FP3. Checkpoint Resume Theater (Fixed: None input)  FP4. UNDERWEIGHT Trim vs Sell (Quant standard)      |
|                   | FP5. Decision Valid-From Hallucination (Fixed snap)                                                     |
+-------------------+-----------------------------------------------------------------------------------------------------+
| VALID IMPROVE (8) | 1. Dual BacktestConfig Schemas Alias   2. HOLD vs WNS Schema Validation  3. Multi-Tenant Config Context |
|                   | 4. Junk CSV Cache Guard                5. CLI Monoliths Decomposition    6. Duplicate TUI Logic         |
|                   | 7. importlib.reload Elimination        8. Untracked Artifacts in Git Index                              |
+-------------------+-----------------------------------------------------------------------------------------------------+
```

### Detailed Breakdown of 8 Real Blockers

1. **B1: Horizon Clamping >63d (`schemas.py:894`)**  
   `PortfolioDecision.time_horizon_days` ($le=252$) passed unclamped to `SignalContract.max_holding_days` ($le=63$). Triggers Pydantic `ValidationError` $\to$ `signal_contract = None` $\to$ CLI crash.  
   *Fix*: `max_holding_days = decision.max_holding_days or min(63, decision.time_horizon_days)`.

2. **B2: Forced Liquidation Limit Order Bug (`walk_forward_runner.py:382`, `broker.py:258`)**  
   EOD margin breach sent `price=close` (Limit Order). On gap-down open ($High < Limit$), order stayed `UNFILLED`. Bankrupt accounts survived in simulation (*survivorship bias*).  
   *Fix*: Forced liquidation sent with `price=None` $\to$ executed at Market Open $T+1$.

3. **B3: Planned Entry Price Pipeline Loss (`schemas.py:559`, `decision_state_manager.py:190, 420`)**  
   `render_pm_decision` omitted `planned_entry_price` and DSM dropped it on `ExtendedDecision`. `TriggerEvaluator` saw `None` and staged limit accumulation never fired.  
   *Fix*: Render field in markdown and pass through `_map_strict` and `_map_legacy`.

4. **B4: Indonesian Benchmark `.JK` Fallback to `SPY` (`default_config.py:124`)**  
   `benchmark_map` lacked `.JK`, falling back to `SPY` (USD). Mismatched IDR return vs USD index without FX adjustment corrupted reflection memory loop.  
   *Fix*: Added `".JK": "^JKSE"`.

5. **B5: Buy Limit Slippage Inversion (`broker.py:258-297`)**  
   Buy limit filled at $\text{limit} \times (1 + \text{friction}) > \text{limit}$, violating exchange limit rules. Gap-down opens also failed to grant price improvement.  
   *Fix*: $P_{fill} = \min(P_{open} \times (1 + \text{friction}), P_{limit})$.

6. **B6: ATR Morning Lookahead Bias (`walk_forward_runner.py:309`)**  
   At open of day $T$, ATR calculated with $df \le T$, including unclosed bar $H_T, L_T, C_T$ (intraday future leak).  
   *Fix*: Anchored risk calculation to `current_date = self.calendar.previous_trading_day(current_date)`.

7. **B7: `yfinance` Date Boundary Truncation (`y_finance.py:106`)**  
   Daily `yfinance.history` is end-date exclusive. Multi-day range queries dropped the final day bar ($T_{end}$).  
   *Fix*: Appended $+1$ day offset: `end_inclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")`.

8. **B8: Unaccrued Borrow & Financing Fees (`portfolio.py:918-927`, `position.py:97`)**  
   Shorts and leverage longs incurred 0 carry cost because fee methods were never called in daily mark-to-market.  
   *Fix*: Deducted daily fee directly from cash in `PortfolioV2.mark_to_market`.

---

## 3. Surgical Implementation Guide (Applied & Verified Diffs)

### Phase 1: Schemas & Decision Pipeline
```python
# File: tradingagents/agents/schemas.py

# 1. Horizon Clamping (Line ~894)
max_holding_days=decision.max_holding_days or min(63, decision.time_horizon_days),

# 2. Render Quantitative Fields (Line ~559-567)
if decision.planned_entry_price is not None:
    parts.extend(["", f"**Planned Entry Price**: {decision.planned_entry_price}"])
if decision.confidence is not None:
    parts.extend(["", f"**Confidence**: {decision.confidence:.2f}"])
if decision.wns_trigger_price is not None:
    parts.extend(["", f"**WNS Trigger Price**: {decision.wns_trigger_price}"])
if decision.wns_recheck_date:
    parts.extend(["", f"**WNS Recheck Date**: {decision.wns_recheck_date}"])

# 3. Decision Authority Hierarchy (Line ~836-840)
raw_planned_entry = (
    getattr(decision, "planned_entry_price", None)
    if getattr(decision, "planned_entry_price", None) is not None
    else planned_entry_price
)
```

```python
# File: tradingagents/backtesting/decision_state_manager.py
# Inside _map_strict (Line ~190) & _map_legacy (Line ~420):
stop_price=decision.stop_price,
take_profit=decision.take_profit,
planned_entry_price=decision.planned_entry_price,
wns_trigger_price=decision.wns_trigger_price,
wns_recheck_date=decision.wns_recheck_date,
next_review_date=decision.next_review_date,
time_horizon_label=decision.time_horizon_label,
time_horizon_days=decision.time_horizon_days,
```

### Phase 2: Simulation Engine & Broker Execution
```python
# File: tradingagents/backtesting/walk_forward_runner.py

# 1. Forced Liquidation at Market Order (Line ~382)
risk_order = self._build_force_close(
    next_valid_date, None, "eod_margin_breach"
)

# 2. Morning ATR Lookahead Elimination (Line ~309)
self._update_risk_levels(
    self.prev_decision,
    self._ohlcv_df,
    current_date=self.calendar.previous_trading_day(current_date),
)
```

```python
# File: tradingagents/backtesting/broker.py (Line ~258-297)
is_limit = order.price is not None and float(order.price) > 0
limit_p = float(order.price) if is_limit else 0.0

if not is_limit:
    base_price = open_price
elif is_buy:
    base_price = min(open_price, limit_p)
else:
    base_price = max(open_price, limit_p)

# Apply friction and enforce limit bound
if is_buy:
    fill_price = base_price * (1 + total_friction_pct)
else:
    fill_price = base_price * (1 - total_friction_pct)

if is_limit:
    fill_price = min(fill_price, limit_p) if is_buy else max(fill_price, limit_p)
```

```python
# File: tradingagents/backtesting/portfolio.py (Line ~918-927)
pos_val = abs(self.position_value(mark))
if pos_val > 0:
    if self.position.is_short():
        carry_fee = pos_val * float(getattr(self.margin_cfg, "borrow_fee_daily", 0.0002))
    else:
        borrowed_cash = max(0.0, pos_val - max(0.0, self.cash + pos_val))
        carry_fee = borrowed_cash * float(getattr(self.margin_cfg, "financing_rate_daily", 0.0001))
    if carry_fee > 0:
        self.cash -= carry_fee
```

### Phase 3: Configuration & Dataflows
```python
# File: tradingagents/default_config.py (Line ~124)
".JK": "^JKSE",    # Indonesia (Jakarta Composite / IHSG)

# File: tradingagents/dataflows/y_finance.py (Line ~106)
end_inclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
```

---

## 4. Next Immediate Actions (Operational Roadmap)

> **Status: ALL EXECUTED (this session).** Actuals below supersede the original asks.

1. **Lock Invariant Unit Tests** — DONE: `tests/test_audit_invariants.py` (12 tests).
   Covers the roundtrip `render_pm_decision → MarkdownDecisionParser →
   DecisionStateManager` asserting `confidence` and `planned_entry_price` survive,
   plus broker limit-cap (`P_fill ≤ P_limit`, exact-cap on gap-up, price
   improvement on gap-down, UNFILLED when untouched), market-order liquidation
   through a gap-down (with the stale-limit stranding case as contrast), daily
   borrow/financing accrual incl. debit-balance compounding, and zero-cost
   unleveraged longs.
2. **Git Index Hygiene** — DONE: `git rm -r --cached` applied to `.kiro/`,
   `backtest_results/`, plus `snapshots/ backtest_cache/ reports/ data/`
   (2,308 index deletions staged; physical files untouched). `.gitignore`
   extended for all six paths. **Not committed — review `git status`, then commit.**
3. **Execution Validation** — DONE: full suite is now **765 passed, 1 skipped**
   (not 751: +12 invariant tests, +6 CLI-split tests, −2 vacuous tests removed
   with a deleted identity function, config-unification re-exports verified).

---

## 4b. Executed Beyond This Document (same session, verified)

- Junk-CSV cache guard + atomic cache write (`stockstats_utils.load_ohlcv`);
  legacy poisoned caches are dropped and refetched.
- Layer inversion removed: `cli/data_fetch.py` → `tradingagents/backtesting/ohlcv_fetch.py`;
  engine lazy-import now targets the library; zero `cli.data_fetch` references remain.
- Dual BacktestConfig family unified onto `position.py`; `decision_schema.py`
  546→246 lines (re-exports + legacy-only shapes); dead knobs deleted
  (`borrow_fee_apr`, `BacktestConfig.default_reduce_pct`, legacy `Order` extras);
  `backtest.yaml` resolution guard: 108 flattened keys, zero value diffs.
- CLI monoliths split mechanically: `main.py` 1405→568
  (`message_buffer/display/report_io/selections.py`), `commands/backtest.py`
  1318→504 (`backtest_tui/backtest_report/backtest_prompts.py`).
- Checkpoint resume wired for real (`invoke(None)` when a checkpoint step exists)
  and far-future `decision_valid_from` now logs a loud per-day wait warning.

**Remaining open (need spec owner / larger design, deliberately not auto-fixed):**
UNDERWEIGHT trim-vs-exit unification; TUI renderer merge + graph-progress observer
into the library; `importlib.reload` env-var config hack → explicit overrides param;
global mutable run-context race (`dataflows/config.py` singleton).

---

## 5. Verification Protocol

```bash
# 1. Targeted Suite (Checkpoint, Signal, 5-Tier, Margin, Risk, Broker)
pytest tests/test_checkpoint_resume.py tests/test_signal_processing.py \
       tests/test_strict_5tier.py tests/test_typed_signal_integration.py \
       tests/test_stock_order_generator.py tests/test_risk_execution.py \
       tests/test_margin.py tests/test_markdown_parser_v2.py -q

# 2. Full Regression Suite
pytest -q
```
**Acceptance Benchmark (actual, this session):** full suite `765 passed, 1 skipped
(live API gate), 83 subtests passed, 0 failed`; targeted §5 suite `101 passed`;
`ruff check cli/ tradingagents/backtesting/` clean; `import cli.main,
cli.commands.backtest` OK.
