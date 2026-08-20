# Comprehensive Implementation Plan: Dynamic Exit Engine, Horizon Evaluation Hardening, and Schema-Indicator Integration

## 1. Executive Summary & Problem Statement

### 1.1 The Critical Quant Flaw
During backtest evaluation across multiple assets, positions with substantial peak Maximum Favorable Excursion (**MFE**) suffered severe profit decay and reversed into full stop-loss hits:
- **AVGO (2025-02-04)**: Gained **+9.27% MFE** by Bar 1–8, hovered in profit for 11 days, but decayed over 18 days into a **-12.33% Stop Loss**.
- **AMZN (2025-01-28)**: Reached **+3.42% MFE**, stalled, and decayed over 14 days into a **-3.88% Stop Loss**.

### 1.2 Root Causes Identified via 10-Subagent Audit
1. **Static Barrier Trap (`horizon_evaluator.py`)**: Positions are evaluated against rigid, immutable $SL$ and $TP$ lines across up to 63 bars without Break-Even Ratchets, Trailing Stops, or Time-Decay exits.
2. **Gap-Open Priority & Same-Bar Conflict Inversion (`horizon_evaluator.py`)**: `b_low <= stop_loss` is checked before `b_open >= take_profit`. A gap-up open over TP is falsely stopped out if the bar's low touches SL.
3. **Phantom Limit Entry Fill (`horizon_evaluator.py`)**: `ASSUMED_AI_ENTRY` unconditionally assigns limit prices without verifying if the bar's range touched the price (`b_low <= entry <= b_high`).
4. **Schema Truncation (`schemas.py`)**: `TraderProposal`, `TradingDecision`, and `SignalContract` lack dynamic exit fields (`trailing_stop_pct`, `break_even_trigger_pct`, `max_holding_days`).
5. **Indicator Void (`stockstats_utils.py`, `structural_levels.py`)**: LLMs receive basic SMA/RSI/MACD but lack ATR volatility bands and Chandelier Exit levels (`High_22 - 3*ATR_22`).
6. **Orphaned Config (`position.py`)**: `RiskConfig.trailing_stop_pct` exists in data structures but is never passed to or evaluated by `HorizonEvaluator`.

---

## 2. Architecture & Invariants

```
+─────────────────────────────────────────────────────────────────────────────+
│                       DYNAMIC HORIZON EVALUATION STATE                      │
│                                                                             │
│  T+1 Entry ───> Bar Evaluation ───> MFE >= Break-Even Threshold (e.g. +3%)  │
│                     │                       │                               │
│                     │                       ▼                               │
│                     │               Ratchet SL = Entry + FeeBuffer          │
│                     │                       │                               │
│                     │               Trailing Stop Activated                 │
│                     │               SL = max(SL, PeakHigh * (1 - Trail%))   │
│                     │                       │                               │
│                     ▼                       ▼                               │
│              Check Exits:           Check Exits:                            │
│              1. Gap Open            1. Gap Open                             │
│              2. Dynamic SL / Trail  2. Dynamic SL / Trail                   │
│              3. Take Profit         3. Take Profit                          │
│              4. Max Holding Days    4. Max Holding Days                     │
+─────────────────────────────────────────────────────────────────────────────+
```

### Core Invariants
1. **Spot Long-Only Discipline**: No short selling. Exit actions liquidate long exposure back to 100% cash.
2. **Monotonic Stop Ratchet**: Once Stop Loss moves up (via Break-Even or Trailing Stop), it **NEVER** moves down.
3. **Gap-First Order Execution**: Gap open beyond price barriers takes priority over intraday highs/lows.
4. **Zero Phantom Fills**: Limit orders only fill if the bar's price range actually touches the limit price.
5. **No Lookahead Leakage**: Trailing stops on Bar $t$ are derived strictly from high/close prices up to Bar $t-1$ or open/current of Bar $t$.

---

## 3. Detailed Phase-by-Phase Action Plan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             EXECUTION PHASES                                │
│                                                                             │
│  Phase 1: Dataflows & Technical Indicators (ATR, Chandelier, Volatility)    │
│  Phase 2: Schemas & Pydantic Contracts (Dynamic Exit Fields & Validators)   │
│  Phase 3: Horizon Evaluator Core Engine (Gap Priority, Ratchet, Trailing)   │
│  Phase 4: Walk-Forward Runner & Risk Alignment                              │
│  Phase 5: Agent Prompts & LLM Structured Outputs                            │
│  Phase 6: Metrics, CLI & Evaluation TUI (MFE Efficiency, Exit Reasons)     │
│  Phase 7: Unit Testing & Full Verification Suite                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Dataflows & Technical Indicators

**Files to modify:**
- `tradingagents/dataflows/stockstats_utils.py`
- `tradingagents/dataflows/structural_levels.py`

#### Tasks
- [ ] **1.1 Add ATR & Chandelier Exit Indicators to `stockstats_utils.py`**:
  - Implement `atr_14`, `atr_20`, and Chandelier Exit bands:
    - $\text{Chandelier Long} = \max(\text{High}_{22}) - 3.0 \times \text{ATR}_{22}$
    - $\text{Chandelier Short} = \min(\text{Low}_{22}) + 3.0 \times \text{ATR}_{22}$
  - Ensure zero lookahead: use causal rolling windows (`rolling(22).max()`) without `.bfill()`.
- [ ] **1.2 Expose Volatility Levels in `structural_levels.py`**:
  - Add volatility metrics (ATR percentage of current price, Chandelier Exit level) to the dictionary returned by `compute_structural_levels` or technical analyst context.

---

### Phase 2: Schemas & Pydantic Data Contracts

**Files to modify:**
- `tradingagents/agents/schemas.py`
- `tradingagents/agents/utils/structured.py`

#### Tasks
- [ ] **2.1 Extend `TraderProposal` in `schemas.py`**:
  - Add optional dynamic exit parameters:
    ```python
    trailing_stop_pct: Optional[float] = Field(
        default=None, ge=0.01, le=0.20,
        description="Trailing stop percentage below peak high once active (e.g. 0.05 for 5%)"
    )
    break_even_trigger_pct: Optional[float] = Field(
        default=0.03, ge=0.01, le=0.15,
        description="Runup percentage required to move SL to break-even entry price"
    )
    max_holding_days: int = Field(
        default=20, ge=1, le=63,
        description="Maximum holding period in trading days before time-based exit"
    )
    ```
- [ ] **2.2 Extend `TradingDecision` and `SignalContract` in `schemas.py`**:
  - Propagate `trailing_stop_pct`, `break_even_trigger_pct`, and `max_holding_days` to `TradingDecision` and `SignalContract`.
  - Add validator ensuring `trailing_stop_pct` is consistent with $SL$ percentage if provided.
- [ ] **2.3 Extend `EvaluationOutcome` Enum in `schemas.py`**:
  - Add explicit exit outcomes:
    - `HIT_TAKE_PROFIT = "HIT_TAKE_PROFIT"`
    - `HIT_STOP_LOSS = "HIT_STOP_LOSS"`
    - `HIT_TRAILING_STOP = "HIT_TRAILING_STOP"`
    - `HIT_BREAK_EVEN = "HIT_BREAK_EVEN"`
    - `HIT_TIME_STOP = "HIT_TIME_STOP"`
    - `EXPIRED = "EXPIRED"`
    - `NO_ORDER = "NO_ORDER"`

---

### Phase 3: Horizon Evaluator Core Engine Hardening

**Files to modify:**
- `tradingagents/backtesting/horizon_evaluator.py`
- `tradingagents/backtesting/position.py`

#### Tasks
- [ ] **3.1 Fix Phantom Fill in `ASSUMED_AI_ENTRY` (`horizon_evaluator.py:202-215`)**:
  - For limit orders, check if $T+1$ bar range touches `actual_entry_price`:
    ```python
    if actual_entry_price is not None:
        if b_low <= actual_entry_price <= b_high:
            entry_price = actual_entry_price
        elif b_open < actual_entry_price:  # Favorable gap-down open for buy
            entry_price = b_open
        else:
            # Price ran away without touching limit order -> NO FILL
            return HorizonEvaluationResult(
                outcome=EvaluationOutcome.NO_ORDER,
                realized_return=0.0,
                ...
            )
    ```
- [ ] **3.2 Fix Gap-Open Priority & Same-Bar Conflict (`horizon_evaluator.py:339-375`)**:
  - Step 1: Check if Bar $t$ open creates a gap exit:
    - Long: If `b_open >= take_profit` $\to$ Exit `HIT_TAKE_PROFIT` at `b_open`.
    - Long: If `b_open <= current_sl` $\to$ Exit `HIT_STOP_LOSS` at `b_open`.
  - Step 2: If both `b_high >= take_profit` and `b_low <= current_sl` occur in same bar:
    - If bar opened closer to TP than SL, or apply conservative execution (`HIT_STOP_LOSS`).
- [ ] **3.3 Implement Dynamic Break-Even Ratchet & Trailing Stop (`horizon_evaluator.py:320-370`)**:
  - Maintain rolling `peak_high` and `current_sl = stop_loss`.
  - On each bar $t$:
    ```python
    # Update peak MFE
    current_runup_pct = (b_high - entry_price) / entry_price
    peak_high = max(peak_high, b_high)
    peak_mfe_pct = max(peak_mfe_pct, current_runup_pct)

    # Break-Even Ratchet
    if break_even_trigger_pct and peak_mfe_pct >= break_even_trigger_pct:
        be_price = entry_price * 1.001  # Cover slip/commission
        if be_price > current_sl:
            current_sl = be_price
            is_break_even_active = True

    # Trailing Stop Ratchet
    if trailing_stop_pct and peak_mfe_pct >= (trailing_stop_pct * 0.8):
        trail_price = peak_high * (1.0 - trailing_stop_pct)
        if trail_price > current_sl:
            current_sl = trail_price
            is_trailing_active = True
    ```
- [ ] **3.4 Implement Time-Stop Exit (`horizon_evaluator.py:375-385`)**:
  - If `bar_idx >= max_holding_days` and position is still open:
    - Exit at `b_close`, outcome `HIT_TIME_STOP`.

---

### Phase 4: Walk-Forward Runner & Risk Alignment

**Files to modify:**
- `tradingagents/backtesting/walk_forward_runner.py`
- `tradingagents/backtesting/risk.py`
- `tradingagents/backtesting/position.py`

#### Tasks
- [ ] **4.1 Synchronize Dynamic Stop Logic in `risk.py`**:
  - Ensure `RiskManager.evaluate_position_risk` supports the same `break_even_trigger_pct` and `trailing_stop_pct` ratchets as `HorizonEvaluator`.
- [ ] **4.2 Wire `RiskConfig` into Runner**:
  - Ensure `RiskConfig` fields (`trailing_stop_pct`, `break_even_trigger_pct`) are passed cleanly between decision states, orders, and evaluation bundles.

---

### Phase 5: Agent Prompts & LLM Structured Outputs

**Files to modify:**
- `tradingagents/agents/trader/trader.py`
- `tradingagents/agents/managers/portfolio_manager.py`
- `tradingagents/agents/analysts/technical_analyst.py`

#### Tasks
- [ ] **5.1 Update Trader & Portfolio Manager Prompts**:
  - Add explicit instructions for:
    1. Realistic Take Profit targets ($1.5R$ to $3.0R$) grounded in ATR / structural resistance.
    2. Dynamic exit parameters: Recommended `trailing_stop_pct` (e.g. $3\text{--}5\%$) and `break_even_trigger_pct` (e.g. $+3\%$).
    3. Maximum holding duration (`max_holding_days`: 10–25 days for swing trades).
- [ ] **5.2 Ground Technical Analyst with Chandelier & ATR Context**:
  - Inject ATR and Chandelier values into prompt templates to anchor stop loss suggestions.

---

### Phase 6: Metrics, CLI & Evaluation TUI

**Files to modify:**
- `tradingagents/backtesting/metrics.py`
- `cli/commands/evaluate_results.py`
- `cli/commands/evaluate_tui.py`
- `cli/commands/evaluate.py`

#### Tasks
- [ ] **6.1 Implement MFE Realization Efficiency Metric (`metrics.py`)**:
  - Formula: $\text{MFE Efficiency} = \frac{\text{Realized Return}}{\max(\text{MFE}, 0.0001)}$.
  - Penalizes trades that gave back large unrealized profits.
- [ ] **6.2 Update TUI & CLI Results Table (`evaluate_results.py`, `evaluate_tui.py`)**:
  - Render new outcome types (`HIT_TRAILING_STOP`, `HIT_BREAK_EVEN`, `HIT_TIME_STOP`).
  - Add `MFE Efficiency` column or metric in summary cards.

---

### Phase 7: Comprehensive Test Suite & Verification

**Files to create / modify:**
- `tests/test_dynamic_exits.py` (New test file)
- `tests/test_horizon_evaluator.py`
- `tests/test_risk.py`

#### Tasks
- [ ] **7.1 Test Break-Even Ratchet**:
  - Test case where stock rises $+5\%$ (trigger at $+3\%$), then drops below entry. Verifies exit at break-even ($+0.1\%$) instead of stop loss.
- [ ] **7.2 Test Trailing Stop Ratchet (AVGO Scenario)**:
  - Test case simulating AVGO (+9.27% runup, then drop). Verifies exit with $+4.27\%$ gain via $5\%$ trailing stop.
- [ ] **7.3 Test Gap-Open Priority**:
  - Test case where bar opens above TP ($Open = 115 > TP = 110$) and low touches SL ($Low = 94 < SL = 95$). Verifies `HIT_TAKE_PROFIT` at $115$.
- [ ] **7.4 Test Limit Fill Touch Validation**:
  - Test case where limit buy is $100$, but bar range is $102\text{--}105$. Verifies `NO_ORDER` / no fill.
- [ ] **7.5 Test Time Stop**:
  - Test case where price stays within $SL$ and $TP$ for $20$ days (`max_holding_days=15`). Verifies `HIT_TIME_STOP` on Day 15.
- [ ] **7.6 Run Full Regression Suite**:
  - Run all 738+ tests and ensure $100\%$ pass rate with 0 regressions.

---

## 4. Acceptance Criteria

| Criteria | Target | Verification Method |
| :--- | :--- | :--- |
| **Full Pytest Suite** | 100% Pass (0 failures, 0 errors) | `pytest` |
| **AVGO Profit Decay Fix** | Exit at profit or break-even | Unit test in `tests/test_dynamic_exits.py` |
| **Gap-Open Resolution** | TP gap open fills at Open price | Unit test in `tests/test_horizon_evaluator.py` |
| **Limit Fill Touch** | No phantom fills outside range | Unit test in `tests/test_horizon_evaluator.py` |
| **Pydantic Validation** | Dynamic exit fields parsed cleanly | Unit test in `tests/test_schemas.py` |
| **CLI / TUI Table** | Outcomes render correctly | `python -m cli.main evaluate-results` |

---

## 5. Implementation Order for the Next AI Agent

1. **Step 1**: Implement Phase 1 (`stockstats_utils.py`, `structural_levels.py`).
2. **Step 2**: Implement Phase 2 (`schemas.py`, `structured.py`).
3. **Step 3**: Implement Phase 3 (`horizon_evaluator.py`, `position.py`).
4. **Step 4**: Implement Phase 4 (`walk_forward_runner.py`, `risk.py`).
5. **Step 5**: Implement Phase 5 (`trader.py`, `portfolio_manager.py`).
6. **Step 6**: Implement Phase 6 (`metrics.py`, `evaluate_results.py`, `evaluate_tui.py`).
7. **Step 7**: Write tests in Phase 7 (`tests/test_dynamic_exits.py`, `tests/test_horizon_evaluator.py`) and run `pytest`.
