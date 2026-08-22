# Implementation Plan: Forward Projections, Mean-Reversion & Breakout Execution Hardening

## 1. Executive Summary & Problem Formulation

Following the multi-agent empirical audit of MSFT (2026-03-31 and 2026-04-15) and the 16-issue root-cause analysis, this plan provides the exact specifications to fix the remaining pipeline gaps that caused **Over-Defensive Analysis Paralysis (excessive WNS on valid setups)**:
1. **Missing Forward Target Projections (`structural_levels.py`)**: The data tool only computed backward-looking retracements and swing highs ($412.70, $417.77), blinding the LLM to forward breakout upside ($440–$460) and artificially collapsing $R:R$ to $< 0.3:1$, forcing false WNS.
2. **Missing Mean-Reversion & Oversold Bounce Mandate (`bull_researcher.py`)**: The Bull prompt only instructed on passive "Limit Accumulation on Dips", preventing the AI from seizing extreme oversold double-bottom reversals (RSI $< 25$, $354 support bounce).
3. **Execution Tactic Selection (`trader.py`)**: The Trader prompt lacked explicit guidance on when to use **Market Buy** (Breakout Reclaims & Momentum) vs **Limit Buy** (Support Retests / Deep Pullbacks).
4. **Tool Context Ingestion & Schema Consistency**: Missing `trade_date` in 3 analyst context calls, indicator alias support (`rsi_14`), and aligning `ResearchPlan` schema docstrings with WNS.

---

## 2. Complete Inventory of Target Files & Action Items

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             MODIFIED COMPONENTS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. `tradingagents/dataflows/structural_levels.py`                           │
│    - Compute Fibonacci Extensions (1.272x, 1.618x) and ATR Target Channels  │
│    - Inject explicit Forward Breakout Targets into prompt summary           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. `tradingagents/dataflows/y_finance.py` & `stockstats_utils.py`           │
│    - Normalize indicator aliases (`rsi_14` -> `rsi`, `atr_14` -> `atr`)     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. `tradingagents/agents/analysts/fundamentals_analyst.py`, etc.            │
│    - Pass `trade_date=current_date` to `build_instrument_context()`         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. `tradingagents/agents/researchers/bull_researcher.py`                    │
│    - Add explicit **Oversold Mean-Reversion & Double-Bottom Reversal** setup│
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. `tradingagents/agents/trader/trader.py`                                  │
│    - Ground Market Buy (Breakouts/Reclaims) vs Limit Buy (Support Retests)  │
│    - Use forward resistance/extensions for Take Profit targets              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6. `tradingagents/agents/schemas.py`                                        │
│    - Include `WNS` in `ResearchPlan.recommendation` field docstring         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase-by-Phase Technical Specifications

---

### Phase 1: Forward Projections & Indicator Dataflow Hardening

#### Target Files
- `tradingagents/dataflows/structural_levels.py`
- `tradingagents/dataflows/y_finance.py`

#### Tasks
- [ ] **1.1 Add Fibonacci Extensions & ATR Target Projections (`structural_levels.py`)**:
  - In `compute_structural_levels()`:
  ```python
  # Fibonacci Extension Targets (1.272x and 1.618x above 60D High for Breakouts)
  fib_ext_1272 = round(h60 + (0.272 * fib_range), 2) if fib_range > 0 else round(last_close * 1.05, 2)
  fib_ext_1618 = round(h60 + (0.618 * fib_range), 2) if fib_range > 0 else round(last_close * 1.10, 2)
  
  # Forward ATR Volatility Target Channels
  atr_val = atr_14_val or (last_close * 0.02)
  atr_target_2x = round(last_close + (2.0 * atr_val), 2)
  atr_target_3x = round(last_close + (3.0 * atr_val), 2)
  ```
  - Expose in return dictionary: `fib_ext_1272`, `fib_ext_1618`, `atr_target_2x`, `atr_target_3x`.
  - In `get_market_structural_summary()`:
  ```python
  summary += (
      f"   - Forward Expansion Targets (Breakout Upside): Fib 1.272x = {levels.get('fib_ext_1272')} | Fib 1.618x = {levels.get('fib_ext_1618')}\n"
      f"   - Expected Volatility Target Channels: +2x ATR = {levels.get('atr_target_2x')} | +3x ATR = {levels.get('atr_target_3x')}\n"
  )
  ```

- [ ] **1.2 Normalize Indicator Aliases in `y_finance.py`**:
  - Map `rsi_14` $\to$ `rsi`, `atr_14` $\to$ `atr`, `close_50_sma` $\to$ `close_50_sma`.

---

### Phase 2: Analyst Context Ingestion & Schema Alignment

#### Target Files
- `tradingagents/agents/analysts/fundamentals_analyst.py`
- `tradingagents/agents/analysts/sentiment_analyst.py`
- `tradingagents/agents/analysts/crypto_fundamentals_analyst.py`
- `tradingagents/agents/schemas.py`

#### Tasks
- [ ] **2.1 Pass `trade_date` in Analyst Context Calls**:
  - In `fundamentals_analyst.py:20`: `build_instrument_context(company_name, asset_type, trade_date=current_date)`
  - In `sentiment_analyst.py:70`: `build_instrument_context(company_name, asset_type=asset_type, trade_date=end_date)`
  - In `crypto_fundamentals_analyst.py:92`: `build_instrument_context(company_name, asset_type="crypto", trade_date=current_date)`

- [ ] **2.2 Update `ResearchPlan` Schema Docstring (`schemas.py:108`)**:
  - Update `recommendation` description to explicitly include `WNS`:
    `"The investment recommendation. Exactly one of Buy / Overweight / Hold / WNS / Underweight / Sell."`

---

### Phase 3: Mean-Reversion Mandate & Breakout Execution Grounding

#### Target Files
- `tradingagents/agents/researchers/bull_researcher.py`
- `tradingagents/agents/trader/trader.py`
- `tradingagents/agents/managers/portfolio_manager.py`

#### Tasks
- [ ] **3.1 Add Mean-Reversion & Double-Bottom Mandate to Bull Researcher (`bull_researcher.py:32-45`)**:
  - Add Setup Archetype #2:
    > "2. **Oversold Mean-Reversion & Double-Bottom Reversals**: When price prints a multi-day support test / double bottom (near 20D/60D swing low) or RSI exits extreme oversold (<35), build an active mean-reversion upside thesis targeting the 20 SMA / 50 SMA / Fib 50% with protective invalidation strictly below the support floor."

- [ ] **3.2 Ground Execution Taxonomy in Trader Prompt (`trader.py:30-65`)**:
  - Instruct on exact order tactic selection:
    1. **Buy Market**: Use for **Momentum Breakouts** (reclaiming 20D high or moving averages) and **Confirmed Oversold Bounces** (immediate entry to avoid missing runaway gaps).
    2. **Buy Limit**: Use for **Orderly Pullbacks** at known support floors.
    3. **Target Selection**: Anchor Take Profit on realistic structural resistance (Fib 50%/61.8%, 60D High, 200 SMA, or Fib Extensions 1.272x / 1.618x for breakouts).
    4. **WNS**: Use when market is choppy, trend is ambiguous, or natural structural R:R < 1.8:1.

---

### Phase 4: Verification & Test Execution

#### Target Files
- `tests/test_dynamic_exits.py`
- `tests/test_horizon_evaluator.py`
- `tests/test_typed_signal_integration.py`

#### Tasks
- [ ] **4.1 Verify All 747+ Unit Tests Pass Cleanly**:
  ```bash
  pytest tests/ -v
  ```
- [ ] **4.2 Verify Structural Levels Output**:
  - Test that `get_market_structural_summary("MSFT", "2026-04-15")` includes Fibonacci Extensions and ATR channels.

---

## 4. Execution Order for the AI Implementer

1. **Step 1**: Update `tradingagents/dataflows/structural_levels.py` with Fibonacci Extensions & ATR Channels.
2. **Step 2**: Update `tradingagents/dataflows/y_finance.py` with indicator alias mappings.
3. **Step 3**: Update analyst context calls in `fundamentals_analyst.py`, `sentiment_analyst.py`, `crypto_fundamentals_analyst.py`.
4. **Step 4**: Update `schemas.py` `ResearchPlan` docstring.
5. **Step 5**: Update `bull_researcher.py` and `trader.py` prompts with mean-reversion and breakout execution rules.
6. **Step 6**: Run `pytest` and verify 100% pass rate.
