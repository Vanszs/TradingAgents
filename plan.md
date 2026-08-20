# Master Implementation Plan: Quantitative Risk, Horizon Evaluation & Metrics Attribution Hardening

This document is an exhaustive, self-contained implementation plan designed for an autonomous AI software engineer. It provides exact file paths, line numbers, failure mechanics, concrete code replacements, and verification commands to resolve all mathematical, attribution, and risk calculation flaws identified in the quantitative audit (`MATH-01` through `MATH-06`).

---

## Table of Contents
1. [Overview & Execution Protocol](#overview--execution-protocol)
2. [Task 1: Clamp MFE/MAE Excursion on Barrier Exits (`horizon_evaluator.py`)](#task-1-clamp-mfemae-excursion-on-barrier-exits-horizon_evaluatorpy)
3. [Task 2: Fix Downside Semi-Deviation for Sortino Ratio (`metrics.py`)](#task-2-fix-downside-semi-deviation-for-sortino-ratio-metricspy)
4. [Task 3: Prevent Trailing Stop Overwrite on Hold / None (`risk.py`)](#task-3-prevent-trailing-stop-overwrite-on-hold--none-riskpy)
5. [Task 4: Fix FIFO Lot Residual Handling on Reversals (`metrics.py`)](#task-4-fix-fifo-lot-residual-handling-on-reversals-metricspy)
6. [Task 5: Correct Limit Price Touch Boundaries (`trigger_evaluator.py`)](#task-5-correct-limit-price-touch-boundaries-trigger_evaluatorpy)
7. [Task 6: Handle Liquidated Account CAGR Total Loss (`metrics.py`)](#task-6-handle-liquidated-account-cagr-total-loss-metricspy)
8. [Final Validation Protocol](#final-validation-protocol)

---

## Overview & Execution Protocol

### Instructions for Implementing Agent:
- Work strictly task-by-task.
- Apply exact file edits using targeted replacements (`Edit` tool).
- Run the provided **Verification Command** after each task before proceeding to the next.
- Do not introduce breaking schema changes.

---

## Task 1: Clamp MFE/MAE Excursion on Barrier Exits (`horizon_evaluator.py`)

### Target File:
`tradingagents/backtesting/horizon_evaluator.py` (lines ~385–475)

### Problem Description (`MATH-01`):
In `HorizonEvaluator.evaluate`, `mfe_bar` and `mae_bar` are calculated from full daily bar extremes (`b_high`, `b_low`) and appended to `max_mfe` / `max_mae` **before** checking if an intraday barrier (`stop_loss` or `take_profit`) was hit. When a trade hits its stop loss at -2.80%, any subsequent drop on that same day (e.g. to -6.25%) is recorded as `max_adverse_excursion_pct = -6.25%`, falsely attributing post-exit market moves to a trade that was already closed.

### Exact Code Replacement:
In `tradingagents/backtesting/horizon_evaluator.py`:

```python
# OLD (around line 387-475):
                mfe_bar = (b_high - entry_price) / entry_price
                mae_bar = (b_low - entry_price) / entry_price
                close_ret = (b_close - entry_price) / entry_price
            else:
                ...
                mfe_bar = (entry_price - b_low) / entry_price
                mae_bar = (entry_price - b_high) / entry_price
                close_ret = (entry_price - b_close) / entry_price

            max_mfe = max(max_mfe, mfe_bar)
            max_mae = min(max_mae, mae_bar)

            trajectory.append(
                DailyExcursionBar(
                    bar_index=idx + 1,
                    date=bar_date,
                    open=b_open,
                    high=b_high,
                    low=b_low,
                    close=b_close,
                    unrealized_return_close_pct=round(close_ret * 100.0, 2),
                    unrealized_mfe_pct=round(mfe_bar * 100.0, 2),
                    unrealized_mae_pct=round(mae_bar * 100.0, 2),
                )
            )

            # Barrier evaluation: check intraday barrier touches
            if is_long:
                if stop_loss is not None and b_low <= stop_loss:
                    outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = stop_loss
                    actual_days = idx + 1
                    break
                if take_profit is not None and b_high >= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = take_profit
                    actual_days = idx + 1
                    break
            else:  # SHORT
                if stop_loss is not None and b_high >= stop_loss:
                    outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = stop_loss
                    actual_days = idx + 1
                    break
                if take_profit is not None and b_low <= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = take_profit
                    actual_days = idx + 1
                    break

# NEW:
            # Intraday barrier evaluation with post-exit clamping
            if is_long:
                # 1. Stop loss barrier hit
                if stop_loss is not None and b_low <= stop_loss:
                    outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = b_open if b_open <= stop_loss else stop_loss
                    actual_days = idx + 1
                    
                    # Clamp MAE strictly to exit execution price
                    exit_mae = (exit_price - entry_price) / entry_price
                    max_mae = min(max_mae, exit_mae)
                    # MFE on stop bar is bounded by open or 0.0 (no post-exit rally credit)
                    exit_mfe = max(0.0, (b_open - entry_price) / entry_price)
                    max_mfe = max(max_mfe, exit_mfe)
                    
                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mae * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 2. Take profit barrier hit
                if take_profit is not None and b_high >= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = b_open if b_open >= take_profit else take_profit
                    actual_days = idx + 1
                    
                    # Clamp MFE strictly to take profit execution price
                    exit_mfe = (exit_price - entry_price) / entry_price
                    max_mfe = max(max_mfe, exit_mfe)
                    exit_mae = min(0.0, (b_open - entry_price) / entry_price)
                    max_mae = min(max_mae, exit_mae)
                    
                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mfe * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # Normal un-triggered Long bar
                mfe_bar = (b_high - entry_price) / entry_price
                mae_bar = (b_low - entry_price) / entry_price
                close_ret = (b_close - entry_price) / entry_price
                max_mfe = max(max_mfe, mfe_bar)
                max_mae = min(max_mae, mae_bar)

            else:  # SHORT
                # 1. Stop loss barrier hit (Short)
                if stop_loss is not None and b_high >= stop_loss:
                    outcome = EvaluationOutcome.HIT_STOP_LOSS
                    exit_date = bar_date
                    exit_price = b_open if b_open >= stop_loss else stop_loss
                    actual_days = idx + 1
                    
                    exit_mae = (entry_price - exit_price) / entry_price
                    max_mae = min(max_mae, exit_mae)
                    exit_mfe = max(0.0, (entry_price - b_open) / entry_price)
                    max_mfe = max(max_mfe, exit_mfe)
                    
                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mae * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # 2. Take profit barrier hit (Short)
                if take_profit is not None and b_low <= take_profit:
                    outcome = EvaluationOutcome.HIT_TAKE_PROFIT
                    exit_date = bar_date
                    exit_price = b_open if b_open <= take_profit else take_profit
                    actual_days = idx + 1
                    
                    exit_mfe = (entry_price - exit_price) / entry_price
                    max_mfe = max(max_mfe, exit_mfe)
                    exit_mae = min(0.0, (entry_price - b_open) / entry_price)
                    max_mae = min(max_mae, exit_mae)
                    
                    trajectory.append(
                        DailyExcursionBar(
                            bar_index=idx + 1,
                            date=bar_date,
                            open=b_open,
                            high=b_high,
                            low=b_low,
                            close=exit_price,
                            unrealized_return_close_pct=round(exit_mfe * 100.0, 2),
                            unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                            unrealized_mae_pct=round(max_mae * 100.0, 2),
                        )
                    )
                    break

                # Normal un-triggered Short bar
                mfe_bar = (entry_price - b_low) / entry_price
                mae_bar = (entry_price - b_high) / entry_price
                close_ret = (entry_price - b_close) / entry_price
                max_mfe = max(max_mfe, mfe_bar)
                max_mae = min(max_mae, mae_bar)

            trajectory.append(
                DailyExcursionBar(
                    bar_index=idx + 1,
                    date=bar_date,
                    open=b_open,
                    high=b_high,
                    low=b_low,
                    close=b_close,
                    unrealized_return_close_pct=round(close_ret * 100.0, 2),
                    unrealized_mfe_pct=round(max_mfe * 100.0, 2),
                    unrealized_mae_pct=round(max_mae * 100.0, 2),
                )
            )
```

### Verification Command:
```bash
python3 -c "
import pandas as pd
from tradingagents.backtesting.horizon_evaluator import HorizonEvaluator, EvaluationOutcome

df = pd.DataFrame({
    'date': ['2025-04-15', '2025-04-16'],
    'open': [107.0, 104.40],
    'high': [108.0, 106.64],
    'low': [106.5, 100.31],
    'close': [107.0, 104.34],
    'volume': [1000, 1000]
})

res = HorizonEvaluator().evaluate(
    ticker='NVDA',
    signal_date='2025-04-15',
    side='LONG',
    planned_entry_price=107.0,
    actual_entry_price=107.0,
    actual_entry_date='2025-04-15',
    stop_loss=104.0,
    take_profit=122.05,
    time_horizon_days=20,
    ohlcv_df=df
)

assert res.outcome == EvaluationOutcome.HIT_STOP_LOSS
assert res.realized_return_pct == -2.80, f'Expected -2.80, got {res.realized_return_pct}'
assert res.max_adverse_excursion_pct == -2.80, f'Expected clamped MAE -2.80, got {res.max_adverse_excursion_pct}'
print('Task 1 verified.')
"
```

---

## Task 2: Fix Downside Semi-Deviation for Sortino Ratio (`metrics.py`)

### Target File:
`tradingagents/backtesting/metrics.py` (lines ~94–99)

### Problem Description (`MATH-02`):
Sortino ratio semi-deviation was computed as `downside = returns[returns < 0]; downside_std = float(downside.std(ddof=0))`. This calculates the sample standard deviation around the *mean of losing days* over $K$ losing days, instead of taking the root-mean-square of negative returns relative to target return ($MAR = 0$) across all $N$ periods:
$$\sigma_D = \sqrt{\frac{1}{N} \sum_{t=1}^N \min(0, r_t)^2}$$
On strategies with 1 losing day or identical loss percentages, `downside_std` was 0.0, returning `sortino = 0.0`.

### Exact Code Replacement:
In `tradingagents/backtesting/metrics.py`:

```python
# OLD:
        downside = returns[returns < 0].astype(float)
        downside_std = float(downside.std(ddof=0))
        sortino = 0.0
        if downside_std > 0:
            sortino = (daily_mean / downside_std) * math.sqrt(252)

# NEW:
        # True Downside Semi-Deviation (root-mean-square of negative returns relative to 0 across ALL N days)
        downside_diff = returns.clip(upper=0.0).astype(float)
        downside_dev = math.sqrt(float((downside_diff ** 2).mean()))
        sortino = 0.0
        if downside_dev > 0:
            sortino = (daily_mean / downside_dev) * math.sqrt(252)
        elif daily_mean > 0:
            sortino = float("inf")
```

### Verification Command:
```bash
python3 -c "
import pandas as pd
from tradingagents.backtesting.metrics import MetricsCalculator

# Strategy with 9 winning days (+1%) and 1 losing day (-1%)
# Mean = 0.8%, Downside Dev = sqrt((-0.01^2)/10) = 0.003162
df = pd.DataFrame({
    'date': [f'2026-01-{i:02d}' for i in range(1, 11)],
    'total_equity': [100000 * (1.01 ** i) for i in range(9)] + [100000 * (1.01 ** 8) * 0.99],
    'drawdown': [0.0]*10,
    'position_qty': [1]*10
})
res = MetricsCalculator().calculate(df, [], 100000.0)
assert res['sortino_ratio'] > 0.0, f'Expected positive sortino, got {res[\"sortino_ratio\"]}'
print('Task 2 verified.')
"
```

---

## Task 3: Prevent Trailing Stop Overwrite on Hold / None (`risk.py`)

### Target File:
`tradingagents/backtesting/risk.py` (lines ~482–505)

### Problem Description (`MATH-03`):
In `update_position_risk_levels`, when a position has moved its stop up to lock in profit (e.g. entry 100, price 150, trailing stop 130), if a subsequent day's agent output is `Hold` or omits explicit stop levels (`agent_stop is None`), execution falls into `elif entry_price > 0:`, re-calculating `position.stop_price = entry_price * (1 - default_stop_pct)` (95) and destroying the active trailing stop.

### Exact Code Replacement:
In `tradingagents/backtesting/risk.py`:

```python
# OLD:
    if agent_stop is not None:
        if position.is_long():
            position.stop_price = max(position.stop_price or 0.0, agent_stop)
        elif position.is_short():
            position.stop_price = min(position.stop_price or float("inf"), agent_stop)
        else:
            position.stop_price = agent_stop
    elif entry_price > 0:
        if cached_atr is not None:
            mult = getattr(risk_config, "atr_stop_multiplier", 2.0)
            if position.is_long():
                position.stop_price = entry_price - (cached_atr * mult)
            elif position.is_short():
                position.stop_price = entry_price + (cached_atr * mult)
        else:
            stop_pct = getattr(risk_config, "default_stop_loss_pct", 0.05)
            if position.is_long():
                position.stop_price = entry_price * (1 - stop_pct)
            elif position.is_short():
                position.stop_price = entry_price * (1 + stop_pct)

# NEW:
    if agent_stop is not None:
        if position.is_long():
            position.stop_price = max(position.stop_price or 0.0, agent_stop)
        elif position.is_short():
            position.stop_price = min(position.stop_price or float("inf"), agent_stop)
        else:
            position.stop_price = agent_stop
    elif entry_price > 0 and position.stop_price is None:
        # ONLY initialize default stop if position does NOT already have an active trailing stop
        if cached_atr is not None:
            mult = getattr(risk_config, "atr_stop_multiplier", 2.0)
            if position.is_long():
                position.stop_price = entry_price - (cached_atr * mult)
            elif position.is_short():
                position.stop_price = entry_price + (cached_atr * mult)
        else:
            stop_pct = getattr(risk_config, "default_stop_loss_pct", 0.05)
            if position.is_long():
                position.stop_price = entry_price * (1 - stop_pct)
            elif position.is_short():
                position.stop_price = entry_price * (1 + stop_pct)
```

### Verification Command:
```bash
python3 -c "
import pandas as pd
from tradingagents.backtesting.risk import update_position_risk_levels
from tradingagents.backtesting.position import Position

# Position entered at 100, trailing stop moved up to 130
pos = Position(ticker='AAPL', side='LONG', quantity=10, avg_entry_price=100.0, stop_price=130.0)
class MockDec:
    stop_price = None # Agent outputs Hold with no new stop
    take_profit = None

df = pd.DataFrame({'date': ['2026-01-01'], 'close': [140.0]})
update_position_risk_levels(pos, MockDec(), df, current_date='2026-01-01')
assert pos.stop_price == 130.0, f'Expected trailing stop 130.0 preserved, got {pos.stop_price}'
print('Task 3 verified.')
"
```

---

## Task 4: Fix FIFO Lot Residual Handling on Reversals (`metrics.py`)

### Target File:
`tradingagents/backtesting/metrics.py` (lines ~240–255)

### Problem Description (`MATH-04`):
In `MetricsCalculator._trade_stats`, when a closing order quantity exceeds existing open lots (e.g. Short 10 shares, Buy order 25 shares $\rightarrow$ net Long 15 shares), the while loop closes the 10 short shares and exhausts `short_lots`. However, the residual 15 shares are discarded and never appended to `long_lots`. Subsequent sell orders find `long_lots` empty and open phantom short positions.

### Exact Code Replacement:
In `tradingagents/backtesting/metrics.py`:

```python
# OLD:
                while qty_to_close > 0 and lots:
                    lot_qty, lot_price, lot_mult, entry_date = lots.pop(0)
                    matched = min(qty_to_close, lot_qty)
                    ...
                    remaining = lot_qty - matched
                    if remaining > 0:
                        lots.insert(0, (remaining, lot_price, lot_mult, entry_date))
                    qty_to_close -= matched

# NEW:
                while qty_to_close > 0 and lots:
                    lot_qty, lot_price, lot_mult, entry_date = lots.pop(0)
                    matched = min(qty_to_close, lot_qty)
                    if is_long:
                        pnl = (close_price - lot_price) * matched * lot_mult
                        realized_pnls.append(pnl)
                        long_pnl_total += pnl
                        long_total += 1
                        if pnl > 0:
                            long_wins += 1
                    else:
                        pnl = (lot_price - close_price) * matched * lot_mult
                        realized_pnls.append(pnl)
                        short_pnl_total += pnl
                        short_total += 1
                        if pnl > 0:
                            short_wins += 1

                    if trade.date and entry_date:
                        try:
                            from datetime import datetime
                            d_open = datetime.fromisoformat(str(entry_date)[:10])
                            d_close = datetime.fromisoformat(str(trade.date)[:10])
                            days = max(0, (d_close - d_open).days)
                            holding_periods.append(days)
                            if is_long:
                                long_holding_periods.append(days)
                            else:
                                short_holding_periods.append(days)
                        except Exception:
                            pass

                    remaining = lot_qty - matched
                    if remaining > 0:
                        lots.insert(0, (remaining, lot_price, lot_mult, entry_date))
                    qty_to_close -= matched

                # If closing order exceeded existing lots (position reversal), add residual to opposing inventory
                if qty_to_close > 0:
                    if is_long:  # Excess SELL becomes a short open lot
                        short_lots.append((qty_to_close, close_price, trade.multiplier, trade.date))
                    else:        # Excess BUY becomes a long open lot
                        long_lots.append((qty_to_close, close_price, trade.multiplier, trade.date))
```

### Verification Command:
```bash
python3 -c "
from tradingagents.backtesting.decision_schema import Trade, OrderSide, OpenClose
from tradingagents.backtesting.metrics import MetricsCalculator

# Trade 1: Open Short 10 @ 100
t1 = Trade('2026-01-01', 'T', OrderSide.SELL, 10, 100.0, 1000.0, 1.0, 1000.0, 0.0, 1000.0, 0.0, 1, 0.0, 0.0, OpenClose.OPEN, '', '1', '1')
# Trade 2: Reversal BUY 25 @ 90 (Closes 10 short @ +100 profit, opens 15 long @ 90)
t2 = Trade('2026-01-05', 'T', OrderSide.BUY, 25, 90.0, 2250.0, 1.0, 2250.0, 0.0, 2250.0, 0.0, 1, 100.0, 0.0, OpenClose.CLOSE, '', '2', '2')
# Trade 3: Close remaining 15 Long @ 110 (+300 profit)
t3 = Trade('2026-01-10', 'T', OrderSide.SELL, 15, 110.0, 1650.0, 1.0, 1650.0, 0.0, 1650.0, 0.0, 1, 300.0, 0.0, OpenClose.CLOSE, '', '3', '3')

stats = MetricsCalculator()._trade_stats([t1, t2, t3])
assert stats['total_realized_pnl'] == 400.0, f'Expected 400.0 realized PnL, got {stats[\"total_realized_pnl\"]}'
print('Task 4 verified.')
"
```

---

## Task 5: Correct Limit Price Touch Boundaries (`trigger_evaluator.py`)

### Target File:
`tradingagents/backtesting/trigger_evaluator.py` (lines ~220–240)

### Problem Description (`MATH-05`):
`_price_level_touched` checked `low <= planned <= high`. For a Buy Limit order, if the market gaps down or trades completely below the planned price (`high < planned`), the order is fully executable at or better than limit price, but `planned <= high` evaluated to `False`.

### Exact Code Replacement:
In `tradingagents/backtesting/trigger_evaluator.py`:

```python
# OLD:
    def _price_level_touched(
        self,
        decision: ExtendedDecision,
        current_bar: Optional[MarketPoint] = None,
    ) -> bool:
        if decision.planned_entry_price is None or current_bar is None:
            return False
        return float(current_bar.low) <= float(decision.planned_entry_price) <= float(current_bar.high)

# NEW:
    def _price_level_touched(
        self,
        decision: ExtendedDecision,
        current_bar: Optional[Union[MarketPoint, dict]] = None,
    ) -> bool:
        if decision.planned_entry_price is None or current_bar is None:
            return False
        
        low = current_bar.get("low") if isinstance(current_bar, dict) else getattr(current_bar, "low", None)
        high = current_bar.get("high") if isinstance(current_bar, dict) else getattr(current_bar, "high", None)
        if low is None or high is None:
            return False

        planned = float(decision.planned_entry_price)
        action = getattr(decision, "futures_action", "BUY_TO_OPEN") or "BUY_TO_OPEN"
        is_buy = "BUY" in str(action).upper() or getattr(decision, "target_position_side", "LONG") == "LONG"
        
        if is_buy:
            # Buy limit executes if market traded down to or through the limit price
            return float(low) <= planned
        else:
            # Sell limit executes if market traded up to or through the limit price
            return float(high) >= planned
```

### Verification Command:
```bash
python3 -c "
from tradingagents.backtesting.trigger_evaluator import TriggerEvaluator
from tradingagents.backtesting.decision_schema import MarketPoint
te = TriggerEvaluator()
class MockDec:
    planned_entry_price = 100.0
    futures_action = 'BUY_TO_OPEN'
    target_position_side = 'LONG'

# Market gapped down: Low 90, High 98 (entirely below limit price 100)
mp = MarketPoint(date='2026-01-01', open=95.0, high=98.0, low=90.0, close=94.0, volume=1000)
assert te._price_level_touched(MockDec(), mp) == True, 'Expected True on gap-down through limit'
print('Task 5 verified.')
"
```

---

## Task 6: Handle Liquidated Account CAGR Total Loss (`metrics.py`)

### Target File:
`tradingagents/backtesting/metrics.py` (lines ~84–88)

### Problem Description (`MATH-06`):
When an account was liquidated (`final_equity <= 0`), `cagr` evaluated to `0.0` (breakeven) because `final_equity > 0` was required. Total loss must report `cagr_pct = -100.0%`.

### Exact Code Replacement:
In `tradingagents/backtesting/metrics.py`:

```python
# OLD:
        cagr = 0.0
        if years > 0 and initial_cash > 0 and final_equity > 0:
            cagr = (final_equity / initial_cash) ** (1 / years) - 1

# NEW:
        if final_equity <= 0:
            cagr = -1.0
        elif years > 0 and initial_cash > 0:
            cagr = (final_equity / initial_cash) ** (1 / years) - 1
        else:
            cagr = 0.0
```

### Verification Command:
```bash
python3 -c "
import pandas as pd
from tradingagents.backtesting.metrics import MetricsCalculator

# Liquidated account (final equity 0.0)
df = pd.DataFrame({'date': ['2025-01-01', '2026-01-01'], 'total_equity': [100000.0, 0.0], 'drawdown': [0.0, -1.0], 'position_qty': [1, 0]})
res = MetricsCalculator().calculate(df, [], 100000.0)
assert res['cagr_pct'] == -100.0, f'Expected -100.0% CAGR on wipeout, got {res[\"cagr_pct\"]}'
print('Task 6 verified.')
"
```

---

## Final Validation Protocol

After applying all 6 tasks, execute the complete test suite:

```bash
pytest tests/ -v
```

**Target Outcome**: 100% pass rate (738/738 tests passing). Verified clamping of MAE/MFE, mathematically sound Sortino semi-deviation, monotonic trailing stop protection, accurate FIFO reversal handling, and reliable limit trigger boundaries.
