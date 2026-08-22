# Single-Shot Forward Evaluator

The Single-Shot Evaluator (`cli/commands/evaluate.py` & `tradingagents/backtesting/horizon_evaluator.py`) simulates forward trade performance bar-by-bar following a generated signal.

---

## 1. Execution Physics & Fill Invariants

1. **Limit Touch Validation**: Limit buy orders on $T+1$ are only filled if the bar's price range actually touches the limit price ($\text{Low}_{T1} \le P_{\text{limit}} \le \text{High}_{T1}$). Unreached limit orders return `EvaluationOutcome.NO_FILL`.
2. **Gap-Open Precedence**:
   - If market opens above Take Profit ($P_{\text{open}} \ge TP$), position exits at $P_{\text{open}}$ (`HIT_TAKE_PROFIT`).
   - If market opens below Stop Loss ($P_{\text{open}} \le SL$), position exits at $P_{\text{open}}$ (`HIT_STOP_LOSS`).
3. **Conservative Same-Bar Conflict Resolution**: If a daily bar's high touches $TP$ and low touches $SL$ in the same session, the evaluator conservatively triggers **`HIT_STOP_LOSS`** first.
4. **Time-Stop Termination**: If the position reaches `max_holding_days` without touching $TP$ or $SL$, the position is closed at market close with `EvaluationOutcome.HIT_TIME_STOP`.

---

## 2. Quantitative Excursion Tracking

The evaluator records daily trajectory bars:
- **Maximum Favorable Excursion (MFE)**: Highest unrealized percentage gain achieved during the holding period.
- **Maximum Adverse Excursion (MAE)**: Deepest unrealized drawdown experienced before exit.
- **MFE Realization Efficiency**: $\frac{\text{Realized Return}}{\max(\text{MFE}, 0.0001)}$, penalizing trades that gave back substantial unrealized gains.
