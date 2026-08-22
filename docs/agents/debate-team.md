# Research & Debate Team

The Research & Debate Team engages in adversarial dialectical reasoning to stress-test raw analyst reports before tactical order formulation.

---

## 1. Bull Researcher (`bull_researcher.py`)
- **Role**: Builds an aggressive, high-expectancy upside thesis under a spot long accumulation mandate.
- **Key Archetypes Covered**:
  1. **Momentum Breakouts**: Expansion above horizontal resistance ranges with volume expansion.
  2. **Oversold Mean-Reversion**: Multi-day support tests / double bottoms with RSI exiting extreme oversold ($<35$), targeting 50 SMA / Fib 50%.
  3. **Pullback Accumulation**: Staging bids at concrete demand floors (Fibonacci retracements, 60D swing floors).

---

## 2. Bear Researcher (`bear_researcher.py`)
- **Role**: Identifies institutional distribution, valuation traps, overhead liquidity supply walls, and macro headwinds.
- **Key Checks**:
  1. **Death Cross / Downtrends**: Flags active markdown phases where price trades below declining 50/200 SMAs.
  2. **CapEx / Margin Drag**: Highlights free-cash-flow compression and negative earnings revisions.
  3. **Support Breakdown**: Points out failed relief bounces and trapped long overhead supply.

---

## 3. Research Manager (`research_manager.py`)
- **Role**: Acts as the senior debate judge. Synthesizes the Bull vs Bear arguments into a structured `ResearchPlan`.
- **Output Schema**:
  - `recommendation`: `PortfolioRating` (`Buy`, `Overweight`, `Hold`, `WNS`, `Underweight`, `Sell`).
  - `rationale`: Balanced summary of the debate winner.
  - `strategic_actions`: High-level strategic directional roadmap.
- **Boundary**: Focuses strictly on strategic directional consensus; leaves exact numerical price geometry (Entry, SL, TP) to the downstream Execution Trader.
