# Memory & Reflection System

TradingAgents features an episodic memory reflection system (`TradingMemoryLog` in `tradingagents/agents/utils/memory.py`) that enables agents to learn from past trade outcomes without fine-tuning or model retraining.

---

## The Two-Phase Memory Lifecycle

```
[Phase A: Decision Recording] (at Signal Date T0)
Signal Generated ──> Append Pending Entry: [2026-03-18 | MSFT | BUY | pending]
                     Stores full decision narrative and pricing geometry.

[Phase B: Outcome Resolution & Reflection] (after trade closes)
Trade Closed ───> Calculate Realized Return & Benchmark Alpha vs SPY
             ───> LLM Reflection Analysis: What went right/wrong?
             ───> Atomic Update: [2026-03-18 | MSFT | BUY | -4.9% | -3.2% | 4d]

[Phase C: Point-in-Time Recall] (on future analysis)
New Run ─────────> Query get_past_context(ticker, as_of=current_date)
             ───> Injects: (1) Same-ticker history + (2) Cross-ticker lessons
             ───> Provided exclusively to Portfolio Manager (CIO).
```

---

## Structure of Injected Past Context

```markdown
Past analyses of MSFT (most recent first):
[2026-03-18 | MSFT | BUY | -4.9% | -3.2% | 4d]
DECISION:
Buy at $390.94 during 1D Death Cross.
REFLECTION:
Position stopped out because price was in an active liquidation waterfall.
Lesson: Do not buy falling knives without 1H breakout confirmation.

Recent cross-ticker lessons:
[2026-02-10 | NVDA | BUY | -3.5% | -1.1% | 3d]
REFLECTION:
Avoid setting Take Profit above 52W High during macro market consolidations.
```

---

## Safety & Integrity Features
- **Atomic Writes**: Uses `.tmp` file writing and `os.replace` semantics to eliminate file corruption during abrupt process termination.
- **Strict Point-in-Time Boundary**: Entries resolved after `as_of` date are filtered out, guaranteeing zero future leakage during historical backtests.
- **Log Rotation**: Automatically caps resolved entries to prevent context window bloat while preserving all pending items.
