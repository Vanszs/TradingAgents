# Interactive TUI Reference

The interactive Terminal User Interface provides real-time visibility into the multi-agent decision process.

---

## 1. Launching Interactive Mode

```bash
tradingagents
# or
python -m cli.main
```

---

## 2. Menu Navigation

- **Live Market Analysis**: Runs real-time or historical date multi-agent analysis.
- **Single-Shot Evaluator**: Evaluates forward $T+1 \dots T+N$ performance of a single ticker on a specific date.
- **Walk-Forward Backtest**: Runs continuous multi-month portfolio simulation.
- **Exit**: Closes the application.

---

## 3. Real-Time 3-Column Display

During execution, the terminal renders:
1. **Header Panel**: Ticker, date, asset class, selected LLM provider, and elapsed timer.
2. **Progress & Messages Stream**: Live thoughts, tool calls, and debate arguments.
3. **Analysis Panel**: Real-time markdown rendering of analyst reports, debate plans, and portfolio decisions.
4. **Stats Footer**: Token count, wall-clock time per agent, and estimated costs.
