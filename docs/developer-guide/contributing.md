# Contributing & Developer Guide

## 1. Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/Vanszs/TradingAgents.git
cd TradingAgents

# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"
```

---

## 2. Running Test Suites

TradingAgents enforces 100% test pass rates before any release.

```bash
# Run full test suite (750+ tests)
pytest

# Run fast unit tests only
pytest -m unit

# Run focused quantitative execution tests
pytest tests/test_dynamic_exits.py tests/test_horizon_evaluator.py tests/test_typed_signal_integration.py

# Check code formatting & linting
ruff check .
```

---

## 3. Code Standards & Architecture Guidelines

1. **State Isolation**: When creating or modifying agent nodes, never mutate shared state in-place. Always return a dictionary containing only the keys your node is responsible for.
2. **Deterministic Dataflows**: All dataflow computations in `tradingagents/dataflows/` must be causal with zero lookahead. Never use future bars or unbuffered filing dates.
3. **Pydantic Validation**: All structured outputs must validate against strict Pydantic models in `tradingagents/agents/schemas.py`.
4. **Broker Realism**: Backtesting features must respect real-world market mechanics (gap fills, limit touch verification, static bracket orders).
