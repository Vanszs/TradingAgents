# Installation

## Prerequisites

- **Python**: Version `3.10`, `3.11`, or `3.12`
- **Package Manager**: `pip` or `uv`
- **Git**: Installed and configured

---

## 1. Clone the Repository

```bash
git clone https://github.com/Vanszs/TradingAgents.git
cd TradingAgents
```

---

## 2. Set Up Virtual Environment

```bash
# Using Python venv
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

---

## 3. Install Package in Editable Mode

```bash
pip install -e .
```

This installs the core dependencies (`langchain`, `langgraph`, `pydantic`, `rich`, `typer`, `stockstats`, `yfinance`, etc.) and registers the `tradingagents` CLI command globally in your virtual environment.

---

## 4. Verify Installation

```bash
tradingagents --help
```

You should see the command-line help menu:
```text
Usage: tradingagents [OPTIONS] COMMAND [ARGS]...

  TradingAgents: Institutional Multi-Agent LLM Trading Framework.
```
