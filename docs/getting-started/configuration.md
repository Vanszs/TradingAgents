# Configuration

TradingAgents uses a centralized configuration system driven by environment variables (`.env`).

---

## 1. Environment Variables Template (`.env`)

Create a `.env` file in the project root:

```bash
# --- LLM API Keys ---
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=sk-...

# --- Provider & Model Selection ---
TRADINGAGENTS_LLM_PROVIDER=openai           # openai, anthropic, google, deepseek, ollama, azure, bluesmind
TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.4        # Model for debate and portfolio decisions
TRADINGAGENTS_QUICK_THINK_LLM=gpt-5.4-mini  # Model for fast data parsing & analyst reports

# --- Concurrency & Performance ---
TRADINGAGENTS_ANALYST_CONCURRENCY=4         # 1 = Sequential (rate-limit safe), 4 = Pure Parallel

# --- Debate & Risk Committee Depth ---
TRADINGAGENTS_MAX_DEBATE_ROUNDS=1           # Number of Bull vs Bear debate rounds
TRADINGAGENTS_MAX_RISK_ROUNDS=1             # Number of Risk Management committee rounds

# --- Localization ---
TRADINGAGENTS_OUTPUT_LANGUAGE=English       # English, Indonesian, Japanese, etc.

# --- Optional Cloud & Vendor Endpoints ---
# OLLAMA_BASE_URL=http://localhost:11434/v1
# ALPHA_VANTAGE_API_KEY=...
```

---

## 2. Dynamic Override Precedence

Configuration values are resolved in the following priority order:
1. **Explicit CLI Flags** (`--provider`, `--depth`, `--lang`, `--date`)
2. **Environment Variables** (`TRADINGAGENTS_*` in `.env`)
3. **Default Config Dictionary** (`tradingagents/default_config.py`)
