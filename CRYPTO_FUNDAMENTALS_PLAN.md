# CRYPTO FUNDAMENTALS PLAN — Integrasi Analisis Fundamental Crypto

> **Status sekarang**: Crypto assets (BTC-USD, ETH-USD, dst) di-handle di codebase TradingAgents, tapi **fundamentals analyst di-skip total** karena tools-nya hanya untuk saham (yfinance financial statements).
>
> **Tujuan**: Ganti behavior "skip fundamentals untuk crypto" dengan **Crypto Fundamentals Analyst** dedicated yang menggunakan CoinGecko, GitHub API, DeFiLlama, dan on-chain sources.

**Codebase target**: `/home/vanszs/Documents/code/tradingagents` (TradingAgents v0.2.5)
**Tanggal**: 2026-05-23

---

## Daftar Isi

1. [Executive Summary](#1-executive-summary)
2. [Test Foundation & Robustness Setup (Phase -1)](#15-test-foundation--robustness-setup-phase--1-wajib-sebelum-mulai)
3. [Architecture Decision](#2-architecture-decision)
4. [Data Source Selection](#3-data-source-selection)
5. [API Key Requirements](#4-api-key-requirements)
6. [New Files to Create](#5-new-files-to-create)
7. [Files to Modify](#6-files-to-modify)
8. [Functions to Modify](#7-functions-to-modify)
9. [New Schema Additions](#8-new-schema-additions)
10. [Prompt Engineering Strategy](#9-prompt-engineering-strategy)
11. [Phased Rollout](#10-phased-rollout)
12. [Crypto-Specific Testing Strategy](#11-testing-strategy)
13. [Backward Compatibility](#12-backward-compatibility)
14. [Risk & Mitigation](#13-risk--mitigation)
15. [Effort Estimation](#14-effort-estimation)
16. [Open Questions / Decisions Needed](#15-open-questions--decisions-needed)

---

## 1. Executive Summary

Replace the "skip fundamentals for crypto" behavior dengan dedicated **Crypto Fundamentals Analyst**.

**Sekarang**: `cli/utils.py:filter_analysts_for_asset_type()` membuang fundamentals analyst untuk crypto, sehingga bull/bear researcher kehilangan data fundamental crucial.

**Fix**: Swap factory + tools dari stock fundamentals → crypto fundamentals saat `asset_type == "crypto"`. Graph topology, state field `fundamentals_report`, dan analyst node key tetap sama.

**Hasil yang diharapkan**:
- Crypto pipeline mendapat insight fundamental: tokenomics, on-chain metrics, dev activity, sentiment
- Stock pipeline tidak terdampak sama sekali
- Bull/bear researchers untuk crypto punya data evidence-based untuk debat

---

## 1.5 Test Foundation & Robustness Setup (Phase -1: WAJIB SEBELUM MULAI)

> **Filosofi**: Sebelum tambah feature baru, **infrastructure test harus robust dulu**. Kalau pipeline crypto fundamentals di-build di atas test foundation yang lemah, bug akan cepat terkubur dan regression tidak ke-catch.

### 1.5.1 Status Existing Test Infrastructure

**Hasil audit per 2026-05-23 (commit `61522e1`)**:

| Aspect | Status | Detail |
|--------|--------|--------|
| Pytest | ✅ Solid | 20 test files, 229 passed + 75 subtests, 16s runtime |
| Conftest fixtures | ✅ Bagus | `_dummy_api_keys` auto-mock semua API keys, `mock_llm_client` |
| Test markers | ✅ Defined | `unit`, `integration`, `smoke` di pyproject.toml |
| Strict markers | ✅ Aktif | `--strict-markers` mencegah typo |
| **CI/CD** | ❌ TIDAK ADA | `.github/workflows/` kosong — tests tidak auto-run di PR |
| **Coverage gate** | ❌ TIDAK ADA | Coverage 39% total, no threshold enforcement |
| **Linter config** | ❌ TIDAK ADA | Tidak ada ruff/black/flake8 |
| **Type checker** | ❌ TIDAK ADA | Tidak ada mypy/pyright |
| **Failing tests** | ⚠️ 2 failing | `test_ollama_base_url.py` ANSI color assertion |
| **E2E smoke test** | ❌ TIDAK ADA | Tidak ada test yang exercise full `propagate()` pipeline |

**Coverage hot path yang lemah**:
- `agents/analysts/*.py`: 17-29%
- `agents/researchers/*.py`: 10%
- `agents/risk_mgmt/*.py`: 11%
- `graph/conditional_logic.py`: 21%
- `graph/setup.py`: 18%
- `llm_clients/azure_client.py`: 0%

### 1.5.2 Phase -1 Deliverables (Sebelum Phase 0 Crypto)

**Estimasi: 2 hari**

#### Deliverable 1 — Fix Failing Tests (~30 menit)

File: `tests/test_ollama_base_url.py`

Test gagal karena assertion `assert 'http://<host>:11434/v1' in output` tidak match dengan output yang punya ANSI color codes (`\x1b[32m✓ Using Ollama at \x1b[0m...`).

**Fix**: Strip ANSI codes sebelum assert:
```python
import re
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')

def strip_ansi(s: str) -> str:
    return ANSI_ESCAPE.sub('', s)

# Before assertion:
output_clean = strip_ansi(captured.out)
assert 'http://<host>:11434/v1' in output_clean
```

#### Deliverable 2 — CI Workflow (~30 menit)

File baru: `.github/workflows/test.yml`

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -e . pytest pytest-cov

      - name: Run tests with coverage
        run: |
          pytest tests/ --cov=tradingagents --cov=cli \
            --cov-report=xml --cov-report=term \
            --cov-fail-under=40 \
            -m "not integration" \
            --strict-markers

      - name: Run integration tests (skip if no API keys)
        run: pytest tests/ -m integration --tb=short || true
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
          fail_ci_if_error: false

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff
      - run: ruff check tradingagents/ cli/ tests/
      - run: ruff format --check tradingagents/ cli/ tests/
```

#### Deliverable 3 — Linter Config (~15 menit)

Tambah ke `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
exclude = ["build", "dist", ".venv", "tradingagents.egg-info"]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "B",   # bugbear
    "UP",  # pyupgrade
    "SIM", # simplify
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
fail_under = 40

[tool.coverage.run]
source = ["tradingagents", "cli"]
omit = [
    "*/tests/*",
    "*/build/*",
    "*/__init__.py",
]
```

#### Deliverable 4 — E2E Smoke Tests (Stock + Crypto) ⭐

**File baru**: `tests/test_e2e_smoke.py`

Test ini **WAJIB** ada — sebagai canary untuk regression detection. Bukan unit test tapi exercise full pipeline dengan mocked LLM dan dataflows.

```python
"""End-to-end smoke tests covering full pipeline for both stock and crypto.

These tests exercise the complete LangGraph pipeline (analyst → researcher →
manager → trader → risk debate → portfolio manager) with mocked LLM responses
and mocked external data sources. They serve as a regression canary: any
break in the graph wiring, state propagation, or output schema will fail here.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared LLM mock — produces deterministic responses for each agent role
# ---------------------------------------------------------------------------

def _make_mock_llm():
    """Mock LLM that returns role-appropriate canned responses."""
    llm = MagicMock()

    def invoke(messages, *args, **kwargs):
        content = ""
        # Inspect message content to detect which agent is calling
        msg_text = str(messages).lower()
        if "market analyst" in msg_text or "technical" in msg_text:
            content = "Market shows bullish momentum, RSI at 55, MACD positive."
        elif "fundamentals" in msg_text or "tokenomics" in msg_text:
            content = "Strong fundamentals: revenue growth 25% YoY (or tokenomics: deflationary supply, active dev)."
        elif "news" in msg_text:
            content = "Positive news: product launch announced; macro supportive."
        elif "sentiment" in msg_text or "social" in msg_text:
            content = "Sentiment positive, social volume increasing."
        elif "bull" in msg_text:
            content = "Bull: Strong fundamentals + positive momentum support entry here."
        elif "bear" in msg_text:
            content = "Bear: Valuation stretched, profit-taking risk near resistance."
        elif "research manager" in msg_text:
            content = "Investment Plan: Buy with 5% sizing, target +10%, stop -5%."
        elif "trader" in msg_text:
            content = "Action: Buy 100 shares at market with stop loss 5%."
        elif "aggressive" in msg_text:
            content = "Aggressive: Increase sizing to 10% — momentum strong."
        elif "conservative" in msg_text:
            content = "Conservative: Reduce sizing to 3% — risk too high."
        elif "neutral" in msg_text:
            content = "Neutral: Stick with 5% — balanced view."
        elif "portfolio manager" in msg_text:
            content = (
                "**Rating**: Buy\n\n"
                "**Executive Summary**: Open 5% position with stop at -5%.\n\n"
                "**Investment Thesis**: Fundamentals + sentiment + momentum aligned."
            )
        else:
            content = "Generic agent response."

        response = MagicMock()
        response.content = content
        response.tool_calls = []
        return response

    llm.invoke = invoke
    llm.bind_tools = lambda tools: llm
    return llm


@pytest.fixture
def mock_llm_factory():
    """Patch create_llm_client to return mock LLMs."""
    with patch("tradingagents.llm_clients.factory.create_llm_client") as mock_create:
        client = MagicMock()
        client.get_llm.return_value = _make_mock_llm()
        mock_create.return_value = client
        yield mock_create


# ---------------------------------------------------------------------------
# Stock data mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_stock_data():
    """Mock yfinance and Alpha Vantage responses for stock data."""
    with patch("tradingagents.dataflows.y_finance.get_YFin_data_online") as mock_yf, \
         patch("tradingagents.dataflows.y_finance.get_fundamentals") as mock_fund, \
         patch("tradingagents.dataflows.yfinance_news.get_news_yfinance") as mock_news:
        mock_yf.return_value = "OHLCV data for ticker (mocked)."
        mock_fund.return_value = "Fundamentals: revenue $100B, EPS $5 (mocked)."
        mock_news.return_value = "News: positive earnings beat (mocked)."
        yield {"yf": mock_yf, "fund": mock_fund, "news": mock_news}


# ---------------------------------------------------------------------------
# Crypto data mocks (placeholder — implement when crypto fundamentals lands)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_crypto_data():
    """Mock CoinGecko, GitHub, DeFiLlama, Fear & Greed responses."""
    with patch("tradingagents.dataflows.y_finance.get_YFin_data_online") as mock_yf, \
         patch("tradingagents.dataflows.yfinance_news.get_news_yfinance") as mock_news:
        mock_yf.return_value = "BTC OHLCV data (mocked)."
        mock_news.return_value = "Crypto news: ETF approval rumor (mocked)."

        # Phase 1+: when crypto fundamentals dataflows exist, mock them here:
        # with patch("tradingagents.dataflows.coingecko.get_tokenomics") as mock_cg, ...

        yield {"yf": mock_yf, "news": mock_news}


# ---------------------------------------------------------------------------
# Stock pipeline E2E
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_e2e_stock_pipeline_returns_decision(mock_llm_factory, mock_stock_data):
    """Full pipeline NVDA → Buy/Hold/Sell decision with all state fields populated."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    ta = TradingAgentsGraph(config=config, debug=False)

    final_state, decision = ta.propagate("NVDA", "2024-05-10", asset_type="stock")

    # Assert decision is in valid set
    assert decision in {"Buy", "Overweight", "Hold", "Underweight", "Sell"}, \
        f"Decision must be a valid rating, got: {decision!r}"

    # Assert all required state fields populated
    required_fields = [
        "company_of_interest", "trade_date",
        "market_report", "sentiment_report", "news_report", "fundamentals_report",
        "investment_debate_state", "investment_plan",
        "trader_investment_plan",
        "risk_debate_state", "final_trade_decision",
    ]
    for field in required_fields:
        assert field in final_state, f"Missing state field: {field}"
        if field.endswith("_report") or field.endswith("_plan") or field == "final_trade_decision":
            assert final_state[field], f"Field '{field}' is empty"

    # Assert ticker preserved
    assert final_state["company_of_interest"] == "NVDA"
    assert final_state["trade_date"] == "2024-05-10"


# ---------------------------------------------------------------------------
# Crypto pipeline E2E
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_e2e_crypto_pipeline_returns_decision(mock_llm_factory, mock_crypto_data):
    """Full pipeline BTC-USD → Buy/Hold/Sell decision with crypto fundamentals included."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    # Phase 1+: pass asset_type="crypto" to constructor
    ta = TradingAgentsGraph(config=config, debug=False, asset_type="crypto")

    final_state, decision = ta.propagate("BTC-USD", "2024-05-10", asset_type="crypto")

    # Assert decision is in valid set
    assert decision in {"Buy", "Overweight", "Hold", "Underweight", "Sell"}

    # Crypto pipeline must produce fundamentals_report (NOT skipped after Phase 0)
    assert final_state["fundamentals_report"], \
        "Crypto pipeline must populate fundamentals_report (no longer skipped)"

    # Verify crypto-specific signals in the report
    fund_lower = final_state["fundamentals_report"].lower()
    crypto_terms = {"tokenomics", "on-chain", "dev activity", "supply", "market cap"}
    assert any(term in fund_lower for term in crypto_terms), \
        f"Crypto fundamentals must reference at least one of {crypto_terms}; got: {fund_lower[:200]}"

    # Asset type preserved through state
    assert final_state.get("asset_type") == "crypto"


# ---------------------------------------------------------------------------
# Regression: stock pipeline UNCHANGED after crypto changes
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_e2e_stock_pipeline_uses_yfinance_fundamentals(mock_llm_factory, mock_stock_data):
    """Verify stock pipeline still uses yfinance financial data tools, not crypto tools."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    ta = TradingAgentsGraph(config=config, debug=False, asset_type="stock")

    # Inspect tool nodes — fundamentals tools must be stock tools
    fundamentals_node = ta.tool_nodes["fundamentals"]
    tool_names = {t.name for t in fundamentals_node.tools_by_name.values()}
    expected_stock_tools = {"get_fundamentals", "get_balance_sheet",
                            "get_cashflow", "get_income_statement"}
    assert tool_names == expected_stock_tools, \
        f"Stock pipeline must use stock fundamentals tools; got: {tool_names}"


@pytest.mark.smoke
def test_e2e_crypto_uses_crypto_fundamentals_tools(mock_llm_factory, mock_crypto_data):
    """Verify crypto pipeline uses crypto fundamentals tools, NOT stock tools."""
    # Phase 1+: enable when crypto fundamentals tools exist
    pytest.importorskip("tradingagents.agents.utils.crypto_fundamental_tools")

    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    ta = TradingAgentsGraph(config=config, debug=False, asset_type="crypto")

    fundamentals_node = ta.tool_nodes["fundamentals"]
    tool_names = {t.name for t in fundamentals_node.tools_by_name.values()}
    expected_crypto_tools = {
        "get_crypto_tokenomics", "get_crypto_dev_activity",
        "get_crypto_network_metrics", "get_crypto_market_sentiment",
    }
    assert tool_names == expected_crypto_tools, \
        f"Crypto pipeline must use crypto tools; got: {tool_names}"


# ---------------------------------------------------------------------------
# Cross-cutting: graph topology unchanged between stock & crypto
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_graph_topology_identical_for_stock_and_crypto(mock_llm_factory):
    """Both stock and crypto produce the same graph topology — only tools/factory differ."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    ta_stock = TradingAgentsGraph(config=DEFAULT_CONFIG.copy(), asset_type="stock")
    ta_crypto = TradingAgentsGraph(config=DEFAULT_CONFIG.copy(), asset_type="crypto")

    # Same node names in graph
    stock_nodes = set(ta_stock.workflow.nodes.keys())
    crypto_nodes = set(ta_crypto.workflow.nodes.keys())
    assert stock_nodes == crypto_nodes, \
        f"Graph topology must be identical. Diff: {stock_nodes ^ crypto_nodes}"
```

**Key properties dari test ini**:
- ✅ Tidak butuh API key (semua di-mock via fixtures)
- ✅ Tidak butuh internet (no real HTTP)
- ✅ Run cepat (< 5 detik per test setelah mock setup)
- ✅ Mengcover full pipeline propagate() — analyst → researcher → manager → trader → risk → PM
- ✅ Test **stock + crypto** secara simetris
- ✅ Regression test: stock fundamentals tools tidak berubah setelah crypto integration
- ✅ Property test: graph topology identical antara stock dan crypto

#### Deliverable 5 — Pre-commit Hook (~10 menit, optional)

File baru: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: pytest-fast
        name: pytest (unit tests only)
        entry: pytest -m unit --tb=short
        language: system
        pass_filenames: false
        stages: [pre-push]
```

Setup:
```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

### 1.5.3 Acceptance Criteria untuk Phase -1

Sebelum mulai Phase 0 (crypto scaffolding), confirm:

- [ ] `pytest tests/` → 0 failures (2 ollama tests fixed)
- [ ] `pytest tests/test_e2e_smoke.py -v` → 5 tests passed (3 selalu pass, 2 di-skip karena crypto belum exist via `importorskip`)
- [ ] `.github/workflows/test.yml` exists dan triggered di PR
- [ ] `ruff check tradingagents/ cli/ tests/` → 0 errors atau di-fix
- [ ] Coverage tetap minimal 39% (baseline) — tidak boleh turun
- [ ] CI run di PR menampilkan: tests passed, coverage uploaded

**Setelah semua check**, baru mulai Phase 0 crypto fundamentals.

### 1.5.4 Test Strategy untuk Setiap Phase Berikutnya

| Phase | Test Type | Wajib | Optional |
|-------|-----------|-------|----------|
| Phase 0 (scaffolding) | Update existing test `test_crypto_asset_mode.py` (filter behavior changed) | ✅ | - |
| Phase 1 (CoinGecko MVP) | Unit test untuk `coingecko.py`, `crypto_id_map.py`, `fear_greed.py`. Update e2e crypto smoke test (`test_e2e_smoke.py`) untuk check report content baru | ✅ | Property test untuk id mapping |
| Phase 2 (GitHub + DeFiLlama) | Unit test untuk `github_activity.py`, `defillama.py`. Tambah dev activity assertion ke e2e smoke | ✅ | Mutation test |
| Phase 3 (On-chain + News) | Unit test untuk `onchain_metrics.py`. Rate limit graceful degradation test | ✅ | - |
| Phase 4 (testing & tuning) | Coverage push to 60%+, prompt regression test | ✅ | Hypothesis property tests |

### 1.5.5 Update untuk Plan Section 11 (Testing Strategy)

Section 11 (Testing Strategy) di plan ini sekarang **specialize** untuk crypto-specific tests. Test foundation umum (CI, linter, e2e smoke) sudah di-cover oleh Phase -1 ini. Section 11 tetap relevan untuk detail crypto-specific test cases.

---

## 2. Architecture Decision

### Pattern 1: Swap (bukan Replace)

Key `"fundamentals"` di `ANALYST_NODE_SPECS` **tetap**. Yang di-swap di runtime hanya:
- Factory function (stock factory vs crypto factory)
- Tool node (4 stock tools vs 4 crypto tools)

**Kenapa**: Topology graph identik, cuma node implementation yang beda. Routing & conditional edges tidak perlu diubah.

### Pattern 2: Single Analyst (bukan Split)

**1 crypto fundamentals analyst** dengan 4 tools (mirror jumlah stock):
- `get_crypto_tokenomics` — supply, market cap, distribution
- `get_crypto_dev_activity` — GitHub commits, contributors
- `get_crypto_network_metrics` — TVL, on-chain, NVT
- `get_crypto_market_sentiment` — Fear & Greed, dominance

**Kenapa bukan split** (e.g., separate tokenomics + on-chain + dev analysts): Akan menambah graph nodes 3-4×, tambah conditional edges, tambah cost LLM. Single analyst dengan multiple tools sudah cukup ekspresif.

### Architectural Gap yang Harus Diatasi

**Masalah**: `TradingAgentsGraph._create_tool_nodes()` jalan di `__init__` tanpa tahu `asset_type`. Sekarang `asset_type` baru di-pass saat `propagate()`.

**Solusi**: Tambah param `asset_type` di `TradingAgentsGraph.__init__()`. CLI sudah detect asset type di `cli/main.py:513` sebelum `TradingAgentsGraph` dibuat — tinggal pass.

```python
# cli/main.py
asset_type = detect_asset_type(selected_ticker)  # line 513 (sudah ada)
graph = TradingAgentsGraph(
    selected_analysts=...,
    config=config,
    asset_type=asset_type.value,  # ← BARU
)
```

---

## 3. Data Source Selection

| Source | Free Tier | Coverage | Latensi | Pilih? |
|--------|-----------|----------|---------|--------|
| **CoinGecko** | 30 calls/min (no key), 500/min (Demo key) | Tokenomics, market data, community stats, dev links — comprehensive | Real-time | ✅ Phase 1 (primary) |
| **CoinMarketCap** | 333 calls/day free | Mirip CoinGecko, lebih ketat | Real-time | ❌ Phase 1 skip (lebih limited) |
| **Messari** | Limited free | Research-grade | Real-time | ❌ Optional Phase 4 |
| **Glassnode** | $29+/mo | On-chain metrics terbaik | Real-time | ❌ Skip — paid only |
| **DeFiLlama** | Unlimited free, no key | TVL, protocol stats | Real-time | ✅ Phase 2 |
| **GitHub API** | 5000/hr (token), 60/hr (no token) | Dev activity, commits | Real-time | ✅ Phase 2 |
| **Etherscan** | 5 calls/sec free (with key) | EVM on-chain (gas, addresses) | Real-time | ✅ Phase 3 |
| **Alternative.me** | Unlimited free | Fear & Greed Index | Daily | ✅ Phase 1 |
| **CryptoPanic** | 5 calls/min (free) | News + sentiment | Real-time | ⚠️ Optional Phase 3 |

**Recommended Stack**:
- **Phase 1 (MVP)**: CoinGecko + Alternative.me (Fear & Greed) — covers ~70% of needed data, no API key required
- **Phase 2**: + GitHub API + DeFiLlama (free tier, no payment)
- **Phase 3**: + Etherscan + CryptoPanic (require key but free)

**Kenapa skip Glassnode**: Paid-only ($29+/mo), CoinGecko + DeFiLlama covers 80% of same data free.

---

## 4. API Key Requirements

| Source | Env Var | Required? | Free Tier | Fallback |
|--------|---------|-----------|-----------|----------|
| **CoinGecko** | `COINGECKO_API_KEY` | Optional | 30/min keyless, 500/min with Demo key | Works without key (degraded rate) |
| **GitHub** | `GITHUB_TOKEN` | Optional | 60/hr without, 5000/hr with token | Reduce frequency or skip dev tool |
| **DeFiLlama** | None | No | Unlimited | N/A |
| **Etherscan** | `ETHERSCAN_API_KEY` | Phase 3 | 5 calls/sec | Skip on-chain section |
| **CryptoPanic** | `CRYPTOPANIC_API_KEY` | Phase 3 | 5 calls/min | Use CoinGecko community data |
| **Alternative.me** | None | No | Unlimited | N/A |

**Tambahan untuk `.env.example`**:
```bash
# Crypto Data Providers (all optional, system degrades gracefully)
COINGECKO_API_KEY=
GITHUB_TOKEN=
ETHERSCAN_API_KEY=
CRYPTOPANIC_API_KEY=
```

---

## 5. New Files to Create

### Dataflow Layer (`tradingagents/dataflows/`)

1. **`coingecko.py`** — CoinGecko API client
   - `get_tokenomics(coin_id)` → market cap, supplies, inflation
   - `get_market_data(coin_id)` → price, volume, dominance
   - `get_community_stats(coin_id)` → twitter/reddit followers, telegram users
   - `get_developer_data(coin_id)` → GitHub repo URLs (used downstream)
   - Map ticker (`BTC-USD`) → CoinGecko ID (`bitcoin`)
   - Dependencies: `requests`

2. **`crypto_id_map.py`** — Symbol → CoinGecko ID mapper
   - Hardcoded map untuk top 100 coins (fast path)
   - Fallback ke CoinGecko `/coins/list` endpoint untuk coin lain
   - Function: `ticker_to_coingecko_id(ticker: str) -> str | None`

3. **`github_activity.py`** — GitHub API client
   - `get_dev_activity(repo_url)` → commits 4w, contributors, stars, forks, last commit
   - Resolve coin → repo via CoinGecko's `repos_url` field
   - Dependencies: `requests`

4. **`defillama.py`** — DeFiLlama client
   - `get_tvl(protocol_slug)` → TVL current + 7d/30d trend
   - `get_protocol_stats(slug)` → revenue, fees, MAU
   - Dependencies: `requests`

5. **`onchain_metrics.py`** — Etherscan + derived metrics
   - `get_onchain_metrics(token_address, chain)` → active addresses 24h, tx count, gas usage
   - Per-chain abstraction (Ethereum, BSC, Polygon, Arbitrum)
   - Dependencies: `requests`, `ETHERSCAN_API_KEY`

6. **`fear_greed.py`** — Alternative.me Fear & Greed Index
   - `get_fear_greed_index()` → integer 0-100 + classification
   - Dependencies: `requests`

7. **`cryptopanic.py`** *(optional Phase 3)* — Crypto news aggregator
   - `get_crypto_news(currency)` → headlines + sentiment + source
   - Dependencies: `requests`, `CRYPTOPANIC_API_KEY`

### Agent Tools Layer (`tradingagents/agents/utils/`)

8. **`crypto_fundamental_tools.py`** — LangGraph `@tool` wrappers
   - `get_crypto_tokenomics(ticker)` — wrap `coingecko.get_tokenomics`
   - `get_crypto_dev_activity(ticker)` — wrap `github_activity.get_dev_activity`
   - `get_crypto_network_metrics(ticker)` — wrap `defillama` + `onchain_metrics`
   - `get_crypto_market_sentiment(ticker)` — wrap `fear_greed` + `coingecko.get_community_stats`
   - Pattern: mirror existing `fundamental_data_tools.py` structure

### Agent Layer (`tradingagents/agents/analysts/`)

9. **`crypto_fundamentals_analyst.py`** — New analyst node factory
   - `create_crypto_fundamentals_analyst(llm)` returns LangGraph node
   - System message customized untuk crypto domain (tokenomics, dev activity, etc.)
   - Same structure as `fundamentals_analyst.py` tapi import crypto tools

### Tests (`tests/`)

10. **`test_crypto_fundamentals.py`** — Unit + integration tests
    - Mock CoinGecko/GitHub/DeFiLlama responses
    - Test parsing, error handling, rate limit behavior
    - Integration test: full analyst node execution with mocked dataflows

---

## 6. Files to Modify

### Critical (must change for feature to work)

1. **`tradingagents/graph/trading_graph.py`**
   - `__init__()`: tambah param `asset_type: str = "stock"`
   - `_create_tool_nodes()`: conditional return berdasarkan `self.asset_type`

2. **`tradingagents/graph/setup.py`**
   - `__init__()`: tambah param `asset_type: str = "stock"`
   - `setup_graph()`: swap `analyst_factories["fundamentals"]` based on asset_type
   - Pass asset_type from `TradingAgentsGraph` constructor

3. **`cli/utils.py`** (line 56)
   - `filter_analysts_for_asset_type()`: **HAPUS filtering fundamentals untuk crypto**
   - Karena sekarang crypto punya analyst-nya sendiri

4. **`cli/main.py`** (line 513-625)
   - Pass `asset_type=asset_type.value` ke `TradingAgentsGraph()` constructor

5. **`tradingagents/agents/__init__.py`**
   - Add `from .analysts.crypto_fundamentals_analyst import create_crypto_fundamentals_analyst`
   - Re-export untuk dipakai `setup.py`

6. **`tradingagents/agents/utils/agent_utils.py`**
   - Tambah imports untuk crypto tools (untuk dipakai dari `_create_tool_nodes`)

### Non-critical (improvement / consistency)

7. **`tradingagents/dataflows/__init__.py`** — Re-export new dataflow modules

8. **`tradingagents/dataflows/interface.py`**
   - Tambah `crypto_fundamental_data` ke `TOOLS_CATEGORIES` dan `VENDOR_METHODS`
   - Add coingecko sebagai vendor

9. **`tradingagents/default_config.py`**
   - Tambah `"crypto_fundamental_data": "coingecko"` ke `data_vendors` dict
   - Tambah env var overrides untuk crypto API keys di `_ENV_OVERRIDES` (optional)

10. **`.env.example`** — Tambah 4 env var baru (lihat section 4)

11. **`pyproject.toml`** — Tidak perlu deps baru (`requests` sudah ada)

12. **`tradingagents/agents/researchers/bull_researcher.py`** (line 19-20)
    - Update label: `"may be unavailable for crypto"` → `"Crypto fundamentals report (tokenomics, on-chain, dev activity)"`

13. **`tradingagents/agents/researchers/bear_researcher.py`** (line 19-20)
    - Same label change

14. **`tradingagents/agents/managers/research_manager.py`** & **`portfolio_manager.py`**
    - Review prompts — apakah ada hardcoded reference ke "company fundamentals" yang perlu disesuaikan untuk crypto?

15. **`tests/test_crypto_asset_mode.py`**
    - Update test `test_filters_out_fundamentals_analyst_for_crypto` — expect fundamentals **TETAP ADA** untuk crypto sekarang
    - Tambah test baru: `test_crypto_fundamentals_uses_crypto_tools`

---

## 7. Functions to Modify (Before/After)

### 7.1 `cli/utils.py::filter_analysts_for_asset_type` (line 56)

```python
# SEBELUM:
def filter_analysts_for_asset_type(
    analysts: List[AnalystType], asset_type: AssetType
) -> List[AnalystType]:
    if asset_type != AssetType.CRYPTO:
        return analysts
    return [
        analyst
        for analyst in analysts
        if analyst != AnalystType.FUNDAMENTALS
    ]

# SESUDAH:
def filter_analysts_for_asset_type(
    analysts: List[AnalystType], asset_type: AssetType
) -> List[AnalystType]:
    """Crypto now has its own fundamentals analyst — keep all analysts."""
    return analysts
```

### 7.2 `tradingagents/graph/trading_graph.py::TradingAgentsGraph.__init__` (line 50)

```python
# SEBELUM:
def __init__(
    self,
    selected_analysts=["market", "social", "news", "fundamentals"],
    debug=False,
    config: Dict[str, Any] = None,
    callbacks: Optional[List] = None,
):

# SESUDAH:
def __init__(
    self,
    selected_analysts=["market", "social", "news", "fundamentals"],
    debug=False,
    config: Dict[str, Any] = None,
    callbacks: Optional[List] = None,
    asset_type: str = "stock",  # ← BARU
):
    ...
    self.asset_type = asset_type  # ← BARU, simpan untuk dipakai _create_tool_nodes
    ...
    # Pass ke GraphSetup juga:
    self.graph_setup = GraphSetup(
        ...,
        asset_type=self.asset_type,
    )
```

### 7.3 `tradingagents/graph/trading_graph.py::TradingAgentsGraph._create_tool_nodes` (line 130)

```python
# SEBELUM:
def _create_tool_nodes(self) -> Dict[str, ToolNode]:
    return {
        "market": ToolNode([get_stock_data, get_indicators]),
        "social": ToolNode([get_news]),
        "news": ToolNode([get_news, get_global_news, get_insider_transactions]),
        "fundamentals": ToolNode([
            get_fundamentals, get_balance_sheet,
            get_cashflow, get_income_statement
        ]),
    }

# SESUDAH:
def _create_tool_nodes(self) -> Dict[str, ToolNode]:
    """Build tool nodes per analyst category, dispatching crypto vs stock."""
    if self.asset_type == "crypto":
        from tradingagents.agents.utils.crypto_fundamental_tools import (
            get_crypto_tokenomics,
            get_crypto_dev_activity,
            get_crypto_network_metrics,
            get_crypto_market_sentiment,
        )
        fundamentals_tools = [
            get_crypto_tokenomics,
            get_crypto_dev_activity,
            get_crypto_network_metrics,
            get_crypto_market_sentiment,
        ]
    else:
        fundamentals_tools = [
            get_fundamentals, get_balance_sheet,
            get_cashflow, get_income_statement,
        ]

    return {
        "market": ToolNode([get_stock_data, get_indicators]),
        "social": ToolNode([get_news]),
        "news": ToolNode([get_news, get_global_news, get_insider_transactions]),
        "fundamentals": ToolNode(fundamentals_tools),
    }
```

### 7.4 `tradingagents/graph/setup.py::GraphSetup.setup_graph` (line 35)

```python
# SEBELUM:
analyst_factories = {
    "market": lambda: create_market_analyst(self.quick_thinking_llm),
    "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
    "news": lambda: create_news_analyst(self.quick_thinking_llm),
    "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
}

# SESUDAH:
if self.asset_type == "crypto":
    from tradingagents.agents.analysts.crypto_fundamentals_analyst import (
        create_crypto_fundamentals_analyst,
    )
    fundamentals_factory = lambda: create_crypto_fundamentals_analyst(self.quick_thinking_llm)
else:
    fundamentals_factory = lambda: create_fundamentals_analyst(self.quick_thinking_llm)

analyst_factories = {
    "market": lambda: create_market_analyst(self.quick_thinking_llm),
    "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
    "news": lambda: create_news_analyst(self.quick_thinking_llm),
    "fundamentals": fundamentals_factory,
}
```

### 7.5 `tradingagents/agents/utils/agent_utils.py::build_instrument_context` (line 30)

```python
# SESUDAH (extend crypto branch dengan tool hint):
def build_instrument_context(ticker: str, asset_type: str = "stock") -> str:
    instrument_label = "asset" if asset_type == "crypto" else "instrument"
    if asset_type == "crypto":
        extra_hint = (
            " This is a crypto asset. Use crypto-specific fundamental tools "
            "(tokenomics, on-chain metrics, dev activity, market sentiment) "
            "rather than traditional financial statements. "
            "Crypto fundamentals include: token supply dynamics, network usage, "
            "developer activity on GitHub, and market sentiment indicators."
        )
    else:
        extra_hint = ""
    return (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + extra_hint
    )
```

---

## 8. New Schema Additions

Tambahkan ke **`tradingagents/agents/schemas.py`** (append):

```python
from typing import Optional
from pydantic import BaseModel, Field


class TokenomicsReport(BaseModel):
    """Tokenomics snapshot from CoinGecko."""
    market_cap_usd: Optional[float] = Field(default=None, description="Current market cap in USD")
    fully_diluted_valuation: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    max_supply: Optional[float] = Field(default=None, description="None if unlimited (e.g., ETH)")
    inflation_rate_pct: Optional[float] = Field(default=None, description="Annual inflation %")
    supply_ratio: Optional[float] = Field(default=None, description="circulating/max ratio (0-1)")
    is_deflationary: Optional[bool] = None


class OnChainMetrics(BaseModel):
    """On-chain activity from Etherscan + DeFiLlama."""
    active_addresses_24h: Optional[int] = None
    transaction_count_24h: Optional[int] = None
    nvt_ratio: Optional[float] = Field(default=None, description="Network Value to Transactions; <50 = potentially undervalued")
    tvl_usd: Optional[float] = Field(default=None, description="Total Value Locked (DeFi protocols only)")
    tvl_to_mcap: Optional[float] = Field(default=None, description="TVL/Market Cap ratio")
    chain: Optional[str] = Field(default=None, description="ethereum/bsc/solana/etc.")


class DevActivityReport(BaseModel):
    """GitHub activity for the project's main repo."""
    github_url: Optional[str] = None
    commits_4w: Optional[int] = Field(default=None, description="Commits in last 4 weeks")
    contributors: Optional[int] = None
    stars: Optional[int] = None
    forks: Optional[int] = None
    last_commit_days_ago: Optional[int] = None
    activity_score: Optional[str] = Field(default=None, description="active/moderate/dormant")


class MarketSentimentReport(BaseModel):
    """Market-wide sentiment indicators."""
    fear_greed_index: Optional[int] = Field(default=None, description="0-100 scale, <25=extreme fear, >75=extreme greed")
    fear_greed_classification: Optional[str] = None
    btc_dominance_pct: Optional[float] = None
    twitter_followers: Optional[int] = None
    reddit_subscribers: Optional[int] = None
    telegram_users: Optional[int] = None


class CryptoFundamentalsReport(BaseModel):
    """Aggregated crypto fundamentals — internal validation, output is still prose."""
    tokenomics: Optional[TokenomicsReport] = None
    onchain: Optional[OnChainMetrics] = None
    dev_activity: Optional[DevActivityReport] = None
    sentiment: Optional[MarketSentimentReport] = None
    summary_grade: Optional[str] = Field(default=None, description="strong/neutral/weak/concerning")
```

**Catatan**: Schemas ini untuk validasi internal & dokumentasi. Analyst tetap output prose `fundamentals_report` string (matching state field type sekarang).

---

## 9. Prompt Engineering Strategy

System message untuk crypto fundamentals analyst harus berbeda dari stock:

### Perbedaan Kunci

| Aspek | Stock Prompt | Crypto Prompt |
|-------|-------------|---------------|
| Object of analysis | "company" | "asset" / "protocol" |
| Financial concepts | Revenue, earnings, P/E | Market cap, supply schedule, NVT |
| Health indicators | Profit margin, debt | Token velocity, holder distribution |
| Growth metrics | Revenue YoY | TVL growth, active addresses, dev velocity |
| Risk factors | Debt levels | Concentration risk, regulatory uncertainty |

### Grading Rubric untuk Crypto

```
**Tokenomics Health**:
- Supply concentration: Top 10 holders <30% = good, >50% = concerning
- Distribution: VC unlock schedule transparent? Burn mechanism active?
- Inflation: Annual <3% = good for store-of-value; >10% = concerning unless utility token

**Network Activity**:
- Active addresses: Growing 7d/30d = bullish
- NVT Ratio: <50 = potentially undervalued, >100 = potentially overvalued
- Transaction count: vs 30d avg

**Developer Commitment**:
- >50 commits/4w = active
- 10-50 commits/4w = moderate
- <10 commits/4w = concerning (could indicate abandoned project)
- Last commit >30 days ago = red flag

**Market Sentiment**:
- Fear & Greed <25 = extreme fear (contrarian buy signal possible)
- Fear & Greed >75 = extreme greed (caution, possible top)
- BTC dominance trends affect altcoin season

**DeFi-specific** (jika applicable):
- TVL/MCap ratio: >0.5 = strong (protocol value backed by capital)
- TVL trend: 7d/30d direction
```

### System Message Template

```python
system_message = (
    "You are a crypto fundamentals researcher analyzing a digital asset. "
    "Unlike traditional company fundamentals, crypto fundamentals focus on:\n"
    "1. **Tokenomics**: supply dynamics, distribution, inflation/deflation\n"
    "2. **On-chain Activity**: active addresses, transaction volume, NVT ratio\n"
    "3. **Network Health**: TVL (for DeFi), validator/hashrate (for L1)\n"
    "4. **Developer Activity**: GitHub commits, contributor count, project momentum\n"
    "5. **Market Sentiment**: Fear & Greed, dominance shifts\n\n"
    "Use the available tools: `get_crypto_tokenomics`, `get_crypto_dev_activity`, "
    "`get_crypto_network_metrics`, `get_crypto_market_sentiment`.\n\n"
    "Apply this evidence-based grading:\n"
    "- Tokenomics: Is supply deflationary or inflationary? Concentration risk?\n"
    "- Activity: Are active addresses growing? Is NVT reasonable?\n"
    "- Dev: Is the team actively shipping? When was the last commit?\n"
    "- Sentiment: Where is Fear & Greed? Are we in extreme territory?\n\n"
    "Append a Markdown table at the end summarizing key metrics with grades. "
    "Provide actionable insights with supporting evidence to help traders decide."
    + get_language_instruction()
)
```

---

## 10. Phased Rollout

### Phase 0 — Scaffolding (0.5 day, ~80 LOC)

- Tambah `asset_type` param di `TradingAgentsGraph.__init__()` dan `GraphSetup.__init__()`
- Update `cli/utils.py:filter_analysts_for_asset_type` (hapus filtering)
- Buat file stub `crypto_fundamentals_analyst.py` (return placeholder report)
- Update test `test_crypto_asset_mode.py` (expect fundamentals tetap)

**Deliverable**: Crypto pipeline runs end-to-end, fundamentals analyst di-include tapi return placeholder.

### Phase 1 — CoinGecko MVP (3 days, ~400 LOC)

- Implement `dataflows/coingecko.py` (get_tokenomics, get_market_data, get_community_stats)
- Implement `dataflows/crypto_id_map.py` (top 50 hardcoded + fallback ke `/coins/list`)
- Implement `dataflows/fear_greed.py`
- Implement `agents/utils/crypto_fundamental_tools.py` (4 tools)
- Implement full `crypto_fundamentals_analyst.py` dengan system message lengkap
- Wire ke graph via `_create_tool_nodes` dispatch

**Deliverable**: BTC-USD analysis menghasilkan fundamentals report dengan tokenomics + sentiment.

### Phase 2 — GitHub + DeFiLlama (2 days, ~250 LOC)

- Implement `dataflows/github_activity.py`
- Implement `dataflows/defillama.py`
- Tambah `get_crypto_dev_activity` tool yang call GitHub API
- Tambah `get_crypto_network_metrics` tool yang call DeFiLlama
- Update analyst prompt untuk reference tools baru

**Deliverable**: Report sekarang include developer activity + TVL trends untuk DeFi tokens.

### Phase 3 — On-chain + News (2 days, ~200 LOC)

- Implement `dataflows/onchain_metrics.py` (Etherscan)
- Optionally: implement `dataflows/cryptopanic.py`
- Enrich `get_crypto_network_metrics` dengan on-chain data

**Deliverable**: Active addresses, transaction count, gas usage tersedia.

### Phase 4 — Testing & Tuning (2 days, ~300 LOC tests)

- Full unit test suite untuk semua dataflows
- Integration test untuk analyst node
- End-to-end test BTC-USD analysis dengan mocked APIs
- Prompt iteration based on output quality
- Rate limit handling validation
- Documentation update

**Deliverable**: Production-ready, tested, documented crypto fundamentals analyst.

**Total: 9.5 hari (~1.5 minggu) untuk 1 dev**

---

## 11. Testing Strategy

### Unit Tests (per dataflow file)

`tests/test_coingecko.py`:
```python
def test_get_tokenomics_btc(monkeypatch):
    mock_response = {"market_cap": {"usd": 850_000_000_000}, ...}
    monkeypatch.setattr("requests.get", lambda *a, **k: MockResponse(mock_response))
    result = get_tokenomics("bitcoin")
    assert result["market_cap_usd"] == 850_000_000_000
```

Sama untuk: `test_github_activity.py`, `test_defillama.py`, `test_fear_greed.py`, `test_onchain_metrics.py`

### Tool Tests

```python
def test_get_crypto_tokenomics_tool_returns_string():
    """LangGraph tools must return strings."""
    result = get_crypto_tokenomics.invoke({"ticker": "BTC-USD"})
    assert isinstance(result, str)
    assert "market cap" in result.lower()
```

### Integration Tests

```python
def test_crypto_fundamentals_analyst_node(mock_dataflows):
    """Run analyst node end-to-end with mocked HTTP."""
    state = {"company_of_interest": "BTC-USD", "trade_date": "2026-05-23", ...}
    result = crypto_fundamentals_analyst_node(state)
    assert result["fundamentals_report"]
    assert "tokenomics" in result["fundamentals_report"].lower()
```

### End-to-End Test

```python
def test_btc_full_analysis(mock_all_apis):
    """Full graph run for BTC-USD."""
    config = DEFAULT_CONFIG.copy()
    ta = TradingAgentsGraph(asset_type="crypto", config=config)
    final_state, decision = ta.propagate("BTC-USD", "2026-05-23", asset_type="crypto")
    assert final_state["fundamentals_report"]
    assert decision in ("Buy", "Overweight", "Hold", "Underweight", "Sell")
```

### Regression Test

```python
def test_stock_pipeline_unchanged():
    """Verify stock fundamentals still works."""
    ta = TradingAgentsGraph(asset_type="stock")  # default
    # ... existing test fundamentals_analyst dipakai untuk NVDA
```

### Rate Limit Tests

```python
def test_coingecko_rate_limit_graceful_degradation(monkeypatch):
    """When CoinGecko returns 429, tool returns 'data unavailable'."""
    monkeypatch.setattr("requests.get", lambda *a, **k: MockResponse({}, status=429))
    result = get_crypto_tokenomics.invoke({"ticker": "BTC-USD"})
    assert "unavailable" in result.lower() or "rate limit" in result.lower()
```

---

## 12. Backward Compatibility

**Garansi kompatibilitas**:

1. **Stock pipeline**: ZERO perubahan execution path saat `asset_type="stock"` (default)
2. **API key opsional**: Semua env var baru optional dengan graceful fallback
3. **State field**: `fundamentals_report` tetap `str` — tidak break downstream consumers
4. **Existing tests pass**: Semua test sekarang lulus tanpa modifikasi
   - Exception: `test_crypto_asset_mode.py:test_filters_out_fundamentals_analyst_for_crypto` — test ini menguji **OLD behavior** (skip), perlu di-update untuk **NEW behavior** (keep)
5. **No required deps**: `requests` sudah di pyproject.toml; tidak ada library baru yang wajib
6. **Config default**: `data_vendors["crypto_fundamental_data"] = "coingecko"` — kalau key kosong, tool fallback ke unauthenticated rate limit

---

## 13. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **CoinGecko rate limit** (30/min keyless) | Analyst stuck mid-analysis | Cache responses 5 menit; user bisa register Demo key untuk 500/min |
| **CoinGecko ID mapping fails** | Wrong coin data fetched | Maintain hardcoded fallback map untuk top 50 coins; fuzzy match ke nama |
| **GitHub repo not found** | Missing dev data section | Graceful "dev activity data unavailable" inline di report |
| **API response format change** | Parsing breaks silently | Wrap semua parsing dalam try/except, return "data unavailable" |
| **Token cost increase** | Higher LLM cost per crypto run | Crypto analyst tetap 4-tool limit; tidak ada extra debate rounds |
| **Data staleness** | Outdated metrics | CoinGecko real-time; cache TTL 5 menit |
| **Test coverage gap** | Regression di stock pipeline | Comprehensive regression suite di Phase 4 |
| **Concurrent rate limit consumption** | Multi-ticker run hits limit | Sequential analyst execution (sudah default), atau add rate limiter library |

---

## 14. Effort Estimation

| Phase | LOC (approx) | Days | Notes |
|-------|-------------|------|-------|
| **Phase -1: Test Foundation** | ~400 | 2 | Fix failing tests, CI, linter, e2e smoke (stock+crypto) |
| Phase 0: Scaffolding | ~80 | 0.5 | Param threading, filter change, file stubs |
| Phase 1: CoinGecko MVP | ~400 | 3 | Primary data source + analyst |
| Phase 2: GitHub + DeFiLlama | ~250 | 2 | Two new dataflows + tools |
| Phase 3: On-chain + News | ~200 | 2 | Etherscan, optional CryptoPanic |
| Phase 4: Testing & Tuning | ~300 | 2 | Crypto-specific tests, prompt iteration |
| **Total** | **~1,630 LOC** | **11.5 days** | One developer, full-time |

**Dependencies**:
- 1 developer with Python + LangGraph familiarity
- Optional: CoinGecko Demo API key (free, 5 menit setup)
- Optional: GitHub token (free)
- Optional: Etherscan API key (free, 5 menit setup)

---

## 15. Open Questions / Decisions Needed

Hal-hal yang harus diputuskan sebelum mulai implementasi:

1. **CoinGecko Demo API key**: Wajibkan user register untuk free Demo key, atau support fully keyless (30/min) operation?
   - **Rekomendasi**: Support keyless, dokumentasikan upgrade path ke Demo key

2. **Glassnode integration**: Worth $29/mo untuk NVT/MVRV official? Atau cukup CoinGecko approximation?
   - **Rekomendasi**: Skip Phase 1-3, evaluasi setelah feedback usage Phase 4

3. **Forex support**: Pattern serupa di-extend untuk forex pairs (EUR-USD, dll) di future?
   - **Rekomendasi**: Pisahkan ke plan terpisah, fokus crypto dulu

4. **Coin-to-repo mapping**: Hardcode top 50 mappings, atau rely pada CoinGecko `developer_data.repos_url`?
   - **Rekomendasi**: Both — hardcode top 50 (fast path), fallback ke CoinGecko (slow path)

5. **Graph recompilation**: Pass `asset_type` di `__init__` (current proposal), atau create BOTH tool nodes dan select runtime via state?
   - **Rekomendasi**: `__init__` time — simpler, matches current architecture, sesuai dengan instance per ticker

6. **Structured output**: Crypto analyst pakai structured output (seperti ResearchPlan), atau free-text seperti current fundamentals?
   - **Rekomendasi**: Free-text untuk Phase 1-3, evaluasi structured di Phase 4

7. **Historical data**: CoinGecko free tier limit historical. Cukup current snapshot, atau perlu 7-day trends?
   - **Rekomendasi**: Current snapshot Phase 1, tambah 7-day delta Phase 2-3

8. **Multi-chain support**: Handle Ethereum/BSC/Solana/Polygon di Phase 1, atau focus Ethereum dulu?
   - **Rekomendasi**: CoinGecko handles multi-chain natively. Etherscan-style on-chain (Phase 3) start dengan Ethereum, expand ke chains lain by need.

9. **Concurrent rate limit handling**: Pakai library seperti `aiolimiter`, atau homegrown semaphore?
   - **Rekomendasi**: Homegrown sederhana di Phase 1 (since execution sequential), evaluasi `aiolimiter` jika butuh paralel di future

10. **Cache layer**: In-memory dict, Redis, SQLite, atau library `requests-cache`?
    - **Rekomendasi**: `requests-cache` — drop-in, sudah handle TTL, persistent option ke SQLite

---

## Ringkasan Aksi Konkret

**Untuk mulai implementasi**, lakukan dalam urutan ini:

1. ✅ Review & approve plan ini
2. ⏸️ Decide pada Open Questions section 15 (terutama #1, #5, #10)
3. ✅ Setup CoinGecko Demo API key (5 menit, free) — opsional tapi recommended
4. ✅ Setup GitHub Personal Access Token (5 menit, free)
5. ✅ Mulai Phase 0 (scaffolding) — paling cepat, validate path threading
6. ✅ Lanjut Phase 1 (CoinGecko MVP) sampai end-to-end works
7. ⏸️ Validate dengan 3-5 popular coins (BTC, ETH, SOL, MATIC, DOGE)
8. ✅ Phase 2-4 sesuai plan

---

*Plan ini dibuat berdasarkan analisis codebase TradingAgents v0.2.5 commit `61522e1` pada 2026-05-23. Untuk update strukturalnya, jalankan ulang `/understand` agar knowledge graph sinkron.*
