# Master Architecture & Integration Plan: Kronos K-Line Foundation Model (shiyu-coder/Kronos) for TradingAgents

## 1. Executive Summary & Problem Formulation

This document defines the production-grade implementation specification to integrate the **Kronos Financial K-Line Foundation Model** (`shiyu-coder/Kronos`, AAAI 2026) into the `tradingagents` quantitative multi-agent trading ecosystem.

### 1.1 The Core Problem & Quantitative Thesis
- **LLM Cognitive Blindspot**: General-purpose Large Language Models (GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Pro) excel at qualitative reasoning, multi-document synthesis, and macro narrative analysis. However, they struggle with high-dimensional numeric candlestick dynamics, price-volume microstructure, and multi-step non-linear autoregressive trajectory forecasting across hundreds of historical bars.
- **Kronos Foundation Model Capability**: Kronos is a specialized autoregressive transformer architecture pre-trained on multi-resolution K-line sequences across 45 global financial exchanges using hierarchical discrete tokenization. It ingests historical OHLCV sequences (up to 512 bars) and autoregressively generates forward trajectories of future price-volume dynamics (Open, High, Low, Close, Volume).
- **The "Code Judo" Architecture**: Instead of altering the core LangGraph multi-agent topology or introducing fragile custom nodes, Kronos is encapsulated strictly as a **Neural Quantitative Forecasting Dataflow Tool** (`get_kronos_forecast`). It is registered in the `Market / Technical Analyst` toolset and fully supported across single-shot signal evaluation (`evaluate-signal`), walk-forward backtesting (`backtest`), and live market analysis (`analyze`).

---

## 2. System-Wide Invariants & Quant Safety Standards

```
+─────────────────────────────────────────────────────────────────────────────+
│                         KRONOS INTEGRATION INVARIANTS                       │
│                                                                             │
│  [Invariant 1] Zero Lookahead Bias: Causal input clamped strictly at T0.    │
│  [Invariant 2] Point-in-Time Causal Normalization: Anchor statistics        │
│                (Close_T0, rolling mean/std) computed strictly on lookback;  │
│                never across future horizon bars.                            │
│  [Invariant 3] Deterministic Backtesting: Temperature T=0.0 in backtests.   │
│  [Invariant 4] Hardware & Attention Optimization: Native PyTorch 2.x SDPA   │
│                (FlashAttention-2 fallback chain) + bfloat16 mixed precision.│
│  [Invariant 5] Singleton VRAM & Memory Isolation: Thread-safe persistent    │
│                weights in VRAM with zero backtest step re-allocation.       │
│  [Invariant 6] Dual-Tier Caching: In-memory LRU + Disk JSON cache.         │
│  [Invariant 7] Graceful Multi-Platform Fallback: CUDA -> MPS -> CPU -> Mock.│
│  [Invariant 8] Graph & Schema Parity: Non-breaking opt-in via config/CLI.   │
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 3. Data Contracts & Architecture Specifications

### 3.1 Input / Output Data Contracts (`tradingagents/dataflows/kronos_types.py` / `kronos.py`)

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
import pandas as pd

@dataclass(frozen=True)
class KronosContextContract:
    symbol: str
    cutoff_date: str          # Strict causal ceiling (T0: YYYY-MM-DD or ISO timestamp)
    df_bars: pd.DataFrame     # Required columns: ['open', 'high', 'low', 'close', 'volume', 'date']
    lookback_bars: int = 512  # Optimal Kronos receptive field (clamped <= 512)
    pred_len: int = 20        # Forward forecast length in trading bars (5, 10, 20)
    temperature: float = 0.0  # 0.0 for deterministic greedy argmax

@dataclass(frozen=True)
class KronosPredictedBar:
    step: int                 # 1..pred_len (T+1 .. T+N)
    projected_open: float
    projected_high: float
    projected_low: float
    projected_close: float
    projected_volume: float
    return_from_t0_pct: float # ((projected_close - last_close) / last_close) * 100.0

@dataclass(frozen=True)
class KronosForecastContract:
    symbol: str
    forecast_date: str        # T0 cutoff date
    horizon_bars: int         # N steps forward
    last_close: float         # Close price at T0
    forecast_high: float      # Maximum projected high over horizon
    forecast_low: float       # Minimum projected low over horizon
    forecast_end_close: float # Projected close at step N
    forecast_return_pct: float # Expected return % at horizon end
    max_upside_pct: float     # ((forecast_high - last_close) / last_close) * 100.0
    max_downside_pct: float   # ((forecast_low - last_close) / last_close) * 100.0
    directional_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence_score: float   # Derived from trajectory consistency & token entropy (0.0 to 1.0)
    model_name: str           # e.g. "NeoQuasar/Kronos-base"
    device_used: str          # e.g. "cuda:0", "mps", "cpu"
    forecast_bars: List[KronosPredictedBar]
    raw_summary_markdown: str # Structured markdown for LLM consumption
```

---

## 4. Hardware Acceleration, Attention & Mathematical Optimization

### 4.1 PyTorch 2.x Attention Fallback Chain
Kronos transformer self-attention must support high-throughput execution across heterogeneous hardware architectures:
1. **FlashAttention-2 (`flash_attention_2`)**: Used if CUDA compute capability $\ge 8.0$ (Ampere, Ada Lovelace, Hopper) and `flash-attn` is installed.
2. **PyTorch SDPA (`sdpa`)**: Default standard via `torch.nn.functional.scaled_dot_product_attention` (C++ kernel with FlashAttention/Memory-Efficient Attention backends).
3. **Eager PyTorch (`eager`)**: Fallback for CPU / older GPUs.

### 4.2 Precision & Memory Allocation Strategy
- **Ampere/Ada/Hopper (RTX 30xx/40xx, A100, H100)**: `torch.bfloat16` (full dynamic range of fp32 with fp16 speed).
- **Turing / Volta (RTX 20xx, GTX 16xx, V100)**: `torch.float16`.
- **Apple Silicon (MPS) & CPU**: `torch.float32`.
- **Inference Mode**: All tensor computations strictly wrapped inside `torch.inference_mode()`.
- **Pinned Memory Async Transfers**: Tensor inputs allocated in page-locked CPU memory before non-blocking device transfer:
  ```python
  tensor_input = tensor_input.pin_memory().to(device, non_blocking=True)
  ```
- **Optional Linux CUDA Kernel Compilation**:
  ```python
  if config.get("kronos_torch_compile", False) and device.type == "cuda":
      model = torch.compile(model, mode="reduce-overhead")
  ```

### 4.3 Causal Normalization & Tokenization Math
Raw price series exhibit non-stationary drift across stocks (e.g. MSFT at \$420 vs penny stock at \$1.50). Feeding un-normalized price data causes codebook saturation:
1. **Causal Anchor Point**: $P_{0} = Close_{T0}$ (price at cutoff date).
2. **Causal Relative Scaling**:
   $$\tilde{O}_t = \frac{O_t - P_0}{P_0}, \quad \tilde{H}_t = \frac{H_t - P_0}{P_0}, \quad \tilde{L}_t = \frac{L_t - P_0}{P_0}, \quad \tilde{C}_t = \frac{C_t - P_0}{P_0}$$
3. **Rolling Volume Standardization**:
   $$\tilde{V}_t = \frac{V_t - \mu_{V, [T-lookback:T]}}{\sigma_{V, [T-lookback:T]} + \epsilon}$$
4. **Inverse Denormalization**:
   $$P_{\text{projected}, T+k} = P_0 \times (1.0 + \tilde{P}_{\text{predicted}, T+k})$$

---

## 5. Phase-by-Phase Implementation Blueprint

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             EXECUTION PHASES                                │
│                                                                             │
│  Phase 1: Dependencies, Inference Engine & Singleton Manager (kronos.py)   │
│  Phase 2: Dataflow Tool Registration & Structural Levels Integration        │
│  Phase 3: Backtest Engine & 512-Bar Snapshot Window Slicing                 │
│  Phase 4: Market Analyst Prompt & Neural Trajectory Confluence              │
│  Phase 5: Trader & Portfolio Manager Target Anchoring (Anti-Prompt Gaming)  │
│  Phase 6: CLI Options, Configuration Schema & TUI Visualizer                │
│  Phase 7: Comprehensive Test Suite & Zero-Leakage Verification              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Dependencies, Inference Engine & Singleton Manager

#### Target Files
- `pyproject.toml`
- `tradingagents/dataflows/kronos.py` (New core engine)
- `tradingagents/default_config.py`

#### Tasks & Implementation Details
- [x] **1.1 Configure Optional Quant Dependencies in `pyproject.toml`**:
  ```toml
  [project.optional-dependencies]
  quant = [
      "torch>=2.1.0",
      "transformers>=4.40.0",
      "huggingface_hub>=0.20.0",
      "accelerate>=0.25.0",
      "einops>=0.7.0",
  ]
  ```
- [x] **1.2 Create `tradingagents/dataflows/kronos.py`**:
  - Implement `KronosModelManager` (Singleton pattern with thread lock `threading.Lock()`):
    - Methods: `get_model_and_tokenizer(model_tier, device, attn_impl)`, `unload_model()`, `clear_vram_cache()`.
    - Handles model caching in `~/.cache/huggingface/hub` or custom local cache directory.
  - Implement `KronosInferenceEngine`:
    - Point-in-time input validation: `assert (df["date"] <= cutoff_date).all()`.
    - Context lookback slice: take last `min(len(df), 512)` bars.
    - Causal normalization -> Discrete hierarchical tokenization -> Autoregressive greedy generation (`temperature=0.0`) -> Inverse denormalization.
  - Implement Dual-Level Caching:
    - Level 1: In-memory `OrderedDict` (LRU cache with capacity 128).
    - Level 2: Persistent JSON file cache at `api_cache/kronos/{symbol}_{cutoff_date}_{model_tier}_{pred_len}.json`.
  - Implement Graceful Hardware Fallback:
    - Auto-detect CUDA -> MPS -> CPU.
    - If `torch` / `transformers` is missing or model download fails, return a graceful typed error/mock message without crashing the parent process.

---

### Phase 2: Dataflow Tool Registration & Structural Levels Integration

#### Target Files
- `tradingagents/dataflows/interface.py`
- `tradingagents/dataflows/structural_levels.py`
- `tradingagents/agents/utils/core_stock_tools.py`

#### Tasks & Implementation Details
- [x] **2.1 Implement `get_kronos_forecast` in `tradingagents/dataflows/interface.py`**:
  - Expose `get_kronos_forecast(symbol: str, curr_date: str, pred_days: int = 20, config: Optional[Dict] = None) -> str`.
  - Connects to existing dataflow router and retrieves historical OHLCV from cache/provider.
- [x] **2.2 Register LangGraph Tool `@tool get_kronos_forecast` in `core_stock_tools.py`**:
  - Complete type annotations, docstrings, and robust exception trapping (returning structured error markdown on failure).
- [ ] **2.3 Integrate Neural Levels into `structural_levels.py`**:
  - In `compute_structural_levels()`: when `kronos_enabled=True`, calculate neural resistance (projected peak) and neural support (projected trough) alongside classical Fibonacci 1.272x/1.618x extension levels.

---

### Phase 3: Backtest Engine & 512-Bar Snapshot Window Slicing

#### Target Files
- `tradingagents/backtesting/data_window.py`
- `tradingagents/backtesting/snapshot_provider.py`
- `tradingagents/backtesting/walk_forward_runner.py`

#### Tasks & Implementation Details
- [x] **3.1 Expand Allowed Lookback Windows**:
  - Update `tradingagents/backtesting/data_window.py:30`:
    ```python
    ALLOWED_LOOKBACKS = (None, 5, 10, 20, 40, 60, 80, 100, 120, 240, 512)
    ```
- [x] **3.2 Ensure 512-Bar Causal Slicing in Snapshot Provider**:
  - In `SnapshotDataProvider.get_ohlcv()`: guarantee that when `lookback=512`, up to 512 daily bars $\le T_0$ are retrieved without lookahead leakage.
- [ ] **3.3 Zero-Copy Backtest Cache Efficiency**:
  - In `walk_forward_runner.py`: Ensure pre-sliced historical windows are reused efficiently without redundant disk reads.

---

### Phase 4: Market Analyst Prompt & Neural Trajectory Confluence

#### Target Files
- `tradingagents/agents/analysts/market_analyst.py`
- `tradingagents/agents/utils/agent_states.py`

#### Tasks & Implementation Details
- [x] **4.1 Bind `get_kronos_forecast` to Market Analyst**:
  - Add `get_kronos_forecast` to Market Analyst's tool binding list when `kronos_enabled` is active.
- [x] **4.2 Update Market Analyst System Prompt**:
  - Guide the Market Analyst to synthesize **Classical Technical Structure** (SMA 50/200, EMA 9/21, RSI, MACD, Fibonacci) with the **Autoregressive Neural Trajectory** (Kronos projected path).
  - Explicitly prompt for **Confluence vs Divergence Analysis**:
    - High Confluence: Classical breakout confirmed by positive Kronos neural velocity.
    - Bearish Divergence: Classical momentum showing overbought while Kronos forecasts sharp mean-reversion exhaustion.

---

### Phase 5: Trader & Portfolio Manager Target Anchoring & Trading Regime Mapping

#### Target Files
- `tradingagents/agents/trader/trader.py`
- `tradingagents/agents/managers/research_manager.py`
- `tradingagents/agents/managers/risk_manager.py`
- `tradingagents/agents/schemas.py`

#### Tasks & Implementation Details
- [x] **5.1 Spot long-only polarity mapping**:
  - Kronos `BULLISH` may support `BUY`; `BEARISH` and `NEUTRAL` map to `WNS`.
  - Legacy labels normalize only at input boundaries. No short, reverse, pyramiding, or agent-driven SELL exit.
  - Exits are static stop-loss, take-profit, time stop, or horizon/end-of-backtest closure.
- [x] **5.2 Target Anchoring in Trader & Research Manager**:
  - In `trader.py`: Instruct the Trader to anchor `take_profit` to `forecast_high` (or neural resistance) and `stop_loss` below `forecast_low` (or neural support).
  - Prevents arbitrary unrealistic hallucinated price targets (prompt gaming).
- [x] **5.3 Validation Bounds Consistency**:
  - In `schemas.py`: Ensure `SignalContract` and `PortfolioDecision` validate realistic long-only targets anchored to Kronos neural boundaries.

---

### Phase 6: CLI Options, Configuration Schema & TUI Visualizer

#### Target Files
- `tradingagents/default_config.py`
- `cli/commands/evaluate.py`
- `cli/commands/backtest.py`
- `cli/commands/analyze.py`
- `cli/display.py`

#### Tasks & Implementation Details
  - [x] **6.1 Add Configuration Keys in `DEFAULT_CONFIG`**:
  ```python
  "kronos_enabled": False,              # Default off for lightweight installations
  "kronos_model_tier": "base",          # "base" (102.3M, Recommended) | "small" (24.7M) | "mini" (4.1M)
  "kronos_model_repo": "NeoQuasar/Kronos-base",
  "kronos_tokenizer_repo": "NeoQuasar/Kronos-Tokenizer-base",
  "kronos_device": "auto",              # "auto", "cuda", "mps", "cpu"
  "kronos_attn_implementation": "sdpa", # "sdpa", "flash_attention_2", "eager"
  "kronos_torch_compile": False,        # Enable torch.compile for Linux CUDA
  "kronos_pred_len": 20,                # Forecast horizon (5, 10, 20 bars)
  "kronos_temperature": 0.0,            # 0.0 for deterministic greedy argmax
  ```
- [ ] **6.2 CLI Command Flags**:
  - Add `--kronos / --no-kronos` flag across `evaluate-signal`, `backtest`, and `analyze`.
  - Add `--kronos-model` (`base`, `small`, `mini`).
- [ ] **6.3 Rich TUI Visualizer Card**:
  - In `cli/display.py`: Format a dedicated Rich panel displaying:
    - 5-Day / 20-Day Neural Return Target.
    - Expected Path Channel (High/Low trajectory).
    - Model Confidence & Hardware Backend (e.g. `CUDA: bfloat16 + SDPA`).

---

### Phase 7: Comprehensive Test Suite & Zero-Leakage Verification

#### Target Files
- `tests/test_kronos_forecast.py` (New comprehensive test suite)
- `tests/test_data_window.py`
- `tests/test_anti_leakage_hardening.py`

#### Tasks & Implementation Details
- [x] **7.1 Unit & Contract Tests (`test_kronos_forecast.py`)**:
  - Causal normalization, future-bar isolation, deterministic output, disk/memory cache, and fallback behavior.
- [x] **7.2 Full Regression Verification**:
  - `720 passed, 1 skipped, 8 warnings, 86 subtests passed`.

---

## 6. Comprehensive File Matrix & Change Signatures

| File | Subsystem | Action | Invariant Maintained |
| :--- | :--- | :--- | :--- |
| `pyproject.toml` | Dependencies | Add optional `[project.optional-dependencies] quant` | Safe Optionality |
| `tradingagents/default_config.py` | Config | Add `kronos_*` configuration keys | Non-breaking Defaults |
| `tradingagents/dataflows/kronos.py` | Quant Core | Create Kronos inference engine, singleton VRAM manager & caching | Zero-Leakage, SDPA/bfloat16 |
| `tradingagents/dataflows/interface.py` | Data Router | Route `get_kronos_forecast` through vendor engine | Single Source of Truth |
| `tradingagents/dataflows/structural_levels.py` | Tech Quant | Integrate neural resistance/support channels | Analytical Confluence |
| `tradingagents/agents/utils/core_stock_tools.py` | LangGraph | Wrap `@tool get_kronos_forecast` | Type Safety & Cleanliness |
| `tradingagents/backtesting/data_window.py` | Backtest | Add 512-bar window to `ALLOWED_LOOKBACKS` | 512 Receptive Field |
| `tradingagents/backtesting/snapshot_provider.py` | Backtest | Support 512-bar causal slicing | Causal Isolation |
| `tradingagents/agents/analysts/market_analyst.py` | Market Analyst | Tool registration & prompt confluence instructions | Separation of Concerns |
| `tradingagents/agents/trader/trader.py` | Trader Agent | Take Profit / Stop Loss neural anchoring | Anti-Prompt Gaming |
| `cli/commands/evaluate.py` | CLI Engine | Pass `--kronos` flag to runner config | CLI Parity |
| `cli/display.py` | CLI UI | Rich UI card for neural trajectory | User Visibility |
| `tests/test_kronos_forecast.py` | Verification | Full unit and regression test suite | 100% Test Pass Rate |
