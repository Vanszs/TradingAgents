# TradingAgents — Limitasi & Roadmap Migrasi ke Real-Time Trading

> **Status workspace saat ini**: Decision support / advisor system untuk **single-shot analysis** per ticker per hari.
> **Bukan**: Auto-trading bot, algo-trader, atau live execution engine.
>
> Dokumen ini menganalisis (a) limitasi spesifik yang membuat sistem belum production-ready untuk live trading dan (b) komponen apa saja yang harus ditambahkan, dimodifikasi, atau diganti untuk handover ke real-time trading.

**Versi codebase yang dianalisis**: TradingAgents v0.2.5 (commit `61522e1`)
**Tanggal analisis**: 2026-05-23

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Limitasi Arsitektural](#2-limitasi-arsitektural)
3. [Limitasi Per Komponen](#3-limitasi-per-komponen)
4. [Limitasi Operasional](#4-limitasi-operasional)
5. [Limitasi Risiko & Compliance](#5-limitasi-risiko--compliance)
6. [Roadmap Migrasi ke Real-Time](#6-roadmap-migrasi-ke-real-time)
7. [Estimasi Effort](#7-estimasi-effort)
8. [Keputusan Desain yang Harus Diambil Tim](#8-keputusan-desain-yang-harus-diambil-tim)

---

## 1. Ringkasan Eksekutif

TradingAgents v0.2.5 adalah **research framework** yang mensimulasikan tim analis trading firm via multi-agent LLM (LangGraph). Output sistem adalah **rekomendasi tertulis** (`Buy`/`Overweight`/`Hold`/`Underweight`/`Sell` + executive summary + thesis), bukan order eksekusi.

Untuk migrasi ke real-time auto-trading, paling minimal dibutuhkan:

| Kategori | Status Saat Ini | Yang Harus Ditambah |
|----------|-----------------|---------------------|
| **State posisi terbuka** | ❌ Tidak ada | Portfolio state machine (cash, positions, margin, exposure) |
| **Risk gate deterministik** | ❌ Hanya LLM debate | Hard limits (position size, sector concentration, daily DD) |
| **Broker integration** | ❌ Tidak ada | Adapter (Alpaca / IBKR / MT5 / dst) |
| **Scheduler / scan loop** | ❌ Manual call | Cron + event-driven trigger |
| **Position monitor** | ❌ Tidak ada | Continuous stop-loss / take-profit watcher |
| **Order management** | ❌ Tidak ada | Order placement, modify, cancel, fill tracking |
| **Audit trail** | ⚠️ Hanya results JSON | Tamper-evident transaction log |
| **Failover & circuit breaker** | ❌ Tidak ada | Health check, auto-pause on anomaly |

**Estimasi effort total**: 2.000 – 5.000 LOC tambahan (bergantung scope), waktu pengerjaan ~2–4 bulan untuk 1 senior dev.

**Realita biaya**: Daily-mode 5 ticker → ~$5–50/hari LLM cost. Intraday hourly → $30–200/hari.

---

## 2. Limitasi Arsitektural

### 2.1 Sistem dirancang untuk **stateless single-shot analysis**

Method utama `TradingAgentsGraph.propagate(ticker, date)` di `tradingagents/graph/trading_graph.py` adalah **fungsi murni dari sisi state portfolio**: tiap pemanggilan menghasilkan rekomendasi independen, tidak tahu posisi yang sudah terbuka atau saldo user.

```python
# main.py — entry sederhana
ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")
# Output: rating string. Tidak ada koneksi ke broker, tidak ada state update.
```

Konsekuensi:
- Kalau dipanggil 2× untuk ticker sama dengan sedikit selisih waktu, dia akan re-analyze dari nol — tidak ingat sudah merekomendasikan apa
- Tidak bisa membedakan "open new position" vs "add to existing" vs "trim partial"
- Tidak bisa membatalkan/modify rekomendasi sebelumnya

### 2.2 `AgentState` (LangGraph state) tidak mengandung informasi portofolio

File: `tradingagents/agents/utils/agent_states.py`

Field yang ada di state:
```python
class AgentState(MessagesState):
    company_of_interest: str
    asset_type: str               # "stock" / "crypto"
    trade_date: str
    sender: str
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    investment_debate_state: InvestDebateState
    investment_plan: str
    trader_investment_plan: str
    risk_debate_state: RiskDebateState
    final_trade_decision: str
    past_context: str             # memory log injection
```

**Yang TIDAK ADA**:
- `cash_balance` / `equity` / `nav`
- `open_positions: dict[ticker -> Position]`
- `pending_orders: list`
- `margin_used` / `margin_available` / `buying_power`
- `realized_pnl` / `unrealized_pnl` / `daily_pnl`
- `position_in_target_ticker` (qty, avg_entry, unrealized)
- `total_exposure_pct` / `sector_exposure`
- `daily_drawdown` / `circuit_breaker_state`

### 2.3 Output `TraderProposal` & `PortfolioDecision` adalah saran tekstual, bukan instruksi mesin

File: `tradingagents/agents/schemas.py`

```python
class TraderProposal(BaseModel):
    action: TraderAction              # Buy / Hold / Sell
    reasoning: str                    # 2-4 kalimat
    entry_price: Optional[float]      # ✅ ada, tapi tidak validated terhadap market
    stop_loss: Optional[float]        # ✅ ada, tapi tidak validated
    position_sizing: Optional[str]    # ⚠️ STRING bebas: "5% of portfolio"

class PortfolioDecision(BaseModel):
    rating: PortfolioRating           # Buy / Overweight / Hold / Underweight / Sell
    executive_summary: str
    investment_thesis: str
    price_target: Optional[float]     # target price, bukan order price
    time_horizon: Optional[str]       # "3-6 months" — string bebas
```

Masalah:
- `position_sizing` tipe **string bebas** → LLM bisa output `"5% of portfolio"`, `"moderate"`, `"aggressive"`, `"100 shares"`, atau apapun
- `time_horizon` juga string bebas — tidak bisa di-parse jadi datetime
- Tidak ada `quantity` (jumlah lot/share) — masih harus dihitung downstream
- Tidak ada `order_type` (market/limit/stop)
- Tidak ada `time_in_force` (DAY/GTC/IOC)
- Tidak ada `bracket_order_params` (linked stop + take profit)

### 2.4 Risk debate adalah **debat tekstual**, bukan risk calculation

File: `tradingagents/agents/risk_mgmt/{aggressive,conservative,neutral}_debator.py`

3 risk debator hanya **berdiskusi** apakah trade plan terlalu agresif/konservatif dengan natural language. Output mereka adalah teks debat, bukan:
- Hitungan VaR (Value at Risk)
- Hitungan max drawdown expected
- Position sizing berdasarkan Kelly Criterion atau fixed fractional
- Correlation analysis dengan posisi existing
- Sector / industry concentration check

LLM bisa **mengabaikan limit risk** kalau prompt tidak ketat — tidak ada hard guarantee.

### 2.5 Tidak ada konsep **trading session / market hours**

Sistem tidak peduli:
- Apakah market open atau closed
- Pre-market vs regular vs after-hours
- Holiday schedule (NYSE close on Memorial Day, dll)
- Timezone handling (NVDA = US market = ET, tapi server bisa di mana saja)

`propagate("NVDA", "2026-05-22")` jalan tanpa cek apakah 22 Mei 2026 adalah trading day atau bukan.

### 2.6 Tidak ada **execution layer** apapun

Meskipun `backtrader>=1.9.78.123` terdaftar di `pyproject.toml`, **tidak ada satupun import `backtrader` di codebase**. Dependency sisa dari rencana yang belum terimplementasi.

```
$ grep -r "import backtrader\|from backtrader" tradingagents/ cli/
(no output)
```

Tidak ada koneksi ke broker live (Alpaca, Interactive Brokers, MetaTrader, OANDA, dst).

---

## 3. Limitasi Per Komponen

### 3.1 `graph/trading_graph.py` — `TradingAgentsGraph`

**Limitasi**:
- Constructor compile graph **sekali**, tapi struktur graph **fixed** (analyst → debate → manager → trader → risk → PM). Tidak bisa diubah jadi *event-driven* tanpa rewire ulang
- `propagate()` adalah method synchronous — block selama 3-10 menit. Tidak ada async / streaming intermediate result ke UI
- Tidak ada hook untuk inject **portfolio state ke prompt agent** (semua agent prompt dibangun di factory, tidak baca state runtime selain dari `AgentState`)

**Untuk real-time**: Constructor harus terima `portfolio_provider` callable yang dipanggil tiap propagate untuk fetch state terbaru.

### 3.2 `graph/setup.py` — `GraphSetup.setup_graph`

**Limitasi**:
- Wiring edge **hard-coded**: `START → analyst1 → analyst2 → analyst3 → analyst4 → Bull → Bear → Manager → Trader → Aggressive → Conservative → Neutral → PortfolioManager → END`
- Loop debate dikontrol oleh `count` di state — kalau LLM lambat, total latensi tidak bisa di-cap
- Tidak ada **early exit**: kalau di tengah jalan diketahui posisi sudah ditutup external (manual oleh user di broker), pipeline tetap jalan sampai END

**Untuk real-time**: Tambahkan node **PortfolioContextLoader** sebagai node pertama (sebelum analyst), yang inject portfolio snapshot ke state. Tambahkan **early-exit conditional edges** yang cek staleness/cancellation flag.

### 3.3 `graph/conditional_logic.py` — `ConditionalLogic`

**Limitasi**:
- Counter `state["investment_debate_state"]["count"]` dan `state["risk_debate_state"]["count"]` di-increment di node bull/bear/risk debator (bukan di router) — kalau LLM lupa increment, loop bisa infinite. `recursion_limit=100` sebagai safety net tapi error-nya kasar
- Routing decision hanya pakai `state` dari pipeline saat ini, tidak baca external signal (e.g., "user manually paused all trading")

**Untuk real-time**: Tambahkan **kill-switch check** di tiap router function:
```python
def should_continue_debate(self, state):
    if KILL_SWITCH.is_active():
        return "END_GRACEFULLY"
    # ... existing logic
```

### 3.4 `graph/propagation.py` — `Propagator`

**Limitasi**:
- `create_initial_state()` hanya tahu `ticker`, `trade_date`, `asset_type`, `past_context`
- Tidak ada parameter untuk inject `current_position`, `portfolio_value`, `risk_budget_remaining`
- `recursion_limit=100` hard-coded → tidak adaptive

**Untuk real-time**: Tambahkan parameter:
```python
def create_initial_state(
    self,
    company_name: str,
    trade_date: str,
    asset_type: str = "stock",
    past_context: str = "",
    portfolio_snapshot: dict = None,    # ← BARU
    risk_budget: dict = None,           # ← BARU (max_position_pct, max_loss, dll)
    market_session: str = "regular",    # ← BARU (pre/regular/post/closed)
):
```

Dan field-field tambahan ini harus diteruskan sebagai context di prompt **trader** dan **portfolio_manager**.

### 3.5 `graph/checkpointer.py` — SqliteSaver wrapper

**Limitasi**:
- 1 SQLite file per ticker — bagus untuk paralelisme, tapi **TIDAK durable di multi-host setup** (file lokal)
- Checkpoint disimpan **per node completion** — kalau crash mid-LLM-call, state setengah jalan tidak tersimpan, harus redo node dari awal
- Tidak ada **TTL / expiry** — file checkpoint menumpuk seiring waktu
- Thread ID = `sha256(ticker:date)[:16]` → kalau jalan 100× per hari per ticker, semua share thread yang sama → checkpoint conflict

**Untuk real-time**:
- Migrasi ke Redis atau Postgres untuk multi-host support
- Tambah **run_id** unik per invocation (e.g., `f"{ticker}:{date}:{uuid.uuid4()}"`)
- Tambah cleanup job (delete checkpoint > 24 jam)

### 3.6 `graph/signal_processing.py` — `SignalProcessor`

**Limitasi**:
- Output cuma **1 dari 5 string**: `Buy/Overweight/Hold/Underweight/Sell`
- Tidak ada confidence score
- Tidak ada conviction level (high/medium/low)
- Tidak ada explanation untuk **kenapa rating berubah** dari run sebelumnya (kalau sebelumnya Buy, sekarang Hold — kenapa?)

**Untuk real-time**: Ganti `parse_rating()` di `agents/utils/rating.py` dengan parser yang juga ekstrak `confidence`, `conviction`, dan `delta_from_previous`.

### 3.7 `graph/reflection.py` — `Reflector`

**Limitasi**:
- Reflection berbasis **5-day holding period hard-coded** (`_fetch_returns(ticker, date, holding_days=5)`)
- Cocok untuk swing trading, **tidak cocok untuk day-trading** (positions opened & closed dalam 1 hari)
- Reflection cuma dipanggil **at the start of next propagate() for same ticker** — kalau ticker tidak pernah dianalisis lagi, reflection tidak pernah dieksekusi
- Tidak ada **batch reflection job** yang resolve semua pending entries periodically

**Untuk real-time**:
- `holding_days` harus di-config per strategy (1 untuk day, 5 untuk swing, 30 untuk position)
- Tambahkan **independent reflection scheduler** yang scan pending entries setiap jam/hari, bukan menunggu next propagate

### 3.8 `agents/utils/memory.py` — `TradingMemoryLog`

**Limitasi**:
- Disimpan sebagai **markdown file** (`~/.tradingagents/memory/trading_memory.md`)
- File-based → tidak atomic untuk concurrent write (kalau 2 ticker selesai bersamaan, race condition)
- Tidak ada index/query — `get_past_context(ticker)` linear scan seluruh file
- Tidak ada concept of **cross-ticker correlation lessons** ("kemarin saya wrong di NVDA dan AMD bersamaan karena sektor semi crash")

**Untuk real-time**: Migrasi ke structured DB (SQLite/Postgres) dengan index per ticker + per sector + per outcome (win/loss).

### 3.9 `dataflows/*` — Data adapters

**Limitasi**:
- **Cuma yfinance + alpha_vantage** untuk market/news/fundamental
- yfinance unofficial (scraping based) — bisa rate-limited atau breaking changes Yahoo tanpa warning
- Tidak ada **realtime quote feed** (websocket) — semua via REST poll
- Latensi data: yfinance lag ~15 menit untuk free tier. Untuk live trading butuh data yang lebih segar
- **StockTwits & Reddit** dipakai sentiment_analyst, tapi:
  - Reddit public JSON limited ke ~10 req/menit per IP
  - StockTwits no auth tapi tidak resmi commercial use
- Tidak ada concept of **market data subscription** (Polygon.io, IEX Cloud, paid Alpha Vantage)
- Tidak ada **historical data cache** beyond yfinance built-in cache

**Untuk real-time**:
- Tambah Polygon.io / IEX Cloud / Alpaca data adapter (low-latency)
- Tambah websocket feed untuk live quote (latensi ms)
- Tambah Redis-cache layer untuk dedup request

### 3.10 `llm_clients/*` — LLM client wrappers

**Limitasi**:
- **Tidak ada response cache** — analisa NVDA hari ini vs besok call LLM dari nol meskipun banyak data sama
- Tidak ada **fallback chain** (kalau OpenAI down, switch ke Anthropic)
- Tidak ada **cost tracking** terintegrasi (cuma `cli/stats_handler.py` yang track basic stats)
- Tidak ada **rate-limit handling** terstandarisasi — kalau provider rate-limited, error langsung naik ke top
- Tidak ada **timeout per node** — 1 LLM call yang gantung bisa stall seluruh pipeline

**Untuk real-time**:
- Tambah Redis-cache untuk response deduplication
- Tambah fallback chain di factory.py
- Tambah cost guard (auto-pause kalau biaya/jam > threshold)
- Tambah per-call timeout (e.g., 60s) dengan retry

---

## 4. Limitasi Operasional

### 4.1 Tidak ada scheduler

**Saat ini**: `propagate()` dipanggil manual via:
```bash
python main.py
# atau
tradingagents  # CLI interactive
```

**Yang dibutuhkan untuk real-time**:
- Cron job atau systemd timer
- Atau event-driven (dipicu oleh news webhook, price alert, dll)
- Atau hybrid (scheduled + event-driven)

### 4.2 Tidak ada concurrent execution per multiple tickers

**Saat ini**: Loop manual untuk multi-ticker:
```python
for ticker in ["NVDA", "AAPL", "MSFT"]:
    ta.propagate(ticker, date)  # serial — total ~30 menit untuk 3 ticker
```

Kalau dijalankan paralel, **share `TradingAgentsGraph` instance tidak thread-safe** (state internal `self.curr_state`, `self.ticker`, `self.log_states_dict`).

**Untuk real-time**: Refactor jadi pure function (tidak share mutable state), atau pakai 1 instance per worker.

### 4.3 Tidak ada monitoring / observability

Tidak ada built-in:
- Prometheus metrics
- Structured logging (cuma `print()` dan `logger.info()`)
- Distributed tracing (OpenTelemetry)
- Alert (latency > threshold, error rate > X)

Untuk live system, ini wajib.

### 4.4 Tidak ada graceful shutdown

`propagate()` di tengah jalan kalau di-interrupt (Ctrl+C, SIGTERM):
- Checkpoint *mungkin* resume kalau `checkpoint_enabled=True`
- Tapi state setengah jalan di memory log bisa corrupt
- Connection ke LLM provider tidak di-close cleanly

**Untuk real-time**: Tambah signal handler, finalizer yang ensure semua write atomic.

### 4.5 Tidak ada deployment story

Tidak ada:
- Dockerfile untuk production (yang ada cuma generic dev)
- Helm chart / k8s manifest
- CI/CD pipeline untuk deploy
- Secret management (.env masih file plain text)
- Health check endpoint

---

## 5. Limitasi Risiko & Compliance

### 5.1 Tidak ada hard risk limit

LLM bisa output rating apapun. Kalau dipakai langsung ke broker tanpa validation:
- Bisa Buy 100% portfolio in 1 ticker (over-concentration)
- Bisa Sell short tanpa cek margin requirement
- Bisa entry tanpa stop-loss

**Wajib**: Deterministic risk gate **di luar LLM**.

### 5.2 Tidak ada audit trail tamper-evident

Saat ini decision disimpan di:
- `~/.tradingagents/logs/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<date>.json`
- `~/.tradingagents/memory/trading_memory.md`

Keduanya **plain file** — bisa di-edit, dihapus, di-overwrite. Untuk regulated environment (jika trading dengan dana orang lain), butuh:
- Append-only log (write-once)
- Cryptographic hash chain
- External backup (S3/cloud)

### 5.3 Tidak ada compliance check

Sistem tidak validate:
- Pattern day trader (PDT) rule (US: <$25k account = max 3 day-trades per 5 hari)
- Wash sale rule (tax)
- Restricted list (insider trading prevention)
- Sanctions list (OFAC, dll)
- Best execution requirement

### 5.4 Tidak ada disclaimer / disclosure

Output sistem tidak mention:
- "This is not investment advice"
- Confidence level / margin of error
- Source of data + timestamp
- LLM model version yang dipakai

Untuk regulated jurisdiction (jika commercial), ini bisa jadi masalah hukum.

---

## 5.5 GAP KRITIS — Decision Logic Tidak Position-Aware

**Observasi tajam yang harus diaddress sebelum apa-apa:**

Rating 5-tier saat ini (`Buy / Overweight / Hold / Underweight / Sell`) **ambigu di context realtime** karena tidak membedakan apakah action itu valid berdasarkan **state posisi saat ini**.

### 5.5.1 Masalah: Same Rating, Different Meaning

```
Rating LLM: "Buy"

Skenario A: User belum punya posisi NVDA
  → Buy = OPEN NEW POSITION ✅ valid

Skenario B: User sudah punya 100 share NVDA, +5% unrealized
  → Buy = ADD TO POSITION (average up) — apakah ini yang user mau?

Skenario C: User sudah punya 500 share NVDA, sudah max sizing 10% portfolio
  → Buy = ❌ Invalid! Sudah max alokasi, harusnya Hold atau Trim
```

```
Rating LLM: "Sell"

Skenario A: User punya 100 share NVDA
  → Sell = CLOSE POSITION ✅ valid

Skenario B: User TIDAK punya posisi NVDA
  → Sell = SHORT SELL? atau SKIP?
  → Default broker: REJECT karena tidak ada posisi untuk dijual
```

```
Rating LLM: "Hold"

Skenario A: User punya posisi → "Hold" = jangan ubah ✅
Skenario B: User tidak punya posisi → "Hold" = ❌ ambigu (Hold what?)
```

### 5.5.2 Solusi: Action Space Conditional pada State

```python
def determine_valid_actions(portfolio_context) -> set[Action]:
    has_position = portfolio_context.current_position_qty > 0
    has_short = portfolio_context.current_position_qty < 0
    at_max_size = portfolio_context.position_pct_of_equity >= MAX_POSITION_PCT

    if not has_position and not has_short:
        return {Action.OPEN_LONG, Action.OPEN_SHORT, Action.SKIP}
    elif has_position and not at_max_size:
        return {Action.ADD_LONG, Action.TRIM_LONG, Action.EXIT_LONG, Action.HOLD}
    elif has_position and at_max_size:
        return {Action.TRIM_LONG, Action.EXIT_LONG, Action.HOLD}
    elif has_short:
        return {Action.ADD_SHORT, Action.TRIM_SHORT, Action.EXIT_SHORT, Action.HOLD}
```

**Key insight**: LLM **TIDAK** boleh dikasih full action space dan diharap output yang valid. Dia harus dikasih **subset action space yang valid** sesuai state, lalu pilih dari subset itu.

---

## 5.6 Detail Gap UX/Logic & File yang Terdampak

### Gap 1 — Prompt LLM tidak tahu state portfolio

**File terdampak**: `agents/trader/trader.py`, `agents/managers/portfolio_manager.py`, `agents/risk_mgmt/*.py`

**Tambalan**: Inject ke prompt:
```
CURRENT POSITION STATE:
- You hold 100 shares NVDA at avg $875.50, +5.2% unrealized
- Position is 7% of equity. Max allowed: 10%
- You can add max 30 more shares before hitting limit

VALID ACTIONS (others will be rejected):
- ADD_LONG (max 30 shares)
- TRIM_LONG (any amount up to 100)
- EXIT_LONG (close all 100)
- HOLD
```

### Gap 2 — Schema masih "rating", bukan "action"

**File terdampak**: `agents/schemas.py`

Replace `PortfolioRating` enum dengan `PortfolioAction` enum (OPEN_LONG, ADD_LONG, TRIM_LONG, EXIT_LONG, HOLD, SKIP, dll). Field tambahan: `quantity` (int, REQUIRED), `stop_loss` (REQUIRED untuk OPEN), `order_type`, `time_in_force`, `confidence` (float).

### Gap 3 — Tidak ada Action Validation Layer

**File baru**: `tradingagents/decision/action_resolver.py`

Dual role:
- **Pre-LLM**: Compute valid actions berdasarkan state, inject ke prompt
- **Post-LLM**: Validate output (action ∈ valid_set, qty within limits, stop_loss present jika required)

### Gap 4 — Risk Gate harus dual-purpose

Yang ada di Phase 1.2: post-validation. Yang dibutuhkan: **juga** pre-filter untuk hitung action space.

### Gap 5 — Tidak ada Position Lifecycle Awareness

Position metadata wajib disimpan: when opened, why opened (thesis), original target, invalidation criteria. Lalu di-inject ke prompt re-evaluation: "Position dibuka 12 hari lalu untuk thesis [X]. Target $950 (sekarang $890, +5%). Apakah thesis masih valid?"

### Gap 6 — Tidak ada SKIP semantic

`HOLD` sekarang ambigu kalau belum ada posisi. Tambah `Action.SKIP` dengan variants:
- `SKIP_TODAY` (re-evaluate besok)
- `SKIP_AND_WATCH` (price alert trigger)
- `SKIP_PERMANENT` (blacklist N days)

### Gap 7 — Edge Case Handling

| Situasi | Sekarang | Yang dibutuhkan |
|---------|----------|------------------|
| LLM Buy, kas tidak cukup | broker reject | Resolver re-prompt dengan available cash |
| Position max, LLM Buy | risk gate reject | Pre-filter, no Buy in valid set |
| LLM Sell, no position | broker reject | Pre-filter, replace dengan Skip atau OpenShort |
| Conflicting analyst signal | LLM bias | Confidence rendah → otomatis Hold |
| Stale data (>5min) | Tetap analisis | Skip ticker, alert |
| Same-day re-analyze | Re-prompt dari nol | "Earlier today: [X]. What changed?" |

### Gap 8 — Tidak ada Action Delta Audit

Kalau hari ini Buy, besok Sell — kenapa? Tambah:
```python
class ActionDelta:
    previous_action, previous_thesis_summary, previous_action_date
    new_action, new_thesis_summary
    what_changed: str
    is_thesis_invalidated: bool
    is_target_reached: bool
    is_stop_triggered: bool
```

---

## 5.7 Matrix Section yang Harus Ditambah/Diedit

| File / Komponen | Status | Perubahan |
|-----------------|--------|-----------|
| `agents/utils/agent_states.py` | **EDIT** | Tambah `portfolio_context` field |
| `agents/schemas.py` | **EDIT** | `PortfolioRating` → `PortfolioAction`, expand fields |
| `agents/trader/trader.py` | **EDIT** | Inject portfolio_context + valid_actions ke prompt |
| `agents/managers/portfolio_manager.py` | **EDIT** | Same |
| `agents/risk_mgmt/{aggressive,conservative,neutral}_debator.py` | **EDIT** | Risk debate vs current portfolio, bukan vacuum |
| `graph/propagation.py` | **EDIT** | `create_initial_state()` terima portfolio_context, valid_actions |
| `graph/trading_graph.py` | **EDIT** | `propagate()` panggil ActionResolver dulu |
| `graph/signal_processing.py` | **EDIT** | Parse PortfolioAction (bukan rating), include qty + stop |
| `agents/utils/structured.py` | **EDIT** | Helper render valid action set ke prompt |
| `agents/utils/memory.py` | **EDIT** | Simpan action + thesis (bukan cuma rating string) |
| `graph/reflection.py` | **EDIT** | Reflection: "thesis valid tapi sizing salah" / "entry harga jelek" |
| `cli/main.py` | **EDIT** | Tampilkan portfolio context, action delta |
| `decision/action_resolver.py` | **NEW** | Pre-LLM action space + post-LLM validation |
| `decision/action_delta.py` | **NEW** | Track change reasoning antar runs |
| `portfolio/state.py` | **NEW** | (sudah disebut di Phase 1.1) |
| `risk/gate.py` | **NEW** | (Phase 1.2 — tambah role: pre-filter action) |
| `broker/adapter.py` | **NEW** | (sudah disebut di Phase 1.3) |

---

## 5.8 Flow End-to-End Sebelum vs Sesudah

**Sebelum (sekarang)**:
```
propagate(ticker, date)
  → analyst reports → bull/bear debate → research manager
  → trader (Buy/Sell/Hold) → risk debate (text) → portfolio manager (rating)
  → return rating
```

**Sesudah (target realtime)**:
```
propagate(ticker, date)
  ├─ FETCH portfolio_snapshot dari PortfolioStateStore           ← NEW
  ├─ COMPUTE valid_actions = ActionResolver.compute(snapshot)    ← NEW
  ├─ CHECK circuit breaker (daily DD, kill switch)               ← NEW
  │   └─ IF active → return SKIP early
  ├─ INJECT portfolio_context + valid_actions ke initial state
  │
  ├─ analyst reports (sama)
  ├─ bull/bear debate (tahu posisi existing)
  ├─ research manager (mention thesis lama jika ada)
  ├─ trader: TraderProposal (action dari valid_actions, qty bukan %)
  ├─ risk debate (debate action vs portfolio existing)
  ├─ portfolio manager: PortfolioDecision (PortfolioAction, qty, stop)
  │
  ├─ VALIDATE output via ActionResolver.validate()               ← NEW
  │   └─ IF invalid → retry 1× dengan corrective context, atau abort
  ├─ COMPUTE ActionDelta vs previous decision                    ← NEW
  ├─ STORE decision + action_delta ke memory log
  └─ RETURN PortfolioDecision (siap dieksekusi)
```

---

## 5.9 UX Pre-Execution (di luar TradingAgents)

Setelah `PortfolioDecision` keluar, sebelum order kirim ke broker:

1. Show **current portfolio state**
2. Show **LLM action recommendation** + confidence
3. Show **action delta** (apa yang berubah dari rekomen lalu?)
4. Show **risk impact preview**:
   - "After this trade: position 12% (was 8%)"
   - "Risk per trade: 0.8% (within 1% policy)"
   - "Sector exposure: Tech 45% (was 38%)"
5. Confirmation gate: Auto / Manual / Paper-only
6. Post-execution tracking: order status, slippage, fill timestamp

---

## 5.10 Test Matrix Wajib (Sebelum Live)

Setiap kombinasi state × action wajib tested:

| Position State | LLM Output | Expected |
|---|---|---|
| No position | OPEN_LONG (qty OK) | ✅ Submit order |
| No position | OPEN_LONG (qty > cash) | ⚠️ Modify qty atau retry |
| No position | OPEN_LONG (no stop_loss) | ❌ Reject — require stop |
| No position | EXIT_LONG | ❌ Invalid — no position |
| No position | SKIP | ✅ Log reason |
| Has long, not max | ADD_LONG | ✅ Submit add-on |
| Has long, at max | ADD_LONG | ❌ Pre-filtered (LLM tidak boleh propose) |
| Has long, at max | TRIM_LONG | ✅ Partial close |
| Has long, at max | HOLD | ✅ No-op |
| Has long | EXIT_LONG | ✅ Submit close, calc P&L |
| Stop_loss hit intraday | (any) | ❌ Position monitor harusnya sudah close sebelum LLM analyze |

Test matrix wajib **3 bulan paper trading** dengan audit log lengkap sebelum real money.

---

## 6. Roadmap Migrasi ke Real-Time

Komponen yang harus dibangun, **diurutkan dari fondasi ke top layer**:

### Phase 1 — Foundation (paling kritis)

#### 1.1 `portfolio_state.py` — Single source of truth untuk posisi

**Lokasi baru**: `tradingagents/portfolio/state.py`

```python
@dataclass
class Position:
    ticker: str
    quantity: int
    avg_entry_price: float
    entry_date: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy_id: str          # which strategy opened this

@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    cash_balance: float
    total_equity: float
    margin_used: float
    margin_available: float
    buying_power: float
    open_positions: dict[str, Position]
    pending_orders: list[Order]
    realized_pnl_today: float
    realized_pnl_total: float
    unrealized_pnl: float
    daily_drawdown_pct: float
    sector_exposure: dict[str, float]   # "Tech": 0.45, "Healthcare": 0.20
    
class PortfolioStateStore:
    """Persistent storage backed by Postgres or Redis."""
    def get_snapshot(self) -> PortfolioSnapshot: ...
    def update_position(self, ticker, position) -> None: ...
    def record_fill(self, fill: Fill) -> None: ...
    def calculate_exposure(self, ticker, qty, price) -> ExposureResult: ...
```

**Estimated LOC**: ~500
**Dependencies**: Postgres / Redis

#### 1.2 `risk_gate.py` — Deterministic guard sebelum order

**Lokasi baru**: `tradingagents/risk/gate.py`

```python
@dataclass
class RiskPolicy:
    max_position_size_usd: float        # contoh: $10,000
    max_position_pct_of_equity: float   # contoh: 10%
    max_sector_pct: float               # contoh: 30%
    max_total_exposure_pct: float       # contoh: 80%
    max_risk_per_trade_pct: float       # contoh: 1% (Kelly-style)
    max_daily_loss_pct: float           # contoh: 3% — circuit breaker
    require_stop_loss: bool             # True untuk semua trade
    blacklisted_tickers: set[str]
    allowed_asset_types: set[str]       # {"stock", "crypto"}

class RiskGate:
    def __init__(self, policy: RiskPolicy, portfolio: PortfolioStateStore):
        self.policy = policy
        self.portfolio = portfolio
    
    def evaluate(self, decision: PortfolioDecision, trader: TraderProposal) -> RiskGateResult:
        """Return APPROVED / REJECTED / MODIFIED with reasons."""
        snapshot = self.portfolio.get_snapshot()
        
        # 1. Position size check
        if self._exceeds_position_limit(trader, snapshot):
            return RiskGateResult.reject("Position size exceeds policy")
        
        # 2. Sector concentration
        if self._exceeds_sector_limit(trader, snapshot):
            return RiskGateResult.reject("Sector concentration exceeds policy")
        
        # 3. Daily drawdown circuit breaker
        if snapshot.daily_drawdown_pct > self.policy.max_daily_loss_pct:
            return RiskGateResult.reject("Daily loss limit hit — no new trades today")
        
        # 4. Stop loss validation
        if self.policy.require_stop_loss and not trader.stop_loss:
            return RiskGateResult.reject("Stop loss required by policy")
        
        # 5. Risk per trade (Kelly Criterion)
        risk_amount = self._calculate_risk(trader)
        if risk_amount > snapshot.total_equity * self.policy.max_risk_per_trade_pct:
            return RiskGateResult.modify(
                new_quantity=self._max_safe_quantity(trader, snapshot),
                reason="Quantity reduced to fit risk per trade limit"
            )
        
        return RiskGateResult.approve()
```

**Estimated LOC**: ~600
**Critical path**: WAJIB di tested unit test 100% coverage

#### 1.3 `broker_adapter.py` — Pluggable broker interface

**Lokasi baru**: `tradingagents/broker/`

```python
class BrokerAdapter(Protocol):
    """Abstract interface — implementations: AlpacaBroker, IBKRBroker, ..."""
    
    async def submit_order(self, order: Order) -> OrderResult: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def modify_order(self, order_id: str, new_params: dict) -> OrderResult: ...
    async def get_positions(self) -> list[Position]: ...
    async def get_account(self) -> AccountInfo: ...
    async def stream_fills(self) -> AsyncIterator[Fill]: ...    # webhook listener
    async def get_market_status(self) -> MarketStatus: ...

class AlpacaBroker(BrokerAdapter):
    def __init__(self, api_key, secret, paper=True):
        self.client = TradingClient(api_key, secret, paper=paper)
    # ... implementation
```

**Estimated LOC**: ~800 (1 broker)
**Recommend**: Mulai dengan Alpaca paper trading (free, easy API)

### Phase 2 — Integration

#### 2.1 Modifikasi `AgentState` — Inject portfolio context

**File**: `tradingagents/agents/utils/agent_states.py`

```python
class PortfolioContext(TypedDict):
    cash_balance: float
    total_equity: float
    margin_available: float
    current_position_qty: int            # untuk ticker yang dianalisis
    current_position_avg_entry: float
    current_unrealized_pnl_pct: float
    sector_exposure_pct: float
    daily_drawdown_pct: float
    risk_budget_remaining_usd: float

class AgentState(MessagesState):
    # ... existing fields ...
    portfolio_context: Annotated[PortfolioContext, "Portfolio snapshot at run start"]
    risk_policy_summary: Annotated[str, "Plain text summary of risk policy for LLM context"]
    market_session: Annotated[str, "pre / regular / post / closed"]
```

#### 2.2 Modifikasi `Propagator.create_initial_state()`

```python
def create_initial_state(
    self,
    company_name, trade_date, asset_type="stock", past_context="",
    portfolio_context: PortfolioContext = None,        # ← BARU
    risk_policy: RiskPolicy = None,                    # ← BARU
    market_session: str = "regular",                   # ← BARU
):
    state = {
        # ... existing ...
        "portfolio_context": portfolio_context or {},
        "risk_policy_summary": risk_policy.to_prompt_text() if risk_policy else "",
        "market_session": market_session,
    }
    return state
```

#### 2.3 Modifikasi prompt Trader & Portfolio Manager

**File**: `tradingagents/agents/trader/trader.py`

Tambah ke prompt:
```python
prompt += f"""
**CURRENT PORTFOLIO STATE FOR {company_name}**:
- You currently hold: {portfolio_context.current_position_qty} shares
- Avg entry: ${portfolio_context.current_position_avg_entry}
- Unrealized P&L: {portfolio_context.current_unrealized_pnl_pct:+.1%}
- Available margin: ${portfolio_context.margin_available:,.0f}
- Risk budget remaining today: ${portfolio_context.risk_budget_remaining_usd:,.0f}

**RISK POLICY**:
{risk_policy_summary}

**ACTIONS YOU CAN PROPOSE**:
- ADD_LONG (open new or add to existing long)
- TRIM_LONG (reduce existing long position)
- EXIT_LONG (close all of existing long)
- HOLD (no action)
- ENTER_SHORT / TRIM_SHORT / EXIT_SHORT (if shorting allowed by policy)

Output specific quantity (in shares), not percentage.
"""
```

#### 2.4 Modifikasi `TraderProposal` schema

```python
class ConcreteAction(str, Enum):
    ADD_LONG = "add_long"
    TRIM_LONG = "trim_long"
    EXIT_LONG = "exit_long"
    HOLD = "hold"
    ENTER_SHORT = "enter_short"
    TRIM_SHORT = "trim_short"
    EXIT_SHORT = "exit_short"

class TraderProposal(BaseModel):
    action: ConcreteAction
    quantity: int                           # ← shares, bukan %
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    limit_price: Optional[float]
    stop_loss: float                        # ← REQUIRED jika policy require_stop_loss
    take_profit: Optional[float]
    time_in_force: Literal["DAY", "GTC", "IOC"]
    reasoning: str
    confidence: float                       # 0.0 — 1.0
```

### Phase 3 — Orchestration

#### 3.1 `scheduler.py` — Trading loop

```python
class TradingLoop:
    def __init__(self, watchlist, ta_graph, risk_gate, broker, portfolio_store):
        self.watchlist = watchlist
        self.ta = ta_graph
        self.risk = risk_gate
        self.broker = broker
        self.portfolio = portfolio_store
        self.kill_switch = KillSwitch()
    
    async def run_daily_premarket(self):
        """Called at 09:00 ET each trading day."""
        if not await self.broker.get_market_status().is_trading_day_today():
            return
        
        if self.kill_switch.is_active():
            return
        
        for ticker in self.watchlist:
            await self._analyze_and_maybe_trade(ticker)
    
    async def _analyze_and_maybe_trade(self, ticker):
        # 1. Fetch current portfolio context
        snapshot = self.portfolio.get_snapshot()
        ctx = build_portfolio_context(snapshot, ticker)
        
        # 2. Run analysis
        try:
            final_state, decision = await self.ta.propagate_async(
                ticker, today_str, portfolio_context=ctx, risk_policy=self.policy
            )
        except Exception as e:
            await alert(f"Analysis failed for {ticker}: {e}")
            return
        
        # 3. Risk gate
        trader_proposal = parse_trader_proposal(final_state["trader_investment_plan"])
        result = self.risk.evaluate(decision, trader_proposal)
        
        if result.is_rejected:
            await audit_log(f"Rejected: {ticker} — {result.reason}")
            return
        
        # 4. Build order
        order = build_order(trader_proposal, result.modifications)
        
        # 5. Submit to broker
        try:
            order_result = await self.broker.submit_order(order)
            await audit_log(f"Submitted: {order_result}")
            await self.portfolio.record_pending_order(order_result)
        except BrokerError as e:
            await alert(f"Broker rejected order: {e}")
```

#### 3.2 `position_monitor.py` — Continuous (deterministic, no LLM)

```python
class PositionMonitor:
    """Runs every 5-30s during market hours.
    Checks stop-loss, take-profit, time-stop, trailing stop."""
    
    async def tick(self):
        snapshot = await self.portfolio.get_snapshot()
        for ticker, pos in snapshot.open_positions.items():
            current_price = await self.broker.get_quote(ticker)
            
            # Hard stop
            if current_price <= pos.stop_loss:
                await self._emergency_exit(ticker, pos, reason="STOP_LOSS_HIT")
            
            # Take profit
            if pos.take_profit and current_price >= pos.take_profit:
                await self._emergency_exit(ticker, pos, reason="TAKE_PROFIT_HIT")
            
            # Trailing stop (optional)
            new_stop = self._calculate_trailing_stop(pos, current_price)
            if new_stop > pos.stop_loss:
                await self._update_stop(ticker, new_stop)
            
            # Time stop (optional, e.g., max holding 30 days)
            if (now() - pos.entry_date).days > MAX_HOLDING_DAYS:
                await self._emergency_exit(ticker, pos, reason="TIME_STOP")
```

#### 3.3 `fill_handler.py` — Webhook receiver

```python
class FillHandler:
    """Listens to broker webhook (atau polling) untuk update state."""
    
    async def on_fill(self, fill: Fill):
        await self.portfolio.record_fill(fill)
        await self.audit_log.append(fill)
        
        # If full close, calculate realized P&L
        if fill.is_close:
            pnl = await self.portfolio.calculate_realized_pnl(fill)
            await self.metrics.record_realized_pnl(pnl)
        
        # Trigger reflection if win/loss
        await self.reflection_queue.enqueue(fill)
```

### Phase 4 — Operations

#### 4.1 Monitoring

- Prometheus metrics endpoint
- Grafana dashboard:
  - LLM call latency p50/p95/p99
  - LLM cost per hour
  - Order success rate
  - P&L realtime
  - Position count
  - Daily drawdown

#### 4.2 Alerting

- Slack/email/PagerDuty integration
- Critical alerts:
  - Daily DD > threshold → auto-pause
  - Broker connection lost
  - LLM provider down (no fallback)
  - Stale data (no quote update > N minutes during market)
  - Order rejected > X consecutive

#### 4.3 Audit Trail

- Append-only log (ClickHouse / S3)
- Immutable record: setiap decision, order, fill, error
- Hash chain untuk tamper-evidence
- Backup harian ke external storage

#### 4.4 Kill Switch

```python
class KillSwitch:
    """Global pause — can be triggered manually or auto."""
    
    @staticmethod
    def activate(reason: str, until: Optional[datetime] = None):
        # Persist to Redis with TTL
        ...
    
    def is_active(self) -> bool:
        return self.store.get("kill_switch") is not None
```

Triggered oleh:
- Manual (panic button di UI)
- Daily DD limit
- N consecutive losing trades
- Broker disconnect
- LLM cost > budget

---

## 7. Estimasi Effort

Asumsi: 1 senior engineer, productivity ~50 LOC/hari (production quality + tests).

| Phase | Komponen | LOC | Effort (hari) |
|-------|----------|-----|---------------|
| 1.1 | PortfolioState + Store | 500 | 10 |
| 1.2 | RiskGate + RiskPolicy | 600 | 12 |
| 1.3 | BrokerAdapter (Alpaca) | 800 | 16 |
| 2.1 | AgentState modifications | 100 | 2 |
| 2.2 | Propagator modifications | 150 | 3 |
| 2.3 | Prompt updates (5 agents) | 300 | 6 |
| 2.4 | Schema modifications | 200 | 4 |
| 3.1 | Scheduler / TradingLoop | 600 | 12 |
| 3.2 | PositionMonitor | 400 | 8 |
| 3.3 | FillHandler | 300 | 6 |
| 4.1 | Monitoring (Prometheus) | 200 | 4 |
| 4.2 | Alerting | 150 | 3 |
| 4.3 | Audit Trail | 400 | 8 |
| 4.4 | Kill Switch | 200 | 4 |
| **Subtotal Code** | | **4,900 LOC** | **98 hari** |
| Integration testing | | - | 20 |
| Paper-trading validation | | - | 30 |
| Documentation | | - | 10 |
| **TOTAL** | | **~5,000 LOC** | **~158 hari (~7.5 bulan part-time, ~4 bulan full-time)** |

**Catatan**: Estimasi tidak termasuk effort untuk multi-broker support, multi-asset (crypto/forex/futures), atau frontend dashboard.

---

## 8. Keputusan Desain yang Harus Diambil Tim

Sebelum mulai implementasi, ini decision points yang harus disepakati:

### 8.1 Trading Frequency

| Mode | Pros | Cons |
|------|------|------|
| **Daily pre-market** (1×/hari) | Murah ($5–50/hari), match dengan paper TradingAgents | Miss intraday opportunity, slow react ke news |
| **Hourly during market** (~7×/hari) | Lebih reaktif | 7× cost, lebih banyak hallucination |
| **Event-driven** (trigger by news/price) | Optimal cost-benefit | Lebih kompleks, butuh news pipeline |
| **Hybrid** (daily + event-driven) | Best of both | Paling kompleks |

**Rekomendasi**: Mulai dengan **daily pre-market**, evaluate, lalu tambah event-driven trigger.

### 8.2 LLM Provider Strategy

| Strategy | Pros | Cons |
|----------|------|------|
| **Single provider (e.g., GPT-4)** | Simple | Single point of failure |
| **Primary + fallback** | Resilient | 2× setup cost |
| **Cost-tiered** (cheap LLM untuk analyst, premium untuk PM) | Hemat | Tuning effort |

**Rekomendasi**: Gunakan default config TradingAgents (deep_think untuk PM, quick_think untuk analyst), tambah fallback chain.

### 8.3 Asset Classes Awal

| Class | Pros | Cons |
|-------|------|------|
| **US Stocks (paper)** | Data lengkap, broker free (Alpaca) | Market hours fixed |
| **Crypto** | 24/7, instant execution | Volatil, paper less mature |
| **Forex** | Liquid, leverage available | Not native to TradingAgents |
| **Futures** | Capital efficient | Margin complex |

**Rekomendasi**: US Stocks paper trading dulu. Setelah stabil 3 bulan, baru migrasi ke real money atau add asset class.

### 8.4 Risk Policy Initial Values

Saran konservatif untuk phase 1:

```python
RiskPolicy(
    max_position_size_usd=1000,         # Max $1k per ticker
    max_position_pct_of_equity=5,        # 5% per ticker
    max_sector_pct=20,                   # 20% per sector
    max_total_exposure_pct=50,           # 50% deployed (50% cash buffer)
    max_risk_per_trade_pct=0.5,          # 0.5% risk per trade (very conservative)
    max_daily_loss_pct=2,                # Auto-pause if down 2% today
    require_stop_loss=True,
    blacklisted_tickers={"GME","AMC","BBBY"},  # avoid memes
    allowed_asset_types={"stock"},
)
```

### 8.5 Failure Modes Strategy

Apa yang harus terjadi jika:

| Scenario | Default Behavior |
|----------|-----------------|
| LLM call timeout | Retry 1×, lalu skip ticker, alert |
| Broker connection lost | Pause new orders, monitor positions, alert |
| Daily DD limit hit | Activate kill switch, close all positions optional |
| Stop loss not filled (gap) | Market order to close, alert on slippage |
| Conflicting signal (Buy + already at max position) | Skip, log reason |
| Stale data (quote > 5 min old) | Skip ticker, alert |

Tim harus sepakati matrix ini sebelum coding.

### 8.6 Audit & Compliance

- Apakah trading dengan dana sendiri atau klien?
- Jurisdiksi mana? (US = SEC/FINRA, ID = OJK, EU = MiFID II)
- Butuh real-time disclosure ke broker/regulator?
- Retention policy untuk log (US: 7 tahun untuk regulated)

Ini menentukan apakah audit trail harus simple (S3) atau enterprise (Snowflake + read-only S3 backup + KMS encryption).

---

## 9. Risiko & Hal yang Tetap Tidak Bisa Diselesaikan dengan TradingAgents

Beberapa fundamental issues yang **tetap ada** meskipun semua komponen di atas dibangun:

1. **LLM bisa salah** — meskipun prompt + schema + risk gate, ada residual risk LLM hallucinate atau biased. Tidak ada cara 100% prevent
2. **Black swan events** — sistem dilatih (via prompt context) di kondisi normal market. Crash 2008 / COVID-2020 / SVB 2023 kemungkinan tidak ter-handle baik
3. **Cost vs benefit at scale** — kalau strategy hasilnya mediocre, biaya LLM bisa eat all alpha
4. **Reproducibility** — LLM output ada randomness; backtest result bisa beda run ke run (mitigasi: pin seed, pin model version, tapi vendor bisa deprecate)
5. **Latensi vs accuracy** — analisis lebih lama = lebih akurat, tapi window opportunity bisa hilang
6. **Concept drift** — paper TradingAgents test di Jan-Mar 2024 (bull market). Performance di bear / sideways market belum diverify

---

## 10. Kesimpulan

**TradingAgents v0.2.5 sebagai brain — siap. Body untuk live execution — belum ada.**

Untuk handover ke real-time trading butuh:
- ~5.000 LOC kode tambahan (foundation + integration + ops)
- ~3-4 bulan full-time development
- Komitmen tim untuk dispatch decision design (frequency, risk policy, broker, asset class)
- Initial capital untuk paper trading validation (1-3 bulan minimal)
- Monitoring & alerting infrastructure
- Compliance review jika commercial / regulated

**Rekomendasi terkuat**: Gunakan TradingAgents **sebagai-adanya** untuk **research dan decision support** (manual confirmation tetap di tangan trader human), sementara membangun execution layer secara paralel di **paper trading** environment selama 3-6 bulan sebelum commit ke real money.

Untuk **personal use di paper trading ringan** (single user, single account, learning), bisa start lebih cepat dengan minimal subset:
- PortfolioState (Redis)
- RiskGate (basic limits)
- AlpacaBroker (paper account)
- TradingLoop (daily, single ticker)
- Modifikasi prompts untuk include portfolio context

Estimasi minimal viable: ~1.500 LOC, ~30 hari kerja.

---

*Dokumen ini dibuat berdasarkan analisis codebase TradingAgents v0.2.5 commit `61522e1` pada 2026-05-23. Untuk update strukturalnya, jalankan ulang `/understand` agar knowledge graph sinkron.*
