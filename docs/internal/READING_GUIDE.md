# TradingAgents — Panduan Membaca Lengkap

> Dokumen ini ditulis untuk dibaca pelan-pelan. Tujuannya: setelah selesai baca, kamu paham project ini secara teknis cukup dalam — bukan sekadar "tahu ada agent A, B, C", tapi paham bagaimana data mengalir, bagaimana state berubah, kenapa ada loop, dan kenapa di-design begini.
>
> Asumsi pengetahuan kamu: Python OK, pernah pakai LangChain, paham konsep RAG, paham API key.

---

## Daftar Isi

1. [Apa Itu TradingAgents](#1-apa-itu-tradingagents)
2. [Stack Teknologi & Kenapa](#2-stack-teknologi--kenapa)
3. [Konsep LangGraph dalam 5 Menit](#3-konsep-langgraph-dalam-5-menit)
4. [Arsitektur Tingkat Tinggi](#4-arsitektur-tingkat-tinggi)
5. [State — Otak Sistem](#5-state--otak-sistem)
6. [Alur Eksekusi Detail](#6-alur-eksekusi-detail)
7. [Bedah Setiap Agent](#7-bedah-setiap-agent)
8. [Conditional Logic — Bagaimana Loop Berhenti](#8-conditional-logic--bagaimana-loop-berhenti)
9. [LLM Layer — Multi-Provider](#9-llm-layer--multi-provider)
10. [Data Layer — Dari Mana Data Datang](#10-data-layer--dari-mana-data-datang)
11. [Memory & Reflection — Belajar dari Trade Sebelumnya](#11-memory--reflection--belajar-dari-trade-sebelumnya)
12. [Konfigurasi & Env Vars](#12-konfigurasi--env-vars)
13. [Map Repo (File-by-File)](#13-map-repo-file-by-file)
14. [Hal-Hal Yang Mudah Salah Paham](#14-hal-hal-yang-mudah-salah-paham)
15. [Glossarium](#15-glossarium)

---

## 1. Apa Itu TradingAgents

TradingAgents adalah simulasi **firma investasi dalam bentuk multi-agent LLM**. Bayangkan firma sungguhan:

- **Analis** mengumpulkan data (market, berita, fundamental, sentimen)
- **Peneliti** berdebat (bull vs bear) untuk membentuk thesis
- **Trader** menerjemahkan thesis jadi rencana eksekusi (entry, stop loss, sizing)
- **Komite risiko** menantang rencana itu (aggressive, conservative, neutral)
- **Portfolio manager** mengambil keputusan akhir

Setiap "orang" di atas → satu **LLM agent** dengan prompt + role spesifik. Mereka berkomunikasi lewat **state** bersama yang dilewati antar node. Outputnya: keputusan trading (Buy/Overweight/Hold/Underweight/Sell + thesis + horizon) untuk satu **ticker** di satu **tanggal**.

**Input:**
```python
ta.propagate("NVDA", "2024-05-10")
```
**Output:** structured decision + 4 reports + debate transcript + trader plan + risk debate.

### Bukan apa
- **Bukan** real-time trading bot (bukan auto-execute order ke broker)
- **Bukan** backtest framework lengkap (tapi punya alpha-vs-SPY untuk reflection)
- **Bukan** RL/training (semua keputusan datang dari prompting + tool use)

### Kenapa multi-agent (bukan satu LLM besar)?
1. **Specialization** — analis teknikal beda prompt dengan analis fundamental. Lebih fokus → output lebih tajam.
2. **Adversarial debate** — bull vs bear memaksa LLM mengeluarkan argumen yang biasanya akan di-skip kalau cuma satu peran. Mengurangi sycophancy.
3. **Hierarchical decision** — analis → researcher → trader → risk → PM. Setiap level filter informasi yang lebih jelek.
4. **Auditability** — kita bisa lihat siapa berkata apa, bukan satu blob "GPT bilang BUY".

---

## 2. Stack Teknologi & Kenapa

| Komponen | Teknologi | Kenapa |
|---|---|---|
| Orchestration | **LangGraph** | Stateful workflow + loop kontrol + conditional edges + checkpoint |
| LLM abstraction | **LangChain** (`langchain_core`) | Tool binding, structured output, message protocol standar |
| LLM providers | OpenAI / Anthropic / Google / Azure / Ollama | Multi-vendor, switchable via env |
| Data: market | **yfinance** (default), Alpha Vantage (opsional) | Free, OHLCV + fundamentals, no key butuh untuk dasar |
| Data: indicators | **stockstats** | RSI, MACD, Bollinger, dst |
| Data: social | StockTwits, Reddit, news APIs | Sentimen retail |
| CLI | **Typer + Rich** | Interactive prompt + live render |
| Persistence | **SqliteSaver** (LangGraph) | Checkpoint per ticker untuk crash-resume |
| Schema | **Pydantic** | Structured output dari LLM (decisions) |
| Package mgr | **uv** (lihat `uv.lock`) | Cepat, deterministic |

### Konsep LangChain yang sudah kamu tahu (refresh cepat)
- `ChatPromptTemplate` — template prompt dengan placeholder
- `MessagesPlaceholder` — slot untuk inject conversation history
- `llm.bind_tools(tools)` — kasih LLM kemampuan call tools
- `with_structured_output(Schema)` — paksa output sesuai Pydantic schema
- `chain = prompt | llm` — pipe operator (LCEL)

Kalau yang di atas familiar, kamu sudah 60% siap baca kodenya.

---

## 3. Konsep LangGraph dalam 5 Menit

LangChain bagus untuk chain linear (A → B → C). Tapi kalau butuh:
- Loop ("call tool sampai cukup")
- Branching ("kalau bull, lanjut ke bear; kalau debate selesai, ke manager")
- Shared state yang mutable
- Persistence/resume

…itu domain **LangGraph**.

### Building blocks LangGraph

**1. State** — `TypedDict` (atau Pydantic) yang menyimpan semua data berjalan.
```python
class AgentState(MessagesState):
    company_of_interest: str
    market_report: str
    # ...
```
Setiap node menerima state, return dict partial yang di-merge ke state.

**2. Node** — fungsi `state -> dict`. Bisa LLM agent, tool, atau plain function.
```python
def market_analyst_node(state):
    # ... call LLM ...
    return {"market_report": report, "messages": [result]}
```

**3. Edge** — koneksi antar node.
- `add_edge(A, B)` — selalu A → B
- `add_conditional_edges(A, fn, mapping)` — fn(state) return label, label dipetakan ke node next

**4. ToolNode (prebuilt)** — node yang otomatis execute tool calls dari pesan terakhir.
```python
ToolNode([get_stock_data, get_indicators])
```

**5. Compile** → `graph.invoke(state)` jalanin sampai END.

### Pattern penting di TradingAgents

**Pattern A — Tool-calling loop:**
```
Analyst → (ada tool_calls?) ─yes→ ToolNode → balik ke Analyst
                            ─no──→ Msg Clear → next node
```
Conditional edge cek `last_message.tool_calls`. Selama LLM mau pakai tool, loop. Begitu LLM sudah cukup data dan jawab tanpa tool_call, lanjut.

**Pattern B — Debate loop (counter-based):**
```
Bull ⇄ Bear (sampai count >= 2 * max_rounds) → Research Manager
```
Conditional cek `state["investment_debate_state"]["count"]`. Setiap turn naikin count.

**Pattern C — 3-way rotation:**
```
Aggressive → Conservative → Neutral → Aggressive ... → Portfolio Manager
```
Conditional cek `latest_speaker` untuk rotasi, dan `count >= 3 * max_rounds` untuk berhenti.

Tiga pattern ini = inti TradingAgents. Begitu kamu paham 3 ini, sisanya cuma detail.


---

## 4. Arsitektur Tingkat Tinggi

```
                              INPUT
                       ticker + trade_date
                              │
                              ▼
   ┌──────────────────── ANALYST LAYER ────────────────────┐
   │                                                       │
   │  Market Analyst   ⇄ tools(get_stock_data, indicators) │
   │       │                                               │
   │       ▼                                               │
   │  Sentiment Analyst (no tools, pre-fetched data)       │
   │       │                                               │
   │       ▼                                               │
   │  News Analyst     ⇄ tools(get_news, get_global_news,  │
   │       │                    get_insider_transactions)  │
   │       ▼                                               │
   │  Fundamentals     ⇄ tools(financials, balance,        │
   │  Analyst                   cashflow, income)          │
   │       │                                               │
   └───────┼───────────────────────────────────────────────┘
           │ produces: market_report, sentiment_report,
           │           news_report, fundamentals_report
           ▼
   ┌──────────────────── RESEARCH LAYER ───────────────────┐
   │                                                       │
   │     Bull Researcher  ⇄  Bear Researcher  (debate)     │
   │                  │                                    │
   │                  ▼                                    │
   │           Research Manager (judges debate)            │
   │                                                       │
   └───────────────────────┬───────────────────────────────┘
                           │ produces: investment_plan
                           ▼
   ┌──────────────────── EXECUTION LAYER ──────────────────┐
   │                                                       │
   │           Trader (proposes entry/exit/sizing)         │
   │                                                       │
   └───────────────────────┬───────────────────────────────┘
                           │ produces: trader_investment_plan
                           ▼
   ┌──────────────────── RISK LAYER ───────────────────────┐
   │                                                       │
   │   Aggressive  →  Conservative  →  Neutral  (rotation) │
   │                       │                               │
   │                       ▼                               │
   │              Portfolio Manager (final)                │
   │                                                       │
   └───────────────────────┬───────────────────────────────┘
                           │ produces: final_trade_decision
                           ▼
                         OUTPUT
                  + reflection → memory log
```

**Pengamatan penting:**
1. Ada **4 layer** sequential: analyst → research → execution → risk
2. Di dalam layer ada **loop** (tools, debate, rotation)
3. Setiap layer **memperkaya state** — analyst nambah report, research nambah plan, dll
4. **2 LLM tier**: `quick_thinking_llm` untuk analyst/debater (banyak panggil, perlu murah), `deep_thinking_llm` untuk Research Manager + Portfolio Manager (sedikit panggil, perlu pintar)

### File-file utama yang merakit ini
| File | Tugas |
|---|---|
| `tradingagents/graph/trading_graph.py` | Class `TradingAgentsGraph` — orchestrator (init LLM, build graph, jalankan, log) |
| `tradingagents/graph/setup.py` | Class `GraphSetup` — kerjaan utama: pasang nodes & edges di `StateGraph` |
| `tradingagents/graph/conditional_logic.py` | Class `ConditionalLogic` — fungsi-fungsi `should_continue_*` |
| `tradingagents/graph/propagation.py` | Class `Propagator` — bikin initial state + invoke args |
| `tradingagents/graph/checkpointer.py` | SqliteSaver wrapper (resume) |
| `tradingagents/graph/reflection.py` | Class `Reflector` — generate lesson dari hasil trade |
| `tradingagents/graph/signal_processing.py` | Class `SignalProcessor` — wrapper tipis di atas `parse_rating()`. Accepts `quick_thinking_llm` arg untuk back-compat tapi **tidak lagi pakai LLM** (structured output PM guarantees parseable rating) |
| `tradingagents/graph/analyst_execution.py` | Builder: `build_analyst_execution_plan()` → `AnalystExecutionPlan` (list of `AnalystNodeSpec` + concurrency_limit). Juga berisi `AnalystWallTimeTracker` untuk CLI timing display |

---

## 5. State — Otak Sistem

State adalah **kontainer tunggal** yang dilewati antar node. Semua agent baca dari sini, semua agent tulis ke sini. File: `tradingagents/agents/utils/agent_states.py`.

### `AgentState` (state utama)

Inherit dari `MessagesState` (sudah punya `messages: list[BaseMessage]` bawaan LangGraph):

```python
class AgentState(MessagesState):
    company_of_interest: str       # "NVDA"
    asset_type: str                # "stock" atau "crypto" — mempengaruhi prompt wording
    trade_date: str                # "2024-05-10"
    sender: str                    # nama agent terakhir yang nulis (hanya diset oleh Trader: "Trader")

    # Hasil layer 1 (analyst)
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str

    # Layer 2 (research debate)
    investment_debate_state: InvestDebateState  # nested dict
    investment_plan: str           # output Research Manager (rendered markdown)

    # Layer 3 (execution)
    trader_investment_plan: str    # output Trader (rendered markdown)

    # Layer 4 (risk debate)
    risk_debate_state: RiskDebateState          # nested dict
    final_trade_decision: str      # output Portfolio Manager (rendered markdown)

    # Memory dari run-run sebelumnya
    past_context: str              # same-ticker decisions + cross-ticker lessons
```

### `InvestDebateState` (state debate bull/bear)

```python
class InvestDebateState(TypedDict):
    bull_history: str      # akumulasi semua statement bull
    bear_history: str      # akumulasi semua statement bear
    history: str           # gabungan (kronologis)
    current_response: str  # response terakhir (untuk routing)
    judge_decision: str    # keputusan Research Manager
    count: int             # jumlah turn (untuk stop)
```

### `RiskDebateState` (3-way: aggressive/conservative/neutral)

```python
class RiskDebateState(TypedDict):
    aggressive_history: str
    conservative_history: str
    neutral_history: str
    history: str
    latest_speaker: str    # "Aggressive" / "Conservative" / "Neutral" — untuk rotasi
    current_aggressive_response: str
    current_conservative_response: str
    current_neutral_response: str
    judge_decision: str
    count: int
```

### Cara node update state

Node return dict partial — LangGraph **merge** ke state.

```python
def market_analyst_node(state):
    # ... process ...
    return {
        "messages": [result],          # append ke messages list
        "market_report": report,       # overwrite market_report
    }
```

Yang di-return saja yang di-update. Field lain tetap.

### Pertanyaan yang sering muncul

**Q: Kenapa `messages` jadi satu list global, bukan per-agent?**
A: Karena tool-calling loop butuh akses ke `last_message.tool_calls`. Setelah analyst selesai, ada **Msg Clear** node yang: (1) menghapus semua messages via `RemoveMessage(id=m.id)` untuk setiap message, dan (2) menambahkan placeholder `HumanMessage(content="Continue")` untuk kompatibilitas Anthropic (yang membutuhkan minimal 1 human message). Lihat `create_msg_delete()` di `agent_utils.py`.

**Q: Bagaimana agent baca `market_report` di Bull Researcher?**
A: Bull Researcher read `state["market_report"]` langsung. Dia tidak pakai `messages` — message protocol cuma untuk tool-calling loop di analyst.

**Q: Apakah state thread-safe?**
A: LangGraph default sequential. Kalau `analyst_concurrency_limit > 1`, perlu hati-hati. Tapi default = 1.


---

## 6. Alur Eksekusi Detail

Mari trace satu run dari `python main.py` (ticker NVDA, tanggal 2024-05-10).

### Step 0 — Load config
`main.py`:
```python
config = DEFAULT_CONFIG.copy()
ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")
```

`DEFAULT_CONFIG` (di `tradingagents/default_config.py`) berisi:
- `llm_provider`: "openai"
- `deep_think_llm`: "gpt-5.4" (atau model lain via env)
- `quick_think_llm`: "gpt-5.4-mini"
- `max_debate_rounds`: 1
- `max_risk_discuss_rounds`: 1
- `data_vendors`: yfinance untuk semua kategori
- ... dan banyak lagi

Env override: `TRADINGAGENTS_LLM_PROVIDER=anthropic` di `.env` akan override `llm_provider`. Tabel mapping ada di `_ENV_OVERRIDES` di file yang sama.

### Step 1 — `TradingAgentsGraph.__init__`
Yang terjadi:
1. **Buat 2 LLM client** (deep + quick) via `create_llm_client(provider, model, ...)` (factory di `tradingagents/llm_clients/factory.py`)
2. **Buat `TradingMemoryLog`** untuk baca/tulis memory file
3. **Buat tool nodes** per kategori (market/social/news/fundamentals) — tiap kategori dapat list tool yang sesuai
4. **Buat `ConditionalLogic`** dengan `max_debate_rounds`, `max_risk_discuss_rounds`
5. **Buat `GraphSetup`** dan panggil `setup_graph(selected_analysts)` → return `StateGraph` (workflow)
6. **Compile** → `self.graph = self.workflow.compile()` (siap di-invoke)

### Step 2 — `propagate("NVDA", "2024-05-10")`
1. **Resolve pending memory entries** untuk NVDA dari run-run sebelumnya:
   - Baca file memory log
   - Untuk tiap entry "pending", fetch return aktual dari yfinance (ticker close N hari setelah trade_date)
   - Hitung alpha vs benchmark (SPY untuk US, ^N225 untuk Tokyo, dst — lihat `benchmark_map`)
   - Generate "lesson" via `Reflector` (LLM call)
   - Simpan kembali ke memory log dengan status "resolved"
2. **(Optional) Setup checkpointer** kalau `checkpoint_enabled=True`
3. **Bikin initial state** via `Propagator.create_initial_state(...)`:
   ```python
   {
       "messages": [("human", company_name)],  # seed message
       "company_of_interest": "NVDA",
       "asset_type": "stock",
       "trade_date": "2024-05-10",
       "past_context": "<isi memory log relevan>",
       "investment_debate_state": InvestDebateState({...semua kosong, count=0}),
       "risk_debate_state": RiskDebateState({...semua kosong, count=0}),
       "market_report": "",
       "fundamentals_report": "",
       "sentiment_report": "",
       "news_report": "",
   }
   ```
   Catatan: `sender`, `investment_plan`, `trader_investment_plan`, `final_trade_decision` TIDAK di-set di initial state — mereka akan di-set oleh node masing-masing saat dieksekusi.
4. **Run graph** — `self.graph.invoke(init_state, **args)` (atau `.stream()` kalau `debug=True`)

### Step 3 — Eksekusi graph (di mana keajaiban terjadi)

LangGraph eksekusi dimulai dari `START`:

**3.1 Market Analyst** (entry point pertama, sesuai urutan `selected_analysts`)
- Prompt template: "kamu trading assistant, pilih max 8 indikator, panggil `get_stock_data` dulu lalu `get_indicators`"
- LLM lihat state, mungkin generate `tool_calls=[get_stock_data(NVDA, ...)]`
- Conditional `should_continue_market(state)`:
  - Cek `last_message.tool_calls` ada → return `"tools_market"` (label)
  - Tidak ada → return `"Msg Clear Market"`
- Di `setup.py`, conditional edges untuk analyst menggunakan **list format** (bukan dict mapping):
  ```python
  workflow.add_conditional_edges(analyst_node, should_continue_fn, [tool_node, clear_node])
  ```
  LangGraph match return value ke node name di list.
- Kalau ada tool_calls → ke node `tools_market` (ToolNode jalanin tool, hasilnya jadi ToolMessage di `messages`)
- Edge `tools_market → Market Analyst` (loop balik)
- LLM panggil lagi, mungkin `get_indicators(...)`, terus sampai LLM cukup data dan generate report sebagai content (no more tool_calls)
- Sekarang conditional return `"Msg Clear Market"`
- **Msg Clear Market**: hapus messages dari list (supaya analyst berikutnya start fresh)
- `state["market_report"]` sudah terisi report

**3.2 Sentiment Analyst** (sequential setelah Market)
- Tidak pakai tool-calling. Pre-fetch data Yahoo News + StockTwits + Reddit di awal node via `get_news.func()`, `fetch_stocktwits_messages()`, `fetch_reddit_posts()`, inject langsung ke prompt sebagai structured blocks
- Single LLM call (tanpa `bind_tools`) → `state["sentiment_report"]`
- Catatan: meskipun ada `tools_social` ToolNode terdaftar di graph (berisi `get_news`), sentiment analyst tidak pernah generate tool_calls karena LLM-nya tidak di-bind dengan tools. Conditional `should_continue_social` selalu langsung return `"Msg Clear Sentiment"`

**3.3 News Analyst** (tool-calling, mirip Market)
- Tools: `get_news`, `get_global_news`, `get_insider_transactions`
- Loop sampai cukup → `state["news_report"]`

**3.4 Fundamentals Analyst** (tool-calling)
- Tools: `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`
- Loop sampai cukup → `state["fundamentals_report"]`

**3.5 Bull Researcher** (entry research layer)
- Baca `state["market_report"]`, `sentiment_report`, `news_report`, `fundamentals_report`
- Baca `state["investment_debate_state"]["history"]` (kosong di run pertama)
- Generate argumen bullish, append ke `bull_history` dan `history`
- Set `current_response = "Bull Analyst: ..."`, naikin `count`
- Conditional `should_continue_debate` (dipasang di KEDUA node Bull dan Bear):
  - Kalau `count >= 2 * max_debate_rounds` → `Research Manager`
  - Kalau `current_response` mulai dengan "Bull" (match "Bull Analyst: ...") → `Bear Researcher`
  - Else → `Bull Researcher`

**3.6 Bear Researcher**
- Sama tapi side berlawanan, append ke `bear_history`
- Set `current_response = "Bear Analyst: ..."`
- Routing: conditional `should_continue_debate` yang sama dipasang di Bear node juga. Karena response mulai "Bear" (bukan "Bull"), routing ke `Bull Researcher` (sampai count cukup)

**3.7 Research Manager** (deep_thinking_llm)
- Baca `investment_debate_state["history"]` (full debate transcript)
- Judge debate menggunakan structured output (`ResearchPlan` schema via `invoke_structured_or_freetext`)
- Output: `state["investment_plan"]` (rendered markdown) + update `judge_decision` dan `current_response` di `investment_debate_state`
- Catatan: Research Manager TIDAK baca 4 reports langsung — dia hanya baca debate history yang sudah mengandung referensi ke reports dari bull/bear

**3.8 Trader** (quick_thinking_llm)
- Baca `investment_plan` + `company_of_interest`
- Output via structured output (`TraderProposal` schema): action (Buy/Hold/Sell), reasoning, entry_price, stop_loss, position_sizing
- Tulis ke `state["trader_investment_plan"]` (rendered markdown) + `state["sender"] = "Trader"` + append AIMessage ke `messages`

**3.9 Risk Debate (3-way)**
- **Aggressive Analyst**: pro high-reward, push trader_plan ke arah lebih agresif
- Conditional `should_continue_risk_analysis`:
  - `count >= 3 * max_risk_discuss_rounds` → Portfolio Manager
  - `latest_speaker = "Aggressive"` → Conservative
  - `latest_speaker = "Conservative"` → Neutral
  - Else → Aggressive
- **Conservative**: protect capital, push ke arah lebih hati-hati
- **Neutral**: moderate, challenge keduanya
- Rotation: A → C → N → A → C → N ... sampai count cukup

**3.10 Portfolio Manager** (deep_thinking_llm — final boss)
- Baca: `risk_debate_state["history"]` + `investment_plan` + `trader_investment_plan` + `past_context`
- Output structured `PortfolioDecision` via `invoke_structured_or_freetext`: rating (5-tier: Buy/Overweight/Hold/Underweight/Sell), executive_summary, investment_thesis, price_target, time_horizon
- Tulis ke `state["final_trade_decision"]` (rendered markdown) + update `risk_debate_state["judge_decision"]` + set `latest_speaker = "Judge"`

**3.11 END** — graph selesai

### Step 4 — Post-processing
1. **Log state** ke disk: `~/.tradingagents/logs/NVDA/TradingAgentsStrategy_logs/full_states_log_2024-05-10.json`
2. **Store decision** ke memory log dengan status "pending" (return aktual belum bisa dihitung sekarang)
3. **Process signal**: extract rating dari `final_trade_decision` markdown via heuristic parser (`parse_rating` di `agents/utils/rating.py`) — **bukan LLM call**, murni regex 2-pass. `SignalProcessor.process_signal` hanya wrapper tipis di atas `parse_rating`.
4. Return `(final_state, decision)` — decision adalah string rating: "Buy"/"Overweight"/"Hold"/"Underweight"/"Sell"

---

## 7. Bedah Setiap Agent

Semua agent ada di `tradingagents/agents/`. Pattern umum:

```python
def create_<role>_agent(llm):
    def <role>_node(state):
        # 1. Bangun prompt dari state
        # 2. (Optional) bind tools
        # 3. Invoke LLM
        # 4. Return dict update untuk state
        return {...}
    return <role>_node
```

`create_*` adalah **factory** — return closure yang capture `llm`. Pattern ini supaya graph wiring bisa swap LLM tanpa rewrite agent.

### Utility Functions di `agent_utils.py`

File ini bukan hanya re-export tools, tapi juga berisi helper functions penting:

| Function | Kegunaan |
|---|---|
| `create_msg_delete()` | Factory yang return node function untuk clear messages. Menghapus semua messages via `RemoveMessage` + tambah placeholder `HumanMessage("Continue")` untuk Anthropic compatibility |
| `get_language_instruction()` | Return `" Write your entire response in {lang}."` kalau `output_language` bukan English, else empty string. Dipanggil oleh SEMUA agent |
| `build_instrument_context(ticker, asset_type)` | Return string instruksi agar agent preserve exact ticker (termasuk exchange suffix). Untuk crypto, tambah hint "treat as crypto asset" |

### 7.1 Market Analyst (`agents/analysts/market_analyst.py`)
- **Tools**: `get_stock_data`, `get_indicators`
- **Job**: pilih ≤8 indikator teknikal yang relevan, fetch CSV, fetch indikator, tulis report dengan tabel markdown
- **Prompt highlight**: panjang, menjelaskan setiap indikator (RSI, MACD, Bollinger, ATR, VWMA, ...). Memandu LLM pilih yang **non-redundan** (jangan pilih RSI dan StochRSI bersamaan).
- **Output**: `state["market_report"]` (string markdown)

### 7.2 Sentiment Analyst (`agents/analysts/sentiment_analyst.py`)
- **Tools**: tidak ada (data pre-fetched di awal node)
- **Job**: baca data dari Yahoo Finance news + StockTwits + Reddit (sudah di-inject ke prompt), simpulkan sentimen retail
- **Kenapa tidak tool-calling?**: karena data sosial perlu agregasi/limit (tidak make sense LLM panggil-panggil terus). Lebih efisien fetch sekali, kasih ke LLM.
- **Output**: `state["sentiment_report"]`

### 7.3 News Analyst (`agents/analysts/news_analyst.py`)
- **Tools**: `get_news` (ticker-specific), `get_global_news` (macro), `get_insider_transactions`
- **Job**: rangkum berita relevan, identify catalyst (earnings, guidance, lawsuit, dll)
- **Output**: `state["news_report"]`

### 7.4 Fundamentals Analyst (`agents/analysts/fundamentals_analyst.py`)
- **Tools**: `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`
- **Job**: analisis valuasi (P/E, P/B, profit margin, debt ratio, FCF growth)
- **Output**: `state["fundamentals_report"]`

### 7.5 Bull Researcher (`agents/researchers/bull_researcher.py`)
- **Job**: advokat sisi bullish. Baca 4 reports + history debate, bantah argumen bear, tambah argumen pro-buy
- **Tidak pakai tools** — murni reasoning di atas data yang sudah ada
- **Update state**: `bull_history`, `history`, `current_response`, `count++`

### 7.6 Bear Researcher (`agents/researchers/bear_researcher.py`)
- **Job**: kebalikan bull. Cari risiko, counter argumen bull
- **Update state**: `bear_history`, `history`, `current_response`, `count++`

### 7.7 Research Manager (`agents/managers/research_manager.py`)
- **LLM**: `deep_thinking_llm` (penting, perlu nuansa)
- **Job**: baca debate transcript (`investment_debate_state["history"]`), putuskan thesis
- **Tidak baca 4 reports langsung** — hanya debate history (yang sudah mengandung referensi ke reports dari bull/bear arguments)
- **Structured output**: `ResearchPlan` schema (recommendation, rationale, strategic_actions) via `invoke_structured_or_freetext`
- **Output**: `state["investment_plan"]` (rendered markdown) + update `investment_debate_state` (set `judge_decision`, `current_response`)

### 7.8 Trader (`agents/trader/trader.py`)
- **LLM**: `quick_thinking_llm`
- **Job**: terjemahkan thesis → rencana eksekusi konkret
- **Structured output**: `TraderProposal` schema (action: Buy/Hold/Sell, reasoning, entry_price, stop_loss, position_sizing) via `invoke_structured_or_freetext`
- **Output**: `state["trader_investment_plan"]` (rendered markdown) + `state["sender"] = "Trader"` + AIMessage appended ke messages
- **Pattern unik**: menggunakan `functools.partial` untuk inject `name="Trader"` ke node function

### 7.9 Aggressive / Conservative / Neutral Debators (`agents/risk_mgmt/*.py`)
- **Aggressive**: cari peluang lebih besar, kritisi sizing terlalu kecil/stop terlalu ketat
- **Conservative**: protect capital, kritisi sizing terlalu besar/stop terlalu jauh
- **Neutral**: balance, challenge ekstrem
- **Update state**: histori masing-masing + `latest_speaker` + `count++`

### 7.10 Portfolio Manager (`agents/managers/portfolio_manager.py`)
- **LLM**: `deep_thinking_llm`
- **Job**: final decision — synthesize risk debate + research plan + trader plan + past lessons
- **Reads**: `risk_debate_state["history"]`, `investment_plan`, `trader_investment_plan`, `past_context`
- **Structured output**: `PortfolioDecision` schema (rating, executive_summary, investment_thesis, price_target, time_horizon) via `invoke_structured_or_freetext`
- **Output**: `state["final_trade_decision"]` (rendered markdown) + update `risk_debate_state` (set `judge_decision`, `latest_speaker = "Judge"`)

### 7.11 Schemas (`agents/schemas.py`)
File ini berisi semua Pydantic schemas untuk structured output:
- `ResearchPlan` — output Research Manager (fields: `recommendation`, `rationale`, `strategic_actions`)
- `TraderProposal` — output Trader (fields: `action`, `reasoning`, `entry_price`, `stop_loss`, `position_sizing`)
- `PortfolioDecision` — output Portfolio Manager (fields: `rating`, `executive_summary`, `investment_thesis`, `price_target`, `time_horizon`)

Plus renderer functions (`render_research_plan`, `render_trader_proposal`, `render_pm_decision`) yang mengubah Pydantic → markdown untuk storage dan downstream consumption.

### 7.12 Structured Output Helper (`agents/utils/structured.py`)

File ini menyediakan 2 fungsi utama yang dipakai oleh Research Manager, Trader, dan Portfolio Manager:

```python
def bind_structured(llm, schema, agent_name) -> Optional[Any]:
    """Wrap LLM with with_structured_output(schema).
    Return None jika provider tidak support (e.g. older Ollama models)."""

def invoke_structured_or_freetext(structured_llm, plain_llm, prompt, render, agent_name) -> str:
    """Coba structured call dulu, render hasilnya ke markdown.
    Kalau gagal (malformed JSON, transient error), fallback ke plain llm.invoke().
    Pipeline tidak pernah block karena structured output failure."""
```

Pattern canonical di ketiga agent:
1. Saat factory (`create_*`): `structured_llm = bind_structured(llm, Schema, "Agent Name")`
2. Saat invocation: `result = invoke_structured_or_freetext(structured_llm, llm, prompt, render_fn, "Agent Name")`

### 7.13 Rating Parser (`agents/utils/rating.py`)

File ini berisi:
- `RATINGS_5_TIER`: tuple `("Buy", "Overweight", "Hold", "Underweight", "Sell")`
- `parse_rating(text, default="Hold")`: heuristic parser yang extract rating dari prose text. Dua-pass: (1) cari "Rating: X" label, (2) fallback ke first rating word di text. **Tidak pakai LLM** — murni regex.

Dipakai oleh `SignalProcessor.process_signal()` dan `TradingMemoryLog.store_decision()`.


---

## 8. Conditional Logic — Bagaimana Loop Berhenti

File: `tradingagents/graph/conditional_logic.py`. Class `ConditionalLogic` punya beberapa metode `should_continue_*`:

### Tool-calling (per analyst kategori)

```python
def should_continue_market(self, state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools_market"      # ada tool call → eksekusi tool
    return "Msg Clear Market"      # tidak ada → lanjut
```

Sama untuk `should_continue_social/news/fundamentals`, hanya beda label.

**Mekanisme**: LLM secara natural berhenti generate tool_calls ketika sudah cukup data untuk menjawab. Prompt mengatakan "tulis report" — begitu LLM menulis report sebagai content (bukan tool call), loop selesai.

### Debate (bull/bear)

```python
def should_continue_debate(self, state) -> str:
    if state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds:
        return "Research Manager"
    if state["investment_debate_state"]["current_response"].startswith("Bull"):
        return "Bear Researcher"
    return "Bull Researcher"
```

- `count >= 2 * max_rounds` → selesai (kalikan 2 karena 1 round = bull + bear)
- Kalau terakhir Bull bicara → giliran Bear, dan sebaliknya

**Penting**: fungsi ini dipasang sebagai conditional edge di **KEDUA** node (Bull Researcher DAN Bear Researcher). Di `setup.py`:
```python
workflow.add_conditional_edges("Bull Researcher", should_continue_debate, {"Bear Researcher": ..., "Research Manager": ...})
workflow.add_conditional_edges("Bear Researcher", should_continue_debate, {"Bull Researcher": ..., "Research Manager": ...})
```

Default `max_debate_rounds = 1` → 2 turn total (1 bull, 1 bear). Kalau set ke 3 → 6 turn.

### Risk debate (3-way rotation)

```python
def should_continue_risk_analysis(self, state) -> str:
    if state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds:
        return "Portfolio Manager"
    if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
        return "Conservative Analyst"
    if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
        return "Neutral Analyst"
    return "Aggressive Analyst"
```

- `count >= 3 * max_rounds` (kalikan 3 karena 1 round = A+C+N)
- Rotasi: A → C → N → A → C → N → ...
- Default `max_risk_discuss_rounds = 1` → 3 turn total

### Kenapa pattern ini elegan
Conditional functions adalah **plain Python**. Tidak ada DSL, tidak ada YAML aneh. Mau ubah aturan? Edit fungsi. Mau debug? `print(state)` di awal fungsi.

LangGraph tinggal panggil fungsi ini setelah node selesai, baca return value (string label), cocokkan ke mapping di `add_conditional_edges`, jalankan node berikutnya.

### Detail: `analyst_execution.py` — Bagaimana Analyst Dirangkai

File ini adalah **blueprint** untuk analyst layer. Konsep utama:

**`AnalystNodeSpec`** — dataclass yang mendefinisikan satu analyst:
```python
@dataclass(frozen=True)
class AnalystNodeSpec:
    key: str          # "market", "social", "news", "fundamentals"
    agent_node: str   # "Market Analyst", "Sentiment Analyst", ...
    clear_node: str   # "Msg Clear Market", "Msg Clear Sentiment", ...
    tool_node: str    # "tools_market", "tools_social", ...
    report_key: str   # "market_report", "sentiment_report", ...
```

**`ANALYST_NODE_SPECS`** — dict statis yang map key → spec. Perhatikan: key `"social"` map ke `agent_node="Sentiment Analyst"` (back-compat rename).

**`build_analyst_execution_plan(selected_analysts, concurrency_limit)`** — validasi input, return `AnalystExecutionPlan` (list specs + concurrency_limit).

**Bagaimana `setup.py` pakai plan ini:**
```python
plan = build_analyst_execution_plan(selected_analysts, ...)
for spec in plan.specs:
    workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
    workflow.add_node(spec.clear_node, create_msg_delete())
    workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])
```
Lalu edges dirangkai sequential: `START → specs[0].agent_node`, dan setiap `clear_node` → `specs[i+1].agent_node` (atau `"Bull Researcher"` kalau terakhir).

**`AnalystWallTimeTracker`** — utility class untuk CLI yang track berapa lama setiap analyst berjalan (wall time). Dipakai oleh `cli/stats_handler.py` untuk live display.

---

## 9. LLM Layer — Multi-Provider

Folder: `tradingagents/llm_clients/`. Tujuan: abstraksi supaya bisa switch antara OpenAI / Anthropic / Google / Azure / Ollama tanpa edit logic agent.

### File-file kunci

| File | Tugas |
|---|---|
| `factory.py` | `create_llm_client(provider, model, ...)` → return client instance (lazy import) |
| `base_client.py` | Class abstract `BaseLLMClient` — kontrak interface |
| `openai_client.py`, `anthropic_client.py`, `google_client.py`, `azure_client.py` | Implementasi per provider |
| `model_catalog.py` | Daftar model + metadata (context length, supports tools, dll) |
| `capabilities.py` | Tabel: model → bagaimana cara structured output (function call vs JSON mode vs schema) |
| `api_key_env.py` | Mapping provider → env var name (e.g. OpenAI butuh `OPENAI_API_KEY`) |
| `validators.py` | Validasi konfigurasi |

### Pattern factory

```python
def create_llm_client(provider, model, base_url=None, **kwargs):
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(model=model, base_url=base_url, **kwargs)
    elif provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(...)
    # ...
```

**Lazy import** (import di dalam if-branch) — supaya kalau user cuma pakai OpenAI, tidak perlu install `langchain-anthropic`.

### `client.get_llm()`

Setiap client expose `get_llm()` yang return objek **LangChain runnable** (e.g. `ChatOpenAI`, `ChatAnthropic`). Object ini yang dipakai di agent:

```python
chain = prompt | llm.bind_tools(tools)
result = chain.invoke(state["messages"])
```

Karena `llm` sudah LangChain runnable, semua API LangChain (binding, structured output) langsung jalan.

### Capabilities table

Beda model beda cara structured output. GPT-4 pakai function calling, Anthropic pakai tool use, Gemini pakai schema-based JSON mode, model lokal mungkin cuma JSON mode plain.

`capabilities.py` mendefinisikan `ModelCapabilities` dataclass:
```python
@dataclass(frozen=True)
class ModelCapabilities:
    supports_tool_choice: bool
    supports_json_mode: bool
    supports_json_schema: bool
    preferred_structured_method: StructuredMethod  # "function_calling" | "json_mode" | "json_schema" | "none"
    requires_reasoning_content_roundtrip: bool = False  # DeepSeek thinking models
    requires_reasoning_split: bool = False              # MiniMax reasoning models
```

Tabel per-model di-lookup via `get_capabilities(model_name)`. Client classes konsultasi tabel ini alih-alih hardcode if-ladder per model.

`agents/utils/structured.py` punya dua helper:
- `bind_structured(llm, schema, agent_name)` — wrap LLM dengan `with_structured_output(schema)`, return `None` kalau provider tidak support (log warning, agent akan pakai free-text)
- `invoke_structured_or_freetext(structured_llm, plain_llm, prompt, render, agent_name)` — coba structured call + render ke markdown; kalau gagal (malformed JSON, transient error), fallback ke plain `llm.invoke` supaya pipeline tidak block

### Konfigurasi 2-tier

Sistem pakai 2 LLM:
- `quick_thinking_llm` (`config["quick_think_llm"]`, default `gpt-5.4-mini`) — untuk analyst, debater, trader (banyak panggil, tidak butuh top-tier)
- `deep_thinking_llm` (`config["deep_think_llm"]`, default `gpt-5.4`) — untuk Research Manager + Portfolio Manager (judgment akhir, butuh nuansa)

Kamu bisa mix: provider sama tapi model beda, atau provider beda total (e.g. quick=GPT-mini, deep=Claude Opus).

### Reasoning effort (untuk model thinking)

Beberapa provider punya "reasoning level" (extended thinking):
- OpenAI: `openai_reasoning_effort` ("low" / "medium" / "high")
- Anthropic: `anthropic_effort`
- Google: `google_thinking_level` ("minimal" / "high")

Di `default_config.py` semua None default. Bisa set via env atau config dict.

---

## 10. Data Layer — Dari Mana Data Datang

Folder: `tradingagents/dataflows/`.

### Vendor-agnostic interface

File `dataflows/interface.py` adalah **routing layer**. Concept: agent panggil tool generic seperti `get_stock_data(ticker, date)`, di-route ke vendor yang dipilih (yfinance, Alpha Vantage, dll).

```python
VENDOR_METHODS = {
    "get_stock_data": {
        "yfinance": yfinance_get_stock_data,
        "alpha_vantage": alpha_vantage_get_stock_data,
    },
    # ... (9 methods total, masing-masing punya 2 vendor implementations)
}

def route_to_vendor(method, *args, **kwargs):
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)  # cek tool_vendors dulu, lalu data_vendors
    primary_vendors = [v.strip() for v in vendor_config.split(',')]
    # Build fallback chain: primary → remaining available vendors
    for vendor in fallback_vendors:
        try:
            return impl_func(*args, **kwargs)
        except AlphaVantageRateLimitError:
            continue  # HANYA rate limit yang trigger fallback
    raise RuntimeError(f"No available vendor for '{method}'")
```

**Fallback logic**: kalau Alpha Vantage hit rate limit (`AlphaVantageRateLimitError`), otomatis fallback ke vendor berikutnya (biasanya yfinance). Error lain (network, parsing) TIDAK trigger fallback — langsung raise.

**Vendor resolution priority**: `tool_vendors[method]` (per-tool override) → `data_vendors[category]` (per-category default).

### Konfigurasi vendor

Di `default_config.py`:
```python
"data_vendors": {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
},
"tool_vendors": {
    # Override per-tool: "get_stock_data": "alpha_vantage"
},
```

Default semua yfinance — gratis, no API key.

**`dataflows/config.py`** — process-level config singleton. `TradingAgentsGraph.__init__` panggil `set_config(self.config)` yang meng-update singleton ini. Semua tool functions (via `route_to_vendor`) baca config dari sini via `get_config()`. Dict-valued keys di-merge satu level deep (partial update aman).

**Bagaimana config sampai ke dataflows?** `TradingAgentsGraph.__init__` panggil `dataflows.config.set_config(config)` yang menyimpan config ke singleton global `_config`. Setiap kali `route_to_vendor()` dipanggil, dia baca `get_config()` untuk tahu vendor mana yang aktif. Pattern singleton ini karena tools (`@tool` functions) tidak bisa terima parameter tambahan selain yang LLM pass.

### Tools yang dipakai agent

Tools didefinisikan di file-file terpisah di `tradingagents/agents/utils/` (decorator `@tool` dari LangChain), lalu di-re-export via `agent_utils.py`. Mereka panggil `route_to_vendor()` dari `interface.py` yang routing ke vendor.

| Tool | File definisi | Dipakai oleh | Kategori | Parameter |
|---|---|---|---|---|
| `get_stock_data` | `core_stock_tools.py` | Market Analyst | core_stock_apis | `symbol`, `start_date`, `end_date` |
| `get_indicators` | `technical_indicators_tools.py` | Market Analyst | technical_indicators | `symbol`, `indicator`, `curr_date`, `look_back_days=30` |
| `get_news` | `news_data_tools.py` | News Analyst, Sentiment (pre-fetch) | news_data | `ticker`, `start_date`, `end_date` |
| `get_global_news` | `news_data_tools.py` | News Analyst | news_data | `curr_date`, `look_back_days=None`, `limit=None` |
| `get_insider_transactions` | `news_data_tools.py` | News Analyst | news_data | `ticker` |
| `get_fundamentals` | `fundamental_data_tools.py` | Fundamentals Analyst | fundamental_data | `ticker`, `curr_date` |
| `get_balance_sheet` | `fundamental_data_tools.py` | Fundamentals Analyst | fundamental_data | `ticker`, `freq="quarterly"`, `curr_date=None` |
| `get_cashflow` | `fundamental_data_tools.py` | Fundamentals Analyst | fundamental_data | `ticker`, `freq="quarterly"`, `curr_date=None` |
| `get_income_statement` | `fundamental_data_tools.py` | Fundamentals Analyst | fundamental_data | `ticker`, `freq="quarterly"`, `curr_date=None` |

**Catatan penting tentang `get_indicators`**: LLM kadang pass multiple indicators sebagai comma-separated string. Tool ini otomatis split dan proses satu per satu, return gabungan hasilnya.

### Caching

`dataflows/stockstats_utils.py` punya CSV caching (di `~/.tradingagents/cache/`). Kalau panggil indikator yang sama untuk ticker+tanggal yang sama, baca dari disk daripada API call ulang. Penting karena 1 run bisa banyak panggilan.

### Sumber data spesifik

- **`y_finance.py`** — wrapper yfinance. OHLCV, financials, balance sheet, cashflow, income.
- **`yfinance_news.py`** — news dari yfinance.
- **`alpha_vantage_*.py`** — Alpha Vantage (butuh `ALPHA_VANTAGE_API_KEY`)
- **`reddit.py`** — Reddit posts (free API, may need rate limit handling)
- **`stocktwits.py`** — StockTwits sentiment
- **`stockstats_utils.py`** — wrapper `stockstats` library untuk RSI/MACD/Bollinger dll, dengan CSV cache + retry/backoff

---


## 11. Memory & Reflection — Belajar dari Trade Sebelumnya

Salah satu fitur menarik: sistem ini "ingat" trade sebelumnya dan inject pelajaran ke run berikutnya.

### File terlibat
- `tradingagents/agents/utils/memory.py` — `TradingMemoryLog` class
- `tradingagents/graph/reflection.py` — `Reflector` class
- `tradingagents/graph/signal_processing.py` — extract rating dari decision

### Format memory log
File markdown append-only di `~/.tradingagents/memory/trading_memory.md`. Setiap entry dipisahkan oleh HTML comment `<!-- ENTRY_END -->`. Format:

```markdown
[2024-05-10 | NVDA | Buy | pending]

DECISION:
**Rating**: Buy
**Executive Summary**: ...
**Investment Thesis**: ...

<!-- ENTRY_END -->
```

Setelah resolved (outcome diketahui):
```markdown
[2024-05-10 | NVDA | Buy | +3.2% | +1.8% | 5d]

DECISION:
**Rating**: Buy
**Executive Summary**: ...

REFLECTION:
The directional call was correct with +1.8% alpha vs SPY. The bull thesis...

<!-- ENTRY_END -->
```

Tag format: `[date | ticker | rating | raw_return | alpha_return | holding_days]`

### Alur memory

**Saat run baru untuk ticker X:**
1. **Resolve pending entries untuk X** (di `_resolve_pending_entries` di `trading_graph.py`):
   - Cari entry "pending" untuk ticker X
   - Untuk tiap entry, fetch return aktual via yfinance: harga close di `trade_date + 5 hari` vs `trade_date`
   - Hitung alpha: ticker_return - benchmark_return (SPY default, atau sesuai exchange)
   - Panggil `Reflector.reflect_on_final_decision(decision, raw_return, alpha)` → LLM (`quick_thinking_llm`, bukan deep) hasilkan 2-4 kalimat lesson
   - Update entry jadi "resolved" dengan return + reflection

2. **Inject `past_context` ke initial state**:
   - `memory_log.get_past_context(ticker)` → ambil:
     - Hingga 5 same-ticker decisions + reflections (most recent first)
     - Hingga 3 cross-ticker reflections (dari ticker lain, untuk general lessons)
   - Masuk ke `state["past_context"]`
   - Portfolio Manager prompt punya placeholder untuk `past_context` → LLM lihat lesson sebelumnya (baik same-ticker maupun cross-ticker)

3. **Setelah run selesai**:
   - `memory_log.store_decision(ticker, trade_date, final_decision)` → entry baru status "pending"
   - Akan di-resolve pas next run untuk ticker yang sama

### Trade-off design
- ✅ Cross-run learning: tiap run jadi lebih "informed"
- ✅ Cross-ticker lessons: general trading lessons dari ticker lain juga di-inject (max 3)
- ❌ **Lazy resolution**: cuma resolve same-ticker. Kalau kamu run NVDA hari ini, AAPL kemarin tidak ke-resolve sampai kamu run AAPL lagi.
- ❌ **Bias risk**: kalau lesson awal salah, lesson berikutnya bisa terikut.
- ⚙️ **Rotation**: set `memory_log_max_entries` (default None = unlimited) untuk cap jumlah resolved entries. Oldest resolved entries di-prune; pending entries tidak pernah di-prune.

---

## 12. Konfigurasi & Env Vars

### Hierarki konfigurasi
1. **`DEFAULT_CONFIG`** (di `default_config.py`) — baseline
2. **Env vars `TRADINGAGENTS_*`** — override otomatis saat module di-import
3. **`config={...}` saat init** — programmatic override

### Env vars yang ada

Dua mekanisme:
1. **`_ENV_OVERRIDES` dict** (di `default_config.py`) — coerce string ke tipe default, applied saat module import
2. **`os.getenv()` langsung** di config dict values — untuk path-based configs

| Env var | Override | Default | Mekanisme |
|---|---|---|---|
| `TRADINGAGENTS_LLM_PROVIDER` | `llm_provider` | `"openai"` | `_ENV_OVERRIDES` |
| `TRADINGAGENTS_DEEP_THINK_LLM` | `deep_think_llm` | `"gpt-5.4"` | `_ENV_OVERRIDES` |
| `TRADINGAGENTS_QUICK_THINK_LLM` | `quick_think_llm` | `"gpt-5.4-mini"` | `_ENV_OVERRIDES` |
| `TRADINGAGENTS_LLM_BACKEND_URL` | `backend_url` | `None` | `_ENV_OVERRIDES` |
| `TRADINGAGENTS_OUTPUT_LANGUAGE` | `output_language` | `"English"` | `_ENV_OVERRIDES` |
| `TRADINGAGENTS_MAX_DEBATE_ROUNDS` | `max_debate_rounds` | `1` | `_ENV_OVERRIDES` |
| `TRADINGAGENTS_MAX_RISK_ROUNDS` | `max_risk_discuss_rounds` | `1` | `_ENV_OVERRIDES` |
| `TRADINGAGENTS_CHECKPOINT_ENABLED` | `checkpoint_enabled` | `False` | `_ENV_OVERRIDES` |
| `TRADINGAGENTS_BENCHMARK_TICKER` | `benchmark_ticker` | `None` | `_ENV_OVERRIDES` |
| `TRADINGAGENTS_RESULTS_DIR` | `results_dir` | `~/.tradingagents/logs` | `os.getenv()` |
| `TRADINGAGENTS_CACHE_DIR` | `data_cache_dir` | `~/.tradingagents/cache` | `os.getenv()` |
| `TRADINGAGENTS_MEMORY_LOG_PATH` | `memory_log_path` | `~/.tradingagents/memory/trading_memory.md` | `os.getenv()` |

Plus API keys per provider (lihat `llm_clients/api_key_env.py`):
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, dll
- `ALPHA_VANTAGE_API_KEY` (kalau pakai Alpha Vantage)
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` (kalau pakai Reddit data)

### Config keys yang TIDAK punya env override (hanya via programmatic config)

| Key | Default | Keterangan |
|---|---|---|
| `max_recur_limit` | `100` | LangGraph recursion limit |
| `analyst_concurrency_limit` | `1` | Parallelism analyst (default sequential) |
| `news_article_limit` | `20` | Max articles per ticker |
| `global_news_article_limit` | `10` | Max articles for global news |
| `global_news_lookback_days` | `7` | Macro news lookback window |
| `global_news_queries` | (5 queries) | Search queries for global news |
| `memory_log_max_entries` | `None` | Cap on resolved entries (None = no rotation) |
| `benchmark_map` | (dict suffix→index) | Auto-detect benchmark by ticker suffix |

### Output language
`output_language` mempengaruhi **semua agent output** (analyst reports, debate, trader plan, final decision). `get_language_instruction()` di `agent_utils.py` append instruksi bahasa ke prompt setiap agent. Ketika English (default), fungsi return empty string sehingga tidak ada overhead token.

---

## 13. Map Repo (File-by-File)

```
tradingagents/                          ← root
├── main.py                             ← entry point minimal (1 propagate call)
├── test.py                             ← variasi quick test
├── pyproject.toml                      ← deps via uv
├── requirements.txt                    ← deps untuk pip
├── uv.lock                             ← deterministic deps
├── Dockerfile, docker-compose.yml      ← containerized deployment
├── .env.example                        ← template env vars
├── .env.enterprise.example             ← enterprise config (Azure, dll)
├── README.md, CHANGELOG.md             ← docs
│
├── tradingagents/                      ← source utama
│   ├── __init__.py                     ← package init: load .env, suppress noisy warnings
│   ├── default_config.py               ← DEFAULT_CONFIG + env override
│   │
│   ├── graph/                          ← orchestration layer
│   │   ├── __init__.py                 ← re-exports: TradingAgentsGraph, ConditionalLogic, dll
│   │   ├── trading_graph.py            ← TradingAgentsGraph (god object)
│   │   ├── setup.py                    ← GraphSetup.setup_graph (pasang nodes/edges)
│   │   ├── conditional_logic.py        ← ConditionalLogic (should_continue_*)
│   │   ├── propagation.py              ← Propagator (initial state, invoke args)
│   │   ├── analyst_execution.py        ← build_analyst_execution_plan (urutan analyst)
│   │   ├── checkpointer.py             ← SqliteSaver wrapper (resume)
│   │   ├── reflection.py               ← Reflector (post-mortem lesson)
│   │   └── signal_processing.py        ← extract rating dari final markdown
│   │
│   ├── agents/                         ← agent definitions
│   │   ├── __init__.py                 ← re-exports semua create_* factory + state types
│   │   ├── schemas.py                  ← Pydantic: ResearchPlan, TraderProposal, PortfolioDecision
│   │   ├── analysts/
│   │   │   ├── market_analyst.py
│   │   │   ├── sentiment_analyst.py    ← (was "social_media_analyst")
│   │   │   ├── social_media_analyst.py ← DEPRECATED shim → import dari sentiment_analyst
│   │   │   ├── news_analyst.py
│   │   │   └── fundamentals_analyst.py
│   │   ├── researchers/
│   │   │   ├── bull_researcher.py
│   │   │   └── bear_researcher.py
│   │   ├── managers/
│   │   │   ├── research_manager.py
│   │   │   └── portfolio_manager.py
│   │   ├── trader/
│   │   │   └── trader.py
│   │   ├── risk_mgmt/
│   │   │   ├── aggressive_debator.py
│   │   │   ├── conservative_debator.py
│   │   │   └── neutral_debator.py
│   │   └── utils/
│   │       ├── agent_states.py         ← TypedDict: AgentState + nested debate states
│   │       ├── agent_utils.py          ← hub: re-export tools + create_msg_delete + helpers
│   │       ├── core_stock_tools.py     ← @tool get_stock_data (OHLCV via interface)
│   │       ├── technical_indicators_tools.py ← @tool get_indicators (via interface)
│   │       ├── fundamental_data_tools.py    ← @tool get_fundamentals/balance/cashflow/income
│   │       ├── news_data_tools.py      ← @tool get_news/get_global_news/get_insider_transactions
│   │       ├── memory.py               ← TradingMemoryLog (append-only markdown log)
│   │       ├── structured.py           ← bind_structured + invoke_structured_or_freetext
│   │       └── rating.py               ← parse_rating: heuristic 5-tier rating extractor
│   │
│   ├── llm_clients/                    ← multi-provider abstraction
│   │   ├── __init__.py                 ← re-exports: BaseLLMClient, create_llm_client
│   │   ├── factory.py                  ← create_llm_client (lazy import per provider)
│   │   ├── base_client.py              ← BaseLLMClient abstract + normalize_content helper
│   │   ├── openai_client.py            ← OpenAI / DeepSeek / xAI / Qwen / GLM / MiniMax / Ollama / OpenRouter
│   │   ├── anthropic_client.py         ← Anthropic Claude
│   │   ├── google_client.py            ← Google Gemini
│   │   ├── azure_client.py             ← Azure OpenAI (deployment-based)
│   │   ├── capabilities.py             ← ModelCapabilities dataclass + per-model table
│   │   ├── model_catalog.py            ← model dropdown options per provider (CLI)
│   │   ├── api_key_env.py              ← PROVIDER_API_KEY_ENV mapping (provider → env var)
│   │   ├── validators.py               ← validate_model: check model name vs catalog
│   │   └── TODO.md                     ← internal dev notes (consistency improvements tracker)
│   │
│   └── dataflows/                      ← data sources
│       ├── __init__.py                 ← (empty, package marker)
│       ├── interface.py                ← VENDOR_METHODS routing + fallback logic
│       ├── config.py                   ← get_config/set_config: process-level config singleton
│       ├── utils.py                    ← safe_ticker_component, save_output, get_next_weekday
│       ├── y_finance.py                ← yfinance wrapper (OHLCV, fundamentals, insider)
│       ├── yfinance_news.py            ← news via yfinance
│       ├── alpha_vantage.py            ← re-export hub: imports dari 4 alpha_vantage_* modules
│       ├── alpha_vantage_common.py     ← get_api_key, AlphaVantageRateLimitError, date fmt
│       ├── alpha_vantage_stock.py      ← get_stock (OHLCV via Alpha Vantage)
│       ├── alpha_vantage_fundamentals.py ← get_fundamentals/balance/cashflow/income (AV)
│       ├── alpha_vantage_indicator.py  ← get_indicator (technical indicators via AV)
│       ├── alpha_vantage_news.py       ← get_news/get_global_news/get_insider_transactions (AV)
│       ├── reddit.py                   ← Reddit posts fetcher
│       ├── stocktwits.py               ← StockTwits sentiment fetcher
│       └── stockstats_utils.py         ← stockstats wrapper + CSV cache + retry/backoff
│
├── cli/                                ← interactive CLI (Typer + Rich)
│   ├── __init__.py                     ← (empty, package marker)
│   ├── main.py                         ← Typer app, Rich Live rendering, MessageBuffer
│   ├── utils.py                        ← questionary prompts (ticker, model, analyst selection)
│   ├── stats_handler.py                ← StatsCallbackHandler (LLM/tool call + token counter)
│   ├── models.py                       ← AnalystType, AssetType enums (Pydantic)
│   ├── config.py                       ← CLI_CONFIG dict (announcements URL, timeout)
│   ├── announcements.py                ← fetch + display remote announcements panel
│   └── static/welcome.txt              ← ASCII art welcome banner
│
├── tests/                              ← pytest tests
│   ├── conftest.py
│   ├── test_signal_processing.py
│   ├── test_structured_agents.py
│   ├── test_memory_log.py              ← terbesar, banyak edge case
│   ├── test_capabilities.py
│   └── ... (banyak test spesifik)
│
├── scripts/
│   └── smoke_structured_output.py      ← E2E smoke test structured output (multi-provider)
│
└── assets/                             ← gambar untuk README
```

### 13.1 Detail File yang Sering Terlewat

#### `tradingagents/__init__.py`
Package-level init. Memuat `.env` dan `.env.enterprise` via `python-dotenv` (supaya API key tersedia tanpa user harus `source .env`). Juga suppress `PendingDeprecationWarning` dari `langgraph-checkpoint` yang noisy.

#### `tradingagents/agents/analysts/social_media_analyst.py`
**Deprecated backwards-compatibility shim.** Hanya re-import `create_sentiment_analyst` dan `create_social_media_analyst` dari `sentiment_analyst.py`, lalu emit `DeprecationWarning`. Akan dihapus di rilis mendatang. Jangan import dari sini — gunakan `sentiment_analyst` langsung.

#### `tradingagents/agents/utils/core_stock_tools.py`
Mendefinisikan satu `@tool`: **`get_stock_data(symbol, start_date, end_date)`** — memanggil `interface.route_to_vendor("get_stock_data", ...)` untuk fetch OHLCV data. Dipakai oleh Market Analyst.

#### `tradingagents/agents/utils/technical_indicators_tools.py`
Mendefinisikan satu `@tool`: **`get_indicators(symbol, indicator, curr_date, look_back_days=30)`** — memanggil `interface.route_to_vendor("get_indicators", ...)` untuk fetch indikator teknikal (RSI, MACD, Bollinger, dll). Otomatis split comma-separated indicators. Dipakai oleh Market Analyst.

#### `tradingagents/agents/utils/fundamental_data_tools.py`
Mendefinisikan 4 `@tool`:
- **`get_fundamentals(ticker, curr_date)`** — overview (P/E, market cap, dll)
- **`get_balance_sheet(ticker, freq="quarterly", curr_date=None)`** — neraca
- **`get_cashflow(ticker, freq="quarterly", curr_date=None)`** — arus kas
- **`get_income_statement(ticker, freq="quarterly", curr_date=None)`** — laba rugi

Semua memanggil `interface.route_to_vendor(...)`. Dipakai oleh Fundamentals Analyst.

#### `tradingagents/agents/utils/news_data_tools.py`
Mendefinisikan 3 `@tool`:
- **`get_news(ticker, start_date, end_date)`** — berita spesifik ticker
- **`get_global_news(curr_date, look_back_days=None, limit=None)`** — berita makro/global
- **`get_insider_transactions(ticker)`** — transaksi insider (hanya butuh ticker, tanpa date)

Semua memanggil `interface.route_to_vendor(...)`. Dipakai oleh News Analyst.

#### `tradingagents/agents/utils/agent_utils.py`
**Hub file** — bukan tempat tool didefinisikan (sejak refactor). Fungsinya:
1. **Re-export** semua 9 tools dari 4 file di atas (core_stock, technical_indicators, fundamental_data, news_data)
2. **`get_language_instruction()`** — return prompt suffix untuk output language (kosong kalau English)
3. **`build_instrument_context(ticker, asset_type)`** — return string instruksi agar agent selalu pakai ticker exact (termasuk suffix `.TO`, `.HK`, dll)
4. **`create_msg_delete()`** — factory yang return node function untuk clear `messages` list + add placeholder (Anthropic compatibility)

#### `tradingagents/agents/utils/rating.py`
Mendefinisikan skala rating 5-tier kanonik (`Buy, Overweight, Hold, Underweight, Sell`) dan fungsi **`parse_rating(text)`** yang extract rating dari prose markdown via regex 2-pass. Dipakai oleh `SignalProcessor` dan memory log.

#### `tradingagents/agents/utils/structured.py`
Dua helper untuk structured output pattern:
- **`bind_structured(llm, schema, agent_name)`** — wrap LLM dengan `with_structured_output(schema)`, return `None` kalau provider tidak support (log warning)
- **`invoke_structured_or_freetext(structured_llm, plain_llm, prompt, render, agent_name)`** — coba structured call + render ke markdown; kalau gagal, fallback ke free-text invoke

Dipakai oleh Research Manager, Trader, dan Portfolio Manager.

#### `tradingagents/dataflows/alpha_vantage.py`
**Re-export hub** — import dan expose semua fungsi dari `alpha_vantage_stock`, `alpha_vantage_indicator`, `alpha_vantage_fundamentals`, dan `alpha_vantage_news` dalam satu namespace. Memudahkan `interface.py` import satu module saja.

#### `tradingagents/dataflows/config.py`
**Process-level config singleton.** Fungsi:
- `initialize_config()` — copy `DEFAULT_CONFIG` ke `_config` global
- `set_config(dict)` — merge partial config (dict-valued keys merged 1-level deep)
- `get_config()` — return deepcopy of current config

Dipanggil oleh `TradingAgentsGraph.__init__` untuk propagate config ke dataflows layer.

#### `tradingagents/dataflows/utils.py`
Utility functions shared across dataflows:
- **`safe_ticker_component(value)`** — validasi ticker string aman untuk filesystem path (reject traversal attacks seperti `../../etc`)
- **`save_output(df, tag, path)`** — simpan DataFrame ke CSV
- **`get_current_date()`** — return today as string
- **`get_next_weekday(date)`** — skip weekend ke hari kerja berikutnya

#### `tradingagents/llm_clients/api_key_env.py`
Dict `PROVIDER_API_KEY_ENV` yang map setiap provider ke nama env var API key-nya (e.g. `"openai" → "OPENAI_API_KEY"`, `"ollama" → None`). Fungsi `get_api_key_env(provider)` dipakai CLI untuk prompt user memasukkan key yang benar.

#### `tradingagents/llm_clients/validators.py`
Fungsi **`validate_model(provider, model)`** — cek apakah model name ada di catalog untuk provider tersebut. Ollama dan OpenRouter di-skip (accept any). Dipanggil saat init client untuk warn user kalau typo model name.

#### `tradingagents/llm_clients/model_catalog.py`
Daftar model options per provider (dengan label deskriptif) untuk dropdown di CLI. Juga expose `get_known_models()` yang dipakai `validators.py`. Berisi model list untuk OpenAI, Anthropic, Google, DeepSeek, xAI, Qwen, GLM, MiniMax, dan OpenRouter.

#### `tradingagents/llm_clients/capabilities.py`
Dataclass `ModelCapabilities` + tabel per-model yang deklaratif: apakah model support `tool_choice`, `json_mode`, `json_schema`, dan method structured output apa yang preferred. Juga flag khusus seperti `requires_reasoning_content_roundtrip` (DeepSeek thinking) dan `requires_reasoning_split` (MiniMax). Client classes konsultasi tabel ini alih-alih hardcode if-ladder.

#### `tradingagents/llm_clients/azure_client.py`
Implementasi `AzureOpenAIClient(BaseLLMClient)` untuk Azure OpenAI deployments. Butuh env vars: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`, `OPENAI_API_VERSION`. Return `NormalizedAzureChatOpenAI` instance dari `get_llm()`.

#### `cli/config.py`
Dict `CLI_CONFIG` berisi URL endpoint announcements (`https://api.tauric.ai/v1/announcements`), timeout (1s), dan fallback message. Dipakai oleh `announcements.py`.

#### `cli/models.py`
Pydantic enums: `AnalystType` (MARKET, SOCIAL, NEWS, FUNDAMENTALS) dan `AssetType` (STOCK, CRYPTO). Dipakai CLI untuk type-safe analyst selection dan asset mode.

#### `cli/stats_handler.py`
Class `StatsCallbackHandler(BaseCallbackHandler)` — LangChain callback yang track jumlah LLM calls, tool calls, input tokens, dan output tokens secara thread-safe. Dipakai CLI untuk display live stats panel selama run.

#### `cli/announcements.py`
Fetch announcements JSON dari remote endpoint (dengan timeout 1s) dan display sebagai Rich Panel. Kalau `require_attention=True`, pause sampai user tekan Enter. Graceful fallback ke pesan default kalau network gagal.

#### `scripts/smoke_structured_output.py`
End-to-end smoke test yang menjalankan 3 structured-output agents (Research Manager, Trader, Portfolio Manager) langsung terhadap real LLM provider tanpa full `propagate()`. Verifikasi bahwa structured output → rendered markdown chain bekerja untuk provider tertentu. Juga test `SignalProcessor` extract rating. Usage: `python scripts/smoke_structured_output.py openai`.

---

## 14. Hal-Hal Yang Mudah Salah Paham

### 14.1 "Social analyst" = "Sentiment analyst"
Di kode lama, namanya `social_media_analyst`. Sekarang `sentiment_analyst`. Tapi label conditional masih `"social"` (untuk back-compat config lama). Perhatikan:
- File utama: `sentiment_analyst.py`
- File shim (deprecated): `social_media_analyst.py` — hanya re-import dari `sentiment_analyst.py` + emit `DeprecationWarning`
- Wire key: `"social"` (di `ANALYST_NODE_SPECS` dan `selected_analysts`)
- Conditional: `should_continue_social`
- Clear node label: `"Msg Clear Sentiment"`
- Agent node label: `"Sentiment Analyst"` (bukan "Social Media Analyst")
- ToolNode key: `"social"` → berisi `get_news` tool (tapi sentiment analyst tidak bind tools, jadi tool node tidak pernah dipakai)

### 14.2 `messages` itu temporary, bukan log
`state["messages"]` cuma dipakai untuk tool-calling loop di analyst. Setelah analyst selesai, ada **Msg Clear** node yang reset ke `[]`. Jangan andalkan `messages` untuk akses report — pakai `state["market_report"]` dst.

### 14.3 Bull/Bear tidak punya `messages` mereka sendiri
Mereka cuma update `bull_history` / `bear_history` (string). Murni state-update, bukan message-passing.

### 14.4 `count` dikalikan
- `2 * max_debate_rounds` karena 1 round = 1 bull + 1 bear
- `3 * max_risk_discuss_rounds` karena 1 round = aggressive + conservative + neutral

### 14.5 `current_response` dipakai untuk routing
Kalau dimulai dengan "Bull" → next ke Bear. Bull researcher prepend `"Bull Analyst: "` di response-nya (bukan hanya "Bull"). `startswith("Bull")` tetap match karena "Bull Analyst: ..." dimulai dengan "Bull". Sama untuk Bear (`"Bear Analyst: ..."`), dan untuk risk debaters dengan `latest_speaker` yang di-set ke string exact "Aggressive"/"Conservative"/"Neutral".

### 14.6 Quick LLM dipakai bukan cuma untuk hal cepat
Trader pakai `quick_thinking_llm` walaupun dia eksekusi keputusan penting. Reasoning: trader cuma menerjemahkan thesis (yang sudah ditimbang Research Manager dengan deep LLM) ke format eksekusi. Heavy lifting sudah selesai.

### 14.7 Reflection itu deferred, bukan real-time
Saat run NVDA hari ini, sistem TIDAK refleksi run NVDA hari ini. Dia refleksi run NVDA SEBELUMNYA (yang return-nya sudah bisa dihitung). Run hari ini akan direfleksi pas kamu run NVDA lagi minimal 5 hari ke depan.

### 14.8 Checkpointer default OFF
`checkpoint_enabled=False`. Aktifkan lewat env atau config kalau butuh resume.

### 14.9 `output_language` mempengaruhi SEMUA agent output
`get_language_instruction()` di `agent_utils.py` menambahkan `" Write your entire response in {lang}."` ke prompt SETIAP agent (analyst, researcher, debater, trader, PM) ketika language bukan English. Jadi seluruh pipeline output dalam bahasa yang dipilih, bukan hanya report final. Ketika English (default), fungsi return empty string (no extra tokens).

### 14.10 LLM dipanggil banyak kali per run
Estimasi 1 run dengan 4 analyst, max_rounds=1:
- Market: 2-5 LLM call (tool loop)
- Sentiment: 1
- News: 2-5
- Fundamentals: 2-5
- Bull + Bear: 2
- Research Manager: 1 (deep)
- Trader: 1
- Aggressive + Conservative + Neutral: 3
- Portfolio Manager: 1 (deep)
- Reflection (kalau ada pending): 1 per pending

**Total: ~15-25 LLM call per run.** Pakai quick_llm yang murah untuk mayoritas.

### 14.11 Debug mode = streaming, bukan invoke
Ketika `debug=True`, `trading_graph.py` pakai `graph.stream()` (bukan `graph.invoke()`). Setiap chunk di-pretty-print. Final state di-merge manual dari semua chunks. Ini berarti debug mode output lebih verbose tapi state akhir identik.

### 14.12 `batch_update_with_outcomes` — atomic write
Memory log update menggunakan pattern: read all → modify in-memory → write to temp file → `os.replace()` (atomic). Ini mencegah korupsi log kalau crash mid-write. Multiple pending entries untuk ticker yang sama di-resolve dalam satu atomic write via `batch_update_with_outcomes`.

---

## 15. Glossarium

| Istilah | Arti |
|---|---|
| **Ticker** | Simbol saham (NVDA, AAPL, BTC-USD) |
| **OHLCV** | Open/High/Low/Close/Volume — data pasar dasar |
| **Alpha** | Return relatif terhadap benchmark. Alpha = ticker_return - benchmark_return |
| **SPY** | ETF S&P 500, benchmark default untuk US stocks |
| **RSI / MACD / Bollinger / ATR / VWMA** | Indikator teknikal |
| **P/E / P/B** | Price-to-Earnings / Price-to-Book — rasio valuasi |
| **Fundamentals** | Data laporan keuangan (income statement, balance sheet, cashflow) |
| **Insider transactions** | Trade saham oleh executives perusahaan (sinyal sentiment) |
| **Bull / Bear** | Bullish (optimis) / Bearish (pesimis) |
| **Debate rounds** | Berapa kali bull-bear bertukar argumen |
| **Sizing** | Berapa besar posisi (% portfolio) |
| **Stop loss** | Harga keluar otomatis kalau rugi |
| **Entry / Exit** | Harga masuk / keluar trade |
| **Catalyst** | Event yang trigger pergerakan harga (earnings, lawsuit, dll) |
| **Holding days** | Berapa hari hold posisi (default 5 untuk reflection) |
| **Rating tier** | Buy / Overweight / Hold / Underweight / Sell (5-tier scale) |
| **Structured output** | LLM dipaksa output sesuai schema (Pydantic) |
| **LangGraph state** | Dict (TypedDict) yang dilewati antar node |
| **Conditional edge** | Edge yang routing-nya tergantung fungsi (kondisi) |
| **ToolNode** | Node bawaan LangGraph yang otomatis execute tool calls |
| **Tool-calling loop** | Pattern: LLM → tool → LLM → tool ... sampai LLM selesai |
| **Checkpointer** | Penyimpan state yang memungkinkan resume kalau crash |
| **Reflection** | Post-mortem analysis untuk learning dari trade sebelumnya |
| **Memory log** | File markdown yang menyimpan history decision + lesson |
| **Quick / Deep LLM** | Tier murah untuk volume / tier mahal untuk judgment |
| **Capability** | Kemampuan model (tool use, JSON mode, schema, dll) |
| **Vendor** | Sumber data (yfinance, Alpha Vantage, Reddit) |

---

## Penutup

Setelah baca dokumen ini sampai habis, kamu sudah punya **mental model** yang lengkap. Yang masih kurang:
- Detail konkret prompt setiap agent (cek file `.py` masing-masing)
- Implementasi schema Pydantic (cek `agents/schemas.py`)
- Tweak-an config spesifik (cek `default_config.py` line-by-line)

Saran urutan baca selanjutnya (langsung ke code):

1. `main.py` (10 baris) — bayangkan jadi entry point
2. `tradingagents/default_config.py` — paham knob-knob yang bisa di-tune
3. `tradingagents/graph/trading_graph.py` `__init__` dan `propagate` — orchestrator detail
4. `tradingagents/graph/setup.py` `setup_graph` — penting, ini yang merakit
5. `tradingagents/graph/conditional_logic.py` — pendek, langsung paham
6. `tradingagents/agents/utils/agent_states.py` — state definition
7. `tradingagents/agents/analysts/market_analyst.py` — contoh tool-calling agent
8. `tradingagents/agents/researchers/bull_researcher.py` — contoh debate agent
9. `tradingagents/agents/managers/portfolio_manager.py` — contoh structured output agent
10. `tradingagents/agents/schemas.py` — Pydantic schemas

Setelah baca semua di atas, kamu siap **modifikasi**. Di file [`LEARNING_PLAN.md`](./LEARNING_PLAN.md) ada ide-ide capstone (tambah analyst baru, tambah vendor data, ubah rating scale, dll).

Kalau ada yang bingung, tanya saja — kita bisa zoom ke bagian spesifik.
