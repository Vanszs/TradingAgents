# TradingAgents Mastery Plan

> Hasil deep-analysis sub-agent (architecture + domain + infra) yang disintesis menjadi panduan belajar bertahap. Tujuan: menguasai project ini seakan kamu yang membangunnya.

---

## 1. Big Picture (Mental Model)

**TradingAgents** adalah multi-agent LLM trading framework yang mensimulasikan firma investasi: beberapa LLM agent dengan peran berbeda (analis pasar, peneliti bull/bear, trader, risk debator, portfolio manager) saling berdebat dan berkolaborasi untuk menghasilkan keputusan trading akhir untuk satu ticker pada satu tanggal.

Backbone-nya **LangGraph** — graph dengan nodes (agent) + edges (alur), state bersama (TypedDict) yang dilewati antar node, conditional edges untuk loop (tool-calling, debate rounds), dan checkpointer untuk crash-resume.

Kenapa LangGraph: butuh stateful workflow dengan loop kontrol (debate berapa round, tool dipanggil sampai cukup), branching kondisional, observability, dan persistence — yang sulit dengan plain function chaining.

**Alur high-level:**
```
Ticker+Date
   ↓
[Analysts: Market | News | Fundamentals | Sentiment]   ← parallel/sequential
   ↓ (4 reports)
[Bull ↔ Bear debate] → [Research Manager]              ← investment thesis
   ↓
[Trader] → TraderProposal                              ← entry/exit plan
   ↓
[Aggressive ↔ Conservative ↔ Neutral debate] → [Portfolio Manager]
   ↓
Final PortfolioDecision (rating + thesis + horizon)
   ↓
[Reflection] → memory log (untuk run berikutnya)
```

---

## 2. Phase Roadmap

### Phase 1 — Setup & Run End-to-End (~2 jam)

**Tujuan:** Bisa menjalankan project, paham apa input dan output-nya, melihat eksekusi mengalir.

**Baca:**
1. `README.md` — context project
2. `pyproject.toml`, `requirements.txt`, `uv.lock` — deps
3. `.env.example` — env vars yang dibutuhkan
4. `main.py` — entry point minimal
5. `test.py` — alternative simple runner
6. `cli/main.py` (skim dulu) — CLI experience

**Konsep kunci:** uv/pip install, env-based config (`TRADINGAGENTS_*`), provider/model selection, cost vs depth (quick vs deep model).

**Hands-on:**
- Install deps, set `OPENAI_API_KEY` (atau provider lain)
- Jalankan `python main.py` untuk 1 ticker (e.g. NVDA)
- Jalankan `python -m cli.main` — pilih analyst, model, ticker
- Tanggap output: 4 report + debate + trader plan + risk debate + final decision

**Self-check:**
1. File mana yang menentukan model default?
2. Apa beda quick_thinking_llm vs deep_thinking_llm?
3. Bagaimana CLI memilih analysts? Apa default-nya?

---

### Phase 2 — Backbone LangGraph (~4 jam)

**Tujuan:** Paham graph dirakit dari kode, state schema, edge wiring, conditional loop, recursion limit, checkpointer.

**Baca (urut):**
1. `tradingagents/graph/__init__.py`
2. `tradingagents/agents/utils/agent_states.py` — TypedDict state
3. `tradingagents/graph/propagation.py` — initial state + invoke args
4. `tradingagents/graph/conditional_logic.py` — fungsi conditional (tool loop, debate count)
5. `tradingagents/graph/setup.py` — `GraphSetup.setup_graph()` (rakitan utama)
6. `tradingagents/graph/analyst_execution.py` — analyst plan & ordering
7. `tradingagents/graph/checkpointer.py` — SqliteSaver per-ticker
8. `tradingagents/graph/trading_graph.py` — orchestrator (god object): init LLMs, build graph, run

**Konsep kunci:**
- `StateGraph(TypedDict)`, `add_node`, `add_edge`, `add_conditional_edges`
- ToolNode loop pattern (`tool_calls?` → ToolNode → back to agent)
- Recursion limit & message clearing
- Checkpointer untuk resume

**Hands-on:**
- Gambar ulang graph (whiteboard/excalidraw): nodes, edges, conditional
- Tambahkan `print()` di setup untuk verify edge order
- Trial: ubah `max_debate_rounds` di config dan amati efek

**Self-check:**
1. Bagaimana state mengalir dari analyst ke researcher? Apa yang shared, apa yang tidak?
2. Apa yang menghentikan tool-calling loop?
3. Apa fungsi message-clear node setelah analyst selesai?

---

### Phase 3 — Agents: Analysts Layer (~5 jam)

**Tujuan:** Paham 4 jenis analyst, perbedaan tool-using vs pre-fetched, prompt engineering pattern.

**Baca:**
1. `tradingagents/agents/__init__.py`
2. `tradingagents/agents/utils/structured.py` — bind_structured + fallback
3. `tradingagents/agents/analysts/market_analyst.py` (technical, tool-calling)
4. `tradingagents/agents/analysts/news_analyst.py` (events, tool-calling)
5. `tradingagents/agents/analysts/fundamentals_analyst.py` (financials, tool-calling)
6. `tradingagents/agents/analysts/sentiment_analyst.py` (pre-fetched, no tools)
7. `tradingagents/agents/utils/agent_utils.py` (toolkit yang dipakai analyst)

**Konsep kunci:**
- Tool-calling agent loop vs pre-fetched data injection
- Sistem prompt + role + output spec
- `bind_tools()` & `invoke_structured_or_freetext()`

**Hands-on:**
- Buat "EconomicCalendarAnalyst" stub yang fetch FRED data
- Tweak prompt market_analyst dan ukur perubahan di output
- Kurangi tools market_analyst jadi 2 saja, amati efek

**Self-check:**
1. Mengapa sentiment_analyst tidak pakai tools?
2. Apa peran `Msg Clear` setelah analyst?
3. Bagaimana analyst memutuskan kapan stop tool-calling?

---

### Phase 4 — Researchers Debate + Manager (~4 jam)

**Tujuan:** Paham orkestrasi debate adversarial dan judging.

**Baca:**
1. `tradingagents/agents/researchers/bull_researcher.py`
2. `tradingagents/agents/researchers/bear_researcher.py`
3. `tradingagents/agents/managers/research_manager.py`
4. `tradingagents/agents/schemas.py` — `ResearchPlan` schema
5. `tradingagents/agents/utils/memory.py` — memory injection

**Konsep kunci:**
- State: `InvestDebateState` (history, count, current side)
- Conditional edge: `count < max_rounds` → loop, else → manager
- Structured output via Pydantic
- Memory: pending dari run lalu di-inject ke prompt

**Hands-on:**
- Set max_rounds=3, baca history-nya
- Ubah persona bull jadi "value investor", lihat efek
- Tambah mock memory entry, verifikasi terbaca research_manager

**Self-check:**
1. Bagaimana siapa giliran (bull/bear)?
2. Bagaimana research_manager membuat keputusan?
3. Apa output structured-nya?

---

### Phase 5 — Trader + Risk Debate + PM (~4 jam)

**Tujuan:** Paham execution layer (entry/exit/sizing) dan multi-perspective risk debate.

**Baca:**
1. `tradingagents/agents/trader/trader.py`
2. `tradingagents/agents/risk_mgmt/aggressive_debator.py`
3. `tradingagents/agents/risk_mgmt/conservative_debator.py`
4. `tradingagents/agents/risk_mgmt/neutral_debator.py`
5. `tradingagents/agents/managers/portfolio_manager.py`
6. `tradingagents/graph/signal_processing.py` — extract rating
7. `tradingagents/agents/utils/rating.py` — parser fallback
8. `tradingagents/graph/reflection.py` — post-mortem

**Konsep kunci:**
- 3-way risk debate dengan turn rotation
- 5-tier rating scale (`STRONG_BUY..STRONG_SELL`)
- `PortfolioDecision` final structured output
- Reflection generates lesson untuk memory log

**Hands-on:**
- Tambah debator ke-4 (e.g. "Quant"): persona + prompt
- Modifikasi rating jadi 7-tier (perlu update schema + parser + memory)
- Trace 1 run: dari trader_plan ke final_decision

**Self-check:**
1. Apa input PortfolioManager?
2. Bagaimana sistem extract rating dari markdown?
3. Apa yang dipersist ke memory pasca-run?

---

### Phase 6 — Infrastructure (~5 jam)

**Tujuan:** Paham layer integrasi: LLM multi-provider, dataflows, config, persistence, CLI.

**Baca:**

LLM Clients:
1. `tradingagents/llm_clients/__init__.py`, `factory.py`
2. `tradingagents/llm_clients/base_client.py`
3. `tradingagents/llm_clients/capabilities.py`, `model_catalog.py`
4. `openai_client.py`, `anthropic_client.py`, `google_client.py`, `azure_client.py`
5. `tradingagents/llm_clients/api_key_env.py`, `validators.py`

Dataflows:
6. `tradingagents/dataflows/interface.py` — VENDOR_METHODS routing
7. `tradingagents/dataflows/y_finance.py`, `yfinance_news.py`
8. `tradingagents/dataflows/alpha_vantage_*.py`
9. `tradingagents/dataflows/reddit.py`, `stocktwits.py`
10. `tradingagents/dataflows/stockstats_utils.py`, `config.py`

Config & CLI:
11. `tradingagents/default_config.py`
12. `cli/main.py`, `cli/utils.py`, `cli/stats_handler.py`, `cli/models.py`

**Konsep kunci:**
- Factory + lazy import
- Capability table untuk structured output method (function call vs json mode vs schema)
- Vendor routing dengan fallback
- `TRADINGAGENTS_*` env override
- CLI: Typer + Rich Live + MessageBuffer

**Hands-on:**
- Tambah provider baru (e.g. Mistral): subclass base_client + factory + catalog + capabilities
- Tambah data vendor (e.g. Finnhub): implement interface + register + fallback
- Override config via env-only run

**Self-check:**
1. Bagaimana factory pilih client dari string provider?
2. Apa beda capability `structured_method=function_call` vs `json_mode`?
3. Bagaimana CLI tahu agent mana yang sedang aktif?

---

### Phase 7 — Capstone: Modifikasi & Extend (~6-8 jam)

**Tujuan:** Buktikan mastery dengan modifikasi nyata.

**Baca (revisit):**
1. `tradingagents/graph/trading_graph.py` (full)
2. `cli/main.py`, `cli/utils.py`, `cli/stats_handler.py`
3. `tests/` — pola pengujian

**Konsep:** extension points (analyst_factories, VENDOR_METHODS, capabilities, ANALYST_NODE_SPECS), CLI streaming, thread-safety, test patterns.

**Hands-on (pilih min. 2):**
1. Tambah analyst baru ("Options Analyst"): factory, register, conditional logic, wire ke graph, masuk CLI
2. Tambah data vendor (Finnhub): interface + VENDOR_METHODS + fallback
3. Modifikasi rating 5-tier → 7-tier (schema + signal + memory)
4. Implement parallel analysts (concurrency_limit di AnalystExecutionPlan)
5. Tambah LLM provider baru (Mistral)

**Self-check:**
1. Berapa file yang harus diubah untuk tambah analyst baru? (6-8)
2. Bagaimana memastikan tidak break existing tests?
3. Apa concern saat tambah vendor dengan rate limit?
4. Mengapa `output_language` cuma affect final report?
5. Bagaimana CLI tracks agent aktif?

---

## 3. Build It Yourself — Capstone Mini-Project

**Objective:** Re-implement minimal version dari scratch (1 analyst + trader + risk debater + PM) dalam 1 file Python ~200-300 baris.

### Spec

**State schema** — `MiniAgentState(MessagesState)`:
- `ticker`, `trade_date`, `market_report`, `trader_plan`, `risk_debate_state` (history+count), `final_decision`

**Nodes:**
1. **Market Analyst** — 1 tool (`get_stock_data` via yfinance), tool-calling loop, output `market_report`, lalu Msg Clear
2. **Trader** — no tools, structured output `TraderProposal(action, reasoning)` dengan `with_structured_output()`
3. **Risk Debater** — 2 rounds self-debate (round1: kenapa risiko, round2: kenapa acceptable), conditional edge `count<2`
4. **Portfolio Manager** — input full state, output `PortfolioDecision(rating, summary)` structured

**Wiring:**
```
START → Market Analyst ↔ Tools (loop) → Msg Clear → Trader → Risk Debater (loop) → PM → END
```

**Bonus:** memory file (append jadi log), regex extract rating dari PM output.

**Deliverable:** `python mini_trading_agent.py NVDA 2024-05-10`

**Sukses:** graph compile, tool dipanggil ≥1×, trader structured output, risk debate 2 round, PM rating extracted.

**Estimasi:** 8-12 jam.

---

## 4. Cheat Sheet (Komponen → File → Fungsi)

| Komponen | File | Fungsi |
|---|---|---|
| Orchestrator | `tradingagents/graph/trading_graph.py` | God object: init LLMs, build graph, run propagation, manage memory |
| Graph wiring | `tradingagents/graph/setup.py` | StateGraph builder: nodes, edges, conditionals |
| State schema | `tradingagents/agents/utils/agent_states.py` | AgentState + InvestDebateState + RiskDebateState |
| Routing | `tradingagents/graph/conditional_logic.py` | Tool loop, debate termination, turn routing |
| Analyst plan | `tradingagents/graph/analyst_execution.py` | Specs builder: list[str] → ordered AnalystNodeSpecs |
| Initial state | `tradingagents/graph/propagation.py` | Empty state + invoke args (recursion limit) |
| Market analyst | `tradingagents/agents/analysts/market_analyst.py` | Tool-calling: stock data + indicators → technical report |
| Sentiment | `tradingagents/agents/analysts/sentiment_analyst.py` | Pre-fetch Yahoo+StockTwits+Reddit → sentiment report |
| News | `tradingagents/agents/analysts/news_analyst.py` | Tool-calling: news + global → event/macro report |
| Fundamentals | `tradingagents/agents/analysts/fundamentals_analyst.py` | Tool-calling: financials → value report |
| Bull researcher | `tradingagents/agents/researchers/bull_researcher.py` | Advocate FOR, counter bear |
| Bear researcher | `tradingagents/agents/researchers/bear_researcher.py` | Advocate AGAINST, counter bull |
| Research mgr | `tradingagents/agents/managers/research_manager.py` | Judge debate → ResearchPlan |
| Trader | `tradingagents/agents/trader/trader.py` | TraderProposal: action+entry+stop+sizing |
| Aggressive debator | `tradingagents/agents/risk_mgmt/aggressive_debator.py` | Risk-seeking |
| Conservative | `tradingagents/agents/risk_mgmt/conservative_debator.py` | Risk-averse |
| Neutral | `tradingagents/agents/risk_mgmt/neutral_debator.py` | Balanced |
| Portfolio mgr | `tradingagents/agents/managers/portfolio_manager.py` | PortfolioDecision: rating+thesis+target+horizon |
| Schemas | `tradingagents/agents/schemas.py` | Pydantic schemas + renderers |
| Structured helper | `tradingagents/agents/utils/structured.py` | bind_structured + fallback |
| Memory | `tradingagents/agents/utils/memory.py` | Append-only markdown log per ticker |
| Reflection | `tradingagents/graph/reflection.py` | Post-mortem lesson (2-4 sentences) |
| Signal extract | `tradingagents/graph/signal_processing.py` | Regex 5-tier rating |
| Rating parser | `tradingagents/agents/utils/rating.py` | Two-pass parsing |
| Checkpointer | `tradingagents/graph/checkpointer.py` | Per-ticker SQLite, crash-resume |
| LLM factory | `tradingagents/llm_clients/factory.py` | provider str → client (lazy import) |
| Capabilities | `tradingagents/llm_clients/capabilities.py` | Model → structured method + quirks |
| Data routing | `tradingagents/dataflows/interface.py` | VENDOR_METHODS + fallback |
| yFinance | `tradingagents/dataflows/y_finance.py` | OHLCV + financials |
| Indicators | `tradingagents/dataflows/stockstats_utils.py` | Technical indicators + cache |
| Config | `tradingagents/default_config.py` | DEFAULT_CONFIG + env override |
| CLI app | `cli/main.py` | Typer + Rich Live streaming |
| CLI prompts | `cli/utils.py` | questionary interactive prompts |

---

## 5. 10 Pertanyaan Diskusi

1. **Scalability**: Sequential vs parallel analyst — tradeoff context pollution.
2. **Debate quality vs cost**: Berapa rounds optimal (default 1)?
3. **Memory**: Deferred reflection per-ticker — apakah cross-ticker lessons cukup?
4. **Structured output**: Free-text vs structured — kapan pakai apa?
5. **Benchmark**: Alpha vs SPY 5-day — adil? Sharpe/drawdown/win rate?
6. **Multi-asset**: Crypto support tools masih equity — bagaimana extend on-chain?
7. **Adversarial robustness**: Bot manipulasi sentiment — perlu data quality analyst?
8. **Real-time vs batch**: Adaptasi untuk streaming data + intraday?
9. **LLM tier**: 2-tier sekarang — kapan butuh 3-tier?
10. **Reproducibility**: Bagaimana ensure deterministic backtest dengan LLM non-deterministic?
