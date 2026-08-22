# TradingAgents × Kronos — Flow Diagrams

> Diagram-diagram Mermaid lengkap dengan anotasi sumber logic. Render otomatis di GitHub atau Markdown viewer mana saja yang support Mermaid.

---

## Legenda Warna

- 🔵 **Biru** = TradingAgents (LLM-driven, sudah ada)
- 🟠 **Oranye** = Kronos (forecast model, baru)
- 🟢 **Hijau** = Hybrid (TradingAgents existing TAPI dimodifikasi untuk consume Kronos)
- ⚪ **Abu-abu** = Infrastructure (state, edges, config)

---

## 1. Flow Utama End-to-End

Ini gambaran besar urutan eksekusi:

```mermaid
flowchart TD
    Start([START: ticker + trade_date])

    subgraph ANALYST["🔵 ANALYST LAYER (TradingAgents) + 🟠 Forecast (Kronos)"]
        direction TB
        MA["1️⃣ Market Analyst<br/>🔵 LLM + tool loop<br/>tools: get_stock_data, get_indicators"]
        SA["2️⃣ Sentiment Analyst<br/>🔵 LLM, no tools<br/>data pre-fetched"]
        NA["3️⃣ News Analyst<br/>🔵 LLM + tool loop<br/>tools: get_news, global_news, insider"]
        FA["4️⃣ Fundamentals Analyst<br/>🔵 LLM + tool loop<br/>tools: financials/balance/cash/income"]
        KA["5️⃣ Forecast Analyst ★<br/>🟠 Kronos predict<br/>NO LLM, deterministic<br/>output: pred_df + summary stats"]
    end

    subgraph RESEARCH["🟢 RESEARCH LAYER (LLM debate, modified to read forecast)"]
        direction TB
        BULL["6️⃣ Bull Researcher 🟢<br/>baca 4 reports + forecast<br/>argue PRO"]
        BEAR["7️⃣ Bear Researcher 🟢<br/>baca 4 reports + forecast<br/>argue CON"]
        RM["8️⃣ Research Manager<br/>🔵 deep LLM<br/>judge → investment_plan"]
    end

    subgraph EXEC["🟢 EXECUTION LAYER"]
        TR["9️⃣ Trader 🟢<br/>baca investment_plan + forecast_summary<br/>sizing & stop dari Kronos volatility/percentile"]
    end

    subgraph RISK["🟢 RISK LAYER (3-way debate, modified)"]
        direction TB
        AGG["🔟 Aggressive Debator 🟢<br/>argue dengan VaR, prob_positive"]
        CON["1️⃣1️⃣ Conservative Debator 🟢<br/>argue dengan p95_drawdown"]
        NEU["1️⃣2️⃣ Neutral Debator 🟢<br/>balance, median forecast"]
        PM["1️⃣3️⃣ Portfolio Manager 🟢<br/>🔵 deep LLM<br/>+ kronos_alignment field<br/>+ wajib justify kalau divergent"]
    end

    End([END: final_decision + memory log])

    Start --> MA
    MA -->|"tools loop sampai cukup"| MA
    MA -->|"selesai"| SA
    SA --> NA
    NA -->|"tools loop"| NA
    NA --> FA
    FA -->|"tools loop"| FA
    FA --> KA
    KA --> BULL

    BULL <-->|"max_debate_rounds"| BEAR
    BULL --> RM
    BEAR --> RM
    RM --> TR
    TR --> AGG
    AGG --> CON
    CON --> NEU
    NEU -->|"max_risk_rounds"| AGG
    AGG --> PM
    CON --> PM
    NEU --> PM
    PM --> End

    classDef tradingagents fill:#1e3a8a,stroke:#3b82f6,color:#fff
    classDef kronos fill:#c2410c,stroke:#f97316,color:#fff
    classDef hybrid fill:#15803d,stroke:#22c55e,color:#fff
    classDef infra fill:#374151,stroke:#9ca3af,color:#fff

    class MA,SA,NA,FA,RM tradingagents
    class KA kronos
    class BULL,BEAR,TR,AGG,CON,NEU,PM hybrid
    class Start,End infra
```

**Catatan:**
- Node `1-4` 100% TradingAgents existing — tidak diubah
- Node `5` (Forecast Analyst) 100% Kronos — tidak ada LLM call di sini
- Node `6-7, 9-13` adalah TradingAgents tapi **prompt/logic-nya dimodifikasi** untuk membaca output Kronos
- Node `8` (Research Manager) tidak diubah — dia membaca debate yang sudah di-inform oleh Kronos
- Loop di Market/News/Fundamentals = tool-calling loop (LangGraph conditional edge)
- Loop Bull↔Bear dan Risk debate = counter-based conditional edge

---

## 2. Detail State Mutation

Bagaimana state berubah di setiap layer (siapa nulis apa):

```mermaid
flowchart LR
    subgraph S0["Initial State"]
        S0_data["company_of_interest=NVDA<br/>trade_date=2024-05-10<br/>past_context=...<br/>(rest empty)"]
    end

    subgraph S1["After Analyst Layer (5 nodes)"]
        S1_data["+ market_report 🔵<br/>+ sentiment_report 🔵<br/>+ news_report 🔵<br/>+ fundamentals_report 🔵<br/>+ forecast_report 🟠 NEW<br/>+ forecast_summary 🟠 NEW (dict)"]
    end

    subgraph S2["After Research Layer"]
        S2_data["+ investment_debate_state 🔵<br/>(bull_history, bear_history,<br/>history, count, judge_decision)<br/>+ investment_plan 🔵"]
    end

    subgraph S3["After Trader"]
        S3_data["+ trader_investment_plan 🟢<br/>(now informed by forecast)"]
    end

    subgraph S4["After Risk Layer"]
        S4_data["+ risk_debate_state 🟢<br/>(3-way history, count)<br/>+ final_trade_decision 🟢<br/>+ kronos_alignment 🟠 NEW<br/>+ kronos_divergence_reason 🟠 NEW"]
    end

    S0 -->|Layer 1| S1
    S1 -->|Layer 2| S2
    S2 -->|Layer 3| S3
    S3 -->|Layer 4| S4

    classDef s0 fill:#374151,stroke:#9ca3af,color:#fff
    classDef s1 fill:#312e81,stroke:#6366f1,color:#fff
    classDef s2 fill:#1e40af,stroke:#3b82f6,color:#fff
    classDef s3 fill:#15803d,stroke:#22c55e,color:#fff
    classDef s4 fill:#7f1d1d,stroke:#ef4444,color:#fff

    class S0 s0
    class S1 s1
    class S2 s2
    class S3 s3
    class S4 s4
```

**Penjelasan field source:**

| Field State | Sumber Logic | Kategori |
|---|---|---|
| `market_report` | Market Analyst (LLM) | 🔵 TradingAgents |
| `sentiment_report` | Sentiment Analyst (LLM + pre-fetched data) | 🔵 TradingAgents |
| `news_report` | News Analyst (LLM) | 🔵 TradingAgents |
| `fundamentals_report` | Fundamentals Analyst (LLM) | 🔵 TradingAgents |
| `forecast_report` | Forecast Analyst (Kronos.predict + renderer) | 🟠 Kronos |
| `forecast_summary` | forecast_stats.compute_forecast_summary() | 🟠 Kronos |
| `investment_debate_state` | Bull + Bear Researcher (LLM, prompt include forecast) | 🟢 Hybrid |
| `investment_plan` | Research Manager (LLM, deep) | 🔵 TradingAgents |
| `trader_investment_plan` | Trader (LLM + sizing rules dari forecast_summary) | 🟢 Hybrid |
| `risk_debate_state` | 3 risk debaters (LLM, prompt include forecast metrics) | 🟢 Hybrid |
| `final_trade_decision` | Portfolio Manager (LLM, deep, + alignment check) | 🟢 Hybrid |
| `kronos_alignment` | Portfolio Manager (LLM structured output) | 🟠 Kronos-driven field |
| `kronos_divergence_reason` | Portfolio Manager (LLM, conditional on alignment) | 🟠 Kronos-driven field |

---

## 3. Detail Forecast Analyst Internal

Apa yang terjadi di dalam node Forecast Analyst (zoom-in):

```mermaid
flowchart TD
    Input["Input dari state:<br/>ticker, trade_date"]

    subgraph KronosFlow["🟠 Kronos Pipeline (no LLM)"]
        direction TB
        CacheCheck{"Cache hit?<br/>key: ticker_date_params"}
        FetchData["Fetch OHLCV history<br/>via yfinance (reuse dataflows/y_finance.py)<br/>lookback bars (default 400)"]
        LoadModel["Lazy load:<br/>KronosTokenizer.from_pretrained()<br/>Kronos.from_pretrained()<br/>(once per process)"]
        Predict["predictor.predict()<br/>sample_count=10<br/>T=1.0, top_p=0.9<br/>pred_len=10 bars"]
        Stats["compute_forecast_summary()<br/>median_return, p5, p95<br/>prob_positive, vol<br/>drawdown, VaR"]
        Render["render_forecast_report()<br/>→ markdown string"]
        SaveCache["Save to disk cache<br/>~/.tradingagents/cache/kronos/"]
        LoadCache["Load from cache"]
    end

    Output["Return state update:<br/>{forecast_report: str,<br/>forecast_summary: dict}"]

    Input --> CacheCheck
    CacheCheck -->|Yes| LoadCache
    LoadCache --> Output
    CacheCheck -->|No| FetchData
    FetchData --> LoadModel
    LoadModel --> Predict
    Predict --> Stats
    Stats --> Render
    Render --> SaveCache
    SaveCache --> Output

    classDef kronos fill:#c2410c,stroke:#f97316,color:#fff
    classDef io fill:#374151,stroke:#9ca3af,color:#fff
    class CacheCheck,FetchData,LoadModel,Predict,Stats,Render,SaveCache,LoadCache kronos
    class Input,Output io
```

**Karakteristik node ini:**
- Tidak ada LLM call → cepat (~1-3s GPU, ~5-15s CPU first time, instant kalau cache hit)
- Deterministic untuk seed yang sama (kalau set `T=0` atau cache)
- Tidak update `messages` (langsung skip Msg Clear)
- Dipanggil hanya kalau `config["kronos_enabled"] = True`

---

## 4. Conflict Resolution (LLM vs Kronos)

Decision logic di Portfolio Manager:

```mermaid
flowchart TD
    Start["PM dipanggil dengan:<br/>4 reports + forecast_report +<br/>investment_plan + trader_plan +<br/>risk_history + forecast_summary"]

    LLMCall["LLM (deep_thinking_llm) generate:<br/>rating + thesis + target +<br/>kronos_alignment field"]

    CheckAlign{"alignment field"}

    StrongAlign["strongly_aligned<br/>🟢 confidence boost"]
    Aligned["aligned<br/>🟢 normal flow"]
    Neutral["neutral<br/>⚪ normal flow"]
    Divergent["divergent<br/>🟡 require<br/>kronos_divergence_reason"]
    StrongDiv["strongly_divergent<br/>🔴 require<br/>kronos_divergence_reason"]

    StrictMode{"strict_mode<br/>enabled?"}
    Downgrade["⚠️ Auto-downgrade<br/>STRONG_BUY → BUY<br/>BUY → HOLD<br/>etc."]
    KeepRating["Keep LLM rating<br/>tapi flag jelas di output"]

    Final["final_trade_decision +<br/>kronos_alignment +<br/>kronos_divergence_reason"]

    Start --> LLMCall
    LLMCall --> CheckAlign
    CheckAlign -->|strongly_aligned| StrongAlign
    CheckAlign -->|aligned| Aligned
    CheckAlign -->|neutral| Neutral
    CheckAlign -->|divergent| Divergent
    CheckAlign -->|strongly_divergent| StrongDiv

    StrongAlign --> Final
    Aligned --> Final
    Neutral --> Final
    Divergent --> Final
    StrongDiv --> StrictMode
    StrictMode -->|Yes| Downgrade
    StrictMode -->|No| KeepRating
    Downgrade --> Final
    KeepRating --> Final

    classDef good fill:#15803d,stroke:#22c55e,color:#fff
    classDef neutral fill:#374151,stroke:#9ca3af,color:#fff
    classDef warn fill:#a16207,stroke:#eab308,color:#fff
    classDef bad fill:#7f1d1d,stroke:#ef4444,color:#fff

    class StrongAlign,Aligned good
    class Neutral neutral
    class Divergent,Downgrade warn
    class StrongDiv bad
```

**Threshold untuk classify alignment** (di `forecast_stats.py`):

| Alignment | Kondisi |
|---|---|
| `strongly_aligned` | Direction sama + median \|return\| > 3% + prob_positive selaras (>65% kalau bull, <35% kalau bear) |
| `aligned` | Direction sama + magnitude moderate |
| `neutral` | Salah satu close to zero (median return between -1% to +1%) |
| `divergent` | Direction beda + magnitude moderate |
| `strongly_divergent` | LLM=STRONG_BUY tapi p95_return < 0; atau LLM=STRONG_SELL tapi p5_return > 0 |

---

## 5. Conditional Flow: Kronos On vs Off

Bagaimana graph wiring berbeda berdasarkan config:

```mermaid
flowchart TD
    Config{"config.kronos_enabled?"}

    subgraph EnabledFlow["🟢 kronos_enabled = True"]
        direction TB
        F1["Fundamentals.MsgClear"] --> F2["Forecast Analyst 🟠"]
        F2 --> F3["Bull Researcher 🟢<br/>(prompt include forecast)"]
    end

    subgraph DisabledFlow["⚪ kronos_enabled = False"]
        direction TB
        D1["Fundamentals.MsgClear"] --> D2["Bull Researcher 🔵<br/>(original prompt, no forecast)"]
    end

    Config -->|True| EnabledFlow
    Config -->|False| DisabledFlow

    classDef on fill:#15803d,stroke:#22c55e,color:#fff
    classDef off fill:#374151,stroke:#9ca3af,color:#fff
    class F1,F2,F3 on
    class D1,D2 off
```

**Implementasi di `setup.py`:**
```python
if self.config.get("kronos_enabled"):
    workflow.add_node("Forecast Analyst", forecast_node)
    workflow.add_edge(last_analyst_clear, "Forecast Analyst")
    workflow.add_edge("Forecast Analyst", "Bull Researcher")
else:
    workflow.add_edge(last_analyst_clear, "Bull Researcher")
```

---

## 6. Komponen Source Map

Mapping file ke sumber logic:

```mermaid
graph LR
    subgraph TA["🔵 TradingAgents (existing, tidak berubah)"]
        TA1[main.py]
        TA2[graph/setup.py base wiring]
        TA3[graph/conditional_logic.py]
        TA4[graph/propagation.py]
        TA5[agents/analysts/* 4 files]
        TA6[agents/managers/research_manager.py]
        TA7[agents/utils/agent_utils.py]
        TA8[dataflows/y_finance.py]
        TA9[llm_clients/*]
    end

    subgraph K["🟠 Kronos (NEW, folder forecast/)"]
        K1[forecast/kronos_client.py]
        K2[forecast/forecast_node.py]
        K3[forecast/forecast_renderer.py]
        K4[forecast/forecast_stats.py]
        K5[forecast/config.py]
    end

    subgraph H["🟢 Hybrid (modified)"]
        H1[graph/setup.py + conditional wiring]
        H2[graph/trading_graph.py + kronos init]
        H3[agents/utils/agent_states.py + 2 fields]
        H4[agents/researchers/bull_researcher.py prompt]
        H5[agents/researchers/bear_researcher.py prompt]
        H6[agents/trader/trader.py + sizing rules]
        H7[agents/risk_mgmt/* 3 files prompt]
        H8[agents/managers/portfolio_manager.py prompt]
        H9[agents/schemas.py + alignment fields]
        H10[default_config.py + kronos_* keys]
    end

    K1 -->|imported by| K2
    K1 -->|imported by| H2
    K2 -->|wired by| H1
    K3 -->|used by| K2
    K4 -->|used by| K2
    K4 -->|used by| H6
    K4 -->|used by| H7
    K4 -->|used by| H8
    H3 -->|state schema| H1
    H1 -->|wires graph for| TA2

    classDef ta fill:#1e3a8a,stroke:#3b82f6,color:#fff
    classDef kr fill:#c2410c,stroke:#f97316,color:#fff
    classDef hy fill:#15803d,stroke:#22c55e,color:#fff
    class TA1,TA2,TA3,TA4,TA5,TA6,TA7,TA8,TA9 ta
    class K1,K2,K3,K4,K5 kr
    class H1,H2,H3,H4,H5,H6,H7,H8,H9,H10 hy
```

**Ringkasan effort per kategori:**

| Kategori | Jumlah file | Effort |
|---|---|---|
| 🔵 TradingAgents existing (tidak berubah) | ~25 | 0 (cuma reuse) |
| 🟠 Kronos baru | 5 | ~3-4 hari (Phase 1-2) |
| 🟢 Hybrid (modified) | 10 | ~5-7 hari (Phase 3-5) |
| **Total integrasi** | 15 file (5 baru + 10 modified) | **~10-14 hari** |

---

## 7. Sequence Diagram: 1 Run Lengkap

Timeline dari user invoke sampai output:

```mermaid
sequenceDiagram
    actor User
    participant TG as TradingAgentsGraph
    participant LG as LangGraph (compiled)
    participant LLM as LLM (OpenAI/etc)
    participant K as KronosClient 🟠
    participant YF as yfinance
    participant Mem as TradingMemoryLog

    User->>TG: propagate("NVDA", "2024-05-10")
    TG->>Mem: resolve_pending_entries(NVDA)
    Mem-->>TG: ok

    TG->>LG: graph.invoke(initial_state)

    Note over LG: ANALYST LAYER

    rect rgb(30, 58, 138)
        LG->>LLM: Market Analyst prompt
        LLM-->>LG: tool_call: get_stock_data
        LG->>YF: fetch OHLCV
        YF-->>LG: data
        LG->>LLM: continue
        LLM-->>LG: market_report
    end

    rect rgb(30, 58, 138)
        LG->>LLM: Sentiment + News + Fundamentals
        LLM-->>LG: 3 reports
    end

    rect rgb(194, 65, 12)
        Note over LG,K: 🟠 NEW: Forecast Analyst
        LG->>K: predict(NVDA, 2024-05-10)
        alt cache hit
            K-->>LG: cached forecast
        else cache miss
            K->>YF: fetch lookback OHLCV
            YF-->>K: history df
            K->>K: load Kronos model (lazy)
            K->>K: predict(sample_count=10)
            K->>K: compute_forecast_summary
            K->>K: render markdown
            K-->>LG: forecast_report + summary
        end
    end

    Note over LG: RESEARCH LAYER

    rect rgb(21, 128, 61)
        loop max_debate_rounds
            LG->>LLM: Bull (with forecast)
            LLM-->>LG: bull statement
            LG->>LLM: Bear (with forecast)
            LLM-->>LG: bear statement
        end
        LG->>LLM: Research Manager (deep)
        LLM-->>LG: investment_plan
    end

    Note over LG: EXECUTION

    rect rgb(21, 128, 61)
        LG->>LG: compute sizing/stop hints from summary
        LG->>LLM: Trader (with hints)
        LLM-->>LG: trader_plan
    end

    Note over LG: RISK LAYER

    rect rgb(21, 128, 61)
        loop max_risk_rounds
            LG->>LLM: Aggressive (forecast metrics)
            LLM-->>LG: argument
            LG->>LLM: Conservative (forecast metrics)
            LLM-->>LG: argument
            LG->>LLM: Neutral (forecast metrics)
            LLM-->>LG: argument
        end
        LG->>LLM: Portfolio Manager (deep, alignment check)
        LLM-->>LG: final_decision + kronos_alignment
    end

    LG-->>TG: final_state

    TG->>Mem: store_decision (pending)
    TG->>TG: log_state to disk
    TG-->>User: (final_state, decision)
```

**Highlight:**
- Box biru tua = TradingAgents original
- Box oranye = Kronos NEW (single insertion point)
- Box hijau = LLM call dengan prompt yang dimodifikasi (input + forecast)

---

## 8. Data Flow: Bagaimana Forecast Dikonsumsi

Visualisasi bagaimana 1 forecast (output Kronos) dipakai oleh banyak downstream agent:

```mermaid
flowchart LR
    K["🟠 Forecast Analyst<br/>output:<br/>• forecast_report (markdown)<br/>• forecast_summary (dict)"]

    subgraph Consumers["Yang konsumsi forecast"]
        Bull["Bull Researcher<br/>baca: forecast_report<br/>untuk: cite metrics di argument"]
        Bear["Bear Researcher<br/>baca: forecast_report<br/>untuk: cite metrics di argument"]
        Trader["Trader<br/>baca: forecast_summary (dict)<br/>untuk: sizing & stop-loss calc"]
        Agg["Aggressive Debator<br/>baca: forecast_summary<br/>untuk: argue prob_positive, p95_return"]
        Cons["Conservative Debator<br/>baca: forecast_summary<br/>untuk: argue p95_drawdown, VaR"]
        Neu["Neutral Debator<br/>baca: forecast_summary<br/>untuk: argue median + spread"]
        PM["Portfolio Manager<br/>baca: forecast_report + summary<br/>untuk: alignment check + justify"]
    end

    K --> Bull
    K --> Bear
    K --> Trader
    K --> Agg
    K --> Cons
    K --> Neu
    K --> PM

    classDef kronos fill:#c2410c,stroke:#f97316,color:#fff
    classDef hybrid fill:#15803d,stroke:#22c55e,color:#fff
    class K kronos
    class Bull,Bear,Trader,Agg,Cons,Neu,PM hybrid
```

**Insight:** 1 forecast → 7 consumers. Cost amortized: meskipun Kronos predict mahal (~1-3s), hasilnya dipakai di 7 tempat. Sangat efisien.

---

## Penutup

Total ada **8 diagram** di dokumen ini:

1. **Flow utama end-to-end** — keseluruhan pipeline
2. **Detail state mutation** — apa yang berubah di setiap layer
3. **Forecast Analyst internal** — zoom-in ke node Kronos
4. **Conflict resolution** — decision tree alignment
5. **Conditional flow on/off** — kalau Kronos disabled
6. **Komponen source map** — file mana milik siapa
7. **Sequence diagram** — timeline 1 run
8. **Data flow forecast consumption** — siapa baca apa

**Cara render:**
- Buka di GitHub langsung — Mermaid auto-render
- VS Code dengan extension "Markdown Preview Mermaid Support"
- Obsidian, Typora, atau viewer lain yang support Mermaid
- Kalau cuma plain markdown viewer, akan tampak sebagai code block

**Untuk implementasi:** Diagram 1 (urutan node) dan diagram 6 (file map) yang paling sering kamu rujuk pas coding.
