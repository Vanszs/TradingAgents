# TradingAgents × Kronos — Integration Plan

> Dokumen ini merancang penggabungan **TradingAgents** (multi-agent LLM trading framework) dengan **Kronos** (foundation model untuk forecasting candlestick / K-line). Ditulis dengan style yang sama seperti `READING_GUIDE.md` — pelan, dalam, low-level technical, untuk dibaca dulu sebelum implementasi.
>
> Asumsi pembaca: sudah membaca `READING_GUIDE.md`, paham LangGraph state + flow, paham basic neural network forecasting.

---

## Daftar Isi

1. [TL;DR + POV Saya](#1-tldr--pov-saya)
2. [Apa Itu Kronos](#2-apa-itu-kronos)
3. [Kenapa Kombinasi Ini Masuk Akal](#3-kenapa-kombinasi-ini-masuk-akal)
4. [Tantangan & Pitfalls](#4-tantangan--pitfalls)
5. [4 Opsi Arsitektur Integrasi](#5-4-opsi-arsitektur-integrasi)
6. [Arsitektur Pilihan: Hybrid Layered](#6-arsitektur-pilihan-hybrid-layered)
7. [Komponen Baru yang Perlu Dibuat](#7-komponen-baru-yang-perlu-dibuat)
8. [Modifikasi pada TradingAgents Existing](#8-modifikasi-pada-tradingagents-existing)
9. [State Schema Updates](#9-state-schema-updates)
10. [Flow Eksekusi Lengkap](#10-flow-eksekusi-lengkap)
11. [Conflict Resolution: LLM vs Forecast](#11-conflict-resolution-llm-vs-forecast)
12. [Konfigurasi Baru](#12-konfigurasi-baru)
13. [Roadmap Implementasi (Phase 1-5)](#13-roadmap-implementasi-phase-1-5)
14. [Risiko Teknis](#14-risiko-teknis)
15. [Pertanyaan Terbuka](#15-pertanyaan-terbuka)
16. [Glossarium Tambahan](#16-glossarium-tambahan)

---

## 1. TL;DR + POV Saya

### Verdict: **Ide bagus**, tapi non-trivial. Realistic effort: 2-4 minggu untuk integrasi yang solid.

### Kenapa bagus
1. **Komplementer, bukan redundan**:
   - LLM (TradingAgents) = bagus untuk narasi, konteks, news interpretation, multi-faktor reasoning.
   - Kronos = bagus untuk pattern statistik di data harga (trend, momentum, volatility).
   - Mereka mengisi *blind spot* satu sama lain.
2. **Open source, ringan**: Kronos-small cuma 24.7M params, bisa run di GPU consumer atau bahkan CPU (dengan latency wajar).
3. **Pre-trained on 45 exchanges**: zero-shot capable untuk banyak ticker tanpa fine-tune. Cocok untuk prototype cepat.
4. **Probabilistic output**: bisa sample beberapa path → ada distribusi → bisa jadi input untuk risk debate (VaR, expected drawdown).

### Kenapa berhati-hati
1. **Forecast horizon mismatch**: Kronos mengeluarkan OHLCV step-by-step (e.g. 120 periode). TradingAgents berpikir dalam horizon trading (5 hari hold). Perlu mapping eksplisit.
2. **Conflict resolution belum jelas**: kalau Kronos predict turun dan LLM bilang BUY, siapa yang menang? Default behavior harus didesain.
3. **Kompute tambahan**: GPU+model loading bikin tiap run lebih lambat dan butuh resource. Kalau jalan di laptop, perlu strategi caching/lazy loading.
4. **Foundation model bias**: Kronos dilatih pada data sampai cutoff tertentu. Untuk asset yang mode-nya berubah drastis (regime change), forecast bisa misleading.

### Kalau cuma satu kalimat
> "Tambah Kronos sebagai *quantitative second opinion* di analyst layer dan risk layer — biarkan LLM tetap pegang keputusan akhir, tapi dengan akses ke prediksi numerik probabilistik."

Saya **tidak** rekomendasi Kronos sebagai pengganti decision-maker (replace Portfolio Manager). LLM tetap di puncak hierarki, Kronos jadi tool yang dipanggil.

---

## 2. Apa Itu Kronos

### Definisi 1 kalimat
**Foundation model decoder-only Transformer yang pre-trained pada candlestick (OHLCV) sequences dari 45+ bursa global, mampu predict OHLCV future autoregressively.**

### Cara kerja konseptual

Mirip GPT untuk teks, tapi untuk K-line:

1. **Tokenizer** (separate model) — menerima continuous OHLCV → quantize jadi discrete tokens hierarchical.
   - Input: tabel `[open, high, low, close, volume, amount]` per timestamp
   - Output: sequence of token IDs
2. **Autoregressive Transformer** — predict token berikutnya berdasarkan token-token sebelumnya.
   - Loss: cross-entropy antara token prediksi dan ground truth
3. **Inverse tokenizer** — token kembali ke OHLCV continuous.

### Model zoo

| Model | Tokenizer | Context | Params | Kasus pakai |
|---|---|---|---|---|
| Kronos-mini | Tokenizer-2k | 2048 | 4.1M | Resource-constrained, eksperimen cepat |
| Kronos-small | Tokenizer-base | 512 | 24.7M | Default rekomendasi (cepat + akurat) |
| Kronos-base | Tokenizer-base | 512 | 102.3M | Akurasi lebih tinggi |
| Kronos-large | Tokenizer-base | 512 | 499.2M | Closed source (tidak rilis) |

Untuk integrasi dengan TradingAgents → mulai dengan **Kronos-small** (24.7M, 512 context).

### API (penting untuk integrasi)

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

pred_df = predictor.predict(
    df=x_df,                  # pandas DF: [open, high, low, close, volume?, amount?]
    x_timestamp=x_timestamp,  # historical timestamps
    y_timestamp=y_timestamp,  # future timestamps untuk predict
    pred_len=120,
    T=1.0,                    # temperature (probabilistic)
    top_p=0.9,                # nucleus sampling
    sample_count=1,           # berapa path di-sample lalu di-average
)
# Return: pandas DataFrame [open, high, low, close, volume, amount] x pred_len
```

**Karakteristik penting:**
- **Probabilistic**: dengan `sample_count > 1`, kita dapat distribusi forecast → bisa derive confidence interval, percentile, dst.
- **Lookback constrained**: Kronos-small max context 512 timestamps. Kalau 5-min bar → ~42 jam. Kalau daily bar → ~512 hari (~2 tahun).
- **OHLCV-only**: tidak terima feature tambahan (sentiment, news embedding, dll). Ini limitasi.
- **Multi-asset agnostic**: zero-shot per ticker, tidak perlu re-train.

### Yang TIDAK dikerjakan Kronos (penting untuk arsitektur)

- ❌ Tidak baca news / sentiment / fundamentals
- ❌ Tidak menjelaskan kenapa harga bergerak (no narrative)
- ❌ Tidak rekomendasi BUY/HOLD/SELL — outputnya angka raw
- ❌ Tidak handle event-driven (earnings, FOMC) — purely pattern-based
- ❌ Tidak risk-aware (no position sizing, stop loss)

**Kesimpulan**: Kronos = **statistical pattern recognizer**. Sangat sempit fokus, tapi sangat tajam di domain itu.

---

## 3. Kenapa Kombinasi Ini Masuk Akal

### Komplementaritas (visual)

```
                  TradingAgents (LLM)         Kronos (Forecast Model)
                  ─────────────────────       ──────────────────────
Reasoning         ✅ multi-factor              ❌ pure pattern
News/Events       ✅ deep                       ❌ blind
Fundamentals      ✅ deep                       ❌ blind
Sentiment         ✅ moderate                   ❌ blind
Pattern (price)   ⚠️ shallow                    ✅ deep
Probabilistic     ⚠️ vague                       ✅ explicit (sample paths)
Long-term horiz.  ✅ adapts via prompt          ⚠️ depends on lookback
Short-term tick   ⚠️ tidak granular             ✅ designed for it
Speed (1 query)   🐢 ~30-60s                    🚀 ~1-3s
Cost              💰 LLM API tokens             💵 GPU compute
Numeric output    ⚠️ tidak ideal                 ✅ native
Audit trail       ✅ verbose                     ⚠️ blackbox
```

### Use case konkret di mana Kronos memperbaiki TradingAgents

**Kasus 1 — Trader sizing:**
- Sebelum: Trader propose entry $X dengan stop $Y. Sizing manual based on prompt.
- Sesudah: Kronos predict 120 periode ke depan + forecast volatility (std dari sample paths). Trader pakai itu untuk sizing dengan ATR-like rule yang lebih grounded.

**Kasus 2 — Risk debate:**
- Sebelum: Conservative analyst bilang "ini terlalu volatile". Subjective.
- Sesudah: Conservative bisa sebut "Kronos forecast 95th percentile drawdown -7.2% dalam 5 hari, posisi terlalu besar".

**Kasus 3 — Bull/Bear debate:**
- Sebelum: Bull bilang "uptrend kuat". Bukti? Indikator teknikal.
- Sesudah: Bull bisa sebut "Kronos median forecast +4.1% dalam 10 periode, sample 70% paths positive". Bear bisa kounter "tapi 30% paths negatif dengan median drawdown -3%, dan Kronos volatility forecast naik".

**Kasus 4 — Portfolio Manager final:**
- Sebelum: Rating dari reasoning kualitatif.
- Sesudah: PM sees both narrative (LLM debate) and quantitative (Kronos distribution). Lebih siap dengan pertanyaan "berapa expected return-mu? berapa downside?".

---

## 4. Tantangan & Pitfalls

### 4.1 Horizon mismatch
Kronos predict step-by-step (e.g. 120 candle ke depan). TradingAgents berpikir dalam holding period (5 hari).

- **Mitigasi**: Pilih frekuensi data + pred_len yang match dengan holding period.
  - Daily candle, pred_len=10 → forecast 10 hari (cukup untuk hold 5)
  - 1-hour candle, pred_len=120 → 5 hari (~120 trading hours)
  - 15-min candle, pred_len=200 → ~1 hari (untuk intraday)
- Tetapi: Kronos akurasi turun seiring horizon panjang. Long horizon = noisy.

### 4.2 Kompute & latency

Tiap run TradingAgents tambah:
- 1 model load (~2-5 detik first time, cached after)
- N predict call (1-3 detik per call kalau GPU, 10+ detik kalau CPU)

- **Mitigasi**: 
  - Lazy load Kronos sekali di `TradingAgentsGraph.__init__`
  - Cache forecast per `(ticker, trade_date, pred_len)` di disk (sama folder dengan `data_cache_dir`)
  - Optional: skip Kronos kalau `kronos_enabled=False` di config

### 4.3 Conflict resolution

Skenario: LLM (Portfolio Manager) bilang STRONG_BUY, Kronos forecast median return -3% (bearish).

Pilihan:
- **A**: LLM tetap menang (dia sudah lihat Kronos data, kalau tetap BUY berarti factor lain dominan)
- **B**: Override mechanism — kalau Kronos bertentangan kuat (>2σ), force HOLD
- **C**: Veto — Kronos tidak boleh override, tapi Portfolio Manager harus secara eksplisit acknowledge dan justify perbedaan
- **D**: Confidence-weighted — kalau LLM low confidence + Kronos high confidence, follow Kronos

**Rekomendasi**: Pilihan **C** untuk default. LLM tetap pegang final, tapi PM prompt mengharuskan justification eksplisit kalau bertentangan dengan Kronos. Ini memaksa reasoning quality + audit-friendly.

### 4.4 Kronos blind spots

Kronos tidak tahu:
- Akan ada earnings besok
- CEO baru saja resign
- Regulator baru saja ban produk

Pattern-pattern ini bukan ada di OHLCV history. Forecast Kronos akan "salah" dalam scenario event-driven.

- **Mitigasi**: Gabungkan dengan News Analyst output. Kalau ada catalyst-pending detected, tag "kronos_caution=true" dan PM disuruh diskon prediksi Kronos.

### 4.5 Distribution shift

Kronos pre-trained pada data sampai tanggal X. Kalau pasar masuk regime baru (e.g. krisis 2020 saat Kronos di-train cuma sampai 2018), forecast bisa systematically off.

- **Mitigasi**: Optional fine-tune pada data terbaru ticker yang sering kamu trade.

### 4.6 Over-reliance risk

Kalau LLM diberi forecast Kronos, bisa-bisa LLM jadi malas reasoning sendiri ("Kronos bilang naik, gw setuju"). Sycophancy ke model lain.

- **Mitigasi**: Prompt eksplisit minta LLM **challenge** Kronos forecast, sebut sumber inconsistency, ekspos ketidakpastian.


---

## 5. 4 Opsi Arsitektur Integrasi

Sebelum memilih, mari pertimbangkan 4 cara berbeda menyatukan Kronos:

### Opsi A — **Kronos sebagai Tool**

Kronos dibungkus sebagai LangChain `@tool`, dipanggil oleh Market Analyst (atau analyst baru) saat tool-calling loop.

```
Market Analyst → tool: get_kronos_forecast(ticker, lookback, pred_len) → return DF
```

**Pro:**
- Minimal change ke arsitektur — cuma tambah 1 tool
- LLM yang putuskan kapan panggil
- Bisa skip kalau LLM tidak butuh

**Kontra:**
- LLM mungkin tidak panggil sama sekali kalau tidak diprompt eksplisit
- Output forecast (numeric) sulit dicerna LLM tanpa pra-processing
- Tidak terstruktur untuk dipakai layer lain (Trader, Risk)

### Opsi B — **Kronos sebagai Analyst Node Baru** ("Forecast Analyst")

Tambah node `Forecast Analyst` di analyst layer. Tidak pakai LLM — purely Python wrapper di Kronos. Outputnya jadi report ke-5.

```
Market → Sentiment → News → Fundamentals → Forecast (Kronos) → Bull/Bear
```

**Pro:**
- Forecast jadi first-class artifact (`forecast_report` di state)
- Semua agent downstream bisa baca
- Deterministic (no LLM call) → cepat & cheap
- Mudah disable (cuma flag config)

**Kontra:**
- Output Kronos (DF angka) harus dirender jadi narrative untuk LLM. Butuh layer translasi.
- Membuat analyst layer asymmetric (4 LLM-driven + 1 model-driven)

### Opsi C — **Kronos di Trader & Risk Layer**

Kronos cuma dipakai di execution + risk debate, bukan di analyst layer. Jadi LLM analyst tidak tahu Kronos. Trader baru pakai Kronos buat sizing/stop. Risk debaters pakai distribusi Kronos buat argument.

**Pro:**
- Kronos dipakai di tempat numerik penting (entry/sizing, risk metric)
- Tidak mempengaruhi reasoning analyst (mempertahankan independen analisis)

**Kontra:**
- Analyst & researcher tidak benefit dari Kronos
- Bull/Bear debate kehilangan satu data point penting

### Opsi D — **Hybrid Layered** (rekomendasi)

Kombinasi B + C: **Forecast Analyst node** + **Kronos consumed di Trader & Risk debate**.

Detail di section berikutnya.

### Tabel keputusan

| Aspek | A (Tool) | B (Analyst) | C (Trader+Risk) | D (Hybrid) |
|---|---|---|---|---|
| Effort impl | Rendah | Menengah | Menengah | Tinggi |
| Cakupan benefit | Sempit | Luas | Luas | Sangat luas |
| Determinism | Rendah | Tinggi | Tinggi | Tinggi |
| LLM-friendly | ⚠️ | ✅ (kalau ada renderer) | ⚠️ | ✅ |
| Audit | ⚠️ | ✅ | ✅ | ✅ |
| Recommended | ❌ | OK | OK | **✅** |

---

## 6. Arsitektur Pilihan: Hybrid Layered

```
                              INPUT
                       ticker + trade_date
                              │
                              ▼
   ┌──────────────────── ANALYST LAYER (5 nodes) ─────────────────┐
   │                                                              │
   │  Market Analyst   ⇄ tools(get_stock_data, indicators)        │
   │       │                                                      │
   │       ▼                                                      │
   │  Sentiment Analyst (no tools, pre-fetched)                   │
   │       │                                                      │
   │       ▼                                                      │
   │  News Analyst     ⇄ tools(get_news, get_global_news, ...)   │
   │       │                                                      │
   │       ▼                                                      │
   │  Fundamentals     ⇄ tools(financials, balance, cashflow,    │
   │  Analyst                income)                              │
   │       │                                                      │
   │       ▼                                                      │
   │  ★ Forecast Analyst (Kronos, deterministic)                  │
   │     → Fetch OHLCV history (yfinance)                         │
   │     → Run Kronos.predict() with sample_count=N               │
   │     → Compute summary stats (median, p5, p95, vol)           │
   │     → Render markdown report                                 │
   │                                                              │
   └──────────────────────────┬───────────────────────────────────┘
                              │ produces: market/sent/news/fund/forecast reports
                              ▼
   ┌──────────────────── RESEARCH LAYER ──────────────────────────┐
   │                                                              │
   │     Bull Researcher  ⇄  Bear Researcher                      │
   │     (kedua-nya baca forecast_report)                         │
   │                  │                                           │
   │                  ▼                                           │
   │           Research Manager                                   │
   │                                                              │
   └──────────────────────────┬───────────────────────────────────┘
                              │ produces: investment_plan
                              ▼
   ┌──────────────────── EXECUTION LAYER ─────────────────────────┐
   │                                                              │
   │     ★ Trader (modified)                                      │
   │       - Baca investment_plan + forecast_report               │
   │       - Pakai forecast volatility untuk ATR-like sizing      │
   │       - Pakai forecast percentile untuk stop-loss            │
   │                                                              │
   └──────────────────────────┬───────────────────────────────────┘
                              │ produces: trader_investment_plan
                              ▼
   ┌──────────────────── RISK LAYER ──────────────────────────────┐
   │                                                              │
   │     ★ Aggressive  →  Conservative  →  Neutral                │
   │       (semua baca forecast_summary; risk metrics dari        │
   │       distribusi forecast: VaR, max drawdown 95p)            │
   │                       │                                      │
   │                       ▼                                      │
   │     ★ Portfolio Manager                                      │
   │       - Wajib justify kalau decision contradict Kronos       │
   │       - Output structured + Kronos alignment flag            │
   │                                                              │
   └──────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                        OUTPUT + reflection
```

**Warna `★` = node yang berubah/baru.**

### Kenapa hybrid lebih kuat dari B atau C saja

- Forecast Analyst di B → semua LLM agent benefit
- Kronos di Trader+Risk → numerik dipakai untuk metric konkret
- Bersama: forecast dipakai sebagai *evidence* di reasoning + *parameter* di kalkulasi

---

## 7. Komponen Baru yang Perlu Dibuat

### 7.1 Module `tradingagents/forecast/` (folder baru)

```
tradingagents/forecast/
├── __init__.py
├── kronos_client.py        ← Wrapper Kronos: load model, predict, cache
├── forecast_node.py        ← LangGraph node: Forecast Analyst
├── forecast_renderer.py    ← Convert pred_df → markdown report
├── forecast_stats.py       ← Compute median/percentile/VaR/etc dari sample paths
└── config.py               ← Forecast-specific config
```

#### `kronos_client.py` — Wrapper Kronos

Tujuan: ringan dipanggil, lazy load, cache.

```python
class KronosClient:
    def __init__(self, model_name="NeoQuasar/Kronos-small",
                 tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 max_context=512,
                 cache_dir=None):
        self.device = device
        self.cache_dir = cache_dir
        self._predictor = None  # lazy
        self._model_name = model_name
        self._tokenizer_name = tokenizer_name
        self._max_context = max_context

    def _ensure_loaded(self):
        if self._predictor is None:
            from model import Kronos, KronosTokenizer, KronosPredictor
            tok = KronosTokenizer.from_pretrained(self._tokenizer_name)
            mdl = Kronos.from_pretrained(self._model_name)
            self._predictor = KronosPredictor(
                mdl, tok, max_context=self._max_context, device=self.device,
            )

    def predict(self, ticker, trade_date, lookback=400, pred_len=10,
                bar="1d", sample_count=10, T=1.0, top_p=0.9):
        # Cache key
        cache_key = f"{ticker}_{trade_date}_{bar}_{lookback}_{pred_len}_{sample_count}"
        cache_path = self._cache_path(cache_key)
        if cache_path.exists():
            return self._load_cache(cache_path)

        self._ensure_loaded()
        # Fetch historical OHLCV via yfinance
        hist_df = self._fetch_history(ticker, trade_date, lookback, bar)
        # Construct timestamps
        x_ts, y_ts = self._build_timestamps(hist_df, pred_len, bar)
        # Predict (with sample_count for distribution)
        pred_df = self._predictor.predict(
            df=hist_df[["open","high","low","close","volume","amount"]],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=T, top_p=top_p, sample_count=sample_count,
        )
        result = {
            "pred_df": pred_df,
            "history_df": hist_df,
            "meta": {"sample_count": sample_count, "bar": bar, ...}
        }
        self._save_cache(cache_path, result)
        return result
```

**Detail penting:**
- **Lazy load**: model di-load saat first `predict()` call, bukan saat `__init__`. Hemat startup time saat config `kronos_enabled=False`.
- **Cache disk**: per-(ticker, date, params). Run yang sama tidak panggil ulang model.
- **Fetch via yfinance**: reuse `dataflows/y_finance.py` yang sudah ada — jangan duplikasi data loading.

#### `forecast_stats.py` — Derive metrics dari distribusi

Karena `sample_count > 1`, kita dapat banyak path. Hitung:

```python
def compute_forecast_summary(pred_df_samples, current_price):
    """
    pred_df_samples: list of pred_df, satu per sample path
    Return dict dengan median, p5, p95 of return, expected vol, max drawdown.
    """
    closes = np.stack([df['close'].values for df in pred_df_samples])  # (N_samples, pred_len)
    returns = (closes[:, -1] - current_price) / current_price          # final return per path
    drawdowns = compute_drawdowns(closes, current_price)               # per path
    return {
        "median_return": np.median(returns),
        "p5_return": np.percentile(returns, 5),
        "p95_return": np.percentile(returns, 95),
        "prob_positive": np.mean(returns > 0),
        "expected_vol": np.std(np.diff(closes, axis=1) / closes[:, :-1]).mean(),
        "median_drawdown": np.median(drawdowns),
        "p95_drawdown": np.percentile(drawdowns, 95),
        "var_5pct": np.percentile(returns, 5),  # Value at Risk 5%
    }
```

**Output ini yang akan di-render ke markdown dan jadi input agent.**

#### `forecast_renderer.py` — Markdown render

```python
def render_forecast_report(summary, pred_df_median, current_price, pred_len, bar):
    return f"""
## Quantitative Forecast Report (Kronos)

**Setup:**
- Lookback: {summary['lookback']} {bar} bars
- Forecast horizon: {pred_len} {bar} bars
- Sample paths: {summary['sample_count']}

**Forecast distribution (terminal value):**
- Median return: {summary['median_return']*100:+.2f}%
- 5th percentile: {summary['p5_return']*100:+.2f}%  ← downside
- 95th percentile: {summary['p95_return']*100:+.2f}%  ← upside
- Probability positive: {summary['prob_positive']*100:.1f}%

**Risk metrics:**
- Expected volatility (per bar): {summary['expected_vol']*100:.2f}%
- Median max drawdown: {summary['median_drawdown']*100:+.2f}%
- 95th percentile drawdown: {summary['p95_drawdown']*100:+.2f}%
- VaR (5%): {summary['var_5pct']*100:+.2f}%

**Median path:**
| Step | Open | High | Low | Close |
|------|------|------|-----|-------|
{rows}

**Interpretation guide:**
- prob_positive > 60% + p5 > -3% → constructive signal
- p95_drawdown > 7% → high tail risk; prefer smaller sizing
- median diverges from p5/p95 widely → high uncertainty
"""
```

#### `forecast_node.py` — LangGraph node

```python
def create_forecast_analyst(kronos_client, config):
    def forecast_node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]

        result = kronos_client.predict(
            ticker, date,
            lookback=config["kronos_lookback"],
            pred_len=config["kronos_pred_len"],
            bar=config["kronos_bar"],
            sample_count=config["kronos_sample_count"],
        )
        summary = compute_forecast_summary(
            result["pred_df_samples"], current_price=result["current_price"]
        )
        report = render_forecast_report(summary, ...)

        return {
            "forecast_report": report,
            "forecast_summary": summary,  # raw dict, untuk Trader & Risk
        }
    return forecast_node
```

Note: Bukan factory yang terima `llm` (karena tidak panggil LLM). Cuma terima `kronos_client + config`.


---

## 8. Modifikasi pada TradingAgents Existing

### 8.1 `tradingagents/agents/utils/agent_states.py`

Tambah 2 field di `AgentState`:

```python
class AgentState(MessagesState):
    # ... existing ...
    forecast_report: Annotated[str, "Markdown report dari Kronos forecast"]
    forecast_summary: Annotated[dict, "Raw stats: median_return, p5/p95, vol, drawdown, VaR"]
```

`forecast_summary` adalah dict — biar Trader & risk debaters bisa akses angka langsung tanpa parsing markdown.

### 8.2 `tradingagents/graph/setup.py`

Tambah Forecast Analyst sebagai node terakhir di analyst layer, sebelum Bull Researcher:

```python
# Di setup_graph
forecast_node = create_forecast_analyst(self.kronos_client, self.config)
workflow.add_node("Forecast Analyst", forecast_node)

# Wire: terakhir di analyst layer → Forecast → Bull Researcher
last_analyst_clear = plan.specs[-1].clear_node
workflow.add_edge(last_analyst_clear, "Forecast Analyst")
workflow.add_edge("Forecast Analyst", "Bull Researcher")
```

**Penting**: Edge dari `last_analyst_clear → "Bull Researcher"` yang ada sekarang harus diganti dengan dua edge: `→ Forecast Analyst → Bull Researcher`. Conditional kalau `kronos_enabled=False`, skip Forecast.

```python
if self.config.get("kronos_enabled"):
    workflow.add_edge(last_analyst_clear, "Forecast Analyst")
    workflow.add_edge("Forecast Analyst", "Bull Researcher")
else:
    workflow.add_edge(last_analyst_clear, "Bull Researcher")
```

### 8.3 `tradingagents/graph/trading_graph.py`

Init Kronos client di `__init__`:

```python
from tradingagents.forecast.kronos_client import KronosClient

class TradingAgentsGraph:
    def __init__(self, ...):
        # ... existing ...
        if self.config.get("kronos_enabled"):
            self.kronos_client = KronosClient(
                model_name=self.config["kronos_model"],
                tokenizer_name=self.config["kronos_tokenizer"],
                device=self.config.get("kronos_device", "cpu"),
                max_context=self.config.get("kronos_max_context", 512),
                cache_dir=self.config["data_cache_dir"],
            )
        else:
            self.kronos_client = None

        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            kronos_client=self.kronos_client,  # NEW
            config=self.config,                # NEW (untuk forecast config)
            ...
        )
```

### 8.4 `tradingagents/agents/researchers/bull_researcher.py` & `bear_researcher.py`

Update prompt template untuk include forecast:

```python
prompt = """You are an aggressive bull researcher. Build the case FOR investing in {ticker}.

Available evidence:
- Market report: {market_report}
- Sentiment report: {sentiment_report}
- News report: {news_report}
- Fundamentals report: {fundamentals_report}
- Quantitative forecast (Kronos): {forecast_report}    ← NEW

Reference past debate:
{history}

Your task:
1. Identify bullish signals across reports.
2. **Critically evaluate the Kronos forecast**: does it support your thesis? If yes, cite the metrics (median return, prob_positive). If no, explain why pattern-based forecast might miss the catalyst you see.
3. Counter the bear's previous arguments (if any).
...
"""
```

Sama untuk Bear, dengan instruksi "kalau Kronos forecast bullish, jelaskan kenapa kamu masih bearish (catalyst belum priced in di pattern, dll)".

**Kunci**: prompt eksplisit minta agent **engage** dengan forecast, bukan ignore.

### 8.5 `tradingagents/agents/trader/trader.py`

Trader sekarang baca `forecast_summary` (dict) untuk sizing & stop-loss:

```python
def trader_node(state):
    forecast_summary = state.get("forecast_summary", {})
    investment_plan = state["investment_plan"]

    # Pre-compute sizing & stop hints
    sizing_hint = compute_sizing_from_forecast(forecast_summary)
    stop_hint = compute_stop_from_forecast(forecast_summary)

    prompt_with_hints = base_prompt.format(
        investment_plan=investment_plan,
        forecast_summary_text=render_summary_text(forecast_summary),
        sizing_hint=sizing_hint,
        stop_hint=stop_hint,
    )
    # ... rest unchanged (LLM call, structured output)
```

`compute_sizing_from_forecast`: rule sederhana, e.g. position_size = base / (1 + p95_drawdown_normalized). Logic ada di `tradingagents/forecast/forecast_stats.py`.

### 8.6 `tradingagents/agents/risk_mgmt/*.py`

Risk debaters baca `forecast_summary` untuk argumen kuantitatif:

- **Conservative**: kalau `p95_drawdown > threshold`, push posisi lebih kecil
- **Aggressive**: kalau `prob_positive > 65% & p95_drawdown < threshold`, push sizing lebih besar
- **Neutral**: balance, sebut median + spread

### 8.7 `tradingagents/agents/managers/portfolio_manager.py`

Portfolio Manager prompt update:

```python
prompt = """You are the Portfolio Manager making the final trade decision.

Inputs:
- Investment plan (from research): {investment_plan}
- Trader proposal: {trader_investment_plan}
- Risk debate transcript: {risk_history}
- Past context (memory): {past_context}
- Quantitative forecast: {forecast_report}                 ← NEW
- Forecast summary: {forecast_summary_text}                ← NEW

Decision rules:
1. Provide rating (5-tier) + thesis + target + horizon.
2. **If your decision contradicts Kronos forecast direction, you MUST explicitly justify the divergence**:
   - What information do the LLM agents have that Kronos doesn't?
   - Is there a catalyst pending (earnings, regulation) that breaks pattern-based forecast?
3. Set field `kronos_alignment`: one of [strongly_aligned, aligned, neutral, divergent, strongly_divergent].
"""
```

Update `PortfolioDecision` schema di `agents/schemas.py`:

```python
class PortfolioDecision(BaseModel):
    rating: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    thesis: str
    target_price: float | None = None
    horizon_days: int
    # NEW
    kronos_alignment: Literal["strongly_aligned", "aligned", "neutral", "divergent", "strongly_divergent"]
    kronos_divergence_reason: str | None = Field(default=None, description="Required jika kronos_alignment in [divergent, strongly_divergent]")
```

### 8.8 `tradingagents/default_config.py`

Tambah env var + default:

```python
_ENV_OVERRIDES = {
    # existing ...
    "TRADINGAGENTS_KRONOS_ENABLED":      "kronos_enabled",
    "TRADINGAGENTS_KRONOS_MODEL":        "kronos_model",
    "TRADINGAGENTS_KRONOS_DEVICE":       "kronos_device",
    "TRADINGAGENTS_KRONOS_LOOKBACK":     "kronos_lookback",
    "TRADINGAGENTS_KRONOS_PRED_LEN":     "kronos_pred_len",
    "TRADINGAGENTS_KRONOS_SAMPLE_COUNT": "kronos_sample_count",
    "TRADINGAGENTS_KRONOS_BAR":          "kronos_bar",
}

DEFAULT_CONFIG = _apply_env_overrides({
    # existing ...
    "kronos_enabled": False,            # opt-in
    "kronos_model": "NeoQuasar/Kronos-small",
    "kronos_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
    "kronos_device": "cpu",
    "kronos_max_context": 512,
    "kronos_lookback": 400,             # bars
    "kronos_pred_len": 10,              # bars (cocok dengan 5-day hold di daily bar)
    "kronos_sample_count": 10,          # path samples untuk distribution
    "kronos_bar": "1d",                 # daily bars default
    "kronos_temperature": 1.0,
    "kronos_top_p": 0.9,
})
```

---

## 9. State Schema Updates

Sebelum:
```python
class AgentState(MessagesState):
    company_of_interest: str
    asset_type: str
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
    past_context: str
```

Sesudah:
```python
class AgentState(MessagesState):
    # ... semua di atas masih ada ...

    # Forecast layer (Kronos)
    forecast_report: Annotated[str, "Kronos forecast markdown report"]
    forecast_summary: Annotated[dict, "Stats: median_return, p5/p95, vol, drawdown, VaR, prob_positive, sample_count"]
```

`forecast_summary` punya struktur (dictionary):
```python
{
    "median_return": float,        # e.g. 0.032 (3.2%)
    "p5_return": float,            # downside
    "p95_return": float,           # upside
    "prob_positive": float,        # e.g. 0.68 (68%)
    "expected_vol": float,         # std return per bar
    "median_drawdown": float,      # negative
    "p95_drawdown": float,         # negative, worst case 95%
    "var_5pct": float,             # Value-at-Risk
    "current_price": float,
    "pred_len": int,
    "bar": str,
    "sample_count": int,
}
```

---

## 10. Flow Eksekusi Lengkap

Trace satu run dengan Kronos enabled:

### Setup phase
1. `TradingAgentsGraph.__init__`:
   - Init LLM clients (deep + quick) — sama seperti sebelumnya
   - **NEW**: Init `KronosClient` (lazy — model belum di-load)
   - Pass kronos_client ke `GraphSetup`
2. `GraphSetup.setup_graph`:
   - Wire 4 analyst nodes (sama seperti sebelumnya)
   - **NEW**: Add `Forecast Analyst` node
   - **NEW**: Edge: last_analyst_clear → Forecast Analyst → Bull Researcher
   - Wire research/risk/PM (sama)

### Eksekusi (focus pada perubahan)

**Step A — Analyst layer (urutan: Market → Sentiment → News → Fundamentals → Forecast)**

1-4. Market, Sentiment, News, Fundamentals — sama seperti sebelumnya, mereka tidak tahu Kronos sama sekali.

5. **Forecast Analyst (NEW)**:
   - Read `state["company_of_interest"]` dan `state["trade_date"]`
   - Call `kronos_client.predict(ticker, date, ...)`:
     - Cek cache disk → kalau ada, load
     - Else: load model (first time saja), fetch OHLCV via yfinance, run predict dengan sample_count
     - Save cache
   - Compute summary stats (`compute_forecast_summary`)
   - Render markdown report
   - Return `{"forecast_report": report, "forecast_summary": summary}`
   - **State update**: 2 field baru terisi
   - **Tidak ada LLM call di node ini** → cepat (~1-3s + initial model load)
   - **Tidak ada Msg Clear** karena tidak pakai messages

**Step B — Research layer**

6. **Bull Researcher** — prompt baru include `forecast_report`. LLM punya akses ke distribusi. Statement bull bisa cite "median return +3.2%, 68% prob positive".

7. **Bear Researcher** — sama, tapi sisi sebaliknya. Bisa cite "p5 return -4.5%, drawdown risk tinggi".

8. **Research Manager** — baca seluruh debate + forecast. Output `ResearchPlan` (no schema change).

**Step C — Execution layer**

9. **Trader (modified)**:
   - Baca `investment_plan` + `forecast_summary` (raw dict)
   - Pre-compute sizing hint (function deterministic dari forecast)
   - Pre-compute stop hint (e.g. stop = entry × (1 + p5_return × 0.7))
   - LLM call dengan hints di prompt
   - Output `TraderProposal` — sama, tapi reasoning sudah inform by Kronos

**Step D — Risk debate**

10-12. **Aggressive / Conservative / Neutral** — semua baca `forecast_summary`. Argumen bisa kuantitatif.

13. **Portfolio Manager (modified)**:
    - Input: semua report + investment_plan + trader_plan + risk_history + forecast
    - LLM call dengan extended prompt (include rule contradict-Kronos must justify)
    - Output `PortfolioDecision` dengan field baru: `kronos_alignment`, `kronos_divergence_reason`

**Step E — Output & post-processing**

14. Log state ke disk (struktur berubah karena ada field baru di state)
15. Memory log (tidak berubah)
16. Process signal (regex extract rating — tidak berubah)

### Total perubahan latency

Estimasi tambahan dari Kronos:
- First run: ~5-10s (model load) + ~1-3s (predict) = ~10-15s tambahan
- Cached runs: ~1-3s atau bahkan instant (kalau cache hit)
- LLM calls tetap sama jumlahnya tapi prompt lebih panjang → ~5-10% lebih lambat di token side

---

## 11. Conflict Resolution: LLM vs Forecast

Ini bagian paling penting dari design. Detail mekanisme:

### 11.1 Levels of disagreement

Definisikan 5 level alignment:

| Alignment | Kondisi | Action |
|---|---|---|
| `strongly_aligned` | LLM = BUY/STRONG_BUY, Kronos median > +3%, prob_positive > 65% | Confidence boost di rating |
| `aligned` | Direction sama, magnitude moderate | Normal flow |
| `neutral` | Salah satu close to zero | Normal flow |
| `divergent` | Direction beda, magnitude moderate | Wajib `kronos_divergence_reason` di output |
| `strongly_divergent` | LLM = STRONG_BUY tapi Kronos p95_return < 0; atau sebaliknya | **Force HOLD or downgrade** kecuali reason kuat |

### 11.2 Implementasi mekanisme override

Pilihan implementasi (lihat section 4.3):
- **Default**: Pilihan C — eksplisit justify, tidak auto-override
- **Optional strict mode** (`config["kronos_strict_mode"] = True`): kalau `strongly_divergent`, downgrade rating 1 tier (STRONG_BUY → BUY, BUY → HOLD)

```python
# Di trading_graph.py setelah PM
if self.config.get("kronos_strict_mode") and \
   final_state["portfolio_decision"]["kronos_alignment"] == "strongly_divergent":
    final_state["portfolio_decision"]["rating"] = downgrade(
        final_state["portfolio_decision"]["rating"]
    )
    final_state["portfolio_decision"]["thesis"] += "\n\n[AUTO-DOWNGRADED: Kronos strongly divergent]"
```

### 11.3 Audit trail

Setiap run, log ke memory:
- Kronos forecast summary
- LLM final rating
- Alignment level
- Kalau divergent: reason

Saat reflection (next run for same ticker), bandingkan:
- Apakah Kronos atau LLM lebih akurat untuk ticker ini di history?
- Pattern-pattern: Kronos lebih akurat di trending stocks, LLM lebih akurat di event-heavy stocks?

Bisa derive **per-ticker trust weights** dari history.

---


## 12. Konfigurasi Baru

Lengkap di `default_config.py`:

```python
# === Kronos Forecast Layer ===
"kronos_enabled": False,                    # opt-in flag

# Model
"kronos_model": "NeoQuasar/Kronos-small",   # mini/small/base
"kronos_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
"kronos_device": "cpu",                     # "cuda" / "cuda:0" kalau GPU
"kronos_max_context": 512,

# Forecast horizon
"kronos_lookback": 400,                     # historical bars feed ke model
"kronos_pred_len": 10,                      # bars ke depan
"kronos_bar": "1d",                         # "1d", "1h", "15m", "5m"

# Sampling
"kronos_sample_count": 10,                  # paths untuk distribution
"kronos_temperature": 1.0,
"kronos_top_p": 0.9,

# Decision policy
"kronos_strict_mode": False,                # kalau True, auto-downgrade strongly_divergent
"kronos_alignment_thresholds": {
    "strong_positive_return": 0.03,         # 3% median
    "strong_negative_return": -0.03,
    "high_prob_positive": 0.65,
    "low_prob_positive": 0.35,
    "high_drawdown": 0.07,                  # 7% p95 drawdown threshold
},
```

Env var:
- `TRADINGAGENTS_KRONOS_ENABLED=true`
- `TRADINGAGENTS_KRONOS_MODEL=NeoQuasar/Kronos-base` (kalau mau model lebih besar)
- `TRADINGAGENTS_KRONOS_DEVICE=cuda`
- dst.

API key: **tidak perlu** untuk Kronos (model di-host gratis di Hugging Face). Tapi Hugging Face download butuh internet first time.

---

## 13. Roadmap Implementasi (Phase 1-5)

### Phase 1 — Standalone Kronos Wrapper (1-2 hari)

**Tujuan**: Pastikan Kronos jalan di environment user, output bisa diandalkan.

**Tugas:**
1. Clone/install Kronos: `git clone https://github.com/shiyu-coder/Kronos`, install requirements
2. Buat `tradingagents/forecast/kronos_client.py` — load model, predict, cache
3. Standalone test: `python -m tradingagents.forecast.kronos_client NVDA 2024-05-10`
4. Verify output: pred_df shape, sample_count works, cache works

**Deliverable**: KronosClient yang bisa dipanggil mandiri.

**Risk check**: GPU vs CPU latency, model download size, requirements collision dengan TradingAgents (PyTorch version, dll).

### Phase 2 — Forecast Stats & Renderer (1 hari)

**Tujuan**: Convert raw predictions jadi business metrics + markdown.

**Tugas:**
1. `forecast_stats.py` — compute_forecast_summary, drawdown calc, VaR
2. `forecast_renderer.py` — markdown format
3. Unit tests dengan mock pred_df samples

**Deliverable**: Function-function deterministic, tested.

### Phase 3 — Forecast Analyst Node (2-3 hari)

**Tujuan**: Wire Kronos sebagai LangGraph node.

**Tugas:**
1. `forecast_node.py` — create_forecast_analyst factory
2. Update `agent_states.py` — tambah field forecast_report, forecast_summary
3. Update `setup.py` — add node + edge wiring (with conditional kronos_enabled)
4. Update `trading_graph.py` — init kronos_client
5. Update `default_config.py` — env vars + defaults
6. Integration test: run end-to-end, verify state has forecast_report

**Deliverable**: Pipeline jalan dengan Kronos enabled, tapi belum ada perubahan downstream agent.

### Phase 4 — Downstream Integration (3-5 hari)

**Tujuan**: Researcher, Trader, Risk debaters, PM consume forecast.

**Tugas:**
1. Update Bull/Bear prompt — include forecast_report
2. Update Trader logic — sizing & stop hints dari forecast_summary
3. Update Risk debater prompts — include forecast_summary text
4. Update Portfolio Manager prompt + schema (`PortfolioDecision`) — kronos_alignment field
5. Test: run dengan Kronos vs tanpa Kronos, bandingkan output reasoning

**Deliverable**: Full integration. Reasoning agent visibly engage dengan forecast.

### Phase 5 — Conflict Resolution & Polish (2-3 hari)

**Tujuan**: Handle disagreement gracefully, audit trail.

**Tugas:**
1. Implement alignment classifier (function deterministic)
2. Optional strict mode (auto-downgrade)
3. Memory log — persist alignment per run
4. CLI display — show forecast summary box di Rich live
5. Documentation update — README + READING_GUIDE

**Deliverable**: Production-quality integrasi.

### Total estimasi: 9-14 hari kerja (~2-3 minggu)

---

## 14. Risiko Teknis

### 14.1 Dependency conflict
Kronos punya `requirements.txt` sendiri. Mungkin clash dengan TradingAgents:
- PyTorch version
- HuggingFace transformers version
- pandas/numpy version

**Mitigasi**: Test di virtual env terpisah dulu. Kalau bentrok, isolasi Kronos ke subprocess (run via FastAPI service) — tapi tambah kompleksitas.

### 14.2 Model size & download
- Kronos-small: 24.7M params ≈ 100 MB
- Kronos-base: 102.3M params ≈ 400 MB
- Tokenizer: 50-200 MB

First run download dari HuggingFace bisa lama. **Mitigasi**: instruksi user pre-download.

### 14.3 GPU availability
Kalau user pakai laptop tanpa GPU, predict di CPU bisa lambat (~5-30s per call).

**Mitigasi**:
- Default device "cpu" tapi kalau detect CUDA, auto-pakai
- Cache aggressively
- Skip Kronos kalau latency budget terlampaui

### 14.4 Forecast quality untuk asset rare
Kronos pre-trained pada 45 exchanges. Kalau ticker eksotis (small cap Indonesia, dll) tidak ada di training set, akurasi turun.

**Mitigasi**:
- Detect data sparsity (kalau lookback gagal fetch cukup data, skip Kronos)
- Optional fine-tune via finetune script Kronos kalau user invest waktu

### 14.5 Hallucination by proxy
LLM mungkin cite Kronos number salah ("Kronos says 5%"... padahal 2%). 

**Mitigasi**:
- Inject `forecast_summary` raw dict ke state (Trader & Risk pakai langsung, tidak via LLM parsing)
- Output PM punya structured field `kronos_alignment` — bisa cross-check
- Optional: post-hoc validator yang cek apakah angka di thesis match dengan forecast_summary

### 14.6 Cache invalidation
Cache key = `(ticker, date, params)`. Kalau Kronos model di-update di Hugging Face, cache lama jadi stale.

**Mitigasi**: Include model SHA / version di cache key. Atau TTL cache (e.g. 7 hari).

### 14.7 Testing complexity
Test integrasi butuh: Kronos jalan + LLM API + yfinance internet. Slow & flaky.

**Mitigasi**: Mock layer di test:
- `MockKronosClient` yang return fixed pred_df
- Test logic sizing/conflict resolution dengan mock
- E2E test optional, di-skip default

---

## 15. Pertanyaan Terbuka

Hal-hal yang perlu diputuskan sebelum atau saat implementasi:

1. **Frequency data**: Daily atau intraday? Trade-off: daily = lebih banyak history fits, tapi forecast horizon panjang noisy. Intraday = horizon pendek precise, tapi data fetch lebih sulit.

2. **Sample count**: 10? 30? 100? Lebih banyak = distribusi presisi tapi latency naik linear.

3. **Apakah Kronos di-input ke Sentiment Analyst?** Sentiment biasanya text-based, tapi forecast quantitative. Mungkin tidak relevan.

4. **Apakah feed Kronos summary ke Memory log?** Berguna untuk track akurasi history, tapi nambah size memory file.

5. **Threshold alignment static atau adaptive?** Bisa jadi, alignment threshold di-adjust per-ticker berdasarkan track record akurasi Kronos untuk ticker itu.

6. **Multi-model ensemble?** Pakai Kronos-small dan Kronos-base bareng, ensemble prediction? Lebih akurat tapi 2x compute.

7. **Fine-tune atau zero-shot?** Untuk asset spesifik (e.g. crypto), fine-tune mungkin nendang. Tapi butuh data + GPU.

8. **Forecast cache TTL**: forever (until model version berubah) atau time-based? Kalau model deterministik, forever fine.

9. **Apakah Kronos override checkpointer?** Kalau crash di tengah Forecast node, saat resume re-fetch atau load cache?

10. **Backtest evaluation**: Cara fair compare TradingAgents-only vs TradingAgents+Kronos? Same tickers, same dates, blind comparison? 

---

## 16. Glossarium Tambahan

| Istilah | Arti |
|---|---|
| **Foundation model** | Model pre-trained di data besar, bisa di-tune/zero-shot ke berbagai task |
| **K-line / Candlestick** | Representasi OHLCV per timestamp di chart finance |
| **OHLCV** | Open/High/Low/Close/Volume |
| **Tokenizer (Kronos)** | Module yang quantize OHLCV continuous → token discrete |
| **Autoregressive** | Predict token berikutnya berdasar token sebelumnya (seperti GPT) |
| **Lookback** | Jumlah bar history yang fed ke model |
| **pred_len** | Jumlah bar yang di-predict ke depan |
| **Sample paths** | Probabilistic forecast: generate N path, derive distribusi |
| **Top-p / Nucleus sampling** | Sampling dari top-p% probability mass |
| **Temperature (T)** | Kontrol entropy sampling — T tinggi = random, T rendah = greedy |
| **Median forecast** | Tengah distribusi N samples — central estimate |
| **Percentile p5/p95** | Batas bawah/atas distribusi (90% confidence band) |
| **VaR (Value-at-Risk)** | Loss maksimum di percentile tertentu (5% biasanya) |
| **Drawdown** | Penurunan dari peak ke trough |
| **Pure alpha** | Return dikurangi market exposure (beta-adjusted) |
| **Quantitative forecast** | Forecast berbasis angka/pattern (vs qualitative) |
| **Alignment** | Sejauh mana keputusan LLM cocok dengan forecast |
| **Strict mode** | Mode di mana sistem auto-override LLM kalau divergent |
| **Forecast Analyst** | Node baru di analyst layer, deterministic, run Kronos |
| **Hybrid Layered** | Arsitektur pilihan: Kronos sebagai analyst + consumed di Trader+Risk+PM |

---

## Penutup

### Ringkasan POV saya

> **Ide bagus.** Kombinasi LLM (narrative reasoning) + Foundation model time-series (pattern recognition) adalah arah modern di quant trading. Banyak hedge fund kelas atas pakai pendekatan serupa (mereka sebut "neural alpha + qualitative overlay"). Kamu basically replicate konsep itu open-source.

> **Effort realistik 2-3 minggu** untuk implementasi solid. Phase 1-3 mungkin bisa selesai 1 minggu, Phase 4-5 butuh tuning & test.

> **Hindari trap utama**: jangan biarkan Kronos jadi blackbox yang LLM ikuti buta. Forecast model bisa salah arah saat ada catalyst event (yang LLM tahu via news). Force LLM untuk **engage critically** dengan forecast — inilah yang membuat sistem jadi cerdas, bukan duplikasi.

### Rekomendasi langkah berikutnya

1. **Baca dokumen ini sampai habis** dulu — pastikan paham desain sebelum coding
2. Clone Kronos repo, jalankan `examples/prediction_example.py` standalone — pastikan jalan di env kamu
3. Mulai Phase 1: bikin `KronosClient` standalone, test dengan ticker NVDA
4. Phase 2 paralel: bikin stats/renderer (test pakai mock data)
5. Baru integrasi (Phase 3+)

### Kalau saya yang implementasi

Urutan saya:
1. Phase 1 (1 hari) — pastikan kronos jalan, cache works
2. Phase 2 (0.5 hari) — stats + renderer
3. Phase 3 (1 hari) — wire ke LangGraph, no downstream change
4. **Run end-to-end** dulu — ada `forecast_report` di output, tapi tidak ada agent yang baca
5. Phase 4 (2-3 hari) — bertahap: Trader → Risk → PM. Test setiap perubahan dengan A/B (with vs without forecast)
6. Phase 5 (1 hari) — conflict resolution + memory tracking
7. **Tambah eval**: run 20-30 backtest dengan vs tanpa Kronos, bandingkan alpha

Total realistis untuk 1 orang full-time: ~10-12 hari.

### Pertanyaan untuk kamu

Sebelum mulai, klarifikasi:
- **Frekuensi trading kamu**: daily, swing (multi-day), atau intraday? Ini menentukan bar config.
- **Asset target**: US stock saja, atau crypto, atau Indo stock? Affecting model availability di pre-training.
- **GPU available**: ya/tidak. Kalau tidak, Kronos-small CPU mode masih OK (1-3s/predict).
- **Tujuan utama**: research/learning, paper trading, atau real money? Affecting risk tolerance untuk auto-override.

Kalau sudah dijawab, saya bisa zoom ke phase tertentu atau bikin proof-of-concept code.
