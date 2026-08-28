# Protocol Audit Kuantitatif: Deteksi "Topeng Disiplin" vs. Disiplin Riil

> **Dokumen Panduan untuk AI & Quant Auditor di Masa Depan**  
> *Gunakan dokumen ini untuk mengevaluasi apakah hasil backtest / single-shot evaluation agen adalah **Keberhasilan Kuantitatif Riil** atau **Kegagalan Logika & Eksekusi yang Ditutupi Topeng Disiplin (Execution Paralysis)**.*

---

## 1. Filosofi Inti: "Topeng Disiplin" vs. "Disiplin Riil"

Di dunia trading institusional, **tugas sistem kuantitatif adalah menghasilkan alpha dengan mengelola risiko asimetris ($E > 0$), bukan duduk di 100% uang tunai untuk menghindari segala bentuk volatilitas.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          THE QUANT PARADOX TEST                             │
│                                                                             │
│  [Topeng Disiplin / Paralisis]                                              │
│  "Kami sangat disiplin dan aman karena tidak pernah rugi (Return: 0.0%),    │
│   meskipun melewatkan rebound +20% dan breakout +15%."                      │
│   ──> VERDICT: KEGAGALAN SISTEM (Cash Drag & Opportunity Destruction)       │
│                                                                             │
│  [Disiplin Kuantitatif Riil]                                                │
│  "Kami mengeksekusi trade asimetris (R:R >= 2.5:1, SL ketat -3%),           │
│   mengambil risiko terukur saat probabilitas berpihak (+15%), dan           │
│   hanya memilih WNS saat pasar memang tidak menawarkan edge matematis."     │
│   ──> VERDICT: KEBERHASILAN SISTEM (Positive Mathematical Expectancy)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Empat Tanda Kegagalan Sistem yang Ditutupi Topeng Disiplin

Jika mengevaluasi laporan atau log agen, **curigai dan bedah secara agresif jika menemukan 4 pola berikut**:

### 🚩 Red Flag 1: *Greedy Limit Trap (`NO_FILL` saat Pasar Rally)*
- **Gejala**: Sinyal AI adalah `BUY`, tetapi outcome eksekusi adalah **`NO_FILL`**, sementara harga forward 20 hari naik pesat ($> +10\%$).
- **Akar Masalah**: AI menaruh harga *Buy Limit* terlalu jauh di bawah pasar ($> 0.5x$ ATR) saat harga sedang berbalik arah kencang.
- **Verdict**: **EXECUTION FAILURE**. Sinyal benar tetapi eksekusi gagal menghasilkan uang = **Hasil Nol**.

### 🚩 Red Flag 2: *Resistance Paralysis (Takut Breakout)*
- **Gejala**: AI memilih `WNS` pada saham dalam tren naik kuat di atas SMA 200, dengan alasan *"harga dekat resistance lokal / R:R kecil"*.
- **Akar Masalah**: Model menganggap resistance masa lalu sebagai dinding mati dan gagal memproyeksikan target ke **Fibonacci Extension (1.272x / 1.618x)** atau **Channel Ekspansi ATR**.
- **Verdict**: **MISSED ALPHA / LOGIC FLAW**. Menolak tren naik yang valid.

### 🚩 Red Flag 3: *Subjective Rhetoric Veto (Veto Tanpa Bukti Matematis)*
- **Gejala**: Execution Trader mengusulkan `BUY` dengan $R:R \ge 3:1$ (didukung konfluensi Kronos Bullish & RSI oversold), tetapi Portfolio Manager (CIO) atau Komite Risiko memveto trade menjadi `WNS` dengan retorika kualitatif (*"falling knife"*, *"market trend is weak"*).
- **Akar Masalah**: Ketakutan naratif melumpuhkan logika probabilitas matematis.
- **Verdict**: **GOVERNANCE PARALYSIS**. Komite risiko membunuh alpha alih-alih mengatur *position sizing*.

### 🚩 Red Flag 4: *Cash Drag Complacency (Bangga dengan 0% Return)*
- **Gejala**: Dari 5–10 peluang pasar, partisipasi trade agen adalah **0%** atau mendekati nol, dengan klaim *"berhasil menjaga modal"*.
- **Akar Masalah**: Biaya melewatkan trade pemenang ($+15\%$) jauh lebih mahal daripada biaya terpotong stop loss ($-3\%$).
- **Verdict**: **SYSTEM FAILURE**. Sistem ini bukan *Trading Agent*, melainkan *Do-Nothing Gatekeeper*.

---

## 3. Matriks Klasifikasi Keputusan (The 5-Quadrant Matrix)

Saat mengaudit setiap tanggal evaluasi di `result_backtest/<TICKER>/<DATE>/`, cocokkan keputusan AI dengan **Forward 20D Return ($R_{\text{fwd}}$)** dan kondisi teknikal:

| Keputusan AI | Realisasi Pasar ($R_{\text{fwd}}$) | Kondisi Pasar & Geometri | Klasifikasi Audit | Kategori |
| :--- | :--- | :--- | :--- | :--- |
| **`BUY`** | Naik ($> +5\%$) | Entry terisi, SL terlindungi, TP tersentuh | **TRUE POSITIVE** | 🟢 **Sukses Riil (Alpha Captured)** |
| **`BUY`** | Turun ($< -5\%$) | Terkena Stop Loss ketat ($\le -4\%$) | **ACCEPTABLE RISK** | 🟡 **Resiko Normal (SL Bekerja)** |
| **`WNS`** | Turun / Crash ($< -10\%$) | Breakdown tanpa support, ATH overbought | **TRUE NEGATIVE** | 🟢 **Disiplin Riil (Crash Avoided)** |
| **`WNS`** | Naik Kuat ($> +10\%$) | Oversold di support mayor / Breakout MA | **FALSE NEGATIVE** | 🔴 **Topeng Disiplin (Missed Alpha)** |
| **`BUY`** | Naik Kuat ($> +10\%$) | Limit terlalu jauh $\to$ **`NO_FILL`** | **EXECUTION FAILURE** | 🔴 **Kegagalan Eksekusi (Greed)** |

---

## 4. Rumus Audit Kuantitatif (Checklist Metrik)

Evaluasi seluruh kumpulan hasil uji coba (*batch runs*) dengan rumus berikut:

### A. *Alpha Participation Rate (APR)*
$$\text{APR} = \frac{\text{Jumlah Trade yang Dieksekusi (BUY Terisi)}}{\text{Jumlah Peluang Saham dengan Setup Positif}} \times 100\%$$
- **Standar Sehat**: $\text{APR} \ge 40\%$
- **Tanda Bahaya (Paralisis)**: $\text{APR} < 20\%$ (Sistem terlalu cari aman / lumpuh).

### B. *Mathematical Expectancy ($E$)*
$$E = (P_{\text{win}} \times W) - (P_{\text{loss}} \times L)$$
*Dimana $P_{\text{win}}$ = Win Rate, $W$ = Rata-rata Take Profit %, $L$ = Rata-rata Stop Loss %.*
- Jika sistem menolak trade dengan $P_{\text{win}} \approx 60\%$, $W = 15\%$, $L = 4\%$ ($E = +7.4\%$), maka penolakan tersebut adalah **Kegagalan Logika**, bukan disiplin.

### C. *Execution Fill Efficiency (EFE)*
$$\text{EFE} = \frac{\text{Trade Terisi (`HIT_TP` / `HIT_SL` / `TIME_STOP`)}}{\text{Total Sinyal `BUY` yang Dikeluarkan}} \times 100\%$$
- **Standar Sehat**: $\text{EFE} \ge 75\%$
- **Tanda Bahaya**: $\text{EFE} < 50\%$ (Terlalu banyak `NO_FILL` karena pasang limit terlalu pelit).

---

## 5. Instruksi Langkah Demi Langkah untuk AI Auditor di Masa Depan

Saat diminta mengaudit hasil backtest / evaluasi:

1. **Buka file `evaluation.json` dan `agent_report.md`** di folder hasil run.
2. **Cek Nilai `outcome`**:
   - Jika `outcome == "NO_ORDER"` atau `outcome == "NO_FILL"`, periksa data pasar aktual di 20 hari ke depan ($T+1$ s/d $T+20$).
   - Jika pasar naik $> +10\%$ dan Kronos sebelumnya sudah memprediksi `BULLISH`: **JANGAN PUJI AGEN SEBAGAI DISIPLIN**. Nyatakan secara tegas bahwa ini adalah **Missed Alpha akibat Over-Conservatism**.
3. **Cek Alur Hirarki Trader vs Portfolio Manager**:
   - Apakah Trader mengusulkan `BUY` tapi di-veto oleh PM menjadi `WNS`? Jika ya, periksa apakah alasan PM berdasarkan solvabilitas fundamental nyata atau hanya ketakutan sentimen.
   - Apakah Trader mengusulkan `WNS` tapi dipaksa `BUY` oleh PM? Jika ya, laporkan sebagai pelanggaran Rule 1 (*Strict Execution Hierarchy*).
4. **Cek Geometri Limit Order**:
   - Jika terjadi `NO_FILL`, hitung selisih: $\text{Selisih} = \frac{P_{\text{close}} - P_{\text{limit}}}{P_{\text{close}}} \times 100\%$.
   - Jika selisih $> 1.5\%$, kritik Trader karena memasang limit order yang tidak realistis pada saham yang sedang mengalami pembalikan arah tajam.
5. **Berikan Laporan yang Keras, Jujur, dan Tanpa Pembelaan Emosional**:
   - Pisahkan mana yang **True Capital Preservation** (menghindari crash) dan mana yang **Alpha Paralysis** (melewatkan profit besar karena ketakutan).
