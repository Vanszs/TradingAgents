# Master Architectural Mapping & Crosscheck Matrix: TradingAgents (Vanszs Edition)

> **Status**: Production-Grade Verification Blueprint & Granular Subsystem Crosscheck Matrix.
> **Target**: Exhaustive mapping across all dataflows, neural models, agent prompts, broker execution engines, schemas, CLI interfaces, and test suites.
> **Goal**: Detect and eliminate any mismatch in timeframes (1D vs intraday), rating contracts (BUY/WNS), order geometry (T1_OPEN vs T1_LIMIT), scale/units, error recovery, and point-in-time zero-leakage guards.

---

## Subsystem 1: Dataflows, Data Ingestion & Vendor Routing

# Master Architectural Mapping: Subsystem 1 (Dataflows & Ingestion)

---

### Section 1.1: Runtime Configuration, Environment Overrides & Thread-Safe State Isolation
- **Exact File Paths & Lines**:
  - `tradingagents/default_config.py:1-146`
  - `tradingagents/dataflows/config.py:1-53`
  - `tradingagents/dataflows/utils.py:1-43`
- **Purpose, Contracts & Data Structures**:
  - Central configuration single source of truth. Dict schema containing paths, provider options, thinking levels, debate rounds, news limits, backtest flags, data vendor overrides (`data_vendors`, `tool_vendors`), benchmark mappings, and Kronos model configurations.
  - `_coerce(value: str, reference)` dynamically casts environment strings to typed config scalars (bool, int, float, str) without script modifications.
  - `tradingagents/dataflows/config.py` provides thread-safe `_lock` (RLock) around deepcopy operations: `initialize_config()`, `set_config(config: Dict)` (merges 1-level deep for dicts), `get_config() -> Dict`, and `is_point_in_time_mode(config: Optional[Dict] = None) -> bool`.
  - `safe_ticker_component(value: str, max_len: int = 32) -> str` guards filesystem path traversal (`_TICKER_PATH_RE = ^[A-Za-z0-9._\-\^]+$`).
- **Potential Mismatch / Inconsistency Vectors**:
  - `point_in_time_mode` vs `backtest_mode`: `is_point_in_time_mode()` checks `bool(active.get("point_in_time_mode") or active.get("backtest_mode"))`. Must cross-check that all runners set one of these flags explicitly.
  - Dict-merging 1-level deep in `set_config()`: scalar keys are replaced, nested dicts (`data_vendors`, `tool_vendors`, `benchmark_map`) are updated. If a key is passed as `None`, it overwrites the nested dict rather than preserving defaults.
  - Benchmark ticker vs map: `benchmark_ticker` overrides `benchmark_map`. If `benchmark_ticker` is `None`, exchange suffix detection runs against `benchmark_map` (US default `SPY`, Indonesia `.JK` -> `^JKSE`, etc.).
- **Plan Checklist**:
  - [ ] Cross-check `tradingagents/default_config.py:10-28` `_ENV_OVERRIDES` map matches all active runtime flags in CLI and runner configurations.
  - [ ] Verify `tradingagents/dataflows/config.py:46-50` `is_point_in_time_mode` returns `True` under both CLI `--backtest` and library backtesting modes.
  - [ ] Ensure `safe_ticker_component` (`tradingagents/dataflows/utils.py:17-42`) blocks directory traversal across cache filenames and checkpoint persistence paths.

---

### Section 1.2: Dynamic Vendor Routing, Fallback Hierarchy & Fail-Closed PIT Guard
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/interface.py:1-218`
  - `tradingagents/dataflows/__init__.py:1-1`
- **Purpose, Contracts & Data Structures**:
  - Category and tool routing layer (`route_to_vendor(method, *args, **kwargs)`).
  - Categorization via `TOOLS_CATEGORIES` (categories: `core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`, `neural_forecasting`).
  - Priority hierarchy: Tool-level (`tool_vendors[method]`) -> Category-level (`data_vendors[category]`) -> Default vendor.
  - Fallback logic: handles comma-separated primary vendors, appends remaining available vendors from `VENDOR_METHODS[method]`, catches `AlphaVantageRateLimitError`, `YFRateLimitError`, `ConnectionError`, `TimeoutError`, `OSError`.
  - Fail-closed Point-in-Time (PIT) guard (`interface.py:201-203`): if `is_point_in_time_mode()` is active, `fallback_vendors` is hard-locked to `["snapshot"]`.
- **Potential Mismatch / Inconsistency Vectors**:
  - `route_to_vendor` error handling: generic `Exception` is not caught, allowing unexpected vendor errors to propagate or raise `RuntimeError("No available vendor for ...")`.
  - Method signature parity: all vendors implementing the same method name must accept compatible `*args` and `**kwargs`. For example, `get_stock_data` expects `(symbol, start_date, end_date)`.
  - Return shape disparities: `yfinance` returns CSV strings with markdown headers, `alpha_vantage` returns CSV strings or JSON, `snapshot` returns CSV strings or formatted markdown blocks.
- **Plan Checklist**:
  - [ ] Verify `tradingagents/dataflows/interface.py:104-160` `VENDOR_METHODS` mapping table includes all methods exposed to LangGraph agents.
  - [ ] Ensure `route_to_vendor` PIT enforcement at `interface.py:201-203` strictly prevents live network fallbacks during backtests.
  - [ ] Validate rate limit error catching (`AlphaVantageRateLimitError`, `YFRateLimitError`) across all vendor implementations.

---

### Section 1.3: Yahoo Finance Core Ingestion & Timezone-Aware Intraday Processing
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/y_finance.py:1-134`
  - `tradingagents/dataflows/symbol_utils.py:1-131`
- **Purpose, Contracts & Data Structures**:
  - `get_intraday_data(symbol, start, end, interval, timezone="UTC") -> pd.DataFrame`: Validates supported intervals (`1m` to `1d`), enforces timezone-awareness on intraday requests, handles exclusive end dates, auto-adjusts daily session dates, validates finite positive prices (`math.isfinite() > 0`).
  - `get_YFin_data_online(symbol, start_date, end_date) -> str`: Fetches daily OHLCV, pads end date by +1 day for inclusive boundaries, rounds floats to 2 decimal places, formats as CSV with comment metadata header.
  - `normalize_symbol(raw: str) -> str`: Resolves commodity futures (`XAUUSD` -> `GC=F`, `WTICOUSD` -> `CL=F`), forex pairs (`EURUSD` -> `EURUSD=X`), crypto pairs (`BTCUSD` -> `BTC-USD`), and index CFDs (`SPX500` -> `^GSPC`, `NAS100` -> `^NDX`).
- **Potential Mismatch / Inconsistency Vectors**:
  - Timeframe / Window bounds: `get_intraday_data` requires ISO strings with timezone info for intraday, but `YYYY-MM-DD` strings for daily intervals.
  - Daily vs Intraday session handling: Daily bars carry session dates (localized without offset shifting), whereas intraday timestamps undergo `tz_convert`.
  - Inclusive vs exclusive end dates: yfinance `history()` treats `end` as exclusive. Both `get_intraday_data` and `get_YFin_data_online` advance `end_date` by 1 day when matching daily intervals.
- **Plan Checklist**:
  - [ ] Audit `tradingagents/dataflows/y_finance.py:22-88` `get_intraday_data` to ensure all execution engines pass timezone-aware timestamps for sub-daily bars.
  - [ ] Validate `tradingagents/dataflows/symbol_utils.py:92-127` symbol normalization maps all asset classes correctly without network latency.
  - [ ] Verify `get_YFin_data_online` output CSV format compatibility with agent parser tools (`comment="#"`).

---

### Section 1.4: Technical Indicator Computations, Caching & Zero-Lookahead Causal Guards
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/y_finance.py:135-309`
  - `tradingagents/dataflows/stockstats_utils.py:1-221`
- **Purpose, Contracts & Data Structures**:
  - `load_ohlcv(symbol: str, curr_date: str = None) -> pd.DataFrame`: Fetches 15-year window anchored to `curr_date`, enforces on-disk caching in `data_cache_dir` with `.tmp` atomic replacement, cleans prices, and strictly drops rows where `Date > curr_date`.
  - In PIT/backtest mode (`stockstats_utils.py:68-90`), `load_ohlcv` retrieves `snapshot_data["ohlcv"]` directly from config and raises `RuntimeError` if missing.
  - `compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series`: Computes True Range and ATR causally using Wilder's EMA (`alpha = 1.0 / period`, `adjust=False`).
  - `get_stock_stats_indicators_window(symbol, indicator, curr_date, look_back_days) -> str`: Formats time series in ascending order (Past -> Present) with indicator parameter guidelines.
  - `StockstatsUtils.get_stock_stats(symbol, indicator, curr_date)`: Static helper for exact single-value extraction.
- **Potential Mismatch / Inconsistency Vectors**:
  - Indicator naming & aliasing: `rsi_14` -> `rsi`, `atr_14` / `atr_20` -> custom Wilder's ATR implementation; stockstats internal indicator naming convention vs Alpha Vantage keys.
  - Weekend / Holiday lookups: If `curr_date` is a non-trading day, `StockstatsUtils` falls back to the latest available causal row `<= curr_date`.
  - Cache poisoning prevention: `_clean_dataframe` ensures required columns (`Date`, `Open`, `High`, `Low`, `Close`) exist and prices are finite numbers before caching.
- **Plan Checklist**:
  - [ ] Confirm `tradingagents/dataflows/stockstats_utils.py:52-90` strictly cuts off data at `curr_date` without future leakage.
  - [ ] Verify `compute_atr` in `stockstats_utils.py:159-177` matches Wilder's formula and produces zero lookahead bias.
  - [ ] Check `_get_stock_stats_bulk` in `y_finance.py:280-308` handles missing values with `"N/A"` without raising uncaught exceptions.

---

### Section 1.5: Company Fundamentals & Publication Lag Buffering
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/y_finance.py:310-505`
  - `tradingagents/dataflows/stockstats_utils.py:145-157`
  - `tradingagents/dataflows/alpha_vantage_fundamentals.py:1-80`
- **Purpose, Contracts & Data Structures**:
  - `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`, `get_insider_transactions`.
  - Filing lag buffer: `filter_financials_by_date(data, curr_date, min_filing_lag_days=45)` drops columns where fiscal period end date is within 45 days of `curr_date` to reflect real SEC filing delays.
  - PIT mode enforcement:
    - In `y_finance.py` (398, 432, 466): In PIT mode, live financials return empty frames (`data.iloc[0:0]`) forcing use of snapshot data.
    - In `alpha_vantage_fundamentals.py`: Checks publication date fields (`_PUBLICATION_FIELDS`) and drops reports published after `curr_date`.
- **Potential Mismatch / Inconsistency Vectors**:
  - Date filtering semantics: yfinance columns represent fiscal period end dates, not filing dates. Without the 45-day lag buffer, backtests suffer from look-ahead leakage.
  - Quarterly vs Annual frequency: `freq="quarterly"` vs `freq="annual"`.
  - Return shapes: yfinance returns formatted CSV strings with headers; Alpha Vantage returns JSON dicts.
- **Plan Checklist**:
  - [ ] Validate `filter_financials_by_date` (`stockstats_utils.py:145-157`) buffer of 45 days is consistently applied to all live financial statements.
  - [ ] Verify PIT mode in `alpha_vantage_fundamentals.py:31-49` excludes records where `published_date > curr_date` or publication metadata is missing.
  - [ ] Ensure agent tools in `fundamental_data_tools.py` parse both CSV and JSON responses without runtime errors.

---

### Section 1.6: News & Sentiment Ingestion (yfinance & Alpha Vantage)
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/yfinance_news.py:1-201`
  - `tradingagents/dataflows/alpha_vantage_news.py:1-123`
  - `tradingagents/dataflows/alpha_vantage_common.py:1-124`
- **Purpose, Contracts & Data Structures**:
  - `get_news_yfinance(ticker, start_date, end_date) -> str`: Normalizes nested article structures (`_extract_article_data`), filters by UTC publication timestamps, enforces PIT upper bound `end_dt`.
  - `get_global_news_yfinance(curr_date, look_back_days, limit) -> str`: Executes macro search queries (`global_news_queries`), deduplicates headlines, clamps to lookback window.
  - `get_news` & `get_global_news` in `alpha_vantage_news.py`: Calls Alpha Vantage `NEWS_SENTIMENT` endpoint with `time_from` and `time_to`, filters by `_PUBLICATION_FIELDS` under PIT mode.
  - `_make_api_request` in `alpha_vantage_common.py`: Manages `ALPHA_VANTAGE_API_KEY`, detects rate limit messages in JSON (`AlphaVantageRateLimitError`), supports CSV responses.
- **Potential Mismatch / Inconsistency Vectors**:
  - Undated articles in PIT mode: Both yfinance and Alpha Vantage news filters discard articles with missing publication dates when `is_point_in_time_mode()` is active.
  - Date formatting: Alpha Vantage requires `YYYYMMDDTHHMM`, while yfinance expects `YYYY-MM-DD`. Handled via `format_datetime_for_api`.
  - Timestamp boundary filtering: Non-PIT mode allows `end_dt + 1 day` buffer for timezones; PIT mode strictly cuts off at `end_dt 23:59:59 UTC`.
- **Plan Checklist**:
  - [ ] Verify `_extract_article_data` (`yfinance_news.py:13-49`) handles both legacy and updated yfinance JSON payload structures.
  - [ ] Ensure `_filter_pit_feed` (`alpha_vantage_news.py:9-41`) drops all post-cutoff articles during historical runs.
  - [ ] Verify `AlphaVantageRateLimitError` triggers vendor fallback to yfinance in `route_to_vendor`.

---

### Section 1.7: Snapshot Provider & Point-in-Time Backtesting Vendor
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/snapshot.py:1-391`
- **Purpose, Contracts & Data Structures**:
  - `_get_snapshot_data() -> dict`: Extracts injected `snapshot_data` from global config.
  - Data retrieval functions:
    - `snapshot_get_stock_data(symbol, start_date, end_date) -> str`
    - `snapshot_get_indicators(symbol, indicator, curr_date, look_back_days) -> str`
    - `snapshot_get_news(ticker, start_date, end_date) -> str`
    - `snapshot_get_global_news(curr_date, look_back_days, limit) -> str`
    - `snapshot_get_insider_transactions(ticker) -> str`
    - `snapshot_get_fundamentals(ticker, curr_date) -> str`
    - `snapshot_get_balance_sheet`, `snapshot_get_cashflow`, `snapshot_get_income_statement`
  - Zero-Lookahead Guards:
    - `_effective_cutoff(value)`: Resolves `min(requested, active_trade_date)`.
    - `_visible_event(value, cutoff_date, cutoff_time="16:30:00")`: Ensures post-market review cutoff (16:30:00) is enforced on event feeds.
    - `_fundamental_available_date(item)`: Filters fundamentals by `available_date <= cutoff`.
- **Potential Mismatch / Inconsistency Vectors**:
  - Lookback window truncation: Snapshot OHLCV is already pre-sliced by `SnapshotDataProvider`; indicator calculations require sufficient history (e.g. 50/200 SMA require 50/200 rows).
  - DataFrame column casing: Converts snapshot columns (`date`, `open`, `high`, `low`, `close`, `volume`) to capitalized for stockstats (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`).
  - Missing snapshot keys: Returns clean user-facing error messages (`"No OHLCV data available..."`) instead of raising `KeyError`.
- **Plan Checklist**:
  - [ ] Validate `_effective_cutoff` (`snapshot.py:67-75`) never allows dates past the current simulation step trade date.
  - [ ] Verify `_visible_event` (`snapshot.py:55-64`) enforces the 16:30:00 post-market boundary on news and broker events.
  - [ ] Check indicator calculation from snapshot OHLCV in `snapshot_get_indicators` (`snapshot.py:144-230`) across all 13 supported indicators.

---

### Section 1.8: Kronos K-Line Foundation Model Ingestion & Forecasting Pipeline
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/kronos.py:1-629`
  - `tradingagents/dataflows/kronos_official/`
- **Purpose, Contracts & Data Structures**:
  - Autoregressive transformer for multi-step candlestick forecasting (AAAI 2026).
  - Data contracts:
    - `KronosPredictedBar(step, projected_open, projected_high, projected_low, projected_close, projected_volume, return_from_t0_pct)`
    - `KronosForecastContract(symbol, forecast_date, horizon_bars, last_close, forecast_high, forecast_low, forecast_end_close, forecast_return_pct, max_upside_pct, max_downside_pct, directional_bias, confidence_score, model_name, device_used, forecast_bars, raw_summary_markdown)`
  - Receptive field & causal anchor:
    - `_compute_causal_normalization(df, lookback_bars=512)`: Computes relative % change from $P_0$ (last close) and volume z-score standardization.
    - `_forecast_input_fingerprint(df)`: SHA256 hash of input OHLCV slice.
  - Dual-tier caching (`KronosCacheManager`): In-memory LRU (128 items) + on-disk JSON cache (`kronos-<sha256>.json`).
  - Singleton Model Manager (`KronosModelManager`): Thread-safe persistent VRAM weight caching with device resolution (CUDA bf16/fp16 -> MPS fp32 -> CPU fp32).
  - Graceful Fallback (`_generate_synthetic_forecast`): Moment-based statistical fallback when official model weights/PyTorch are unavailable.
- **Potential Mismatch / Inconsistency Vectors**:
  - Horizon validation: Only horizons `(5, 10, 20)` are supported; others raise `ValueError`.
  - Directional Bias thresholds: `forecast_return_pct >= +2.0%` -> `BULLISH`, `<= -2.0%` -> `BEARISH`, else `NEUTRAL`.
  - Cache fingerprint collisions: Input fingerprint includes exact OHLCV records, model tier, repo, device, attention implementation, temperature, top_p, and sample count.
- **Plan Checklist**:
  - [ ] Verify `generate_kronos_forecast` (`kronos.py:520-629`) properly caches forecasts to disk and enforces input fingerprinting.
  - [ ] Verify official predictor inference (`kronos.py:408-518`) output OHLCV format conforms to `KronosForecastContract`.
  - [ ] Check statistical fallback generator (`kronos.py:295-406`) outputs realistic drift and volatility bounds without NaN/inf.

---

### Section 1.9: Multi-Horizon Structural Levels & Fibonacci Price Geometry
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/structural_levels.py:1-250`
- **Purpose, Contracts & Data Structures**:
  - `compute_structural_levels(df, trade_date, df_1h=None) -> Dict[str, Any]`:
    - Macro Horizon: 52-Week High / Low (or available history if $<180$ bars).
    - Intermediate Horizon: 60D Swing High / Low; Directional Fibonacci retracements (Bullish impulse: 50% & 61.8% pullback support floors; Bearish leg: bounce resistance); Fibonacci extension targets (1.272x, 1.618x for breakouts).
    - Tactical Horizon: 20D Swing High / Low; Volatility channels (+2x, +3x ATR-14); ATR-14 and ATR-20 with % of price.
    - Intraday Supplement: 1H 24-bar swing high/low, EMA 20/50 alignment (`compute_1h_micro_levels`).
  - `get_market_structural_summary(symbol, trade_date, df_1h=None) -> str`: Formats structural geometry into clean prompt text for analyst consumption.
- **Potential Mismatch / Inconsistency Vectors**:
  - Fibonacci Directionality: Bullish impulse (`pos_l60 < pos_h60`) calculates pullback demand floors below 60D high ($H - \text{ratio} \times \text{range}$). Bearish leg calculates bounce resistance above 60D low ($L + \text{ratio} \times \text{range}$).
  - Price Units & Scales: All calculated levels are rounded to 2 decimal places in native asset currency.
  - Intraday Data Availability: In PIT mode, `df_1h` is extracted from `snapshot_data["ohlcv_1h"]` if present; gracefully omitted if unavailable.
- **Plan Checklist**:
  - [ ] Verify `compute_structural_levels` (`structural_levels.py:76-192`) handles short histories ($<20$ bars or $<60$ bars) gracefully.
  - [ ] Validate Fibonacci calculation directionality and extension formulas at lines 128-156.
  - [ ] Check 1H micro-level trend alignment string output at `structural_levels.py:59-65`.

---

### Section 1.10: Alternative Crypto Dataflows (CoinGecko, DeFiLlama, On-Chain, Fear & Greed)
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/coingecko.py:1-159`
  - `tradingagents/dataflows/crypto_id_map.py:1-103`
  - `tradingagents/dataflows/defillama.py:1-246`
  - `tradingagents/dataflows/fear_greed.py:1-130`
  - `tradingagents/dataflows/onchain_metrics.py:1-234`
  - `tradingagents/dataflows/github_activity.py:1-192`
  - `tradingagents/dataflows/crypto_news.py:1-159`
- **Purpose, Contracts & Data Structures**:
  - `coingecko.py`: `get_tokenomics(ticker)`, `get_market_data(ticker)`. Supports Demo (`CG-...`) and Pro API keys. Formats market cap, FDV, circulating/total/max supply, deflationary status, ATH change %, community stats.
  - `crypto_id_map.py`: Fast-path top-50 map + slow-path cached `/coins/list` lookup.
  - `defillama.py`: `get_tvl(ticker, trade_date)`. Excludes L1 base assets (`_NON_DEFI_BASE`), resolves protocol slugs dynamically, parses multi-chain TVL breakdowns and 7d/30d changes.
  - `fear_greed.py`: `get_fear_greed_index(trade_date)`. Queries alternative.me; supports historical date lookups (`limit=0`) and live 7-day trend summaries.
  - `onchain_metrics.py`: `get_onchain_metrics(ticker)`. Resolves EVM contracts via CoinGecko platforms, queries Etherscan for ETH supply / EIP-1559 burns or ERC-20 token supply using contract decimals.
  - `github_activity.py`: `get_dev_activity(ticker)`. Queries repo stats (stars, forks, open issues, 4-week commit velocity) and outputs activity grade.
  - `crypto_news.py`: Aggregates multi-source RSS feeds (CoinDesk, CoinTelegraph, Decrypt, TheBlock) with keyword matching.
- **Potential Mismatch / Inconsistency Vectors**:
  - PIT Safety: In PIT mode, `crypto_fundamental_tools.py` blocks live network calls via `_pit_unavailable()` unless pre-sliced data exists in `snapshot_data["sentiment"]`.
  - Non-DeFi / Native L1 Handling: Calling `get_tvl` on BTC/SOL or `get_onchain_metrics` on native L1s returns informative explanations rather than API errors.
  - Decimal scaling: Etherscan ERC-20 token supply queries contract decimals via `eth_call (0x313ce567)` (defaults to 18) to avoid $10^{18}$ unit errors.
- **Plan Checklist**:
  - [ ] Confirm `crypto_fundamental_tools.py:44-126` fail-closed guards block un-snapshotted live crypto calls in backtest mode.
  - [ ] Verify `defillama.py:199-216` historical TVL slicing applies `cutoff_ts <= trade_date`.
  - [ ] Check Etherscan decimals decoding in `onchain_metrics.py:197-224` to prevent token supply scale distortions.

---

### Section 1.11: Social Sentiment & Public Stream Ingestion
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/stocktwits.py:1-84`
  - `tradingagents/dataflows/reddit.py:1-106`
  - `tradingagents/dataflows/mastodon.py:1-69`
  - `tradingagents/dataflows/bluesky.py:1-62`
- **Purpose, Contracts & Data Structures**:
  - Keyless, unauthenticated public social streams returning formatted prompt-ready strings:
    - `stocktwits.py`: Fetches symbol stream (`api.stocktwits.com`), parses Bullish/Bearish/Unlabeled counts, calculates sentiment %, outputs recent message stream.
    - `reddit.py`: Searches public JSON endpoints across finance subreddits (`r/wallstreetbets`, `r/stocks`, `r/investing`) with inter-request rate limit delays.
    - `mastodon.py`: Queries public hashtag timelines (`mastodon.social` or `MASTODON_INSTANCE`).
    - `bluesky.py`: AT Protocol public post search (`public.api.bsky.app`).
  - Error resilience: All fetchers catch network/parsing exceptions and return descriptive placeholder strings (`<stocktwits unavailable: ...>`) rather than raising.
- **Potential Mismatch / Inconsistency Vectors**:
  - Point-in-time leakage: Live social streams reflect current sentiment and must never be queried during historical backtesting.
  - Rate limiting & timeouts: 10.0s timeout per call with rate limiting (Reddit 0.4s delay, StockTwits stream limit).
- **Plan Checklist**:
  - [ ] Verify sentiment agent tools disable live social stream calls during PIT backtest mode.
  - [ ] Ensure HTML tag stripping and 280-char truncation across all social message formatters.
  - [ ] Check graceful degradation placeholder output across all four social data sources.

---

### Section 1.12: Web Search Ingestion (Exa Time-Travel & SearXNG)
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/exa_search.py:1-211`
  - `tradingagents/dataflows/searxng.py:1-76`
  - `tradingagents/agents/utils/web_search_tools.py:1-46`
- **Purpose, Contracts & Data Structures**:
  - `ExaTimeTravelSearch.search(query, trade_date, num_results=5, category="news")`:
    - Zero-lookahead historical search: clamps `endPublishedDate` to `YYYY-MM-DDT23:59:59.000Z`.
    - Local disk caching: caches Exa responses as JSON under `data_cache_dir/exa_search/<sha256>.json`.
    - Post-filter verification: `_filter_historical_results` re-verifies `publishedDate <= cutoff` and `updatedDate <= cutoff`.
    - Fail-closed PIT behavior: In PIT mode, if `EXA_API_KEY` is missing or Exa rate-limits, returns `"Historical search unavailable; no point-in-time result was used"` (never falls back to live SearXNG).
  - `searxng.py`: Live fallback search against local/remote SearXNG instance (`SEARXNG_URL`).
- **Potential Mismatch / Inconsistency Vectors**:
  - Time-travel vs Live search boundary: Live SearXNG search has no date ceiling. Exa time-travel search is strictly required whenever `trade_date` is present.
  - Category restrictions: Only `"news"` and `"general"` categories are accepted; invalid category raises `ValueError`.
- **Plan Checklist**:
  - [ ] Validate `exa_search.py:160-166` strictly blocks fallback to live SearXNG when `end_published_date` or PIT mode is active.
  - [ ] Verify `_filter_historical_results` (`exa_search.py:54-69`) rejects any article whose update timestamp exceeds `cutoff`.
  - [ ] Check `get_web_search` tool bridge in `web_search_tools.py` correctly passes active trade date from runtime config.

---

### Section 1.13: Deterministic Market Data Validation Snapshot & Analyst Bridges
- **Exact File Paths & Lines**:
  - `tradingagents/dataflows/market_data_validator.py:1-123`
  - `tradingagents/agents/utils/market_data_validation_tools.py:1-23`
  - `tradingagents/agents/utils/core_stock_tools.py:1-79`
  - `tradingagents/agents/utils/technical_indicators_tools.py:1-35`
  - `tradingagents/agents/utils/fundamental_data_tools.py:1-91`
  - `tradingagents/agents/utils/news_data_tools.py:1-60`
  - `tradingagents/agents/utils/rating.py:1-28`
- **Purpose, Contracts & Data Structures**:
  - `build_verified_market_snapshot(symbol, curr_date, look_back_days=30, indicators=None) -> str`:
    - Deterministic ground-truth snapshot for LLM market analysts to prevent numeric hallucinations.
    - Generates markdown tables with latest verified OHLCV row, technical indicators (`close_10_ema`, `close_50_sma`, `close_200_sma`, `rsi`, `boll`, `boll_ub`, `boll_lb`, `macd`, `macds`, `macdh`, `atr`), and recent close prices.
    - Re-applies defensive causal cutoff (`df["Date"] <= curr_date`).
  - Analyst Tool Bridges:
    - `get_stock_data`, `get_kronos_forecast` (`core_stock_tools.py`).
    - `get_indicators` (`technical_indicators_tools.py` - splits comma-separated indicator lists).
    - `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` (`fundamental_data_tools.py` - handles inverted argument orders via `_normalize_freq_and_date`).
    - `get_news`, `get_global_news`, `get_insider_transactions` (`news_data_tools.py`).
  - Canonical Rating Normalization:
    - `parse_rating(text: str, default: str = "WNS") -> str`: Normalizes all 5-tier recommendations (`Buy`, `Overweight`, `Hold`, `Wait & See`, `Underweight`, `Sell`) to canonical binary set: `BUY` or `WNS`.
- **Potential Mismatch / Inconsistency Vectors**:
  - Rating enum consistency: Downstream execution layers require binary `BUY` / `WNS`. `parse_rating` must map `Hold`, `Underweight`, `Sell`, and `WNS` to `WNS`, and `Buy` / `Overweight` to `BUY`.
  - Tool argument order handling: LLMs often swap `freq` and `curr_date` in `get_balance_sheet(ticker, "2026-05-25", "quarterly")`. Handled by regex detection in `_normalize_freq_and_date`.
  - Multi-indicator batching: `get_indicators` splits comma-separated strings (e.g. `"rsi,macd"`) to prevent tool execution failures.
- **Plan Checklist**:
  - [ ] Confirm `build_verified_market_snapshot` (`market_data_validator.py:62-122`) executes without invoking LLMs and enforces exact date filtering.
  - [ ] Verify `_normalize_freq_and_date` (`fundamental_data_tools.py:28-32`) correctly corrects swapped `(freq, curr_date)` arguments.
  - [ ] Validate `parse_rating` (`rating.py:17-28`) maps all legacy rating variations strictly to canonical `BUY` or `WNS`.

---

### Section 1.14: Subsystem Crosscheck Matrix (Summary of Mismatch Vectors)

| Domain | Vector / Contract | Verification Rule & Source Reference |
| :--- | :--- | :--- |
| **Timeframe / Horizon** | 1D vs Intraday vs Multi-Horizon | `get_intraday_data` requires tz-aware ISO timestamps for intraday; daily bars localized without calendar shift (`y_finance.py:65-76`). `structural_levels.py` checks 52W, 60D, 20D, and 1H micro-levels. |
| **Rating / Signal Enum** | Canonical binary signals (`BUY`, `WNS`) | Legacy labels (`Buy`, `Overweight`, `Hold`, `Underweight`, `Sell`, `Wait and See`) normalized to `BUY` / `WNS` by `parse_rating` (`rating.py:17-28`). |
| **Order Types & Geometry** | Causal price levels & Fibonacci ranges | Directional Fibonacci pullback support vs expansion targets calculated causally from 60D swings (`structural_levels.py:128-156`). |
| **Units & Scales** | % vs Decimals, Currencies, Decimals | Etherscan queried for ERC-20 contract decimals (`onchain_metrics.py:197-208`); ATR formatted in points and % of price (`structural_levels.py:151`). |
| **Config & Precedence** | Priority resolution hierarchy | Tool-level override (`tool_vendors`) > Category-level (`data_vendors`) > Default (`interface.py:169-183`). `_ENV_OVERRIDES` coerces types (`default_config.py:31-49`). |
| **PIT Safety & Fallbacks** | Fail-closed backtesting guarantee | `is_point_in_time_mode()` hard-locks `route_to_vendor` to `["snapshot"]` (`interface.py:201-203`); Exa time-travel search blocks live SearXNG fallbacks (`exa_search.py:160-166`). |
| **Schema Parity** | Headers, comments & CSV parsing | OHLCV CSVs generated with comment headers `#`; agent tools strip comments with `pd.read_csv(comment="#")` (`core_stock_tools.py:68`). |

- **Plan Checklist**:
  - [ ] Integrate all 14 granular checklist blocks directly into the master `plan.md` crosscheck matrix.

---

## Subsystem 2: Kronos Neural K-Line Foundation Model Engine

# Subsystem 2: Kronos Neural Foundation Engine (`subsystem-2-kronos-neural-foundation-engine`) — Architectural Mapping & Crosscheck Matrix

---

## 1. Overview & Architectural Boundary

The **Kronos Neural Foundation Engine** integrates the autoregressive financial K-line transformer (`shiyu-coder/Kronos`, AAAI 2026) into the multi-agent trading system. It operates as a modular quantitative dataflow tool (`get_kronos_forecast`) accessible by the `Market Analyst` node, with strict Point-In-Time (PIT) causal isolation, dual-tier caching, hardware acceleration fallbacks, and deterministic greedy decoding ($T=0.0$).

---

## 2. Granular Architectural Sections & Crosscheck Matrix

### Section 2.1: Model Registry, Model Manager & Hardware Acceleration Hierarchy
- **File Paths & Line Ranges**:
  - `tradingagents/dataflows/kronos.py:30-48` (`SUPPORTED_KRONOS_HORIZONS`, `KRONOS_MODEL_REGISTRY`)
  - `tradingagents/dataflows/kronos.py:82-210` (`KronosModelManager` singleton, `get_device_and_dtype`, `load_model`)
  - `tradingagents/dataflows/kronos_official/__init__.py:1-17` (`get_model_class`, model registration)
- **Purpose, Contracts & Data Structures**:
  - **Registry**: Supports tiers `base` (102.3M params, 512 context), `small` (24.7M params, 512 context), `mini` (4.1M params, 2048 context).
  - **Singleton Manager**: `KronosModelManager` uses `threading.Lock()` to maintain persistent model weights in VRAM, eliminating step-wise re-instantiation overhead during backtests.
  - **Hardware Hierarchy**: Auto-resolves execution target:
    1. `CUDA`: `torch.bfloat16` (if Ampere/Ada/Hopper supported) else `torch.float16`, SDPA attention backend.
    2. `MPS`: Apple Silicon `torch.float32`, eager attention backend.
    3. `CPU`: `torch.float32`, eager attention backend.
  - **Local Model Priority**: Prefers local weights at `<root>/models/kronos/base` and `<root>/models/kronos/tokenizer` over remote HuggingFace Hub downloads (`kronos.py:172-176`).
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Context Length Mismatch**: `mini` model uses `context_length: 2048` and `Kronos-Tokenizer-2k`, while `base`/`small` use 512. If `mini` is selected with the default 512 slice, verify it does not trigger indexing anomalies.
  - `[ ]` **PyTorch Compile Boundary**: `torch_compile=True` is restricted to CUDA devices (`kronos.py:184-186`). Non-CUDA environments must safely ignore `torch_compile` without raising exceptions.
  - `[ ]` **Thread Lock Contention**: `KronosModelManager._lock` synchronizes model loading across parallel analyst threads (`analyst_concurrency_limit > 1`). Verify no deadlock occurs during concurrent graph tool executions.

---

### Section 2.2: Dual-Tier Caching Engine & Input Fingerprinting (PIT Isolation)
- **File Paths & Line Ranges**:
  - `tradingagents/dataflows/kronos.py:212-254` (`KronosCacheManager`, in-memory LRU + disk JSON)
  - `tradingagents/dataflows/kronos.py:283-293` (`_forecast_input_fingerprint`)
  - `tradingagents/dataflows/kronos.py:557-584` (`cache_identity`, `cache_key`, disk cache resolution)
- **Purpose, Contracts & Data Structures**:
  - **Tier 1 (Memory)**: `OrderedDict` LRU cache with capacity 128 guarded by `threading.RLock()`.
  - **Tier 2 (Disk JSON)**: Written to `<data_cache_dir>/kronos/kronos-<hash>.json`.
  - **Fingerprint Invariant**: `_forecast_input_fingerprint(df)` hashes normalized causal CSV rows (`date,open,high,low,close,volume`) with SHA-256. The composite cache key includes: `symbol`, `cutoff`, `fingerprint`, `model_tier`, `model_repo`, `tokenizer_repo`, `device_pref`, `attn_pref`, `torch_compile`, `pred_days`, `temperature`, `top_p`, `sample_count`.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Cache Collision / Leakage Guard**: Verify that forecasts computed on different backtest snapshots with identical cutoff dates but varying lookback windows generate distinct cache keys via the OHLCV fingerprint (`kronos.py:283-293`).
  - `[ ]` **Cache Deserialization Integrity**: Deserialized disk cache dictionaries must reconstruct exact `KronosForecastContract` dataclasses (`kronos.py:579-584`).

---

### Section 2.3: Mathematical Normalization, Autoregressive Inference & Fallback Engine
- **File Paths & Line Ranges**:
  - `tradingagents/dataflows/kronos.py:51-80` (`KronosPredictedBar`, `KronosForecastContract`)
  - `tradingagents/dataflows/kronos.py:256-281` (`_compute_causal_normalization`)
  - `tradingagents/dataflows/kronos.py:295-406` (`_generate_synthetic_forecast`)
  - `tradingagents/dataflows/kronos.py:408-518` (`_generate_predictor_forecast`)
  - `tradingagents/dataflows/kronos.py:520-629` (`generate_kronos_forecast`)
- **Purpose, Contracts & Data Structures**:
  - **Causal Normalization**: Anchor price $P_0 = Close_{T0}$ (last bar); relative price scaling: $\tilde{P}_t = (P_t - P_0) / P_0$. Volume standardized via rolling $z$-score: $\tilde{V}_t = (V_t - \mu_V) / (\sigma_V + \epsilon)$ (`kronos.py:256-281`).
  - **Official Predictor Adapter**: Feeds clean 512-bar window to `KronosPredictor.predict()`, maps predicted OHLCV, computes trajectory extremes (`forecast_high`, `forecast_low`, `forecast_end_close`), return % (`forecast_return_pct`), and directional bias (`BULLISH` for $\ge +2.0\%$, `BEARISH` for $\le -2.0\%$, `NEUTRAL` otherwise) (`kronos.py:480-481`).
  - **Statistical Drift/Vol Fallback**: If PyTorch/Transformers are missing or official inference fails, automatically falls back to mathematical drift/vol trajectory from historical moments (`kronos.py:295-406`), labeling output `(Statistical-Fallback)` in `model_name` and `device_used`.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Calendar vs Business Day Timestamps**: `_generate_predictor_forecast` calculates future prediction timestamps using `pd.bdate_range` (`kronos.py:432`). In 24/7 crypto markets, `bdate_range` skips weekends; verify alignment with asset type.
  - `[ ]` **Percentage vs Decimal Scale**: Forecast contract fields (`forecast_return_pct`, `max_upside_pct`, `max_downside_pct`, `return_from_t0_pct`) are formatted as percentages (e.g. `+5.25` for 5.25%), whereas raw returns inside intermediate math are decimal multipliers.
  - `[ ]` **Confidence Score Boundary**: Confidence is clamped to $[0.50, 0.95]$ in both official and fallback generators (`kronos.py:361, 481`).

---

### Section 2.4: Official Kronos Neural Architecture & Tokenizer
- **File Paths & Line Ranges**:
  - `tradingagents/dataflows/kronos_official/kronos.py:12-178` (`KronosTokenizer`)
  - `tradingagents/dataflows/kronos_official/kronos.py:179-340` (`Kronos` model)
  - `tradingagents/dataflows/kronos_official/kronos.py:483-663` (`KronosPredictor`, `calc_time_stamps`)
  - `tradingagents/dataflows/kronos_official/module.py:1-570` (`BinarySphericalQuantizer`, `HierarchicalEmbedding`, `TemporalEmbedding`, `DependencyAwareLayer`, `DualHead`, `TransformerBlock`)
- **Purpose, Contracts & Data Structures**:
  - **Quantization**: `BinarySphericalQuantizer` performs hybrid pre/post token discrete spherical quantization ($s_1 = 6$ bits, $s_2 = 10$ bits).
  - **Transformer Architecture**: Autoregressive dual-head transformer predicting $s_1$ and $s_2$ logits, conditioned via `DependencyAwareLayer` and `TemporalEmbedding` (`minute`, `hour`, `weekday`, `day`, `month`).
  - **Inference Mode**: `KronosPredictor.predict()` standardizes historical inputs ($z$-score with clipping $[-5, 5]$) and auto-regressively predicts $N$ bars ahead.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Amount Column Synthesis**: `KronosPredictor.predict` synthesizes `amount = volume * mean(OHLC)` if missing (`kronos_official/kronos.py:533`). Ensure zero-volume assets do not generate `NaN` amount values.
  - `[ ]` **Deterministic Decoding**: In backtesting, `T=0.0` (greedy decoding) must be strictly enforced to avoid stochastic divergence across runs.

---

### Section 2.5: Dataflow Interface, Vendor Routing & Fail-Closed PIT Isolation
- **File Paths & Line Ranges**:
  - `tradingagents/dataflows/interface.py:95-100` (`TOOLS_CATEGORIES["neural_forecasting"]`)
  - `tradingagents/dataflows/interface.py:155-160` (`VENDOR_METHODS["get_kronos_forecast"]`)
  - `tradingagents/dataflows/interface.py:184-218` (`route_to_vendor`, PIT mode lock)
- **Purpose, Contracts & Data Structures**:
  - **Category Registration**: `neural_forecasting` category exposes `get_kronos_forecast`.
  - **Vendor Dispatch**: Maps `get_kronos_forecast` to `generate_kronos_forecast` across `kronos`, `snapshot`, and `default` vendors.
  - **PIT Fail-Closed Enforcement**: When `is_point_in_time_mode()` is active, `route_to_vendor()` locks `fallback_vendors = ["snapshot"]`, preventing any network escape or un-sandboxed data fetch (`interface.py:201-203`).
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Vendor Signature Alignment**: `generate_kronos_forecast` expects `(symbol, df_bars, cutoff_date, pred_days, config)`. Ensure `route_to_vendor` forwards all arguments without tuple truncation.

---

### Section 2.6: LangGraph Tool Registration & Multi-Horizon Receptive Field Windowing
- **File Paths & Line Ranges**:
  - `tradingagents/agents/utils/core_stock_tools.py:39-79` (`@tool get_kronos_forecast`)
  - `tradingagents/agents/utils/agent_utils.py:9` (Re-export of `get_kronos_forecast`)
  - `tradingagents/graph/trading_graph.py:196-201` (`_create_tool_nodes`, dynamic Kronos tool injection)
- **Purpose, Contracts & Data Structures**:
  - **Receptive Field Fetching**: Tool fetches up to 800 calendar days of historical OHLCV via `route_to_vendor("get_stock_data", symbol, dt_start, curr_date)` to guarantee a full 512 trading-bar receptive field (`core_stock_tools.py:61-63`).
  - **Input Sanitization**: Strips comment headers (`#`), coerces dates to `YYYY-MM-DD`, and clamps history strictly $\le curr\_date$.
  - **Opt-In Dynamic Tool Binding**: `get_kronos_forecast` is conditionally appended to `market_tools` if and only if `config.get("kronos_enabled", False)` is True (`trading_graph.py:197-199`).
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Horizon Validation Gate**: `pred_days` must strictly be one of `SUPPORTED_KRONOS_HORIZONS = (5, 10, 20)`. Invalid horizons return an explicit error string without raising uncaught exceptions (`core_stock_tools.py:49-53`).
  - `[ ]` **Snapshot Window Truncation**: In backtests with restricted `lookback_days` (e.g. 60D or 120D), the returned OHLCV slice has fewer than 512 bars. Kronos handles dynamic contexts $\le 512$, but verify that warning/info logs correctly reflect the truncated receptive field.

---

### Section 2.7: Market Analyst Prompt Integration & Multi-Horizon Structural Confluence
- **File Paths & Line Ranges**:
  - `tradingagents/agents/analysts/market_analyst.py:26-42` (Market Analyst system prompt & workflow)
  - `tradingagents/dataflows/structural_levels.py:76-191` (`compute_structural_levels`, Fib & ATR extensions)
  - `tradingagents/dataflows/structural_levels.py:194-250` (`get_market_structural_summary`)
- **Purpose, Contracts & Data Structures**:
  - **Analytical Workflow**:
    1. Regime identification (Trending, Range-Bound, Pullback, Falling Knife).
    2. Query Kronos forecast: `get_kronos_forecast(symbol, curr_date, pred_days=20)`; report engine type (Official vs Fallback).
    3. Query technical indicators (`close_10_ema`, `close_50_sma`, `close_200_sma`, `macd`, `rsi`, `boll`, `atr`).
    4. Confluence analysis: Compare classical Fibonacci levels (1.272x / 1.618x extensions) and ATR channels (+2x / +3x ATR) against Kronos neural trajectory without claiming a statistical fallback is neural.
    5. Invalidation & Asymmetry: Define Stop Loss below support floors ensuring $R:R \ge 2:1$.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Statistical vs Neural Labelling**: Prompt enforces that LLM distinguishes `Official-Kronos` from `Statistical-Fallback` to prevent hallucinated deep learning certainty (`market_analyst.py:36, 38`).
  - `[ ]` **Level Synthesis Confluence**: Verify that neural target levels in the market report align with Fibonacci breakout extensions rather than conflicting with 60D/52W structure.

---

### Section 2.8: Downstream Agent Anchoring, Capitulation Reversals & Anti-Prompt Gaming
- **File Paths & Line Ranges**:
  - `tradingagents/agents/trader/trader.py:36-46` (Trader execution taxonomy & capitulation reversal rules)
  - `tradingagents/agents/risk_mgmt/aggressive_debator.py:33-37` (Aggressive analyst alpha capture & guardrails)
  - `tradingagents/agents/risk_mgmt/conservative_debator.py:33-37` (Conservative analyst vetoes & mean-reversion rules)
  - `tradingagents/agents/managers/portfolio_manager.py:106-116` (Portfolio Manager capital allocation governance)
  - `tradingagents/agents/schemas.py:182-306` (`TraderProposal`, `render_trader_proposal`)
  - `tradingagents/agents/schemas.py:339-455` (`PortfolioDecision`, `render_pm_decision`)
  - `tradingagents/agents/schemas.py:715-831` (`SignalContract`, validation invariants)
- **Purpose, Contracts & Data Structures**:
  - **Execution Rules**:
    - `Buy Market (T+1 Open)`: Default for momentum breakouts and capitulation reversals.
    - `Buy Limit (T+1 Limit)`: Reserved for resting bids at structural support floors ($Risk \le 5\%$, $R:R \ge 2:1$).
    - `WNS (Wait and See)`: Mandatory for falling knives breaking below 200 SMA/support without structural base.
  - **Capitulation Reversal Confluence**: When price tests major 52W support with extreme oversold $RSI < 28$ **AND Kronos confirms a BULLISH trajectory**, Trader executes `Buy Market` with tight Stop Loss below 52W floor ($Risk \le 5\%$, $R:R \ge 2.5:1$) rather than defaulting to WNS.
  - **Strict Hierarchy Invariant**: If Trader issues `WNS`, Portfolio Manager cannot upgrade to `BUY` (`portfolio_manager.py:101`).
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Order Geometry Bounds**: For all `BUY` signals, $StopLoss < PlannedEntry < TakeProfit$ is strictly validated by Pydantic validators (`schemas.py:292-298, 450-454, 826-830`).
  - `[ ]` **WNS Re-evaluation Gates**: Every `WNS` must specify `wns_recheck_date` (ISO date) or `wns_trigger_price` (numeric level) (`schemas.py:281-286, 438-443, 807-810`).

---

### Section 2.9: Configuration Hierarchy, YAML / Env-Var Resolution & CLI Flags
- **File Paths & Line Ranges**:
  - `tradingagents/default_config.py:24-28, 136-146` (Default config & `TRADINGAGENTS_KRONOS_*` env overrides)
  - `tradingagents/backtesting/position.py:238-259` (`AgentConfig` dataclass)
  - `tradingagents/backtesting/config_resolver.py:88-97, 131-138` (`resolve_agent_config`, `validate_backtest_config`)
  - `tradingagents/backtesting/agent_runner.py:188-195` (`_safe_agent_runtime_config`)
  - `backtest.yaml:130-137` (`agent:` Kronos config block)
  - `cli/main.py:478-483, 513` (`--kronos / --no-kronos` CLI option in `run_analysis`)
  - `cli/commands/evaluate.py:74-78, 105-108` (`--kronos / --no-kronos` in `evaluate-signal`)
- **Purpose, Contracts & Data Structures**:
  - **Resolution Hierarchy**: CLI Flag $\rightarrow$ YAML (`backtest.yaml`) $\rightarrow$ Environment Variables (`TRADINGAGENTS_KRONOS_*`) $\rightarrow$ `DEFAULT_CONFIG`.
  - **Config Keys**:
    - `kronos_enabled`: `bool` (default: `False`)
    - `kronos_model_tier`: `str` (`"base"`, `"small"`, `"mini"`; default: `"base"`)
    - `kronos_model_repo`: `str` (default: `"NeoQuasar/Kronos-base"`)
    - `kronos_tokenizer_repo`: `str` (default: `"NeoQuasar/Kronos-Tokenizer-base"`)
    - `kronos_device`: `str` (`"auto"`, `"cuda"`, `"mps"`, `"cpu"`; default: `"auto"`)
    - `kronos_attn_implementation`: `str` (`"sdpa"`, `"flash_attention_2"`, `"eager"`; default: `"sdpa"`)
    - `kronos_torch_compile`: `bool` (default: `False`)
    - `kronos_pred_len`: `int` (`5`, `10`, `20`; default: `20`)
    - `kronos_temperature`: `float` (default: `0.0`)
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Backtest AgentConfig Parity**: Verify that `position.py:AgentConfig` carries all Kronos fields and `config_resolver.py` forwards them without dropouts (`config_resolver.py:88-97`).
  - `[ ]` **Temperature in AgentConfig**: `AgentConfig` in `position.py` does not define `kronos_temperature`; `generate_kronos_forecast` defaults to `0.0` when absent. Confirm deterministic backtesting is preserved.

---

### Section 2.10: Regression Test Suite & Verification Matrix
- **File Paths & Line Ranges**:
  - `tests/test_kronos_forecast.py:1-177` (Full unit & regression test suite)
- **Purpose, Contracts & Data Structures**:
  - **PIT Causal Isolation Test** (`test_point_in_time_causal_isolation`): Asserts that appending future bars $> T_0$ produces identical outputs to a clean slice $\le T_0$ (`test_kronos_forecast.py:62-87`).
  - **Anchor Math Test** (`test_causal_normalization_anchor_math`): Asserts $(Close_{T0} - P_0) / P_0 == 0.0$ and volume $z$-score standardization (`test_kronos_forecast.py:51-60`).
  - **Determinism Test** (`test_deterministic_forecast_reproducibility`): Asserts repeated calls with $T=0.0$ return identical contracts (`test_kronos_forecast.py:89-111`).
  - **Cache Isolation Test** (`test_dual_tier_cache_isolation`): Asserts snapshot-bound JSON files are written to disk cache and retrievable via `KronosCacheManager.get` (`test_kronos_forecast.py:113-141`).
  - **LangGraph Tool Invocation Test** (`test_langgraph_tool_invocation`): Asserts `@tool get_kronos_forecast` returns formatted markdown summary table (`test_kronos_forecast.py:143-177`).
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` **Test Suite CI Execution**: Confirm `pytest tests/test_kronos_forecast.py` executes without optional PyTorch/Transformers dependencies present by verifying mock/statistical fallback execution paths.

---

## 3. Master Checklist for Direct Merge into `plan.md`

```markdown
### Subsystem 2: Kronos Neural Foundation Engine Crosscheck Matrix
- [ ] 2.1 Model Registry & Weights: Verify base/small/mini context lengths (512 vs 2048) and local repo fallback in `tradingagents/dataflows/kronos.py:30-48`.
- [ ] 2.2 Dual-Tier Caching: Verify SHA-256 OHLCV input fingerprinting and disk JSON cache isolation in `tradingagents/dataflows/kronos.py:212-254, 283-293`.
- [ ] 2.3 Causal Normalization: Verify Close_T0 anchor scaling and volume z-score standardization in `tradingagents/dataflows/kronos.py:256-281`.
- [ ] 2.4 Official vs Fallback Parity: Verify `_generate_predictor_forecast` and `_generate_synthetic_forecast` adhere to `KronosForecastContract` in `tradingagents/dataflows/kronos.py:51-80, 408-518`.
- [ ] 2.5 PIT Fail-Closed Router: Verify `interface.py:155-160, 201-203` enforces `fallback_vendors = ["snapshot"]` in point-in-time mode.
- [ ] 2.6 LangGraph Tool Binding: Verify `@tool get_kronos_forecast` in `core_stock_tools.py:39-79` is dynamically bound in `trading_graph.py:196-201` when `kronos_enabled=True`.
- [ ] 2.7 Market Analyst Confluence: Verify `market_analyst.py:26-42` prompts compare neural trajectories with Fibonacci extensions (1.272x/1.618x) and ATR channels.
- [ ] 2.8 Downstream Execution Anchoring: Verify `trader.py:36-46` and `portfolio_manager.py:106-116` enforce Capitulation Reversal rules and WNS falling knife vetoes.
- [ ] 2.9 Config & CLI Resolution: Verify CLI `--kronos` flag (`cli/main.py:478-483`, `cli/commands/evaluate.py:74-78`), YAML (`backtest.yaml:130-137`), and env vars resolve across `position.py` and `config_resolver.py`.
- [ ] 2.10 Zero Leakage Test Suite: Verify `tests/test_kronos_forecast.py` covers causal isolation, determinism, and caching.
```

---

## Subsystem 3: Structural Levels, Directional Fibonacci & Indicator Math

# Subsystem 3: Structural Levels & Indicators — Architectural Crosscheck Mapping

## Overview & Architecture Scope
Subsystem 3 delivers institutional-grade quantitative price structure, multi-horizon swing/Fibonacci levels, causal volatility channels, and technical indicator computations. It bridges raw OHLCV time series to agent prompts, structured schemas, and backtesting risk/evaluation engines.

---

### Section 3.1: Quantitative Multi-Horizon Structural Levels Engine
- **Files & Line Ranges**:
  - `tradingagents/dataflows/structural_levels.py:1-250`
- **Purpose & Core Contracts**:
  - `compute_1h_micro_levels(df_1h: pd.DataFrame, trade_date: str) -> Dict[str, Any]` (`lines 14-74`): Calculates 24-bar intraday swing high/low, EMA 20, EMA 50, and 3-state trend bias (`BULLISH (Close > EMA20 > EMA50)`, `BEARISH (Close < EMA20 < EMA50)`, `NEUTRAL / MIXED (Consolidation or Pullback)`).
  - `compute_structural_levels(df: pd.DataFrame, trade_date: str, df_1h: Optional[pd.DataFrame] = None) -> Dict[str, Any]` (`lines 76-192`): Calculates:
    - **Macro Horizon**: 52-Week (252-bar) or available history High/Low; sets `is_full_52w = len(history) >= 180`.
    - **Intermediate Horizon**: 60D Swing High/Low, 20D Swing High/Low.
    - **Directional Fibonacci Retracements**: Bullish impulse (`pos_l60 < pos_h60`): $\text{Fib}_{50\%} = H_{60} - 0.50 \times \text{range}$, $\text{Fib}_{61.8\%} = H_{60} - 0.618 \times \text{range}$; Bearish leg (`pos_l60 \ge pos_h60`): $\text{Fib}_{50\%} = L_{60} + 0.50 \times \text{range}$, $\text{Fib}_{61.8\%} = L_{60} + 0.618 \times \text{range}$.
    - **Fibonacci Extension Targets**: $\text{Fib}_{1.272\text{x}} = H_{60} + 0.272 \times \text{range}$, $\text{Fib}_{1.618\text{x}} = H_{60} + 0.618 \times \text{range}$ (fallback to $1.05\times\text{Close}$ and $1.10\times\text{Close}$ if $\text{range} \le 0$).
    - **Volatility Channels**: $\text{ATR}_{14}$ (Wilder's smoothing), $\text{ATR}_{20}$, $\text{ATR}_{14\%} = (\text{ATR}_{14}/\text{Close}) \times 100$, $\text{ATR}_{\text{half}} = 0.5 \times \text{ATR}_{14}$, $\text{Target}_{+2\text{x}} = \text{Close} + 2.0 \times \text{ATR}$, $\text{Target}_{+3\text{x}} = \text{Close} + 3.0 \times \text{ATR}$.
  - `get_market_structural_summary(symbol: str, trade_date: str, df_1h: Optional[pd.DataFrame] = None) -> str` (`lines 194-250`): Generates structured markdown prompt text with sections 1 (Macro), 2 (Intermediate), 3 (Tactical), and conditional 4 (Intraday 1H Micro).
- **Mismatch & Inconsistency Vectors**:
  - *Timeframe assumptions*: Macro expects daily bars; 1H micro requires explicit `df_1h` or `snapshot_data["ohlcv_1h"]`. If missing, 1H section is cleanly omitted without crashing.
  - *Zero-range handling*: `fib_range <= 0` collapses `fib_50` and `fib_618` to `last_close`.
  - *Rounding scale*: All price levels rounded to 2 decimal places (`round(val, 2)`). Crypto sub-cent assets (< $0.01) risk severe precision loss if price is truncated to 0.00.
  - *PIT compliance*: Filters `work_df["date_str"] <= str(trade_date)[:10]` to strictly prevent look-ahead bias.

#### Crosscheck Checklist
- [ ] `structural_levels.py:31-35`: Confirm 1H cutoff logic handles both full ISO timestamps (`YYYY-MM-DDTHH:MM:SS`) and date-only strings (`YYYY-MM-DD 23:59:59 UTC`).
- [ ] `structural_levels.py:118-119`: Verify threshold for `is_full_52w` ($N \ge 180$ bars) matches prompt label rendering (`52-Week Range` vs `Available History Range (N bars)`).
- [ ] `structural_levels.py:133-142`: Validate directional Fibonacci index detection (`argmax()` vs `argmin()`) when multiple identical highs/lows occur in the 60-bar window.
- [ ] `structural_levels.py:154-155`: Verify breakout upside extension targets anchor to $H_{60}$ ($\text{fib\_ext\_1272} = H_{60} + 0.272 \times \text{range}$).
- [ ] `structural_levels.py:207-210`: Verify fallback retrieval of `snapshot_data["ohlcv_1h"]` in PIT mode when `df_1h=None`.

---

### Section 3.2: Stockstats & OHLCV Caching Utility Layer
- **Files & Line Ranges**:
  - `tradingagents/dataflows/stockstats_utils.py:1-221`
- **Purpose & Core Contracts**:
  - `yf_retry(func, max_retries=3, base_delay=2.0)` (`lines 17-34`): Exponential backoff specifically for `YFRateLimitError` (HTTP 429).
  - `_clean_dataframe(data: pd.DataFrame) -> pd.DataFrame` (`lines 36-49`): Normalizes Date, converts `Open`, `High`, `Low`, `Close`, `Volume` to numeric, drops NaNs, and applies forward fill.
  - `load_ohlcv(symbol: str, curr_date: str = None) -> pd.DataFrame` (`lines 52-143`):
    - In PIT/backtest mode: Reads exclusively from `config["snapshot_data"]["ohlcv"]`, enforces uppercase column normalization (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`), rejects live fallback, and filters `Date <= curr_date`.
    - In live mode: Downloads 15-year historical window anchored to `curr_date` (or today), uses atomic `.tmp` cache file writing (`{safe_symbol}-YFin-data-{start_str}-{end_str}.csv`), validates required columns, drops poisoned caches, and filters `Date <= curr_date`.
  - `filter_financials_by_date(data: pd.DataFrame, curr_date: str, min_filing_lag_days: int = 45) -> pd.DataFrame` (`lines 145-156`): Enforces 45-day filing lag buffer on financial statements.
  - `compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series` (`lines 159-176`): Causal Wilder's Exponential Moving Average: $\text{TR} = \max(H - L, |H - C_{t-1}|, |L - C_{t-1}|)$; $\text{ATR} = \text{TR}.\text{ewm}(\alpha = 1/\text{period}, \text{min\_periods}=\text{period}, \text{adjust}=\text{False}).\text{mean}()$.
  - `StockstatsUtils.get_stock_stats(symbol, indicator, curr_date)` (`lines 179-221`): Evaluates indicator on `curr_date` or falls back to the most recent causal trading day row.
- **Mismatch & Inconsistency Vectors**:
  - *ATR implementation discrepancy*: `stockstats_utils.py:175` uses Wilder's EMA (`ewm(alpha=1/period)`), whereas `backtesting/risk.py:346` uses simple rolling mean (`rolling(window=period).mean()`).
  - *Weekend/Holiday non-trading day query*: Returns string `"N/A: Not a trading day (weekend or holiday)"` or last causal row if available.

#### Crosscheck Checklist
- [ ] `stockstats_utils.py:68-80`: Verify PIT mode raises `RuntimeError` if `config["snapshot_data"]["ohlcv"]` is missing or empty, with zero live network calls.
- [ ] `stockstats_utils.py:61`: Verify `safe_ticker_component(symbol)` sanitizes path traversals (e.g. `../../etc/passwd`).
- [ ] `stockstats_utils.py:123-125`: Verify atomic file write pattern (`.tmp` renamed to `.csv` via `os.replace`) prevents race conditions and corrupted cache reads.
- [ ] `stockstats_utils.py:159-176`: Verify `compute_atr` produces identical series lengths and handles NaN for initial `period - 1` bars.

---

### Section 3.3: YFinance Indicator Calculation & Window Formatting
- **Files & Line Ranges**:
  - `tradingagents/dataflows/y_finance.py:1-309`
- **Purpose & Core Contracts**:
  - `get_intraday_data(symbol, start, end, interval, timezone="UTC") -> pd.DataFrame` (`lines 22-89`): Fetches validated OHLCV for intervals (`1m`, `5m`, `10m`, `15m`, `30m`, `1h`, `2h`, `3h`, `4h`, `1d`). Enforces tz-awareness, non-empty, finite positive prices, and single-day inclusive advance (`end_dt + 1 day` for daily).
  - `get_YFin_data_online(symbol, start_date, end_date) -> str` (`lines 91-133`): Formats OHLCV as CSV string with comment headers (`# Stock data for SYMBOL`).
  - `get_stock_stats_indicators_window(symbol, indicator, curr_date, look_back_days) -> str` (`lines 135-278`):
    - Resolves indicator aliases: `rsi_14` $\to$ `rsi`, `atr_14` $\to$ `atr`, `close_50_sma`, `close_200_sma`, `close_10_ema`.
    - Supported indicators dictionary: `close_50_sma`, `close_200_sma`, `close_10_ema`, `macd`, `macds`, `macdh`, `rsi`, `boll`, `boll_ub`, `boll_lb`, `atr`, `atr_14`, `atr_20`, `vwma`, `mfi`.
    - Calls `_get_stock_stats_bulk` once, formats date-value series in ascending order (Past $\to$ Present), skips non-trading days to optimize token density, and appends parameter descriptions.
  - `_get_stock_stats_bulk(symbol, indicator, curr_date) -> dict` (`lines 280-309`): Bulk calculation returning `Dict[str, str]` mapping `YYYY-MM-DD` to indicator value. Special-cases `atr`, `atr_14`, `atr_20` to use `stockstats_utils.compute_atr`.
- **Mismatch & Inconsistency Vectors**:
  - *Indicator Aliases*: `y_finance.py` handles `rsi_14` and `atr_14`, but `alpha_vantage_indicator.py` and `snapshot.py` reject `rsi_14` if queried without aliasing.
  - *Output format*: Generates Markdown string with `# Time Series (Ascending Order: Past -> Present)`.

#### Crosscheck Checklist
- [ ] `y_finance.py:46-52`: Verify daily interval single-day queries advance `end_arg` by 1 day to guarantee inclusive bar return.
- [ ] `y_finance.py:144-153`: Verify alias dictionary normalizes `rsi_14` $\to$ `rsi` and `atr_14` $\to$ `atr` before key validation against `best_ind_params`.
- [ ] `y_finance.py:249-260`: Verify date iterator ascends from `curr_date - look_back_days` to `curr_date`, omitting weekend keys absent from `indicator_data`.
- [ ] `y_finance.py:297-303`: Verify bulk calculation routes `atr`, `atr_14`, `atr_20` to `compute_atr` and formats NaNs as `"N/A"`.

---

### Section 3.4: Alpha Vantage Indicator Vendor & REST API Integration
- **Files & Line Ranges**:
  - `tradingagents/dataflows/alpha_vantage_indicator.py:1-224`
- **Purpose & Core Contracts**:
  - `get_indicator(symbol, indicator, curr_date, look_back_days, interval="daily", time_period=14, series_type="close") -> str` (`lines 4-224`):
    - Supported indicators mapping: `close_50_sma` (`SMA`, period 50), `close_200_sma` (`SMA`, period 200), `close_10_ema` (`EMA`, period 10), `macd` (`MACD`), `macds` (`MACD`), `macdh` (`MACD`), `rsi` (`RSI`, period 14), `boll`/`boll_ub`/`boll_lb` (`BBANDS`, period 20), `atr` (`ATR`, period 14), `vwma` (unsupported direct endpoint).
    - Makes REST API calls via `_make_api_request` requesting CSV datatypes.
    - Parses CSV header to extract target column (`MACD`, `MACD_Signal`, `MACD_Hist`, `Real Middle Band`, `Real Upper Band`, `Real Lower Band`, `RSI`, `ATR`, `EMA`, `SMA`).
    - Filters data rows to `before <= date <= curr_date_dt`, sorts ascending, formats date-value lines, and attaches usage guidance.
- **Mismatch & Inconsistency Vectors**:
  - *VWMA limitation*: Alpha Vantage API lacks direct VWMA; `alpha_vantage_indicator.py:147-150` returns an informational notice rather than tabular data.
  - *Rate limits*: Raises `AlphaVantageRateLimitError` or returns formatted error strings.

#### Crosscheck Checklist
- [ ] `alpha_vantage_indicator.py:32-45`: Confirm supported indicator list matches names expected by LLM tool prompt (`close_50_sma`, `close_200_sma`, `close_10_ema`, `macd`, `macds`, `macdh`, `rsi`, `boll`, `boll_ub`, `boll_lb`, `atr`, `vwma`).
- [ ] `alpha_vantage_indicator.py:167-172`: Verify column mapping dictionary matches exact Alpha Vantage CSV response headers (`Real Middle Band`, `MACD_Signal`, `MACD_Hist`).
- [ ] `alpha_vantage_indicator.py:197`: Confirm strict date range bounding (`before <= date_dt <= curr_date_dt`) prevents lookahead data leakage.

---

### Section 3.5: Snapshot Indicator Engine for Backtesting & PIT Integrity
- **Files & Line Ranges**:
  - `tradingagents/dataflows/snapshot.py:140-230`
- **Purpose & Core Contracts**:
  - `snapshot_get_indicators(symbol, indicator, curr_date, look_back_days) -> str` (`lines 144-230`):
    - Reads from `config["snapshot_data"]["ohlcv"]`.
    - Validates indicator against `best_ind_params` (`close_50_sma`, `close_200_sma`, `close_10_ema`, `macd`, `macds`, `macdh`, `rsi`, `boll`, `boll_ub`, `boll_lb`, `atr`, `vwma`, `mfi`).
    - Normalizes columns to `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
    - Filters using `_effective_cutoff(curr_date)` ensuring `Date <= curr_date_dt`.
    - Computes indicator values via `stockstats.wrap(df)`.
    - Builds lookback window lines (`date_str >= before and date_str <= curr_date`) and returns markdown block.
- **Mismatch & Inconsistency Vectors**:
  - *Missing alias translation*: `snapshot.py` lacks alias dictionary (`rsi_14`, `atr_14`). If an agent queries `rsi_14` in snapshot mode, it receives `Indicator 'rsi_14' not supported`.
  - *Single OHLCV source*: Snapshot indicator relies entirely on pre-injected daily OHLCV dataframe.

#### Crosscheck Checklist
- [ ] `snapshot.py:176-180`: Verify behavior when unsupported indicator is passed (returns helpful error listing valid keys).
- [ ] `snapshot.py:193-196`: Verify `_effective_cutoff` enforces the minimum of requested `curr_date` and runtime `trade_date`.
- [ ] `snapshot.py:202-206`: Verify `stockstats.wrap(df)` triggers indicator computation without mutating source snapshot records.

---

### Section 3.6: Dataflow Interface Routing, Vendor Dispatch & Fail-Closed Guardrails
- **Files & Line Ranges**:
  - `tradingagents/dataflows/interface.py:1-218`
- **Purpose & Core Contracts**:
  - `TOOLS_CATEGORIES["technical_indicators"]` (`lines 72-77`): Maps `get_indicators` tool to `technical_indicators` category.
  - `VENDOR_METHODS["get_indicators"]` (`lines 112-116`):
    - `alpha_vantage`: `get_alpha_vantage_indicator`
    - `yfinance`: `get_stock_stats_indicators_window`
    - `snapshot`: `snapshot_get_indicators`
  - `get_vendor(category, method=None)` (`lines 169-183`): Resolves tool-level override (`tool_vendors[method]`) before category-level default (`data_vendors[category]`).
  - `route_to_vendor(method, *args, **kwargs)` (`lines 184-218`):
    - Resolves vendor execution order.
    - **PIT Fail-Closed Guardrail**: If `is_point_in_time_mode()` is True, `fallback_vendors` is hard-locked to `["snapshot"]` (`lines 201-203`).
    - Handles retry/fallback on `AlphaVantageRateLimitError`, `YFRateLimitError`, and network errors (`ConnectionError`, `TimeoutError`, `OSError`).
- **Mismatch & Inconsistency Vectors**:
  - *Default vendor config*: `default_config.py:110` sets `"technical_indicators": "yfinance"`.
  - *CLI/Backtest mode override*: Backtest runners set `data_vendors["technical_indicators"] = "snapshot"` and `point_in_time_mode = True`.

#### Crosscheck Checklist
- [ ] `interface.py:112-116`: Confirm all 3 vendor methods (`alpha_vantage`, `yfinance`, `snapshot`) are registered under `VENDOR_METHODS["get_indicators"]`.
- [ ] `interface.py:201-203`: Verify that under `is_point_in_time_mode()`, `fallback_vendors` cannot fall back to `yfinance` or `alpha_vantage`.
- [ ] `interface.py:211-217`: Verify rate limit exceptions trigger next vendor in live mode, but raise `RuntimeError` in PIT mode if snapshot is exhausted.

---

### Section 3.7: Deterministic Ground-Truth Market Data Validator
- **Files & Line Ranges**:
  - `tradingagents/dataflows/market_data_validator.py:1-123`
  - `tradingagents/agents/utils/market_data_validation_tools.py:1-23`
- **Purpose & Core Contracts**:
  - `DEFAULT_SNAPSHOT_INDICATORS` (`lines 21-25`): Fixed tuple of indicators (`close_10_ema`, `close_50_sma`, `close_200_sma`, `rsi`, `boll`, `boll_ub`, `boll_lb`, `macd`, `macds`, `macdh`, `atr`).
  - `_verified_rows(symbol, curr_date) -> pd.DataFrame` (`lines 28-46`): Retrieves OHLCV via `load_ohlcv`, defensively enforces `Date <= pd.to_datetime(curr_date)`.
  - `build_verified_market_snapshot(symbol, curr_date, look_back_days=30, indicators=None) -> str` (`lines 62-123`):
    - Computes verified OHLCV and indicator values for latest bar.
    - Generates markdown verification table with:
      - Latest verified OHLCV row (`Open`, `High`, `Low`, `Close`, `Volume`).
      - Verified technical indicators table.
      - Recent verified closes table (last $N$ trading rows).
      - Strict instruction prohibiting hallucination of ungrounded bounces/levels.
  - `get_verified_market_snapshot` LangChain tool (`market_data_validation_tools.py:8-23`): Exposes snapshot validator to agent toolchains.
- **Mismatch & Inconsistency Vectors**:
  - *Source of truth*: Explicitly instructs LLMs that if tool outputs conflict with the deterministic snapshot, they must flag the discrepancy rather than reconciling with imagined numbers.

#### Crosscheck Checklist
- [ ] `market_data_validator.py:42`: Verify defensive filtering `df["Date"] <= pd.to_datetime(curr_date)` is applied independently of `load_ohlcv`.
- [ ] `market_data_validator.py:81-82`: Confirm indicator calculation exceptions are caught individually (`N/A (ExceptionName)`) without failing the entire snapshot table.
- [ ] `market_data_validator.py:114-122`: Verify prompt instruction anchors support/resistance verification to exact data rows.

---

### Section 3.8: LangChain Indicator & Stock Tool Wrappers
- **Files & Line Ranges**:
  - `tradingagents/agents/utils/technical_indicators_tools.py:1-35`
  - `tradingagents/agents/utils/core_stock_tools.py:1-79`
- **Purpose & Core Contracts**:
  - `get_indicators` (`technical_indicators_tools.py:8-35`):
    - LangChain tool wrapper annotated with `symbol`, `indicator`, `curr_date`, `look_back_days=30`.
    - Handles comma-separated multi-indicator queries (`lines 28-34`): splits `indicator.split(",")` and invokes `route_to_vendor` per indicator, joining results with `\n\n`.
  - `get_stock_data` (`core_stock_tools.py:14-37`):
    - LangChain tool wrapper annotated with `symbol`, `start_date` (defaults to 60 days before `end_date`), `end_date` (defaults to `trade_date` or today).
    - Routes to `route_to_vendor("get_stock_data", symbol, start_date, end_date)`.
  - `get_kronos_forecast` (`core_stock_tools.py:40-79`):
    - Queries neural K-line foundation model for `pred_days` (5, 10, 20).
    - Fetches 800 days of historical OHLCV data to feed receptive field (512 bars).
- **Mismatch & Inconsistency Vectors**:
  - *Comma-separated queries*: LLMs frequently pass `"rsi, macd, close_50_sma"` to `indicator`; splitting logic prevents vendor errors.
  - *Date normalization*: Strips ISO time components (`.split("T")[0].split(" ")[0]`).

#### Crosscheck Checklist
- [ ] `technical_indicators_tools.py:28`: Verify comma-separated splitting trims whitespace and lowercases indicator tokens.
- [ ] `core_stock_tools.py:30-35`: Verify default `start_date` (60 days before `end_date`) provides sufficient bar count for 20D/50D moving averages.
- [ ] `core_stock_tools.py:68-74`: Confirm comment lines (`#`) in CSV output are skipped before passing dataframe to Kronos forecast generator.

---

### Section 3.9: Agent Prompt Context Injection & Multi-Horizon Structural Prompt Topology
- **Files & Line Ranges**:
  - `tradingagents/agents/utils/agent_utils.py:79-101`
  - `tradingagents/agents/analysts/market_analyst.py:1-81`
  - `tradingagents/agents/researchers/bull_researcher.py:1-81`
  - `tradingagents/agents/researchers/bear_researcher.py:1-79`
  - `tradingagents/agents/risk_mgmt/neutral_debator.py:1-58`
  - `tradingagents/agents/risk_mgmt/conservative_debator.py:1-57`
  - `tradingagents/agents/risk_mgmt/aggressive_debator.py:1-57`
  - `tradingagents/agents/trader/trader.py:1-82`
- **Purpose & Core Contracts**:
  - `build_instrument_context(ticker, asset_type="stock", trade_date=None)` (`agent_utils.py:79-101`): Injects pre-calculated `get_market_structural_summary` directly into the system prompt of every downstream agent.
  - `create_market_analyst(llm)` (`market_analyst.py:12-81`):
    - System prompt specifies Institutional Technical Market Analyst evaluating **Multi-Horizon Daily Structure** (Macro 52W/200 SMA $\to$ Intermediate 60D/50 SMA $\to$ Tactical 20D/20 EMA/ATR).
    - Enforces 5-step analytical workflow: Market Regime identification, Kronos K-line forecast query, complementary indicator selection, invalidation/asymmetry check ($R:R \ge 2:1$), and concluding Markdown level summary table.
    - Binds tools: `[get_stock_data, get_indicators]` (+ `get_kronos_forecast` if `kronos_enabled`).
  - Researcher & Debator Prompt Topology:
    - `bull_researcher.py:39-41`: Directs bull researcher to anchor limit accumulation zones to structural support (Fibonacci, 20D/60D swing floors, tactical support).
    - `bear_researcher.py:38-40`: Directs downside specialist to stress-test structural breakdown levels and invalidation vectors.
    - `neutral_debator.py:34-36`: Mandates verification of Multi-Horizon Daily Structure, sub-200 SMA Falling Knife veto, Fib Extensions ($1.272\text{x}, 1.618\text{x}$), and tight limits/market entry for oversold reversals ($R:R \ge 2:1$).
    - `conservative_debator.py:34-36`: Enforces Falling Knife veto below 200 SMA and validates Trader WNS execution hierarchy.
    - `aggressive_debator.py:34-36`: Targets Fib extensions on breakouts and prevents dip-buying into freefall without support base.
  - `trader.py:34-48`: Execution taxonomy rules: `Buy Market` (T+1 Open) for momentum breakout/oversold inflection; `Buy Limit` (support floor accumulation only, within 0.2x ATR); `WNS` (invalidation / falling knife gate).
- **Mismatch & Inconsistency Vectors**:
  - *Terminology harmonization*: Replaced legacy "1D Macro / 1H Micro" prompt requirements with "Multi-Horizon Daily Structure" across all agents.
  - *Crypto asset differences*: Appends note on 24/7 trading, weekend gaps, and absence of circuit breakers for crypto.

#### Crosscheck Checklist
- [ ] `agent_utils.py:88-93`: Verify `build_instrument_context` invokes `get_market_structural_summary` when `trade_date` is provided and appends to context string.
- [ ] `market_analyst.py:31-43`: Confirm prompt enforces $R:R \ge 2:1$ and breakout expansion targets (Fib 1.272x/1.618x, +2x/+3x ATR) rather than capping at local resistance.
- [ ] `trader.py:37-44`: Verify Trader guidelines strictly anchor Take Profit to Fibonacci Extensions and enforce explicit `wns_recheck_date` / `wns_trigger_price` on WNS.
- [ ] `neutral_debator.py:34-37`: Verify neutral debator prompt explicitly checks Multi-Horizon Daily Structure and enforces Trader WNS hierarchy.

---

### Section 3.10: Execution Geometry, Invalidation Guards, and Backtest Evaluator Crosscheck
- **Files & Line Ranges**:
  - `tradingagents/agents/schemas.py:53-125, 280-332, 339-455`
  - `tradingagents/backtesting/risk.py:316-350, 352-425`
  - `tradingagents/backtesting/horizon_evaluator.py:1-120`
- **Purpose & Core Contracts**:
  - Rating Enums & Normalization (`schemas.py:53-87, 109-125`):
    - `PortfolioRating`: Canonical `BUY = "Buy"`, `WNS = "WNS"`. Legacy aliases mapped via `normalize_portfolio_rating`: `OVERWEIGHT` $\to$ `Buy`, `HOLD` / `UNDERWEIGHT` / `SELL` $\to$ `WNS`.
    - `TraderAction`: `BUY = "Buy"`, `BUY_MARKET = "Buy Market"`, `BUY_LIMIT = "Buy Limit"`, `WNS = "WNS"`.
    - `EntryMode`: `ASSUMED_AI_ENTRY`, `T1_OPEN`, `T1_LIMIT`.
  - Proposal Validation & Price Bounds (`schemas.py:280-305, 437-455`):
    - For `BUY`: Mandates `stop_loss` and `take_profit`. Enforces $\text{stop\_loss} < \text{take\_profit}$ and $\text{stop\_loss} < \text{entry\_price} < \text{take\_profit}$.
    - For `WNS`: Requires either `wns_recheck_date` (YYYY-MM-DD) or `wns_trigger_price`.
    - Desk threshold warning: Logs warning if $(\text{take\_profit} - \text{entry}) / (\text{entry} - \text{stop\_loss}) < 1.95$.
  - Backtest Risk Engine & Static Exits (`risk.py:352-425`):
    - Entry-time invariants: Once initialized, `position.stop_price` and `position.take_profit` are immutable (zero ratcheting/widening post-entry).
    - Default ATR-based stops: $\text{Stop} = \text{Entry} - (\text{ATR} \times 2.0)$; $\text{TP} = \text{Entry} + (\text{ATR} \times 3.0)$.
  - Horizon Evaluator Execution Modes (`horizon_evaluator.py`):
    - `T1_OPEN`: Fills at the open of the first trading day following signal date.
    - `T1_LIMIT`: Fills only if bar low $\le \text{planned\_entry\_price} \le$ bar high; otherwise reports `NO_FILL`.
    - `WNS`: Evaluates to `EvaluationOutcome.NO_ORDER` with $0.0\%$ realized return.
- **Mismatch & Inconsistency Vectors**:
  - *Desk R:R vs Hard Validator*: Schema warns on $R:R < 1.95$, while prompt instructs $R:R \ge 2:1$.
  - *ATR Calculation Method*: `stockstats_utils.py` uses Wilder's EMA for prompt indicators; `risk.py` uses 14-period rolling mean for fallback stop calculation if agent leaves stops undefined.

#### Crosscheck Checklist
- [ ] `schemas.py:290-298`: Verify Pydantic validator rejects BUY proposals where $\text{stop\_loss} \ge \text{take\_profit}$ or $\text{entry} \le \text{stop\_loss}$.
- [ ] `schemas.py:438-443`: Verify Pydantic validator rejects WNS decisions lacking both `wns_recheck_date` and `wns_trigger_price`.
- [ ] `risk.py:389-395`: Verify position stop and take-profit levels cannot be overwritten once set (`position.stop_price is not None`).
- [ ] `horizon_evaluator.py`: Verify `T1_OPEN` entry uses exact next session open price without lookahead.

---

## Master Checklist: Subsystem 3 Crosscheck Matrix

```markdown
### Subsystem 3: Structural Levels and Indicators
- [ ] **3.1 Multi-Horizon Price Structure**
  - [ ] 52-Week Macro Range: 252-bar High/Low correctly identified (`structural_levels.py:115-119`).
  - [ ] 60D Intermediate Swings: High/Low and directional impulse detected (`structural_levels.py:121-131`).
  - [ ] Directional Fibonacci: Bullish pullback floors vs Bearish bounce resistance (`structural_levels.py:133-142`).
  - [ ] Forward Breakout Targets: Fibonacci 1.272x and 1.618x expansion targets (`structural_levels.py:154-155`).
  - [ ] Tactical Volatility Channels: ATR 14/20 and +2x/+3x forward target channels (`structural_levels.py:146-162`).
  - [ ] 1H Intraday Micro Swings: 24-bar high/low and EMA 20/50 alignment (`structural_levels.py:14-73`).
- [ ] **3.2 Indicator Dataflows & Calculations**
  - [ ] Stockstats Caching: 15-year historical window cached atomically with `.tmp` replace (`stockstats_utils.py:98-126`).
  - [ ] Point-in-Time Enforcement: Snapshot isolation in backtest mode (`stockstats_utils.py:68-80`, `interface.py:201-203`).
  - [ ] Indicator Aliases: `rsi_14` $\to$ `rsi`, `atr_14` $\to$ `atr` correctly mapped (`y_finance.py:144-153`).
  - [ ] Causal Wilder's ATR: Exponential smoothing with alpha=1/period (`stockstats_utils.py:159-176`).
  - [ ] Deterministic Market Validator: Ground-truth table generation preventing hallucinations (`market_data_validator.py:62-123`).
- [ ] **3.3 Tool Wrappers & Routing**
  - [ ] Multi-indicator query parsing: comma-separated string splitting (`technical_indicators_tools.py:28-34`).
  - [ ] Fail-closed vendor routing: Rate-limit fallbacks in live mode, strict snapshot lock in PIT mode (`interface.py:184-218`).
  - [ ] Tool node binding: Market analyst receives `get_stock_data`, `get_indicators`, `get_kronos_forecast` (`trading_graph.py:196-198`).
- [ ] **3.4 Prompt Topology & Schema Invariants**
  - [ ] Context Injection: `build_instrument_context` embeds structural summary in all agent prompts (`agent_utils.py:79-100`).
  - [ ] Asymmetry Mandate: Prompts enforce $R:R \ge 2:1$ with stop loss below confirmed support floors (`market_analyst.py:39`, `neutral_debator.py:23`).
  - [ ] Invalidation / Falling Knife Gate: Sub-200 SMA breakdown strictly enforces WNS (`trader.py:39-44`, `conservative_debator.py:34`).
  - [ ] Execution Geometry: Buy Market (T1_OPEN) for momentum breakouts vs Buy Limit (T1_LIMIT) for support accumulation (`trader.py:37-38`).
  - [ ] Pydantic Schemas: Price bound validation ($\text{SL} < \text{Entry} < \text{TP}$) and WNS condition enforcement (`schemas.py:280-305, 437-455`).
  - [ ] Static Exits: Risk engine forbids post-entry stop ratcheting or widening (`risk.py:389-395`).
```

---

## Subsystem 4: Analyst Agents, Prompt Templates & Context Injection

# Subsystem 4: Analyst Agents, Prompt Templates, Tools & Schemas Mapping

---

## 1. Executive Summary & Architectural Topology

Subsystem 4 comprises the entire analytical and decision-making agent hierarchy in `tradingagents`. It runs on a two-tier LLM engine (`quick_thinking_llm` for domain analysts, debaters, and trader; `deep_thinking_llm` for managers):

```
                                [START]
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   [Market Analyst]       [Sentiment Analyst]         [News Analyst]       [Fundamentals / Crypto]
   (Tool-loop: OHLCV,     (Pre-fetched: StockTwits,  (Tool-loop: News,    (Tool-loop: Balance Sheet,
    Indicators, Kronos)    Reddit, Bluesky, Mastodon)  Macro, Insiders)    Cashflow, Tokenomics)
         │                         │                         │                         │
         └─────────────────────────┼─────────────────────────┴─────────────────────────┘
                                   ▼
                        [Msg Clear / Sync Nodes]
                                   ▼
                         [Bull/Bear Debate]
                   (Bull Researcher ⇆ Bear Researcher)
                                   ▼
                          [Research Manager]
                      (Structured: ResearchPlan)
                                   ▼
                                [Trader]
                     (Structured: TraderProposal)
                                   ▼
                        [Risk Committee Debate]
             (Aggressive ⇆ Conservative ⇆ Neutral Debator)
                                   ▼
                        [Portfolio Manager]
                    (Structured: PortfolioDecision)
                                   ▼
                         [SignalContract & PIT]
                                   ▼
                                 [END]
```

---

## 2. Granular Module Breakdown & Checklists

### Section 4.1: Technical Market Analyst (`tradingagents/agents/analysts/market_analyst.py:1-81`)
- **Core Purpose**: Evaluates multi-horizon daily structure (Macro 52W/200 SMA $\to$ Intermediate 60D/50 SMA $\to$ Tactical 20D/20 EMA/ATR) and identifies market regimes under long-only spot equity constraints.
- **Tools Bound**:
  - `get_stock_data` (`tradingagents/agents/utils/core_stock_tools.py:14-36`)
  - `get_indicators` (`tradingagents/agents/utils/technical_indicators_tools.py:8-35`)
  - `get_kronos_forecast` (`tradingagents/agents/utils/core_stock_tools.py:39-79`) (conditionally appended if `kronos_enabled=True`).
- **Data Flow / Contract**: Ingests `state["trade_date"]`, `state["company_of_interest"]`, `state["asset_type"]`, and isolated `state["market_messages"]`. Outputs `messages` and string `market_report`.
- **Cross-Check Vector Checklist**:
  - [ ] `market_analyst.py:28` - `kronos_enabled` checked dynamically via `get_config()` at node invocation; verify sync with runtime config overrides.
  - [ ] `market_analyst.py:39` - Prompt enforces $R:R \ge 2:1$ and Fibonacci extensions $1.272x / 1.618x$; cross-check with Trader/PM schemas where $R:R \ge 1.8:1$ or $2.0:1$ warning thresholds exist.
  - [ ] `market_analyst.py:16-24` - Crypto asset mode: injects 24/7 trading note, disables circuit breaker assumptions.

---

### Section 4.2: Multi-Source Sentiment Analyst (`tradingagents/agents/analysts/sentiment_analyst.py:1-366`)
- **Core Purpose**: Aggregates multi-source retail and institutional sentiment (News, StockTwits cashtags, Reddit, Bluesky, Mastodon, Fear & Greed Index, Exa Time-Travel Search).
- **Execution Model**: Zero tool-calling in-graph. Pre-fetches all sources prior to LLM invocation and enforces structured output via `SentimentReport` (`schemas.py:593-678`) rendered via `render_sentiment_report`.
- **Data Flow / Contract**: Ingests `trade_date`, `company_of_interest`, `asset_type`. In backtest mode, strictly enforces snapshot-only data (`runtime_config["snapshot_data"]`) and fails closed for missing historical sources. Outputs `sentiment_report` and `AIMessage`.
- **Cross-Check Vector Checklist**:
  - [ ] `sentiment_analyst.py:49-50` - Lookback window hardcoded to `_seven_days_back(end_date)`; verify alignment with crypto sentiment velocity.
  - [ ] `sentiment_analyst.py:77-105` - PIT guard: In backtest mode, live network calls (`Exa`, `StockTwits`, `Reddit`, `Bluesky`, `Mastodon`) strictly disabled; returns fallback placeholders if snapshot is missing.
  - [ ] `sentiment_analyst.py:117-121` - IDX (`.JK`) ticker handling routes Exa queries to Indonesian retail platforms (Stockbit, X `#saham`).
  - [ ] `sentiment_analyst.py:615-626` - Schema field `overall_score` (0.0 to 10.0) vs `overall_band` (6 tiers: Bullish, Mildly Bullish, Neutral, Mixed, Mildly Bearish, Bearish).

---

### Section 4.3: Corporate Fundamentals Analyst (`tradingagents/agents/analysts/fundamentals_analyst.py:1-75`)
- **Core Purpose**: Analyzes multi-year quarterly/annual financial statements, corporate solvency, operating cash flows, debt maturity, and valuation metrics for equities.
- **Tools Bound**:
  - `get_fundamentals` (`tradingagents/agents/utils/fundamental_data_tools.py:8-23`)
  - `get_balance_sheet` (`tradingagents/agents/utils/fundamental_data_tools.py:34-52`)
  - `get_cashflow` (`tradingagents/agents/utils/fundamental_data_tools.py:54-72`)
  - `get_income_statement` (`tradingagents/agents/utils/fundamental_data_tools.py:74-91`)
  - `get_web_search` (`tradingagents/agents/utils/web_search_tools.py:14-46`) (live mode only, disabled in `backtest_mode`).
- **Data Flow / Contract**: Ingests `state["trade_date"]`, `state["company_of_interest"]`, `state["fundamentals_messages"]`. Injects exchange regulatory filing context via `build_exchange_filing_context` (`SEC EDGAR`, `IDX/BEI`, `TSX/SEDAR+`, `LSE/RNS`, `HKEXnews`, `TSE/TDnet`). Outputs string `fundamentals_report`.
- **Cross-Check Vector Checklist**:
  - [ ] `fundamental_data_tools.py:28-32` - Parameter inversion tolerance: handles `_normalize_freq_and_date` when LLM passes `curr_date` into `freq` positional argument.
  - [ ] `fundamentals_analyst.py:29-32` - `backtest_mode` PIT check disables `get_web_search` to prevent future corporate earnings leaks.

---

### Section 4.4: Crypto Fundamentals & Protocol Analyst (`tradingagents/agents/analysts/crypto_fundamentals_analyst.py:1-109`)
- **Core Purpose**: Specialized replacement for `fundamentals_analyst` when `asset_type == "crypto"`. Analyzes tokenomics, supply schedules (FDV vs Circulating), GitHub developer velocity, on-chain metrics, TVL, and BTC dominance.
- **Tools Bound**:
  - `get_crypto_tokenomics` (`crypto_fundamental_tools.py:49-57`)
  - `get_crypto_dev_activity` (`crypto_fundamental_tools.py:60-68`)
  - `get_crypto_network_metrics` (`crypto_fundamental_tools.py:71-82`)
  - `get_crypto_market_sentiment` (`crypto_fundamental_tools.py:85-113`)
  - `get_crypto_onchain_news` (`crypto_fundamental_tools.py:115-125`)
  - `get_web_search` (`web_search_tools.py:14-46`) (live mode only).
- **Cross-Check Vector Checklist**:
  - [ ] `crypto_fundamental_tools.py:44-47` - Point-in-time mode enforcement: `_pit_unavailable()` blocks live HTTP network queries when `is_point_in_time_mode() == True`.
  - [ ] `crypto_fundamentals_analyst.py:58-63` - Evaluation rubric enforces TVL fee capture analysis vs token emission inflation.

---

### Section 4.5: News & Macro Catalyst Analyst (`tradingagents/agents/analysts/news_analyst.py:1-82`)
- **Core Purpose**: Catalysts classification (Structural Growth, Transitory Panic, Regulatory Clearance, Fundamental Deterioration) and market pricing status assessment.
- **Tools Bound**:
  - `get_news` (`news_data_tools.py:8-25`)
  - `get_global_news` (`news_data_tools.py:26-47`)
  - `get_insider_transactions` (`news_data_tools.py:48-60`) (stocks only; excluded for crypto).
  - `get_web_search` (`web_search_tools.py:14-46`) (live mode only).
- **Cross-Check Vector Checklist**:
  - [ ] `news_analyst.py:28-29` - Dynamic tool filtering: `get_insider_transactions` suppressed for crypto.
  - [ ] `news_analyst.py:31-33` - `backtest_mode` PIT check disables live `get_web_search`.
  - [ ] `default_config.py:89-100` - Global macro news queries and lookback parameters (`global_news_lookback_days=7`, `global_news_article_limit=10`).

---

### Section 4.6: Dialectical Research Debate (`bull_researcher.py:1-81`, `bear_researcher.py:1-79`, `research_manager.py:1-68`)
- **Core Purpose**: Adversarial debate between Long Bull and Downside Bear researchers, adjudicated into a structured `ResearchPlan` by the Lead Research Manager.
- **State Schema**: `InvestDebateState` (`agent_states.py:9-20`) managing `bull_history`, `bear_history`, `history`, `current_response`, `judge_decision`, `count`.
- **Termination Logic**: `conditional_logic.py:54-64` terminates after $2 \times \text{max\_debate\_rounds}$ turns and routes to `ResearchManager`.
- **Manager Structured Output**: `ResearchPlan` (`schemas.py:131-165`) with fields `recommendation` (`PortfolioRating`: `BUY` / `WNS`), `rationale`, `strategic_actions`.
- **Cross-Check Vector Checklist**:
  - [ ] `bull_researcher.py:39-41` - Bull mandate enforces oversold bounce and limit accumulation dip targeting with $R:R \ge 2:1$.
  - [ ] `bear_researcher.py:38-40` - Bear mandate enforces sub-200 SMA / support breakdown invalidation checks.
  - [ ] `research_manager.py:36-40` - Enforces WNS invalidation with required temporal gate ("check after date X") or structural price gate ("check after touch Y").
  - [ ] `research_manager.py:46-52` - Uses `invoke_structured_or_freetext` with fallback to `render_research_plan`.

---

### Section 4.7: Senior Execution Trader (`tradingagents/agents/trader/trader.py:1-82`)
- **Core Purpose**: Translates Research Manager's plan into concrete execution parameters under Spot Long-Only constraints.
- **Output Schema**: `TraderProposal` (`schemas.py:182-305`):
  - `action`: `TraderAction` (`BUY`, `BUY_MARKET`, `BUY_LIMIT`, `WNS`). Legacy `SELL`/`HOLD` normalized to `WNS` at validator.
  - `entry_price`: Optional float (None for `BUY_MARKET`).
  - `stop_loss`: Required for BUY, strictly below entry.
  - `take_profit`: Required for BUY, strictly above entry.
  - `max_holding_days`: Integer 1 to 63 (default 20).
  - `wns_condition_type`, `wns_recheck_date`, `wns_trigger_price`: Required fields if `action == WNS`.
- **Cross-Check Vector Checklist**:
  - [ ] `trader.py:37` - Directional buy orders MUST use `Buy Market` (fill at T+1 Open, `entry_price=None`) to eliminate `NO_FILL` execution starvation.
  - [ ] `trader.py:38` - `Buy Limit` restricted to resting demand floors in multi-day consolidations within $\le 0.2x$ ATR.
  - [ ] `trader.py:41` - Capitulation bottom exception: 52W support test + RSI < 28 + Bullish Kronos executes `Buy Market` with tight stop (Risk $\le 5\%$, $R:R \ge 2.5:1$).
  - [ ] `schemas.py:280-304` - Model validator `_validate_trader_proposal` enforces stop_loss < entry < take_profit and logs warning if $R:R < 1.95:1$.

---

### Section 4.8: Risk Management Committee & Portfolio Manager (`aggressive_debator.py:1-57`, `conservative_debator.py:1-57`, `neutral_debator.py:1-58`, `portfolio_manager.py:1-235`)
- **Core Purpose**: 3-way risk debate (Alpha Desk vs Capital Preservation vs Quantitative Arbiter) synthesized into the binding capital allocation decision `PortfolioDecision`.
- **State Schema**: `RiskDebateState` (`agent_states.py:23-46`) managing 3-party conversation histories, `latest_speaker`, `judge_decision`, `count`.
- **Termination Logic**: `conditional_logic.py:65-75` terminates after $3 \times \text{max\_risk\_discuss\_rounds}$ turns and routes to `PortfolioManager`.
- **Strict Execution Hierarchy Governance**:
  - **Rule 1 Invariant (`portfolio_manager.py:101, 134-150`)**: If Trader proposed `WNS`, Portfolio Manager **CANNOT** override to `BUY`. Python logic hard-downgrades PM decision to `WNS` and clears stop/target levels.
  - **Rule 2**: If Trader proposed `BUY`, PM can accept `BUY` or veto/downgrade to `WNS`.
- **Output Schema**: `PortfolioDecision` (`schemas.py:339-535`) converted to `SignalContract` via `portfolio_decision_to_signal_contract()` (`schemas.py:833-894`).
- **Cross-Check Vector Checklist**:
  - [ ] `portfolio_manager.py:67-77` - Auto-detects and resolves `EntryMode.T1_OPEN` vs `EntryMode.T1_LIMIT` based on Trader action and entry price bounds.
  - [ ] `portfolio_manager.py:78-84` - Past memory injection (`past_context`) from `TradingMemoryLog` (`memory.py:71-100`).
  - [ ] `schemas.py:458-485` - Model validator `_use_upper_bound_for_horizon_range` parses natural language ranges (e.g. "6-12 months", "2-4 weeks") and clamps trading days $\le 252$.
  - [ ] `schemas.py:807-817` - Fail-safe auto-recovery: missing WNS date defaults to `signal_date` or current date without throwing validation error.

---

### Section 4.9: Graph Topology & Concurrency Execution Engine (`tradingagents/graph/setup.py:1-232`, `tradingagents/graph/analyst_execution.py:1-146`)
- **Core Purpose**: Dynamic assembly of LangGraph state machine, supporting customizable analyst selection, isolated tool-loop channels, parallel branch fan-out, and wall-time tracking.
- **Node Topology**:
  - Analyst nodes: `Market Analyst`, `Sentiment Analyst`, `News Analyst`, `Fundamentals Analyst` (or `Crypto Fundamentals Analyst`).
  - Tool nodes: `tools_market`, `tools_social`, `tools_news`, `tools_fundamentals`.
  - State Isolation: Each analyst operates on its own dedicated channel (`market_messages`, `social_messages`, `news_messages`, `fundamentals_messages`) via `AgentState` (`agent_states.py:56-59`).
  - Synchronization: `analyst_execution.py` coordinates branch joins into `Bull Researcher`.
- **Cross-Check Vector Checklist**:
  - [ ] `setup.py:58-69` - Asset-type dynamic factory switching: automatically replaces `create_fundamentals_analyst` with `create_crypto_fundamentals_analyst` when `asset_type == "crypto"`.
  - [ ] `setup.py:103-108` - Error containment: unhandled exceptions inside analyst nodes return typed failure strings instead of crashing graph execution.
  - [ ] `analyst_execution.py:33-45` - Wire key `"social"` mapped to display name `"Sentiment Analyst"` for backwards compatibility with saved configs.
  - [ ] `analyst_execution.py:91-131` - `AnalystWallTimeTracker` records per-analyst latency for CLI summary displays.

---

## 3. Comprehensive Cross-Check Inconsistency & Edge-Case Matrix

| Component / File | Parameter / Contract | Runtime Truth / Implementation | Potential Mismatch Risk / Invariant Guard |
| :--- | :--- | :--- | :--- |
| `market_analyst.py:28` | Kronos Forecast Tool | Registered only if `get_config().get("kronos_enabled")` is True | Ensure CLI flags (`--kronos`) properly update global config before graph compile. |
| `sentiment_analyst.py:79` | Backtest Point-in-Time | Live social APIs (`StockTwits`, `Reddit`, `Exa`) blocked in backtest | Snapshot data required in `runtime_config["snapshot_data"]`; returns typed fallback if absent. |
| `fundamentals_analyst.py:29` | Web Search Tool | `get_web_search` added only when `backtest_mode == False` | Guarantees zero forward-looking financial announcement leakage during backtests. |
| `crypto_fundamental_tools.py:54` | Crypto PIT Guard | Calls `is_point_in_time_mode()` to block live CoinGecko/Etherscan | Returns typed string notice without raising unhandled HTTP exceptions. |
| `schemas.py:53-78` | Rating Scale Enum | Canonical: `PortfolioRating.BUY`, `PortfolioRating.WNS` | Deserialization layer normalizes legacy `HOLD`, `SELL`, `OVERWEIGHT`, `UNDERWEIGHT` to canonical. |
| `schemas.py:182-305` | Trader Proposal | `TraderAction.BUY_MARKET` vs `BUY_LIMIT` | `BUY_MARKET` mandates `entry_price=None` (T+1 Open); `BUY_LIMIT` requires $SL < Entry < TP$. |
| `portfolio_manager.py:134` | Execution Hierarchy | Rule 1: PM cannot override Trader `WNS` to `BUY` | Python level hard-downgrade to `WNS` with warning log if LLM violates prompt invariant. |
| `schemas.py:458-485` | Time Horizon Units | Range strings (e.g. "6-12 months") $\to$ Integer trading days | Multipliers: Day=1, Week=5, Month=21, Year=252. Clamped between 1 and 252 days. |
| `agent_states.py:56-59` | Message Channel Isolation | Separate channels for each analyst tool loop | Prevents context window explosion and cross-talk during parallel analyst execution. |
| `structured.py:44-74` | Structured Recovery | Primary native schema $\to$ secondary JSON fence extraction | Tolerates model format degradation without pipeline crashes; falls back to free-text. |

---

## 4. Master `plan.md` Subsystem-4 Integration Checklist

- [x] **4.1 Market Analyst Prompt & Multi-Horizon Technical Structure**
  - [x] Integrate Macro (52W/200 SMA), Intermediate (60D/50 SMA), Tactical (20D/20 EMA/ATR) hierarchy.
  - [x] Connect `get_kronos_forecast` tool when `kronos_enabled=True`.
  - [x] Enforce $R:R \ge 2:1$ with Fibonacci extension targets ($1.272x / 1.618x$).
- [x] **4.2 Sentiment Analyst Multi-Source Ingestion & Zero Tool-Calling Architecture**
  - [x] Pre-fetch Yahoo News, StockTwits, Reddit, Bluesky, Mastodon, Fear & Greed Index, Exa Time-Travel.
  - [x] Enforce typed `SentimentReport` schema (`overall_band`, `overall_score`, `confidence`, `narrative`).
  - [x] Verify fail-closed snapshot fallback in backtest mode.
- [x] **4.3 Equity vs Crypto Fundamentals Analyst Dynamic Routing**
  - [x] Implement multi-jurisdiction regulatory filing context (`SEC`, `BEI/IDX`, `TSX`, `LSE`, `HKEX`, `TSE`).
  - [x] Implement `CryptoFundamentalsAnalyst` for tokenomics, dev activity, and TVL metrics.
  - [x] Guard `get_web_search` and live crypto APIs against lookahead leakage in PIT/backtest modes.
- [x] **4.4 News & Catalyst Analyst Pricing Framework**
  - [x] Classify catalysts: Structural Growth, Transitory Panic, Regulatory Clearance, Fundamental Deterioration.
  - [x] Disable `get_insider_transactions` dynamically for crypto assets.
- [x] **4.5 Dialectical Bull/Bear Debate & Research Manager**
  - [x] Adversarial debate loop with state tracking (`InvestDebateState`).
  - [x] Output structured `ResearchPlan` (`BUY` vs `WNS` with mandatory temporal or price gate).
- [x] **4.6 Senior Execution Trader Taxonomies & Bounds Validation**
  - [x] Default directional long setups to `Buy Market` (T+1 Open) to eliminate `NO_FILL`.
  - [x] Restrict `Buy Limit` to resting demand floors within $\le 0.2x$ ATR.
  - [x] Enforce strict Pydantic geometry validation ($SL < Entry < TP$, $R:R \ge 1.95:1$).
- [x] **4.7 Risk Committee Debate & Portfolio Manager Allocation**
  - [x] 3-way risk debate (`Aggressive`, `Conservative`, `Neutral`) with `RiskDebateState`.
  - [x] Enforce Rule 1 Execution Hierarchy invariant (Trader `WNS` vetoes PM `BUY`).
  - [x] Map `PortfolioDecision` into canonical `SignalContract` with parsed horizon days.
- [x] **4.8 Graph Topology, Isolated Message Channels & Concurrency**
  - [x] Independent tool-loop message state channels (`market_messages`, `social_messages`, etc.).
  - [x] Dynamic execution plan generation and latency tracking (`AnalystWallTimeTracker`).
  - [x] Fail-safe error wrapping per analyst node.

---

## Subsystem 5: Bull/Bear Researchers, Debate Synthesis & Risk Committee

# Subsystem 5 Architectural Mapping: Researchers & Risk Debators

## Overview & Subsystem Boundary
Subsystem 5 encompasses the dialectical debate teams, the synthesis managers, and the risk arbitration committee within `tradingagents`:
1. **Research Team (Dialectical Stage 1)**: `Bull Researcher` (`bull_researcher.py`), `Bear Researcher` (`bear_researcher.py`), and `Research Manager` (`research_manager.py`).
2. **Execution Gate Bridge**: `Trader` (`trader.py`) & `TraderProposal` schema.
3. **Risk Committee (Dialectical Stage 2)**: `Aggressive Analyst` (`aggressive_debator.py`), `Conservative Analyst` (`conservative_debator.py`), `Neutral Analyst` (`neutral_debator.py`), and `Portfolio Manager` / CIO (`portfolio_manager.py`).
4. **State Machine & Graph Topology**: `AgentState`, `InvestDebateState`, `RiskDebateState` (`agent_states.py`), `ConditionalLogic` (`conditional_logic.py`), `GraphSetup` (`setup.py`), and `TradingAgentsGraph` (`trading_graph.py`).
5. **Contract & Schema Layer**: `PortfolioRating`, `TraderAction`, `ResearchPlan`, `TraderProposal`, `PortfolioDecision`, `SignalContract` (`schemas.py`), and `invoke_structured_with_recovery` (`structured.py`).

---

### Section 5.1: Research Debate Layer (Bull & Bear Researchers)
- **Files & Line Ranges**:
  - `tradingagents/agents/researchers/bull_researcher.py:1-81`
  - `tradingagents/agents/researchers/bear_researcher.py:1-79`
  - `tradingagents/agents/utils/agent_states.py:8-20` (`InvestDebateState`)
- **Core Contracts & Data Structures**:
  - **Inputs**: `state["company_of_interest"]`, `state["asset_type"]`, `state["trade_date"]`, `state["market_report"]`, `state["fundamentals_report"]`, `state["news_report"]`, `state["sentiment_report"]`, `state["investment_debate_state"]`.
  - **Outputs**: Dict `{"investment_debate_state": InvestDebateState}` updating `history`, `bull_history`/`bear_history`, `current_response`, and incrementing `count` by 1.
  - **State Contract**:
    ```python
    class InvestDebateState(TypedDict):
        bull_history: str
        bear_history: str
        history: str
        current_response: str
        judge_decision: str
        count: int
    ```
- **Architectural Checkpoints & Crosscheck Vectors**:
  - [ ] **Prompt Timeframe Uniformity**: Verify `bull_researcher.py:37-42` and `bear_researcher.py:37-40` adhere strictly to Institutional Multi-Horizon Daily Structure (Macro 52W $\to$ Intermediate 60D $\to$ Tactical 20D) without legacy "1H micro" or intraday framing leaks.
  - [ ] **Mandate Consistency**: Verify Bull Researcher strictly adheres to Spot Equity Long-Only accumulation (`bull_researcher.py:38`) and prohibits shorting or speculative leverage.
  - [ ] **Turn-taking & Rebuttal Section**: Verify dynamic formatting of `bear_rebuttal_section` (`bull_researcher.py:26-30`) and `bull_rebuttal_section` (`bear_researcher.py:26-30`) when `current_response` is empty (Opening Round) vs populated (Counter-argument).
  - [ ] **Speaker Prefix Consistency**: Ensure Bull outputs prepend `"Bull Analyst: "` (`bull_researcher.py:68`) and Bear outputs prepend `"Bear Analyst: "` (`bear_researcher.py:66`) to maintain parser compatibility with `conditional_logic.py:61` (`startswith("Bull")`).
  - [ ] **Internationalization Guard**: Verify `get_language_instruction()` (`agent_utils.py:63-76`) is appended to both prompts so non-English runs produce fully localized argument transcripts.

---

### Section 5.2: Research Synthesis & Strategic Planning (Research Manager)
- **Files & Line Ranges**:
  - `tradingagents/agents/managers/research_manager.py:1-68`
  - `tradingagents/agents/schemas.py:131-175` (`ResearchPlan`, `render_research_plan`)
  - `tradingagents/agents/utils/structured.py:76-118` (`bind_structured`, `invoke_structured_or_freetext`)
- **Core Contracts & Data Structures**:
  - **Inputs**: `state["company_of_interest"]`, `state["asset_type"]`, `state["trade_date"]`, `state["investment_debate_state"]["history"]`.
  - **Outputs**: Dict `{"investment_debate_state": ..., "investment_plan": str}`.
  - **Pydantic Model (`ResearchPlan`)**:
    - `recommendation`: `PortfolioRating` (`BUY` or `WNS`). Normalizes legacy `Overweight`, `Hold`, `Underweight`, `Sell` (`schemas.py:160-164`).
    - `rationale`: `str` (Conversational summary of debate winner).
    - `strategic_actions`: `str` (Tactical instructions & sizing guidelines).
- **Architectural Checkpoints & Crosscheck Vectors**:
  - [ ] **Binary Rating Scale Guarantee**: Ensure `ResearchPlan.recommendation` can only be `Buy` or `WNS` (`schemas.py:53-86, 140-145`). Legacy rating words must be mapped deterministically (`_LEGACY_RATING_TO_CANONICAL`).
  - [ ] **Separation of Concerns (Strategic vs Geometry)**: Research Manager must only establish strategic consensus and mandate rating; exact numerical limit prices, stop-loss, and take-profit geometry are strictly deferred to Trader (`docs/agents/debate-team.md:30-31`).
  - [ ] **Fallback Robustness**: Verify `invoke_structured_or_freetext` (`structured.py:93-118`) captures provider JSON parse exceptions and falls back gracefully to plain LLM text generation rendered via `render_research_plan`.
  - [ ] **State Synchronization**: Ensure `judge_decision` and `current_response` in `investment_debate_state` are synchronized with the generated `investment_plan` markdown string (`research_manager.py:54-66`).

---

### Section 5.3: Execution Proposal Bridge (Senior Execution Trader)
- **Files & Line Ranges**:
  - `tradingagents/agents/trader/trader.py:1-82`
  - `tradingagents/agents/schemas.py:109-125, 182-332` (`TraderAction`, `TraderProposal`, `render_trader_proposal`)
- **Core Contracts & Data Structures**:
  - **Inputs**: `state["company_of_interest"]`, `state["asset_type"]`, `state["trade_date"]`, `state["investment_plan"]`.
  - **Outputs**: Dict `{"messages": [AIMessage], "trader_investment_plan": str, "trader_proposal": Optional[TraderProposal], "sender": "Trader"}`.
  - **Pydantic Model (`TraderProposal`)**:
    - `action`: `TraderAction` (`Buy`, `Buy Market`, `Buy Limit`, `WNS`).
    - `reasoning`: `str`.
    - `entry_price`: `Optional[float]`.
    - `stop_loss`: `Optional[float]`.
    - `take_profit`: `Optional[float]`.
    - `position_sizing`: `Optional[str]`.
    - `max_holding_days`: `int` (1 to 63, default 20).
    - `wns_condition_type`: `Optional[WNSConditionType]` (`TIME_GATE`, `PRICE_TOUCH`, `BOTH`).
    - `wns_recheck_date`: `Optional[str]` (ISO `YYYY-MM-DD`).
    - `wns_trigger_price`: `Optional[float]`.
- **Architectural Checkpoints & Crosscheck Vectors**:
  - [ ] **Price Geometry Validation Invariants**:
    - For `BUY`: `stop_loss < entry_price < take_profit` when `entry_price` is specified (`schemas.py:292-298`).
    - For `BUY`: `stop_loss < take_profit` mandatory (`schemas.py:290-291`).
    - For `WNS`: At least one of `wns_recheck_date` or `wns_trigger_price` MUST be non-null (`schemas.py:281-286`).
  - [ ] **Unit & Price Cleaning**: Coercion of stringified prices (e.g. `"$150.50"` or `"150.50%"`) into positive finite floats (`schemas.py:250-273`).
  - [ ] **Expectancy Guardrail**: Warning logged if $R:R = \frac{\text{take\_profit} - \text{entry}}{\text{entry} - \text{stop\_loss}} < 1.95$ (`schemas.py:299-303`).
  - [ ] **Execution Mode Alignment**: Default directional momentum entries use `Buy Market` (`trader.py:37`); `Buy Limit` reserved for resting consolidation accumulation within $0.2\times\text{ATR}$ (`trader.py:38`).

---

### Section 5.4: Risk Management Committee (Aggressive, Conservative, & Neutral Debators)
- **Files & Line Ranges**:
  - `tradingagents/agents/risk_mgmt/aggressive_debator.py:1-57`
  - `tradingagents/agents/risk_mgmt/conservative_debator.py:1-57`
  - `tradingagents/agents/risk_mgmt/neutral_debator.py:1-58`
  - `tradingagents/agents/utils/agent_states.py:23-46` (`RiskDebateState`)
- **Core Contracts & Data Structures**:
  - **Inputs**: `state["company_of_interest"]`, `state["asset_type"]`, `state["trade_date"]`, `state["trader_investment_plan"]`, `state["risk_debate_state"]`.
  - **Outputs**: Dict `{"risk_debate_state": RiskDebateState}` updating transcripts and tracking `latest_speaker`.
  - **State Contract**:
    ```python
    class RiskDebateState(TypedDict):
        aggressive_history: str
        conservative_history: str
        neutral_history: str
        history: str
        latest_speaker: str  # "Aggressive", "Conservative", "Neutral", "Judge"
        current_aggressive_response: str
        current_conservative_response: str
        current_neutral_response: str
        judge_decision: str
        count: int
    ```
- **Architectural Checkpoints & Crosscheck Vectors**:
  - [ ] **Committee Role Mandates**:
    - `Conservative Analyst`: CRO Capital Preservation desk. Mandates Falling Knife / sub-200 SMA breakdown vetoes (`conservative_debator.py:22-37`).
    - `Neutral Analyst`: Quantitative Risk Arbiter. Evaluates multi-horizon daily structure, Fibonacci extensions (1.272x, 1.618x), and mathematical expectancy $R:R \ge 2:1$ (`neutral_debator.py:22-38`).
    - `Aggressive Analyst`: Alpha Desk. Defends high-asymmetry momentum continuation and oversold capitulation turns ($RSI < 25$) (`aggressive_debator.py:22-37`).
  - [ ] **Strict Execution Hierarchy Prompt Invariant**: All 3 risk debator prompts explicitly mandate respecting Trader WNS: if Trader proposed WNS, debators must not force Long entries (`conservative_debator.py:35`, `neutral_debator.py:37`, `aggressive_debator.py:36`).
  - [ ] **State Field Integrity**: Ensure each debator updates its respective `*_history` and `current_*_response` while preserving other debators' existing state strings (`aggressive_debator.py:44-53`, `conservative_debator.py:44-53`, `neutral_debator.py:44-53`).
  - [ ] **Speaker State Transition**: Ensure `latest_speaker` is set to `"Aggressive"`, `"Conservative"`, or `"Neutral"` so `conditional_logic.py:71-75` routes to the next committee member.

---

### Section 5.5: Final Capital Allocation & Signal Contract Engine (Portfolio Manager / CIO)
- **Files & Line Ranges**:
  - `tradingagents/agents/managers/portfolio_manager.py:1-235`
  - `tradingagents/agents/schemas.py:339-571` (`PortfolioDecision`, `render_pm_decision`)
  - `tradingagents/agents/schemas.py:701-894` (`SignalContract`, `portfolio_decision_to_signal_contract`)
  - `tradingagents/agents/utils/structured.py:33-74` (`invoke_structured_with_recovery`)
- **Core Contracts & Data Structures**:
  - **Inputs**: `state["company_of_interest"]`, `state["asset_type"]`, `state["trade_date"]`, `state["investment_plan"]`, `state["trader_investment_plan"]`, `state["trader_proposal"]`, `state["risk_debate_state"]`, `state["past_context"]`.
  - **Outputs**: Dict `{"risk_debate_state": ..., "final_trade_decision": str, "signal_contract": Optional[SignalContract]}`.
  - **Signal Contract Schema (`SignalContract`)**:
    - `ticker`: `str`, `signal_date`: `str` (`YYYY-MM-DD`).
    - `rating`: `PortfolioRating` (`BUY` or `WNS`), `action`: `Literal["BUY", "WNS"]`.
    - `entry_mode`: `Optional[EntryMode]` (`T1_OPEN`, `T1_LIMIT`, `ASSUMED_AI_ENTRY`).
    - `planned_entry_price`: `Optional[float]`.
    - `take_profit`: `Optional[float]`, `stop_loss`: `Optional[float]`.
    - `time_horizon_days`: `int` (1 to 252), `max_holding_days`: `int` (1 to 63).
    - `confidence`: `float` (0.0 to 1.0).
    - `wns_condition_type`, `wns_recheck_date`, `wns_trigger_price`.
- **Architectural Checkpoints & Crosscheck Vectors**:
  - [ ] **Rule 1 Invariant Enforcement (Strict Hierarchy Code Judo)**:
    - If Trader proposed WNS, PM **CANNOT** issue BUY. If LLM outputs BUY, `portfolio_manager.py:134-166` forces an automatic programmatic downgrade to `PortfolioRating.WNS`, clears price levels, and inherits Trader's WNS parameters.
  - [ ] **Structured Output Multi-Tier Recovery**:
    - Primary: Native `structured_llm.invoke()` (`portfolio_manager.py:122`).
    - Secondary: Plain LLM call + markdown JSON code fence regex extraction (`structured.py:33-74`).
    - Tertiary: Deterministic failsafe fallback `SignalContract(action="WNS", ...)` (`portfolio_manager.py:189-214`).
  - [ ] **Date & Horizon Normalization**:
    - Deterministic review date generation (`+7d` for BUY, `+21d` for WNS, `+14d` for fallback) if omitted by LLM (`portfolio_manager.py:175-180, 191, 204`).
    - Conversion of textual ranges (e.g. `"6-12 months"`) to clamped trading days upper bound ($252$) (`schemas.py:457-484`).
  - [ ] **T1_OPEN vs T1_LIMIT Entry Mode Resolution**:
    - `BUY_MARKET` / directional breakout maps to `EntryMode.T1_OPEN` with `planned_entry_price = None` (`portfolio_manager.py:68-74, 171-173, schemas.py:863-871`).
    - `BUY_LIMIT` on support pullback maps to `EntryMode.T1_LIMIT` with validated `planned_entry_price` bounded by `stop_loss < planned_entry < take_profit` (`schemas.py:866-868`).

---

### Section 5.6: Graph Flow, State Propagation, & Conditional Routing
- **Files & Line Ranges**:
  - `tradingagents/graph/conditional_logic.py:54-75`
  - `tradingagents/graph/setup.py:77-88, 154-230`
  - `tradingagents/graph/propagation.py:28-82`
  - `tradingagents/graph/trading_graph.py:119-146, 403-538`
- **Core Contracts & Data Structures**:
  - **Research Debate Loop**:
    - Node Sequence: `Bull Researcher` $\leftrightarrow$ `Bear Researcher` until `count >= 2 * max_debate_rounds`, then routes to `Research Manager` (`conditional_logic.py:54-64`).
  - **Risk Committee Loop**:
    - Node Sequence: `Aggressive Analyst` $\to$ `Conservative Analyst` $\to$ `Neutral Analyst` $\to$ `Aggressive Analyst` until `count >= 3 * max_risk_discuss_rounds`, then routes to `Portfolio Manager` (`conditional_logic.py:65-75`).
  - **Linear Pipeline Edges**:
    - `Analyst Barrier Join` $\to$ `Bull Researcher` (`setup.py:176`).
    - `Research Manager` $\to$ `Trader` (`setup.py:197`).
    - `Trader` $\to$ `Aggressive Analyst` (`setup.py:198`).
    - `Portfolio Manager` $\to$ `END` (`setup.py:230`).
- **Architectural Checkpoints & Crosscheck Vectors**:
  - [ ] **Round Counting Arithmetic**:
    - Research debate threshold is $2 \times \text{max\_debate\_rounds}$ (2 turns per round) (`conditional_logic.py:58`).
    - Risk committee threshold is $3 \times \text{max\_risk\_discuss\_rounds}$ (3 turns per round) (`conditional_logic.py:68`).
    - Verify counter resets or increments in initial state (`InvestDebateState.count = 0`, `RiskDebateState.count = 0`) (`propagation.py:58, 72`).
  - [ ] **Config vs CLI Flag Discrepancies**:
    - Verify `max_debate_rounds` and `max_risk_discuss_rounds` properly resolve from `DEFAULT_CONFIG` (`default_config.py:82-83`), environment variables `TRADINGAGENTS_MAX_DEBATE_ROUNDS` / `TRADINGAGENTS_MAX_RISK_ROUNDS` (`default_config.py:16-17`), or runtime CLI flags without silent overrides.
  - [ ] **Recursion Limit Guard**: `max_recur_limit` configured to 100 (`default_config.py:84`, `propagation.py:18-20`) to prevent LangGraph infinite loop crashes during expanded debate rounds.

---

### Section 5.7: Quantitative Structural Context & Indicator Dataflows
- **Files & Line Ranges**:
  - `tradingagents/agents/utils/agent_utils.py:79-100` (`build_instrument_context`)
  - `tradingagents/dataflows/structural_levels.py:76-250` (`compute_structural_levels`, `get_market_structural_summary`)
- **Core Contracts & Data Structures**:
  - **Inputs**: Symbol, `trade_date`, daily OHLCV dataframe (`load_ohlcv`).
  - **Outputs**: Formatted structural level markdown block injected into researcher, debator, trader, and manager prompts.
  - **Calculated Horizons**:
    - **Macro Horizon**: 52-Week High/Low, multi-month base (`structural_levels.py:114-119`).
    - **Intermediate Horizon**: 60D Swing High/Low, Fibonacci 50% & 61.8% pullback floors, Fibonacci Extensions (1.272x, 1.618x) (`structural_levels.py:121-155`).
    - **Tactical Horizon**: 20D Swing High/Low, ATR(14), ATR(20), ATR% of price, Forward ATR channels (+2x, +3x ATR), 0.5x ATR buffer (`structural_levels.py:146-162, 178-184`).
    - **Intraday Supplement (1H Micro)**: 24-bar swing high/low, EMA 20/50, trend alignment (conditional when 1H data available) (`structural_levels.py:14-74, 186-190, 241-249`).
- **Architectural Checkpoints & Crosscheck Vectors**:
  - [ ] **Fibonacci Directional Awareness**: Ensure Fibonacci retracement calculation checks whether $pos_{L60} < pos_{H60}$ (bullish impulse) or $pos_{L60} > pos_{H60}$ (bearish leg) to place demand support vs bounce resistance levels accurately (`structural_levels.py:133-142`).
  - [ ] **Point-In-Time (PIT) Anti-Leakage Guard**: Verify `compute_structural_levels` filters history strictly with `work_df["date_str"] <= str(trade_date)[:10]` (`structural_levels.py:97`), guaranteeing zero lookahead into future bars during backtesting.
  - [ ] **Unit & Scale Robustness**: ATR% expressed in percentage format ($[0, 100]$) while prices and absolute ATR are kept in native instrument quote currency (`structural_levels.py:151, 178-184`).
  - [ ] **Crypto vs Stock Context**: Ensure `build_instrument_context` checks `asset_type == "crypto"` to tailor phrasing ("asset" vs "company") and suppresses traditional fundamental assumptions (`agent_utils.py:81-86`).

---

### Section 5.8: CLI Presentation, Message Buffering, & Report Serialization
- **Files & Line Ranges**:
  - `cli/progress_contract.py:1-46` (`FIXED_AGENTS`, `ALL_TEAMS`, `REPORT_SECTIONS`)
  - `cli/report_io.py:13-122` (`save_report_to_disk`)
  - `cli/display.py:30-100` (`update_display`)
  - `tradingagents/graph/trading_graph.py:491-536` (`_log_state`)
- **Core Contracts & Data Structures**:
  - **Disk Directory Hierarchy**:
    - `1_analysts/` (`market.md`, `sentiment.md`, `news.md`, `fundamentals.md`)
    - `2_research/` (`bull.md`, `bear.md`, `manager.md`)
    - `3_trading/` (`trader.md`)
    - `4_risk/` (`aggressive.md`, `conservative.md`, `neutral.md`)
    - `5_portfolio/` (`decision.md`)
    - `signal.json`, `config.json`, `complete_report.md`
  - **JSON Full State Log**: `results_dir/<ticker>/TradingAgentsStrategy_logs/full_states_log_<date>.json`
- **Architectural Checkpoints & Crosscheck Vectors**:
  - [ ] **Agent Label Alignment**:
    - `ALL_TEAMS["Research"]`: `["Bull Researcher", "Bear Researcher", "Research Manager"]` (`progress_contract.py:32`).
    - `ALL_TEAMS["Risk"]`: `["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"]` (`progress_contract.py:34`).
    - `short_agent_label` correctly strips suffixes for clean terminal TUI columns (`progress_contract.py:44-46`).
  - [ ] **Atomic Serialization Safety**:
    - Verify `safe_ticker_component(self.ticker)` is used when building file paths to prevent directory traversal vulnerabilities on dot-suffixed tickers (e.g. `DEWA.JK`, `BINA.JK`, `BRK.B`) (`trading_graph.py:529-530`).
    - `signal.json` is exported using `SignalContract.model_dump(mode="json")` ensuring ISO string date formatting and null safety (`report_io.py:98-108`).
  - [ ] **Config Sanitization**: API keys and secrets stripped before writing `config.json` via `sanitize_config()` (`report_io.py:111-116`).

---

## Master Crosscheck Matrix for Subsystem 5

| Section / Dimension | Timeframe Assumptions | Rating / Action Enums | Order Geometry & Types | Units & Scales | Fail-Closed & Fallback Guards |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5.1 Bull & Bear Researchers** | Multi-Horizon Daily (Macro 52W, Inter 60D, Tact 20D) | Spot Long accumulation vs Invalidation / Downside | Support test / Double bottom; hard invalidation floors | Native currency quote price levels | Rebuttal defaults to Opening Round if `current_response` empty |
| **5.2 Research Manager** | Evaluates intermediate catalysts & multi-day setups | Binary `BUY` vs `WNS` (normalizes legacy aliases) | High-level roadmap only; defers pricing geometry to Trader | Percentage sizing hints | `bind_structured` + `invoke_structured_or_freetext` retry |
| **5.3 Senior Execution Trader** | Max holding: 1–63 trading days (default 20D) | `Buy`, `Buy Market`, `Buy Limit`, `WNS` | `stop_loss < entry < take_profit`; $R:R \ge 2:1$ check | Coerces strings & percents to finite floats | WNS requires `wns_recheck_date` or `wns_trigger_price` |
| **5.4 Risk Debators (Agg/Cons/Neut)** | Multi-Horizon Daily; Falling Knife regime detection | Defends or challenges proposed action | Validates Fib 1.272x/1.618x, ATR channels, Stop Loss | $R:R \ge 2.0:1$ mathematical expectancy | Enforces Trader WNS invariant; prevents Long in freefall |
| **5.5 Portfolio Manager (CIO)** | 1–252 trading days horizon; 1–63 max holding | Strictly binary `BUY` vs `WNS` | Validates `T1_OPEN` (planned=None) vs `T1_LIMIT` (planned=val) | Clamps confidence $[0.0, 1.0]$ | Programmatic downgrade if Trader proposed WNS; failsafe WNS contract |
| **5.6 Graph Topology & Flow** | $2 \times$ debate rounds, $3 \times$ risk rounds | N/A (State routing) | N/A | Turn counters | `max_recur_limit=100` prevents infinite loop traps |
| **5.7 Structural Levels Dataflow** | 52W (Macro), 60D (Inter), 20D (Tact), 1H (Micro) | N/A (Quantitative indicators) | Fib 50%/61.8% floors, Fib 1.272x/1.618x, ATR +2x/+3x | ATR% $[0, 100]$, ATR absolute in price units | Strict PIT filtering (`ts <= trade_date`); empty dataframe safety |
| **5.8 CLI & Serialization** | Date-indexed logs & reports | Displays and serializes canonical ratings | Serializes typed `SignalContract` to `signal.json` | JSON native format | `safe_ticker_component` path sanitization; secret scrub |

---

## Verification & Test Traceability
- `tests/test_quant_expectancy_code_judo.py:34-197`: Validates Rule 1 Invariant (PM cannot override Trader WNS to BUY; auto-downgraded to WNS with parameters).
- `tests/test_quant_expectancy_code_judo.py:199-243`: Validates Rule 2 Breakout Targets (Fib extensions 1.272x, 1.618x and ATR channels).
- `tests/test_quant_expectancy_code_judo.py:245-316`: Validates Rule 3 Entry Mode Flexibility (`T1_OPEN` market entry vs `T1_LIMIT` tight limit).
- `tests/test_quant_expectancy_code_judo.py:318-347`: Validates Rule 4 Falling Knife WNS Discipline (sub-200 SMA breakdowns evaluate to `NO_ORDER` with 0% loss).
- `tests/test_quant_expectancy_code_judo.py:349-433`: Validates prompt alignment for Multi-Horizon Daily Structure across market analyst, neutral debator, and bull researcher.
- `tests/test_graph_topology.py:1-135`: Validates analyst fan-out barrier join into Bull Researcher and graph routing stability.

---

## Subsystem 6: Execution Trader, Portfolio Manager & Pydantic Schemas

# Subsystem 6: Trader, Portfolio Manager & Schemas
## Architectural Mapping & Master Crosscheck Matrix

---

### SECTION 1: Core Pydantic Schemas & Shared Types
- **Exact File Paths & Lines**: `tradingagents/agents/schemas.py:1-125`, `tradingagents/agents/schemas.py:698-894`
- **Purpose**: Canonical typed contracts across the entire agent lifecycle and backtest evaluations.
- **Contracts, Data Structures & Models**:
  - `PortfolioRating(str, Enum)` (`tradingagents/agents/schemas.py:53-66`): Canonical `BUY = "Buy"`, `WNS = "WNS"`. Legacy input aliases: `OVERWEIGHT = "Buy"`, `HOLD = "WNS"`, `UNDERWEIGHT = "WNS"`, `SELL = "WNS"`.
  - `_LEGACY_RATING_TO_CANONICAL` & `normalize_portfolio_rating` (`tradingagents/agents/schemas.py:68-87`): Normalizes raw input strings.
  - `EntryMode(str, Enum)` (`tradingagents/agents/schemas.py:89-95`): `ASSUMED_AI_ENTRY`, `T1_OPEN`, `T1_LIMIT`.
  - `EvaluationOutcome(str, Enum)` (`tradingagents/agents/schemas.py:97-107`): `HIT_TAKE_PROFIT`, `HIT_STOP_LOSS`, `HIT_TIME_STOP`, `EXPIRED`, `NO_ORDER`, `INSUFFICIENT_DATA`, `NO_FILL`.
  - `TraderAction(str, Enum)` (`tradingagents/agents/schemas.py:109-119`): `BUY = "Buy"`, `BUY_MARKET = "Buy Market"`, `BUY_LIMIT = "Buy Limit"`, `WNS = "WNS"`, `HOLD = "WNS"`, `SELL = "WNS"`.
  - `WNSConditionType(str, Enum)` (`tradingagents/agents/schemas.py:120-124`): `TIME_GATE`, `PRICE_TOUCH`, `BOTH`.
  - `SignalContract(BaseModel)` (`tradingagents/agents/schemas.py:701-831`): Single-shot typed execution signal with validation rules:
    * `ticker: str`, `signal_date: str (YYYY-MM-DD)`.
    * `rating: PortfolioRating`, `action: Literal["BUY", "WNS"]`.
    * `planned_entry_price: Optional[float]`, `take_profit: Optional[float]`, `stop_loss: Optional[float]`.
    * `time_horizon_days: int (1..252)`, `confidence: float (0.0..1.0)`, `max_holding_days: Optional[int] (1..63)`.
    * WNS terms: `wns_condition_type`, `wns_recheck_date`, `wns_trigger_price`.
  - `portfolio_decision_to_signal_contract` (`tradingagents/agents/schemas.py:833-894`): Maps `PortfolioDecision` into `SignalContract`.
- **Potential Mismatch / Inconsistency Vectors**:
  - **Rating / Enum Inconsistency**: Legacy code in `tradingagents/backtesting/decision_schema.py:47-55` defines `Rating(BUY, SELL, HOLD, WNS, UNDERWEIGHT, OVERWEIGHT, INVALID)` and `Action(BUY, SELL, OPEN_SHORT, COVER_SHORT, ADD, REDUCE, HOLD, SELL_ALL, INVALID)`. `portfolio_decision_to_signal_contract` enforces strict BUY/WNS mapping (`tradingagents/agents/schemas.py:840`).
  - **Geometry & Price Bounds**: `SignalContract._validate_entry_contract` (`tradingagents/agents/schemas.py:783-831`) strictly requires `stop_loss < planned_entry_price < take_profit` for limit orders and `stop_loss < take_profit` for market orders. If `planned_entry_price` is omitted on BUY, defaults to `EntryMode.T1_OPEN`.
  - **Holding Horizon Clamping**: `PortfolioDecision.time_horizon_days` can be up to 252 (`tradingagents/agents/schemas.py:382`), but `SignalContract.max_holding_days` is bounded `1 <= max_holding_days <= 63` (`tradingagents/agents/schemas.py:740`). `portfolio_decision_to_signal_contract` (`tradingagents/agents/schemas.py:890`) clamps via `min(63, decision.time_horizon_days)`.
  - **Timezone Awareness Requirement**: `signal_timestamp` and `reference_price_timestamp` must be timezone-aware ISO strings (`tradingagents/agents/schemas.py:791-796`).
  - **Auto-Recovery Fallback**: For WNS without dates or prices, auto-recovers `wns_recheck_date` to `signal_date` or today (`tradingagents/agents/schemas.py:810`).

```markdown
- [ ] Crosscheck `PortfolioRating` enum aliases vs canonical `Buy`/`WNS` outputs (`tradingagents/agents/schemas.py:53-66`).
- [ ] Verify `SignalContract` price ordering validation (`stop_loss < planned_entry < take_profit`) across all limit entries (`tradingagents/agents/schemas.py:783-831`).
- [ ] Ensure `portfolio_decision_to_signal_contract` clamps `max_holding_days` to `min(63, time_horizon_days)` preventing `ValidationError` (`tradingagents/agents/schemas.py:890`).
- [ ] Verify `SignalContract` timezone-aware timestamp validation on `signal_timestamp` and `reference_price_timestamp` (`tradingagents/agents/schemas.py:791-796`).
```

---

### SECTION 2: Research Manager Structured Schemas & Node
- **Exact File Paths & Lines**: `tradingagents/agents/schemas.py:127-176`, `tradingagents/agents/managers/research_manager.py:1-68`
- **Purpose**: Synthesizes bull/bear debate into a structured investment plan for the execution trader.
- **Contracts, Data Structures & Models**:
  - `ResearchPlan(BaseModel)` (`tradingagents/agents/schemas.py:131-165`):
    * `recommendation: PortfolioRating` (normalized via `normalize_portfolio_rating`).
    * `rationale: str` (conversational summary of bull/bear debate).
    * `strategic_actions: str` (concrete execution guidance for Trader).
  - `render_research_plan(plan: ResearchPlan) -> str` (`tradingagents/agents/schemas.py:166-174`): Formats `**Recommendation**`, `**Rationale**`, `**Strategic Actions**`.
  - `create_research_manager(llm)` (`tradingagents/agents/managers/research_manager.py:16-68`):
    * System Prompt (`tradingagents/agents/managers/research_manager.py:31-45`): Directs Lead Research Manager under Spot Long-Only (BUY vs WNS) mandate; mandates temporal gate ("check after date X") or structural price gate ("check again after touch price level Y") on WNS.
    * Fallback handling: Uses `invoke_structured_or_freetext` (`tradingagents/agents/managers/research_manager.py:46-52`).
- **Potential Mismatch / Inconsistency Vectors**:
  - **Mandate Consistency**: Research Manager is instructed in BUY vs WNS mandate (`tradingagents/agents/managers/research_manager.py:35-40`). Legacy prompt text referencing shorting or trimming must remain absent.
  - **Language Instructions**: Prompt appends `get_language_instruction()` (`tradingagents/agents/managers/research_manager.py:44`).
  - **State Key Integrity**: Returns updated `investment_debate_state` with `judge_decision`, `current_response`, and `investment_plan` (`tradingagents/agents/managers/research_manager.py:54-66`).

```markdown
- [ ] Verify `ResearchPlan` normalization maps legacy `Overweight` to `Buy` and `Hold`/`Sell`/`Underweight` to `WNS` (`tradingagents/agents/schemas.py:160-164`).
- [ ] Check `create_research_manager` system prompt for spot long-only BUY/WNS constraints and temporal/price gates (`tradingagents/agents/managers/research_manager.py:31-45`).
- [ ] Verify `render_research_plan` markdown format matches headers expected downstream by Trader and Markdown parser (`tradingagents/agents/schemas.py:166-174`).
```

---

### SECTION 3: Execution Trader Node & Proposal Schema
- **Exact File Paths & Lines**: `tradingagents/agents/schemas.py:178-333`, `tradingagents/agents/trader/trader.py:1-82`
- **Purpose**: Translates `ResearchPlan` and analyst reports into a concrete transaction proposal with execution levels.
- **Contracts, Data Structures & Models**:
  - `TraderProposal(BaseModel)` (`tradingagents/agents/schemas.py:182-306`):
    * `action: TraderAction` (BUY, BUY_MARKET, BUY_LIMIT, WNS).
    * `reasoning: str`, `entry_price: Optional[float]`, `stop_loss: Optional[float]`, `take_profit: Optional[float]`.
    * `position_sizing: Optional[str]`, `max_holding_days: int = 20` (1..63).
    * `wns_condition_type: Optional[WNSConditionType]`, `wns_recheck_date: Optional[str]`, `wns_trigger_price: Optional[float]`.
    * Coercion & Validation: `_coerce_positive_price` (`tradingagents/agents/schemas.py:250-272`) handles string cleaning, % to fraction, strips currency symbols.
    * Model validation: BUY requires `stop_loss < entry_price < take_profit` (when `entry_price` is given) and warnings on R:R < 1.95:1 (`tradingagents/agents/schemas.py:280-305`). WNS requires `wns_recheck_date` or `wns_trigger_price`.
  - `render_trader_proposal(proposal: TraderProposal) -> str` (`tradingagents/agents/schemas.py:307-332`): Generates structured markdown with `FINAL TRANSACTION PROPOSAL: **BUY**` or `**WNS**`.
  - `create_trader(llm)` (`tradingagents/agents/trader/trader.py:20-82`):
    * System Prompt (`tradingagents/agents/trader/trader.py:33-49`): Enforces Spot Long-Only execution taxonomy:
      1. `Buy Market` (Default directional entry for momentum breakouts, MA reclaim, oversold inflection; T+1 Open, `entry_price=None`, Fib Extensions 1.272x/1.618x, 60D High, 52W High, +2x/+3x ATR, R:R >= 2:1).
      2. `Buy Limit` (Resting Demand Floor Accumulation only, within 0.2x ATR, Stop Loss <= 5% risk, R:R >= 2:1).
      3. `WNS` (Breakdowns below support, RSI > 35 falling knives without base, overbought ATH RSI > 75 where R:R < 1.5:1). Capitulation reversals (52W support + RSI < 28 + Bullish Kronos) execute `Buy Market` with SL <= 5%, R:R >= 2.5:1.
    * Output: `messages`, `trader_investment_plan`, `trader_proposal` (`tradingagents/agents/trader/trader.py:75-80`).
- **Potential Mismatch / Inconsistency Vectors**:
  - **Market vs Limit Price Representation**: `Buy Market` has `entry_price=None` (filled at T+1 Open), whereas `Buy Limit` has numeric `entry_price`.
  - **Unit & String Coercion**: `_coerce_positive_price` parses string percentages (e.g. `"5%"` $\to$ `0.05`), but price targets like `"$189.50"` strip `$` correctly (`tradingagents/agents/schemas.py:262-272`).
  - **Trader WNS Guarantee**: `render_trader_proposal` guarantees stable final line `FINAL TRANSACTION PROPOSAL: **WNS**` or `**BUY**` for regex consumers (`tradingagents/agents/schemas.py:328-330`).

```markdown
- [ ] Check `TraderProposal` model validator for bounds checking (`stop_loss < entry < take_profit`) and WNS condition requirement (`tradingagents/agents/schemas.py:280-305`).
- [ ] Verify `_coerce_positive_price` regex cleans currency symbols (`$`, `Rp`, `,`) and converts string percentages (`tradingagents/agents/schemas.py:258-272`).
- [ ] Verify `create_trader` taxonomy prompt aligns with Fibonacci extensions (1.272x, 1.618x) and ATR channel targets (`tradingagents/agents/trader/trader.py:36-46`).
- [ ] Ensure `trader_node` sets both string `trader_investment_plan` and typed `trader_proposal` in agent state (`tradingagents/agents/trader/trader.py:75-80`).
```

---

### SECTION 4: Portfolio Manager Node & Decision Governance
- **Exact File Paths & Lines**: `tradingagents/agents/schemas.py:335-572`, `tradingagents/agents/managers/portfolio_manager.py:1-235`
- **Purpose**: Synthesizes risk debate, trader proposal, and research plan into final capital allocation decision (`PortfolioDecision` and `SignalContract`).
- **Contracts, Data Structures & Models**:
  - `PortfolioDecision(BaseModel)` (`tradingagents/agents/schemas.py:339-535`):
    * `rating: PortfolioRating`, `executive_summary: str`, `investment_thesis: str`.
    * `stop_loss: Optional[float]`, `take_profit: Optional[float]`, `price_target: Optional[float]`, `planned_entry_price: Optional[float]`.
    * `entry_mode: Optional[EntryMode]`, `time_horizon_days: int (1..252)`, `confidence: float (0.0..1.0)`.
    * `max_holding_days: Optional[int] (1..63)`, `next_review_date: Optional[str]`.
    * WNS fields: `wns_condition_type`, `wns_recheck_date`, `wns_trigger_price`.
    * Validators: `_use_upper_bound_for_horizon_range` (`tradingagents/agents/schemas.py:456-485`) parses strings like `"6-12 months"` $\to$ `252` days; Indonesian units (`hari`, `minggu`, `bulan`, `tahun`) supported.
  - `render_pm_decision(decision: PortfolioDecision) -> str` (`tradingagents/agents/schemas.py:536-571`): Renders markdown headers `**Rating**`, `**Executive Summary**`, `**Investment Thesis**`, `**Stop Loss**`, `**Price Target**`, `**Planned Entry Price**`, etc.
  - `create_portfolio_manager(llm)` (`tradingagents/agents/managers/portfolio_manager.py:40-235`):
    * **Rule 1 Invariant (Trader WNS Invariant)** (`tradingagents/agents/managers/portfolio_manager.py:58-65`, `100-101`, `134-166`): If Trader proposal is WNS, PM CANNOT override to BUY. If LLM outputs BUY, node automatically downgrades rating to WNS, clears stop/take profit/planned entry, carries over Trader WNS recheck date/trigger price, and defaults `wns_recheck_date` to `trade_date + 14d`.
    * **Planned Entry & Mode Resolution** (`tradingagents/agents/managers/portfolio_manager.py:67-77`): Resolves `EntryMode.T1_LIMIT` with `planned_entry_price` if Trader proposed `BUY_LIMIT`; resolves `EntryMode.T1_OPEN` with `planned_entry_price = None` if `BUY_MARKET`.
    * **Next Review Date Determinism** (`tradingagents/agents/managers/portfolio_manager.py:175-181`): Deterministically computes `trade_date + 7d` (for BUY) or `trade_date + 21d` (for WNS) if omitted by LLM.
    * **Fallback SignalContract Construction** (`tradingagents/agents/managers/portfolio_manager.py:182-215`): If conversion fails or LLM is free-text, constructs deterministic fallback `SignalContract(rating=WNS, action=WNS, time_horizon_days=20, confidence=0.7, wns_recheck_date=trade_date + 14d)`.
- **Potential Mismatch / Inconsistency Vectors**:
  - **Strict Hierarchy Violation Defense**: Prompt warning + programmatic hard downgrade check (`tradingagents/agents/managers/portfolio_manager.py:134-166`).
  - **Price Target Alias**: `take_profit` vs `price_target` field alias resolution (`tradingagents/agents/schemas.py:445`, `553`).
  - **Review Date Regex vs Model Coercion**: `_validate_portfolio_dates` (`tradingagents/agents/schemas.py:431-435`) ensures ISO YYYY-MM-DD or None.

```markdown
- [ ] Verify Rule 1 Strict Hierarchy Invariant: programmatic downgrade of PM BUY to WNS when Trader proposed WNS (`tradingagents/agents/managers/portfolio_manager.py:134-166`).
- [ ] Check `_use_upper_bound_for_horizon_range` multi-language horizon parser for English and Indonesian terms (`tradingagents/agents/schemas.py:456-485`).
- [ ] Verify deterministic calculation of `next_review_date` (T+7 for BUY, T+21 for WNS) when omitted (`tradingagents/agents/managers/portfolio_manager.py:175-181`).
- [ ] Verify fallback `SignalContract` fail-safe generation when structured output conversion encounters exceptions (`tradingagents/agents/managers/portfolio_manager.py:189-215`).
```

---

### SECTION 5: Structured Invocation & Graceful JSON Recovery Utilities
- **Exact File Paths & Lines**: `tradingagents/agents/utils/structured.py:1-118`
- **Purpose**: Universal execution wrapper for LangChain structured LLMs with secondary JSON extraction and free-text markdown repair.
- **Contracts, Data Structures & Models**:
  - `extract_json_from_text(text: str, schema: type[T]) -> Optional[T]` (`tradingagents/agents/utils/structured.py:33-42`): Extracts and validates JSON from markdown code fences ` ```json ... ``` ` or raw text.
  - `invoke_structured_with_recovery(...) -> tuple[Optional[T], str]` (`tradingagents/agents/utils/structured.py:44-74`):
    1. Primary: Invokes native `structured_llm.invoke(prompt)`.
    2. Secondary: If primary fails, invokes `plain_llm.invoke(prompt)` and runs `extract_json_from_text`.
    3. Final: If JSON extraction fails, returns `(None, raw_text)`.
  - `bind_structured(llm, schema, agent_name) -> Optional[Any]` (`tradingagents/agents/utils/structured.py:76-91`): Safe wrapper for `llm.with_structured_output(schema)`.
  - `invoke_structured_or_freetext(...) -> str` (`tradingagents/agents/utils/structured.py:93-118`): Fallback wrapper returning rendered markdown string.
- **Potential Mismatch / Inconsistency Vectors**:
  - **Provider Capability Disparities**: OpenAI uses `json_schema`, Gemini uses `response_schema`, Anthropic uses tool calling. Models without structured support return `None` from `bind_structured`, cleanly routed to free-text + regex repair.
  - **JSON Fence Variations**: Regex `r"```(?:json)?\s*(\{.*?\})\s*```"` correctly handles optional `json` language identifier.

```markdown
- [ ] Check `extract_json_from_text` against malformed and markdown-fenced LLM responses (`tradingagents/agents/utils/structured.py:33-42`).
- [ ] Verify `invoke_structured_with_recovery` returns valid Pydantic instance on secondary recovery pass (`tradingagents/agents/utils/structured.py:44-74`).
- [ ] Ensure `bind_structured` catches `NotImplementedError` and `AttributeError` for legacy/local LLM providers (`tradingagents/agents/utils/structured.py:76-91`).
```

---

### SECTION 6: Agent States & Graph Propagation Channels
- **Exact File Paths & Lines**: `tradingagents/agents/utils/agent_states.py:1-84`, `tradingagents/graph/propagation.py:1-112`, `tradingagents/graph/trading_graph.py:1-544`
- **Purpose**: LangGraph typed state management, memory log injection, and state propagation across nodes.
- **Contracts, Data Structures & Models**:
  - `InvestDebateState(TypedDict)` (`tradingagents/agents/utils/agent_states.py:9-20`): `bull_history`, `bear_history`, `history`, `current_response`, `judge_decision`, `count`.
  - `RiskDebateState(TypedDict)` (`tradingagents/agents/utils/agent_states.py:23-46`): `aggressive_history`, `conservative_history`, `neutral_history`, `history`, `latest_speaker`, `judge_decision`, `count`.
  - `AgentState(MessagesState)` (`tradingagents/agents/utils/agent_states.py:48-84`):
    * Core: `company_of_interest`, `asset_type`, `trade_date`, `sender`, `past_context`.
    * Analyst tool-loop message channels: `market_messages`, `social_messages`, `news_messages`, `fundamentals_messages`.
    * Reports: `market_report`, `sentiment_report`, `news_report`, `fundamentals_report`.
    * Decision artifacts: `investment_plan`, `trader_investment_plan`, `trader_proposal: Optional[TraderProposal]`, `final_trade_decision`, `signal_contract: Optional[SignalContract]`.
  - `Propagator.create_initial_state` (`tradingagents/graph/propagation.py:28-82`): Initializes empty channels and sets `trader_proposal: None`, `signal_contract: None`.
  - `TradingAgentsGraph._log_state` & `process_signal` (`tradingagents/graph/trading_graph.py:490-544`): Serializes `signal_contract` to JSON log and extracts canonical signal.
- **Potential Mismatch / Inconsistency Vectors**:
  - **State Key Synchronization**: Every key defined in `AgentState` must be populated during `create_initial_state` to prevent `KeyError` in LangGraph transitions.
  - **Serialization Format**: `_log_state` calls `model_dump(mode="json")` on `signal_contract` (`tradingagents/graph/trading_graph.py:520-524`).
  - **Signal Priority**: `process_signal` (`tradingagents/graph/trading_graph.py:537-544`) prioritizes typed `signal_contract` over regex parsing of `final_trade_decision`.

```markdown
- [ ] Crosscheck `AgentState` schema keys against `create_initial_state` dict definition (`tradingagents/agents/utils/agent_states.py:48-84` vs `tradingagents/graph/propagation.py:39-80`).
- [ ] Verify `process_signal` prioritizes `signal_contract` before falling back to `parse_rating` (`tradingagents/graph/trading_graph.py:537-544`).
- [ ] Verify `TradingAgentsGraph._log_state` dumps `signal_contract` with `mode="json"` for date/enum serialization (`tradingagents/graph/trading_graph.py:520-524`).
```

---

### SECTION 7: Memory Log Reflection & PM Decision Context Injection
- **Exact File Paths & Lines**: `tradingagents/agents/utils/memory.py:1-309`, `tradingagents/agents/utils/rating.py:1-28`
- **Purpose**: Append-only markdown memory log maintaining historical decision context and post-trade reflections injected into PM prompt.
- **Contracts, Data Structures & Models**:
  - `TradingMemoryLog` (`tradingagents/agents/utils/memory.py:10-309`):
    * Separator: `<!-- ENTRY_END -->` (`line 14`).
    * Tag format: `[{date} | {ticker} | {rating} | pending]` $\to$ `[{date} | {ticker} | {rating} | {raw_pct} | {alpha_pct} | {holding_days}d]`.
    * `store_decision(ticker, trade_date, final_trade_decision)` (`lines 31-50`): Parses canonical rating via `parse_rating` and appends pending entry.
    * `get_past_context(ticker, n_same=5, n_cross=3, as_of=None)` (`lines 71-106`): Injects past analyses of same ticker + cross-ticker lessons, respecting point-in-time cutoff `as_of`.
    * `update_with_outcome` & `batch_update_with_outcomes` (`lines 109-227`): Atomic update replacing pending tag and appending `REFLECTION:` section.
  - `parse_rating(text: str, default: str = "WNS") -> str` (`tradingagents/agents/utils/rating.py:17-28`): Regex `_RATING_LABEL_RE` (`line 12-14`) parses 5-tier legacy ratings (`Buy`, `Overweight`, `Hold`, `WNS`, `Underweight`, `Sell`) and normalizes strictly to `"BUY"` or `"WNS"`.
- **Potential Mismatch / Inconsistency Vectors**:
  - **Point-in-Time Leakage Protection**: `get_past_context` filters `e.get("date", "") <= str(as_of)` when `as_of` is provided (`tradingagents/agents/utils/memory.py:81`).
  - **Rating Canonicalization**: `parse_rating` maps `Buy`/`Overweight` to `"BUY"`, everything else (`Hold`, `Sell`, `Underweight`, `WNS`) to `"WNS"`.

```markdown
- [ ] Check `parse_rating` regular expressions across bilingual markdown headers (`tradingagents/agents/utils/rating.py:12-28`).
- [ ] Verify `get_past_context` strictly enforces PIT cutoff `as_of` filter to prevent lookahead leakage (`tradingagents/agents/utils/memory.py:81`).
- [ ] Verify atomic write safety (`.tmp` file and rename) in `TradingMemoryLog.update_with_outcome` (`tradingagents/agents/utils/memory.py:170-173`).
```

---

### SECTION 8: Backtest Decision Schemas, Legacy Enums & ExtendedDecision
- **Exact File Paths & Lines**: `tradingagents/backtesting/position.py:21-67`, `tradingagents/backtesting/position.py:568-646`, `tradingagents/backtesting/decision_schema.py:1-246`
- **Purpose**: Dataclass models representing parsed trading decisions and legacy backward compatibility shims.
- **Contracts, Data Structures & Models**:
  - `PositionSide(str, Enum)` (`tradingagents/backtesting/position.py:21-27`): `LONG`, `FLAT`, `SHORT` (compat only).
  - `OrderType(str, Enum)` (`tradingagents/backtesting/position.py:29-45`): Canonical: `BUY_TO_OPEN`, `SELL_TO_CLOSE`, `NO_ORDER`. Legacy boundary-only: `BUY_TO_ADD`, `SELL_TO_REDUCE`, `SELL_TO_OPEN`, `SELL_TO_ADD`, `BUY_TO_REDUCE`, `BUY_TO_CLOSE`, `REVERSE_TO_LONG`, `REVERSE_TO_SHORT`.
  - `ExtendedDecision / ParsedDecision` (`tradingagents/backtesting/position.py:568-646`):
    * Fields: `decision_id`, `ticker`, `trade_date`, `agent_rating`, `report_generated_at`, `last_data_date`, `decision_valid_from`, `normalized_rating`, `allocation_pct`, `reduce_pct`, `leverage`, `confidence`, `short_allowed`, `allow_new_position`, `market_mode = "SPOT_LONG_ONLY"`, `allowed_position_sides = "LONG"`, `position_intent`, `current_position_side`, `target_position_side`, `futures_action`, `stop_price`, `take_profit`, `planned_entry_price`, `wns_trigger_price`, `wns_recheck_date`, `time_horizon_days`, `time_horizon_label`, `thesis_summary`, `valid`, `invalid_reason`, `next_review_date`.
    * Trigger fields: `prev_rating`, `rating_changed`, `triggered`, `trigger_reasons`, `trigger_details`.
  - Legacy `tradingagents/backtesting/decision_schema.py`:
    * Re-exports modern configs from `position.py` (`lines 21-37`).
    * Legacy enums: `AssetClass`, `Rating`, `Action`, `OrderSide`, `OpenClose`, `OrderStatus` (`lines 43-85`).
    * Legacy record shapes: `ParsedDecision`, `Order`, `Trade`, `Position` (`lines 91-246`).
- **Potential Mismatch / Inconsistency Vectors**:
  - **Duplicate Class Names**: `tradingagents/backtesting/decision_schema.py` defines legacy `ParsedDecision` with `.action` and `.rating`, whereas `tradingagents/backtesting/position.py` defines modern `ParsedDecision` with `.agent_rating` and `.futures_action`.
  - **Schema Detection**: `OrderGenerator.generate` (`tradingagents/backtesting/order_generator.py:138-158`) detects old vs new schema via `hasattr(decision, "action") and hasattr(decision, "rating")`.

```markdown
- [ ] Verify `ExtendedDecision` field defaults ensure `market_mode="SPOT_LONG_ONLY"` and `short_allowed=False` (`tradingagents/backtesting/position.py:588-591`).
- [ ] Check schema autodetection in `OrderGenerator.generate` correctly discriminates legacy `decision_schema.ParsedDecision` vs `position.ParsedDecision` (`tradingagents/backtesting/order_generator.py:138-158`).
- [ ] Ensure `ExtendedDecision.invalid()` constructor safely populates all required fallback fields (`tradingagents/backtesting/position.py:622-641`).
```

---

### SECTION 9: Markdown Decision Parser & Horizon Extraction
- **Exact File Paths & Lines**: `tradingagents/backtesting/markdown_parser.py:1-411`
- **Purpose**: Extracts structured `ExtendedDecision` fields from raw LLM markdown reports with bilingual regex patterns.
- **Contracts, Data Structures & Models**:
  - `MarkdownDecisionParser` (`tradingagents/backtesting/markdown_parser.py:18-411`):
    * Rating extraction (`_extract_rating`, `lines 262-282`): Scans `_preferred_section` for English (`Buy`, `Sell`, `Hold`, `WNS`, `Wait and See`, `Underweight`, `Overweight`) and Indonesian (`Beli`, `Jual`, `Tahan`). Maps to `Rating` enum.
    * Agent Rating mapping (`lines 98-99`): `Buy`/`Overweight` $\to$ `"Buy"`, all others $\to$ `"WNS"`.
    * Point-in-Time validation: Requires `Decision Valid From` metadata; if missing, returns `ExtendedDecision.invalid` with `"Missing Decision Valid From. Refuse trading to avoid same-bar execution."` (`lines 81-87`).
    * Numerical Extraction:
      - `confidence` (`lines 101-109`): Normalized to fraction if > 1.
      - `allocation_pct` / `reduce_pct` (`lines 110-129`).
      - `stop_price`, `take_profit`, `planned_entry_price`, `wns_trigger_price` (`lines 138-172`).
      - `wns_recheck_date` (`lines 173-181`): Extracts `YYYY-MM-DD`.
      - `time_horizon_days` (`lines 190-198`): Calls `_upper_bound_horizon_days` (`lines 333-352`) to parse intervals like `"1-3 months"` to upper bound in trading days (63 days).
      - `next_review_date` (`lines 390-411`): Multi-pattern matcher for English and Indonesian headers.
- **Potential Mismatch / Inconsistency Vectors**:
  - **Lookahead & Same-Bar Execution Rejection**: Parser strictly rejects decisions lacking `decision_valid_from` (`tradingagents/backtesting/markdown_parser.py:81-87`).
  - **Horizon Multipliers**: `day/hari` = 1, `week/minggu` = 5, `month/bulan` = 21, `year/tahun` = 252 (`tradingagents/backtesting/markdown_parser.py:345-351`).
  - **Section Prioritization**: `_preferred_section` (`tradingagents/backtesting/markdown_parser.py:290-312`) searches specific PM/Trader headers to prevent matching earlier debate quotes.

```markdown
- [ ] Verify `MarkdownDecisionParser` rejects missing `Decision Valid From` metadata to prevent lookahead execution (`tradingagents/backtesting/markdown_parser.py:81-87`).
- [ ] Check `_upper_bound_horizon_days` unit multipliers for days, weeks, months, and years across English and Indonesian text (`tradingagents/backtesting/markdown_parser.py:333-352`).
- [ ] Check `_preferred_section` window slicing (2500 chars from PM/Trader headers) isolates final decisions from debate logs (`tradingagents/backtesting/markdown_parser.py:290-312`).
```

---

### SECTION 10: Decision State Manager & Spot Long-Only Mapper
- **Exact File Paths & Lines**: `tradingagents/backtesting/decision_state_manager.py:1-111`
- **Purpose**: Maps agent rating and current position state into execution intent (`PositionIntent`) and order action (`OrderType`).
- **Contracts, Data Structures & Models**:
  - Rating constants: `RATING_BUY = "Buy"`, `RATING_WNS = "WNS"`.
  - `_canonical_rating(value)` (`tradingagents/backtesting/decision_state_manager.py:23-29`): Maps `buy`, `overweight`, `strong buy` $\to$ `Buy`, all others $\to$ `WNS`.
  - `DecisionStateManager.resolve(rating, current_position, allow_new_position)` (`tradingagents/backtesting/decision_state_manager.py:41-62`):
    * If `rating == WNS`: returns `(PositionIntent.HOLD, side, OrderType.NO_ORDER)`.
    * If `side == "FLAT"`:
      - If `allow_new_position == False`: `(PositionIntent.HOLD, "FLAT", OrderType.NO_ORDER)`.
      - If `allow_new_position == True`: `(PositionIntent.OPEN, "LONG", OrderType.BUY_TO_OPEN)`.
    * If `side == "LONG"`:
      - Returns `(PositionIntent.HOLD, "LONG", OrderType.NO_ORDER)`. (One-shot spot entries: never pyramid from repeated daily BUY ratings).
    * If `side == "SHORT"`:
      - Returns `(PositionIntent.HOLD, side, OrderType.NO_ORDER)` (Legacy short state never covered or reversed).
  - `DecisionStateManager.map(decision, current_position) -> ExtendedDecision` (`tradingagents/backtesting/decision_state_manager.py:63-111`): Produces mapped `ExtendedDecision` with `allocation_pct = initial_entry_pct` (on OPEN) or `0.0`.
- **Potential Mismatch / Inconsistency Vectors**:
  - **No Pyramiding Invariant**: When `side == "LONG"`, receiving another BUY rating yields `OrderType.NO_ORDER` (`tradingagents/backtesting/decision_state_manager.py:56-58`).
  - **No Agent-Driven Exits**: SELL_TO_CLOSE is never emitted by DSM; exits are exclusively governed by static risk controls.

```markdown
- [ ] Verify `DecisionStateManager.resolve` enforces one-shot entry rule (no pyramiding when `side == "LONG"`) (`tradingagents/backtesting/decision_state_manager.py:56-58`).
- [ ] Verify `DecisionStateManager` never emits `SELL_TO_CLOSE` or short orders (`tradingagents/backtesting/decision_state_manager.py:41-62`).
- [ ] Check `DecisionStateManager.map` correctly propagates planned entry, stop loss, take profit, and WNS trigger prices (`tradingagents/backtesting/decision_state_manager.py:99-105`).
```

---

### SECTION 11: Order Generation & Sizing Mechanics
- **Exact File Paths & Lines**: `tradingagents/backtesting/order_generator.py:1-411`
- **Purpose**: Translates mapped `ExtendedDecision` into concrete pending `Order` dataclasses with lot-size rounding and leverage/cash capping.
- **Contracts, Data Structures & Models**:
  - `OrderGenerator.decide(decision, current_position, current_equity, current_cash, reference_price)` (`tradingagents/backtesting/order_generator.py:50-103`):
    * Rejects non-BUY_TO_OPEN orders or orders when not FLAT.
    * Sets `exec_date = decision.decision_valid_from` (or `trade_date + 1d`).
    * Resolves `limit_price = decision.planned_entry_price or 0.0`.
    * Computes `target_qty = _allocation_to_qty(...)` $\to$ `_round_to_lot(...)` $\to$ `_cap_by_leverage(...)`.
  - `_allocation_to_qty` (`tradingagents/backtesting/order_generator.py:287-338`):
    * Allocates `target_notional = equity * alloc`.
    * **Spot Cash Clamp** (`lines 313-321`): In spot long-only mode (`initial_margin_pct >= 1.0`), clamps `target_notional = min(target_notional, free_cash / (1 + buy_fee + slippage))`.
    * Converts to shares using `price_ref = limit_price or reference_price or mark_price`.
  - `_round_to_lot(qty)` (`tradingagents/backtesting/order_generator.py:339-345`): Rounds down to integer multiple of `lot_size` (e.g. 100 for IDX, 1 for US).
  - `_cap_by_leverage(qty)` (`tradingagents/backtesting/order_generator.py:346-367`): Ensures total exposure $\le \text{equity} \times \text{max\_leverage}$.
  - `generate_stop_order` & `generate_liquidation_order` (`tradingagents/backtesting/order_generator.py:371-411`): Emits `SELL_TO_CLOSE` risk orders with `is_risk_order = True`.
- **Potential Mismatch / Inconsistency Vectors**:
  - **Cash Clamping with Fees**: Prevents negative cash balance on spot fills by factoring `1 + buy_fee + slippage` (`tradingagents/backtesting/order_generator.py:318-321`).
  - **Lot Size Truncation**: Fractional shares are truncated via integer division `(qty // lot) * lot` (`tradingagents/backtesting/order_generator.py:344`).
  - **Limit Order Reason String**: Emits `reason="agent_buy_limit"` if `limit_price > 0`, else `reason="agent_buy_open"` (`tradingagents/backtesting/order_generator.py:269`).

```markdown
- [ ] Verify `_allocation_to_qty` spot cash clamp accounts for `buy_fee` and `slippage` fee factors (`tradingagents/backtesting/order_generator.py:318-321`).
- [ ] Check `_round_to_lot` rounds down cleanly for lot sizes of 1 and 100 (`tradingagents/backtesting/order_generator.py:339-345`).
- [ ] Verify `_cap_by_leverage` enforces max leverage cap against existing portfolio exposure (`tradingagents/backtesting/order_generator.py:346-367`).
```

---

### SECTION 12: Trigger Evaluator
- **Exact File Paths & Lines**: `tradingagents/backtesting/trigger_evaluator.py:1-342`
- **Purpose**: Evaluates setup invalidation, price touch, and R:R deterioration conditions before allowing ratings to fire orders.
- **Contracts, Data Structures & Models**:
  - `TriggerConfig` (`tradingagents/backtesting/trigger_evaluator.py:35-64`):
    * `enabled: bool = True`, `trigger_on_tp_sl_hit: bool = True`, `trigger_on_setup_invalid: bool = True`.
    * `trigger_on_rr_deteriorated: bool = True`, `trigger_on_strong_exit_signal: bool = True`.
    * `trigger_on_better_candidate: bool = True`, `trigger_on_entry_condition_changed: bool = True`.
    * `trigger_on_rating_confirmed: bool = True`.
    * Thresholds: `rr_deterioration_pct = 0.30`, `better_candidate_confidence_ratio = 1.5`, `setup_invalid_keyword_threshold = 2`.
  - `TriggerResult` (`tradingagents/backtesting/trigger_evaluator.py:66-77`): `triggered: bool`, `reasons: list[str]`, `details: dict`.
  - Invalidation Patterns: `SETUP_INVALID_PATTERNS` (`tradingagents/backtesting/trigger_evaluator.py:21-33`) matches English and Indonesian thesis breakdown phrases.
  - `_price_level_touched` (`tradingagents/backtesting/trigger_evaluator.py:177-195`): Evaluates whether `current_bar.low <= planned_entry_price`.
  - `_compute_real_rr_drop` (`tradingagents/backtesting/trigger_evaluator.py:253-291`): Computes explicit R:R distance changes from stop/target legs.
- **Potential Mismatch / Inconsistency Vectors**:
  - **Flat Portfolio Gating**: When `position.is_flat()`, `triggered = (is_entry_signal and not is_invalid)` (`tradingagents/backtesting/trigger_evaluator.py:150-165`).
  - **No Agent Strong Exits**: `_strong_exit_signal` always returns `False` in spot long-only mode (`tradingagents/backtesting/trigger_evaluator.py:293-300`).

```markdown
- [ ] Check `SETUP_INVALID_PATTERNS` regex matches bilingual thesis invalidation phrases (`tradingagents/backtesting/trigger_evaluator.py:21-33`).
- [ ] Verify `_price_level_touched` checks `low <= planned_entry_price` for pending limit accumulation (`tradingagents/backtesting/trigger_evaluator.py:177-195`).
- [ ] Verify `_compute_real_rr_drop` handles division by zero safely when stop equals reference price (`tradingagents/backtesting/trigger_evaluator.py:283-285`).
```

---

### SECTION 13: Forward Horizon Evaluator & Causal Backtest Engine
- **Exact File Paths & Lines**: `tradingagents/backtesting/horizon_evaluator.py:1-311`, `cli/commands/evaluate.py:1-344`
- **Purpose**: Point-in-time forward evaluator testing causal single-shot signals across forward OHLCV daily bars.
- **Contracts, Data Structures & Models**:
  - `DailyExcursionBar` (`tradingagents/backtesting/horizon_evaluator.py:27-37`): Bar-by-bar MFE/MAE excursion tracking.
  - `EvaluationResult` (`tradingagents/backtesting/horizon_evaluator.py:40-75`): Complete single-shot evaluation summary:
    * `outcome: EvaluationOutcome`, `realized_return_pct: float`, `max_favorable_excursion_pct: float`, `max_adverse_excursion_pct: float`.
    * `planned_rr_ratio`, `realized_rr_ratio`, `mfe_efficiency`, `actual_holding_days`.
  - `HorizonEvaluator.evaluate(...)` (`tradingagents/backtesting/horizon_evaluator.py:107-311`):
    * Evaluates `BUY` signals against future daily bars (`date > signal_date`).
    * Fill Logic:
      - `T1_OPEN`: Fills at `first_open`.
      - `T1_LIMIT`: Fills at requested `actual_entry_price` if `low <= actual_entry_price <= high`; if `first_open < actual_entry_price`, price improvements to `first_open`; if `low > actual_entry_price`, outcomes `NO_FILL`.
    * Intra-Bar Stop/Target Execution Order:
      1. Gap checks at `bar_open <= stop_loss` or `bar_open >= take_profit`.
      2. Intraday touch checks at `bar_low <= stop_loss` or `bar_high >= take_profit`.
      3. Time-stop check at `idx + 1 >= effective_time_stop`.
  - CLI `evaluate_signal_cmd` (`cli/commands/evaluate.py:55-344`): Single-shot CLI execution with rich TUI, snapshot isolation, and result bundle generation.
- **Potential Mismatch / Inconsistency Vectors**:
  - **No-Fill Outcome Handling**: `NO_FILL` outcomes are recorded with `actual_entry_price = None` and 0 realized return (`tradingagents/backtesting/horizon_evaluator.py:208-217`).
  - **Gap Open Execution Order**: Gap-down open below stop loss fills at open, not at stop level (`tradingagents/backtesting/horizon_evaluator.py:252-255`).
  - **Single-Shot Evaluation Side**: `WNS`, `HOLD`, `SELL` from FLAT inventory map to `EvaluationOutcome.NO_ORDER` with 0% realized return (`tradingagents/backtesting/horizon_evaluator.py:133-137`).

```markdown
- [ ] Verify `HorizonEvaluator.evaluate` price improvement on gap-down open for limit buy orders (`tradingagents/backtesting/horizon_evaluator.py:205-207`).
- [ ] Verify `HorizonEvaluator.evaluate` gap-down open fill price at `bar_open` (not `stop_loss`) when `bar_open <= stop_loss` (`tradingagents/backtesting/horizon_evaluator.py:252-254`).
- [ ] Check CLI `evaluate_signal_cmd` snapshot isolation and result bundle writing (`cli/commands/evaluate.py:180-245`, `308-337`).
- [ ] Ensure non-BUY actions (`WNS`, `HOLD`, `SELL`) correctly map to `EvaluationOutcome.NO_ORDER` (`tradingagents/backtesting/horizon_evaluator.py:133-137`).
```

---

### SECTION 14: Default Config & CLI Crosscheck Matrix
- **Exact File Paths & Lines**: `tradingagents/default_config.py:1-146`, `tradingagents/backtesting/position.py:72-296`
- **Purpose**: Global defaults, environment variable overrides, and backtest dataclass config resolution.
- **Config Keys & Mappings**:
  - `DEFAULT_CONFIG` (`tradingagents/default_config.py:52-146`):
    * `llm_provider = "openai"`, `deep_think_llm = "gpt-5.4"`, `quick_think_llm = "gpt-5.4-mini"`.
    * `max_debate_rounds = 1`, `max_risk_discuss_rounds = 1`, `analyst_concurrency_limit = 1`.
    * `kronos_enabled = False`, `kronos_model_tier = "base"`, `kronos_pred_len = 20`.
    * `point_in_time_mode = False`, `backtest_mode = False`, `memory_enabled = True`.
  - `BacktestConfig` & Child Configs (`tradingagents/backtesting/position.py:72-296`):
    * `ExecutionConfig`: `buy_fee = 0.0015`, `sell_fee = 0.0025`, `slippage = 0.001`, `lot_size = 1`, `contract_multiplier = 1.0`.
    * `MarginConfig`: `initial_margin_pct = 1.0`, `maintenance_margin_pct = 1.0`, `max_leverage = 1.0` (Spot Cash-Only default in `BacktestConfig:495-499`).
    * `RiskConfig`: `default_stop_pct = 0.08`, `default_take_profit_pct = 0.20`, `max_holding_days = 20`, `atr_stop_multiplier = 1.5`, `atr_tp_multiplier = 3.0` (R:R >= 2.0:1).
    * `DecisionMappingConfig`: `mode = "spot_long_only"`, `initial_entry_pct = 0.30`.
- **Potential Mismatch / Inconsistency Vectors**:
  - **Spot Cash-Only Invariant**: In `BacktestConfig.__init__`, `margin` defaults to `initial_margin_pct = 1.0`, `max_leverage = 1.0`, whereas standalone `MarginConfig()` defaults to `0.50` and `2.0`. Backtests must use `BacktestConfig` to ensure cash-only constraints.
  - **Fee Defaults**: `ExecutionConfig` uses percentage-based fees (`buy_fee = 0.0015`, `sell_fee = 0.0025`) vs legacy tick-based configs.

```markdown
- [ ] Check `BacktestConfig` default factory overrides `MarginConfig` to cash-only (`initial_margin_pct=1.0`, `max_leverage=1.0`) (`tradingagents/backtesting/position.py:495-499`).
- [ ] Verify `_ENV_OVERRIDES` in `default_config.py` correctly maps and coerces all `TRADINGAGENTS_*` variables (`tradingagents/default_config.py:10-49`).
- [ ] Check `BacktestConfig.validate()` asserts `asset_class == 'stock'`, `data.provider == 'snapshot'`, and `memory_enabled == False` (`tradingagents/backtesting/position.py:511-563`).
```

---

## Subsystem 7: Simulated Broker, Position Accounting & Margin Realism

# Subsystem 7: Backtest Engine, Simulated Broker, Portfolio & Margin System — Architectural Mapping & Crosscheck Matrix

---

## 1. Architectural Overview & Component Inventory

Subsystem 7 provides causal walk-forward simulation, order routing, portfolio accounting, margin checking, risk containment, signal evaluation, and anti-leakage verification for spot equity instruments (with historical multi-asset compatibility shims).

### Key Files & Modules
- `tradingagents/backtesting/__init__.py:1-154`: Module entry point and public exports.
- `tradingagents/backtesting/position.py:1-772`: Canonical V2 configuration, types, enums, `Position`, `Order`, `Fill`, `Trade`, `InstrumentSpec`, and `ExtendedDecision`.
- `tradingagents/backtesting/decision_schema.py:1-246`: Backward-compatibility shims, legacy enums (`Rating`, `Action`, `OrderSide`, `OpenClose`), legacy `ParsedDecision`, `Order`, `Trade`, `Position`.
- `tradingagents/backtesting/margin_engine.py:1-185`: Stateless pure math helpers for margin, excess margin, leverage, and margin breach checks.
- `tradingagents/backtesting/margin.py:1-148`: `MarginAccount` and `MarginResult` wrapper interfaces.
- `tradingagents/backtesting/portfolio.py:1-817`: `Portfolio` (V1 cash/margin ledger) and `PortfolioV2` (canonical spot long-only cash ledger).
- `tradingagents/backtesting/broker.py:1-262`: `SimulatedBroker` (order queue, execution with slippage/fees/spread, immediate risk execution).
- `tradingagents/backtesting/order_generator.py:1-411`: `OrderGenerator` (translates `ExtendedDecision` into `BUY_TO_OPEN` orders, cash clamping, lot sizing, leverage clamping).
- `tradingagents/backtesting/risk.py:1-425`: `RiskEngine` (bar-by-bar stop-loss, take-profit, liquidation checks, ATR stops).
- `tradingagents/backtesting/trigger_evaluator.py:1-342`: `TriggerEvaluator` and `TriggerConfig` (evaluates setup validity, R:R changes, price triggers).
- `tradingagents/backtesting/horizon_evaluator.py:1-311`: `HorizonEvaluator` (causal forward evaluation of BUY signals across N days).
- `tradingagents/backtesting/decision_state_manager.py:1-111`: `DecisionStateManager` (canonical rating normalizer: BUY vs WNS).
- `tradingagents/backtesting/markdown_parser.py:1-411`: `MarkdownDecisionParser` (extracts structured decisions without speculative action inference).
- `tradingagents/backtesting/cutoff_validator.py:1-192`: `DecisionCutoffValidator` (fail-closed PIT leakage validator).
- `tradingagents/backtesting/data_window.py:1-282`: `compute_window`, `slice_ohlcv`, `slice_news`, `slice_fundamentals`, `slice_sentiment`, `slice_broker_activity`.
- `tradingagents/backtesting/metrics.py:1-336`: `MetricsCalculator` (returns, Sharpe, Sortino with downside deviation, turnover, drawdowns, trigger stats).
- `tradingagents/backtesting/reports.py:1-304`: `BacktestReportGenerator` (exports `summary.json`, CSVs, and markdown reports).
- `tradingagents/backtesting/walk_forward_runner.py:1-949`: `WalkForwardBacktestRunner` (orchestrates 10-step daily simulation loop).
- `tradingagents/backtesting/engine.py:1-250`: `BacktestEngine` (YAML/dict configuration parsing and execution).
- `tradingagents/backtesting/config_resolver.py:1-141`: `resolve_agent_config` (YAML + environment variable resolution).
- `cli/commands/backtest.py:1-498`: CLI interactive command, rich UI, and progress reporting.

---

## 2. Granular Section Mapping & Verification Matrix

### Section 7.1: Configuration Hierarchy, Defaults & Resolution
- **Files & Line Ranges:**
  - `tradingagents/backtesting/position.py:72-296` (`ExecutionConfig`, `MarginConfig`, `RiskConfig`, `DecisionMappingConfig`, `DataConfig`, `AgentConfig`, `LeakageGuardConfig`, `OutputConfig`)
  - `tradingagents/backtesting/position.py:474-563` (`BacktestConfig.validate`)
  - `tradingagents/backtesting/engine.py:156-250` (`BacktestEngine.from_yaml`, `from_dict`, `_build_config`)
  - `tradingagents/backtesting/config_resolver.py:25-141` (`resolve_agent_config`, `validate_backtest_config`)
- **Purpose & Contracts:**
  - Construct typed configuration with validated parameter boundaries before runtime initialization.
  - Resolves YAML parameters, falling back to environment variables (`TRADINGAGENTS_LLM_PROVIDER`, `TRADINGAGENTS_LLM_MODEL`).
- **Core Data Structures:**
  - `BacktestConfig`: Aggregates all sub-configs; default initial cash = 100,000.0; default lookback = 240 days; default margin = 1.0 (cash-only spot).
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Lookback Days Discrepancy:** `position.py:491` sets default `lookback_days: Optional[int] = 240`, while `data_window.py:30-32` defines `ALLOWED_LOOKBACKS = (None, 5, 10, 20, 40, 60, 80, 100, 120, 240, 512)`. CLI prompt (`backtest.py:145`) defaults to 240 if not overridden.
  - [ ] **Margin Default Alignment:** `position.py:494-500` sets default margin to `initial_margin_pct=1.0, maintenance_margin_pct=1.0, max_leverage=1.0` in `BacktestConfig`, but standalone `MarginConfig` (`position.py:127-130`) defaults to `initial_margin_pct=0.50, maintenance_margin_pct=0.35, max_leverage=2.0`. Verify that `_build_config` in `engine.py:216-220` enforces 1.0 when `margin` is omitted from YAML.
  - [ ] **Mode Enum Overrides:** `DecisionMappingConfig.__post_init__` (`position.py:207-214`) forcefully overwrites `self.mode = "spot_long_only"` and `self.spot_mode = True` even if "strict_5tier" or "aggressive" is passed, while raising ValueError if any short flags are True.

---

### Section 7.2: Canonical Data Types, Records & Legacy Shims
- **Files & Line Ranges:**
  - `tradingagents/backtesting/position.py:21-67` (`PositionSide`, `OrderType`, `MarketMode`, `PositionIntent`, `FillRule`)
  - `tradingagents/backtesting/position.py:301-326` (`InstrumentSpec`)
  - `tradingagents/backtesting/position.py:331-469` (`Position`, `Fill`, `Order`)
  - `tradingagents/backtesting/position.py:568-755` (`ParsedDecision`, `SnapshotMetadata`, `MarketPoint`, `PortfolioSnapshot`, `MarginEvent`, `Trade`)
  - `tradingagents/backtesting/decision_schema.py:43-246` (Legacy enums `Rating`, `Action`, `OrderSide`, `OpenClose`, `OrderStatus`, and legacy dataclasses)
- **Purpose & Contracts:**
  - Bridge legacy multi-asset/short/reverse data structures with the PRD spot long-only architecture.
  - `Position`: positive `quantity` = Long, 0 = Flat; negative inventory rejected at execution.
  - `Order`: explicit `OrderType` (`BUY_TO_OPEN`, `SELL_TO_CLOSE`, `NO_ORDER`).
- **Core Data Structures:**
  - `InstrumentSpec`: `multiplier: float = 1.0`, `tick_size: float = 0.01`, `currency: str = "USD"`, `lot_size: int = 100`.
  - `Fill`: execution record linking `order_id`, `decision_id`, `price`, `fee`, `slippage_amount`.
  - `Trade`: legacy & V2 compatible trade audit record.
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Dual Definition of Dataclasses:** `Trade`, `Position`, `ParsedDecision`, `Order` are defined in *both* `decision_schema.py` and `position.py`. In `decision_schema.py:178-246`, `Trade` uses `gross_amount, net_amount, tick_size, slippage_ticks` whereas in `position.py:726-755` it uses `slippage_amount, realized_pnl_delta, mark_price`.
  - [ ] **Side String vs Enum:** In `position.py:454-461`, `Order.side` returns string `"BUY"` or `"SELL"`. In `decision_schema.py:69-72`, `OrderSide` is an Enum. `SimulatedBroker._execute_order` (`broker.py:244`) casts side using `OrderSide.BUY if is_buy else OrderSide.SELL`.
  - [ ] **Lot Size Discrepancy:** `InstrumentSpec.auto_from_ohlcv` (`position.py:324`) sets `lot_size: int = 100`, but `ExecutionConfig` (`position.py:81`) defaults `lot_size: int = 1`. Sizing in `order_generator.py:340-344` reads `self.exec_cfg.lot_size`.

---

### Section 7.3: Margin Math & Margin Engine
- **Files & Line Ranges:**
  - `tradingagents/backtesting/margin_engine.py:22-185` (`notional_value`, `initial_margin`, `maintenance_margin`, `max_contracts_by_margin`, `excess_margin`, `is_margin_call`, `is_intraday_margin_breach`, `leverage`, `margin_utilization`)
  - `tradingagents/backtesting/margin.py:32-148` (`MarginResult`, `MarginAccount`)
- **Purpose & Contracts:**
  - Stateless mathematical functions for initial/maintenance margin, leverage calculations, and intraday breach detection.
- **Core Formulas:**
  - `notional = |quantity| * mark_price * multiplier` (`margin_engine.py:28`)
  - `initial_margin = notional * initial_margin_pct` (`margin_engine.py:40`)
  - `maintenance_margin = notional * maintenance_margin_pct` (`margin_engine.py:52`)
  - `excess_margin = account_equity - maintenance_margin` (`margin_engine.py:83`)
  - `margin_call = account_equity < maintenance_margin * (1 + buffer_pct)` (`margin_engine.py:108`)
  - `worst_equity_intraday = account_equity_open - ((open_price - intraday_low) * quantity * multiplier)` (`margin_engine.py:131-133`)
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Short Position Handling in Margin Engine:** `is_intraday_margin_breach` (`margin_engine.py:125-127`) explicitly raises `ValueError("spot long-only margin check rejects short positions")` if `quantity < 0`. Ensure no test passes negative quantity expecting a short breach calculation.
  - [ ] **Margin Account Margin Call Formula:** In `margin.py:104`, `margin_call` checks `equity < maintenance_margin * (1.0 + threshold)` where default threshold is `0.40`. In `margin_engine.py:108`, `buffer_pct` default is `0.0`. Crosscheck threshold naming and scaling across both modules.

---

### Section 7.4: Portfolio State & Accounting (V1 vs V2)
- **Files & Line Ranges:**
  - `tradingagents/backtesting/portfolio.py:55-435` (`Portfolio` - Legacy)
  - `tradingagents/backtesting/portfolio.py:437-817` (`PortfolioV2` - PRD Canonical)
- **Purpose & Contracts:**
  - Tracks free cash, margin posted, signed position, mark-to-market valuations, equity curve, realized and unrealized PnL.
  - `PortfolioV2`: Spot cash accounting model. Long open: `cash -= (notional + fee)`. Long close: `cash += (notional - fee)`. Margin posted is tracked for reporting but does not double-deduct from cash (`portfolio.py:696, 720`).
- **Core Invariants:**
  - No pyramiding: `_apply_open` raises `ValueError` if position is not flat (`portfolio.py:640, 692`).
  - Cash non-negativity: `_apply_open` raises `InsufficientMarginError` if `(notional + fee) > self.cash + 1e-7` (`portfolio.py:682-685`).
  - Equity calculation: `account_equity = self.cash + position_value(mark_price)` (`portfolio.py:519`).
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Cash Accounting Mismatch (V1 vs V2):** In V1 `Portfolio._post_margin` (`portfolio.py:268-276`), cash is reduced only by `initial_margin` (futures style), and daily mark settlement credits/debits cash (`portfolio.py:297`). In V2 `PortfolioV2._apply_open` (`portfolio.py:696`), cash is reduced by full notional plus fees (spot style).
  - [ ] **Daily Settlement Flag:** `Portfolio` has `enable_daily_settlement: bool` (`portfolio.py:73, 218`). In `PortfolioV2`, daily mark-to-market (`portfolio.py:749-805`) updates unrealized PnL and equity curve without altering cash balances.
  - [ ] **Currency Unit Assumptions:** Hardcoded log string in `portfolio.py:684` assumes `"IDR"` (`"...requires {total_cost:.2f} IDR, available {self.cash:.2f} IDR"`), while `InstrumentSpec.currency` (`position.py:310`) defaults to `"USD"`.

---

### Section 7.5: Simulated Broker & Order Execution
- **Files & Line Ranges:**
  - `tradingagents/backtesting/broker.py:35-262` (`SimulatedBroker`)
- **Purpose & Contracts:**
  - Queue pending orders and execute them on future market bars at `next_session_open`.
  - Immediate execution path (`execute_pending_orders_immediate`, `broker.py:77-106`) for stop-loss, take-profit, and liquidation orders.
  - Computes slippage (percentage-based or tick-based), bid-ask spread friction (`spread_bps`), asymmetric commissions (`buy_fee`, `sell_fee`), and limit price bounds.
- **Execution Rules:**
  - Limit Order Condition: `traded_past_limit = low > limit_p` for Buy, `high < limit_p` for Sell. If true, order becomes `UNFILLED` (`broker.py:178-185`).
  - Price Improvement: For Buy limit, `base_price = min(open_price, limit_p)` (`broker.py:190`).
  - Friction Calculation: `total_friction_pct = slippage_pct + (spread_bps / 10000.0) / 2.0` (`broker.py:204-205`).
  - Limit Bound Preservation: `fill_price = min(fill_price, limit_p)` for Buy (`broker.py:213`).
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Immediate Risk Execution vs Pending Queue:** Risk orders bypass the pending queue and execute on the same bar (`broker.py:88-99`), whereas agent decision orders execute at T+1 Open (`walk_forward_runner.py:288-296`).
  - [ ] **Non-Positive Quantity / Short Rejection:** Orders with `quantity <= 0` or types other than `BUY_TO_OPEN` / `SELL_TO_CLOSE` are rejected with `order.status = "REJECTED"` (`broker.py:54-66`).
  - [ ] **Order Status "UNFILLED" vs "PENDING":** Limit orders that did not trade in range are tagged `UNFILLED` and discarded from trades list (`broker.py:184`), not carried over to future days.

---

### Section 7.6: Order Generator & Sizing Logic
- **Files & Line Ranges:**
  - `tradingagents/backtesting/order_generator.py:38-411` (`OrderGenerator`)
- **Purpose & Contracts:**
  - Converts mapped `ExtendedDecision` into concrete `Order` objects.
  - Enforces spot cash clamping, lot-size rounding, and maximum leverage clamping.
- **Sizing Mechanics:**
  - Allocation conversion: `target_notional = equity * allocation_pct` (`order_generator.py:311`).
  - Spot Cash Clamp: `target_notional = min(target_notional, free_cash / (1 + buy_fee + slippage))` (`order_generator.py:321`).
  - Lot Rounding: `qty = (qty // lot_size) * lot_size` (`order_generator.py:344`).
  - Leverage Clamping: `max_qty_by_lev = (equity * max_leverage - existing_exposure) // (price_ref * multiplier)` (`order_generator.py:365`).
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Percentage vs Decimal Scaling:** `_allocation_to_qty` checks if `alloc > 1.0` and divides by 100.0 (`order_generator.py:307-308`).
  - [ ] **Reference Price Fallback Hierarchy:** `price_ref` resolves: `limit_price -> reference_price -> mark_price -> stop_price -> 100.0` (`order_generator.py:323-331`). If default fallback 100.0 is used on high/low value stocks, sizing will distort.
  - [ ] **Pyramiding Rejection:** If `not current_position.is_flat()`, `decide()` returns empty list `[]` (`order_generator.py:74-75`).

---

### Section 7.7: Risk Engine & Position Risk Management
- **Files & Line Ranges:**
  - `tradingagents/backtesting/risk.py:22-315` (`RiskEvent`, `RiskEngine`)
  - `tradingagents/backtesting/risk.py:316-350` (`compute_atr`)
  - `tradingagents/backtesting/risk.py:352-425` (`update_position_risk_levels`)
- **Purpose & Contracts:**
  - Evaluates 8-step risk hierarchy on every daily bar (Hard risk -> Liquidation -> Stop-loss -> Take-profit -> Hold).
  - Stop-loss triggers if `low <= stop_price`; fill price is `min(open, stop_price)` (`risk.py:250-252`).
  - Take-profit triggers if `high >= take_profit`; fill price is `max(open, take_profit)` (`risk.py:281-283`).
  - Stop-loss takes precedence over take-profit when both trigger on the same bar (`risk.py:107-124`).
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Static Exit Invariance:** `update_position_risk_levels` (`risk.py:391-394`) prohibits ratcheting: once `position.stop_price` or `position.take_profit` is set, subsequent agent reports cannot move or widen them.
  - [ ] **ATR Stop Sizing R:R Guarantee:** `RiskConfig` (`position.py:174-175`) sets `atr_stop_multiplier: float = 1.5` and `atr_tp_multiplier: float = 3.0`, guaranteeing R:R >= 2.0. In `risk.py:411, 421`, fallback default multipliers are 2.0 and 3.0.
  - [ ] **Lookahead in ATR Computation:** `compute_atr` (`risk.py:326-330`) filters `ohlcv_df` strictly to `date <= current_date` to prevent future data leakage.

---

### Section 7.8: Trigger Evaluator (Gated Execution)
- **Files & Line Ranges:**
  - `tradingagents/backtesting/trigger_evaluator.py:19-64` (`SETUP_INVALID_PATTERNS`, `TriggerConfig`)
  - `tradingagents/backtesting/trigger_evaluator.py:66-342` (`TriggerResult`, `TriggerEvaluator`)
- **Purpose & Contracts:**
  - Gates BUY orders from firing unless a trigger condition is met:
    - `tp_sl_hit`: Previous risk order triggered stop/take-profit.
    - `setup_invalid`: Raw text / thesis matches invalidation regexes.
    - `rr_deteriorated`: R:R drop >= 30% (`rr_deterioration_pct: 0.30`).
    - `better_candidate`: Confidence ratio >= 1.5x.
    - `entry_condition_changed`: Rating becomes Buy/Overweight while flat.
    - `rating_confirmed`: Rating remains Buy while long.
    - `price_level_touched`: Current bar low <= `planned_entry_price`.
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Flat State Entry Condition:** While flat, `triggered = bool(is_entry_signal and not is_invalid)` (`trigger_evaluator.py:164`). Invalidation phrases override entry signals.
  - [ ] **Regex Multilingual Patterns:** `SETUP_INVALID_PATTERNS` matches both English ("setup invalid", "thesis broken") and Indonesian ("skenario gagal", "tesis rusak", "batalkan posisi") (`trigger_evaluator.py:21-33`).
  - [ ] **Confidence Proxy vs Explicit R:R:** When `use_real_rr_when_available=True` (`trigger_evaluator.py:59`), calculates true reward-to-risk drop from price band. Falls back to confidence drop if explicit legs are missing (`trigger_evaluator.py:248-251`).

---

### Section 7.9: Forward Horizon Evaluator
- **Files & Line Ranges:**
  - `tradingagents/backtesting/horizon_evaluator.py:16-75` (`EvaluationOutcome`, `DailyExcursionBar`, `EvaluationResult`)
  - `tradingagents/backtesting/horizon_evaluator.py:107-311` (`HorizonEvaluator.evaluate`)
- **Purpose & Contracts:**
  - Evaluates point-in-time forward outcome of a single BUY signal over 1 to 252 days against actual future daily bars.
  - Computes Max Favorable Excursion (`max_mfe`), Max Adverse Excursion (`max_mae`), MFE efficiency, and realized R:R ratio.
- **Outcomes Evaluated:**
  - `HIT_STOP_LOSS`, `HIT_TAKE_PROFIT`, `HIT_TIME_STOP`, `EXPIRED`, `NO_ORDER`, `INSUFFICIENT_DATA`, `NO_FILL`.
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Stop < Entry < Take Profit Invariant:** `HorizonEvaluator.evaluate` raises `ValueError` if `stop_loss >= take_profit` or `not (stop_loss < entry_price < take_profit)` unless gap open occurred (`horizon_evaluator.py:218-225`).
  - [ ] **Timezone Enforcement:** If `entry_timestamp` is provided, it must be timezone-aware or `ValueError` is raised (`horizon_evaluator.py:151-153`).
  - [ ] **Excursion Unit Scale:** `unrealized_return_close_pct`, `unrealized_mfe_pct`, `unrealized_mae_pct` are stored as percentages (multiplied by 100), whereas `realized_rr_ratio` and `mfe_efficiency` are unit ratios (`horizon_evaluator.py:269-272, 294-295`).

---

### Section 7.10: Decision State Manager & Rating Normalization
- **Files & Line Ranges:**
  - `tradingagents/backtesting/decision_state_manager.py:14-30` (Rating aliases & `_canonical_rating`)
  - `tradingagents/backtesting/decision_state_manager.py:31-111` (`DecisionStateManager`)
- **Purpose & Contracts:**
  - Pure mapping of arbitrary LLM rating outputs to canonical `BUY` or `WNS`.
  - Canonical outputs: `RATING_BUY = "Buy"`, `RATING_WNS = "WNS"`.
  - Aliases normalized: `overweight`, `strong_buy`, `strong buy` -> `Buy`; `hold`, `underweight`, `sell`, `wait and see` -> `WNS`.
- **Mapping Logic:**
  - When flat: `Buy` -> `PositionIntent.OPEN`, `target_side: LONG`, `OrderType.BUY_TO_OPEN` (`decision_state_manager.py:54`).
  - When long: `Buy` -> `PositionIntent.HOLD`, `target_side: LONG`, `OrderType.NO_ORDER` (no pyramiding) (`decision_state_manager.py:58`).
  - `WNS` -> `PositionIntent.HOLD`, `OrderType.NO_ORDER` (`decision_state_manager.py:50`).
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Sell Action Deprecation:** Agent reports recommending "Sell" or "Underweight" map to `WNS` (no agent-driven exit order). Exits are governed exclusively by risk engine stop-loss / take-profit / time stops.

---

### Section 7.11: Markdown Decision Parser
- **Files & Line Ranges:**
  - `tradingagents/backtesting/markdown_parser.py:18-411` (`MarkdownDecisionParser`)
- **Purpose & Contracts:**
  - Robust regex parsing of unformatted or Markdown LLM outputs into `ExtendedDecision`.
  - Extracts ticker, trade date, last data date, valid-from date, rating, confidence, allocation, stop price, take profit, planned entry price, WNS trigger price, WNS recheck date, time horizon.
- **Parsing Invariants & Fallbacks:**
  - Missing `Decision Valid From` triggers `ExtendedDecision.invalid` with reason `"Missing Decision Valid From. Refuse trading to avoid same-bar execution."` (`markdown_parser.py:82-87`).
  - Multilingual support for Indonesian and English trading report headers.
  - Confidence normalization: if `confidence > 1`, divided by 100.0 (`markdown_parser.py:363-364`).
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Next Review Date Multi-line Regex:** Indonesian report sections (`## TANGGAL REVIEW BERIKUTNYA\n\n**YYYY-MM-DD**`) require multi-line regex matching (`markdown_parser.py:398-406`).
  - [ ] **No Action Inference:** Parser extracts raw metadata only; `futures_action` is never populated by the parser (left to `DecisionStateManager`).

---

### Section 7.12: Cutoff Validator & Point-in-Time (PIT) Anti-Leakage
- **Files & Line Ranges:**
  - `tradingagents/backtesting/cutoff_validator.py:23-192` (`DecisionCutoffValidator`, `LeakageValidationError`)
- **Purpose & Contracts:**
  - Fail-closed audit of every daily decision and snapshot metadata before order execution.
  - Validates:
    - Provider mode == `"snapshot"` (`cutoff_validator.py:89`).
    - `max_ohlcv_date <= trade_date` and `last_data_date <= trade_date` (`cutoff_validator.py:106-115`).
    - `max_news_time.date() < trade_date` (previous-day news cutoff) (`cutoff_validator.py:131-137`).
    - `max_sentiment_time.date() < trade_date` (previous-day sentiment cutoff) (`cutoff_validator.py:172-178`).
    - `max_fundamental_available_date + buffer_days <= trade_date` (`cutoff_validator.py:151-158`).
    - `decision_valid_from > trade_date` (strict next-bar execution) (`cutoff_validator.py:183-190`).
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Fundamental Buffer Days:** `fundamental_buffer_days` defaults to 3 in `LeakageGuardConfig` (`position.py:271`), `cutoff_validator.py:27`, and `walk_forward_runner.py:83`. Ensure all paths pass consistent buffer.
  - [ ] **Timezone Stripping in Date Comparison:** `_parse_dt` replaces `"Z"` with `"+00:00"` (`cutoff_validator.py:45`). Date comparison extracts `.date()`.

---

### Section 7.13: Rolling Data Window & Slicing
- **Files & Line Ranges:**
  - `tradingagents/backtesting/data_window.py:30-37` (`ALLOWED_LOOKBACKS`, `DEFAULT_CALENDAR_BUFFER_DAYS`)
  - `tradingagents/backtesting/data_window.py:80-282` (`compute_window`, `slice_ohlcv`, `slice_news`, `slice_fundamentals`, `slice_sentiment`, `slice_broker_activity`, `assert_window_is_valid`)
- **Purpose & Contracts:**
  - Bounds agent history per decision to a fixed window (e.g. 5, 10, 20, 60, 240 days).
  - Normalizes timestamps to UTC-naive (`_normalize_to_utc`, `data_window.py:68-77`) to prevent timezone comparison bugs.
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Lookback Window Buffer:** `compute_window` adds `buffer_days = 7` to `min_date` for OHLCV to absorb non-trading days/weekends (`data_window.py:100`), but `slice_ohlcv` applies `masked.tail(lookback_days)` (`data_window.py:133`) to ensure exact row count.

---

### Section 7.14: Performance Metrics & Realized Trade Statistics
- **Files & Line Ranges:**
  - `tradingagents/backtesting/metrics.py:23-336` (`MetricsCalculator`)
- **Purpose & Contracts:**
  - Calculates portfolio equity curve returns, CAGR, Max Drawdown, Sharpe, Sortino (using true downside semi-deviation), Calmar, win rate, profit factor, turnover, fee drag, slippage drag, leverage stats, margin calls, and trigger statistics.
- **Formulas & Computations:**
  - Downside Semi-Deviation: `downside_diff = returns.clip(upper=0.0); downside_dev = sqrt((downside_diff ** 2).mean())` (`metrics.py:98-99`).
  - Sortino Ratio: `(daily_mean / downside_dev) * sqrt(252)` (`metrics.py:102`).
  - Fee Drag: `(total_fees / initial_cash) * 100` (`metrics.py:161`).
  - FIFO Lot Matching: `_trade_stats` (`metrics.py:181-239`) matches BUY lots to SELL lots via FIFO to calculate holding period days and realized PnL.
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Sortino Infinite Handling:** If `downside_dev == 0` and `daily_mean > 0`, Sortino is `float("inf")` (`metrics.py:104`). Summary JSON serializer must handle `inf` (`reports.py:96` uses `default=str`).
  - [ ] **Zero Initial Cash Handling:** Returns and drags check `initial_cash > 0` before dividing (`metrics.py:78, 161, 163`).

---

### Section 7.15: Reports & Export Artifacts
- **Files & Line Ranges:**
  - `tradingagents/backtesting/reports.py:35-304` (`BacktestReportGenerator`, `build_leakage_audit`, `build_trigger_stats`)
- **Purpose & Contracts:**
  - Serializes run artifacts into `backtest_results/{ticker}/`:
    - `summary.json`, `trade_log.csv`, `equity_curve.csv`, `decision_log.jsonl`, `position_log.csv`, `margin_log.csv`, `trigger_log.jsonl`, `leakage_audit.json`, `account_state.csv`, `margin_events.csv`, `report.md`, `config.json`.
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Export Config Serializer:** `export_config` uses lambda `getattr(o, "__dict__", str(o))` (`reports.py:162`) to handle nested dataclasses.

---

### Section 7.16: Walk-Forward Runner Orchestration
- **Files & Line Ranges:**
  - `tradingagents/backtesting/walk_forward_runner.py:63-949` (`WalkForwardBacktestRunner`)
- **Purpose & Contracts:**
  - Implements the 10-step walk-forward backtest loop across all trading calendar days:
    1. Load market point.
    2. Execute pending orders at today's open.
    3. Check risk events bar-by-bar (intraday stop/TP).
    4. Mark-to-market portfolio at close.
    5. Build point-in-time snapshot up to current date.
    6. Run LLM agent (Daily Review Agent) or check WNS reanalysis date gate.
    7. Parse decision from Markdown report.
    8. Validate PIT leakage constraints.
    9. Save decision, update static risk levels, map action, evaluate trigger.
    10. Generate orders for T+1 open and queue to broker.
  - Final cleanup: force-closes remaining positions at end-of-backtest date close (`walk_forward_runner.py:202-234`).
- **Mismatch Vectors & Potential Traps:**
  - [ ] **WNS Reanalysis Skip Logic:** If `should_skip_agent` is true (before `_next_reanalysis_date` and price trigger not touched), agent LLM execution is bypassed to save tokens/time (`walk_forward_runner.py:401-417`).
  - [ ] **Immediate Risk Orders vs Morning Execution:** `_check_risk_events` immediately executes stop/TP trades on the current bar (`walk_forward_runner.py:663-692`).
  - [ ] **Last Date Duplicate Mark-to-Market:** End-of-backtest force close pops the duplicate equity curve entry before re-marking to market (`walk_forward_runner.py:223-228`).

---

### Section 7.17: CLI Command, UI & Progress Feedback
- **Files & Line Ranges:**
  - `cli/commands/backtest.py:69-498` (`backtest`)
  - `cli/commands/backtest_prompts.py:1-250` (Prompts, lookback choices)
  - `cli/commands/backtest_tui.py:1-350` (`_BacktestUI`, Rich layouts)
  - `cli/commands/backtest_report.py:1-200` (Results tables)
- **Purpose & Contracts:**
  - User CLI interface for configuring and executing walk-forward backtests with interactive prompts, live Rich terminal layout, watchdog thread, and execution logs.
- **Mismatch Vectors & Potential Traps:**
  - [ ] **Interactive Override Preservation:** `backtest.py:171-175` calls `BacktestEngine.from_dict` instead of `from_yaml` so interactive user prompts (ticker, start date, end date, initial cash) are not overwritten by disk YAML re-reading.
  - [ ] **Watchdog Stuck Detection:** Watchdog thread (`backtest.py:379-414`) detects thread idle > 30s and logs warnings if LLM calls take extended time.

---

## 3. Comprehensive Crosscheck Matrix & Markdown Checklist

```markdown
### Subsystem 7: Backtest Engine, Simulated Broker, Portfolio & Margin Crosscheck Checklist

#### 7.1 Configuration & Defaults
- [ ] Verify `BacktestConfig.lookback_days` default in `position.py:491` matches `data_window.ALLOWED_LOOKBACKS` (240).
- [ ] Verify `engine.py:216-220` enforces 1.0 cash-only margin defaults when YAML omits `margin`.
- [ ] Verify `DecisionMappingConfig.__post_init__` (`position.py:207-214`) enforces `spot_mode=True` and rejects short flags.
- [ ] Verify `resolve_agent_config` in `config_resolver.py:52-72` resolves `TRADINGAGENTS_LLM_PROVIDER` and `TRADINGAGENTS_LLM_MODEL`.

#### 7.2 Dataclass Harmonization & Enums
- [ ] Crosscheck `Trade` in `position.py:726` and `decision_schema.py:178` for field naming compatibility (`slippage_amount` vs `slippage_ticks`).
- [ ] Verify `Position.side` property returns `PositionSide` enum (`position.py:371-378`).
- [ ] Verify `InstrumentSpec.auto_from_ohlcv` precision deduction (`position.py:314-325`).
- [ ] Verify `OrderType` enum rejects legacy reverse / short types during execution (`position.py:29-45`, `broker.py:59-66`).

#### 7.3 Margin & Leverage Calculation
- [ ] Verify `notional_value` uses absolute quantity (`margin_engine.py:28`).
- [ ] Verify `is_intraday_margin_breach` rejects negative quantities with `ValueError` (`margin_engine.py:125-127`).
- [ ] Verify `MarginAccount.margin_call` and `MarginAccount.liquidation_guard` formulas align with PRD §15 (`margin.py:89-105`).

#### 7.4 Portfolio Accounting & Cash Flow
- [ ] Verify `PortfolioV2._apply_open` deducts full notional + buy_fee (`portfolio.py:696`).
- [ ] Verify `PortfolioV2._apply_close` credits notional - sell_fee and calculates realized PnL (`portfolio.py:716-721`).
- [ ] Verify `PortfolioV2.account_equity` formula: `cash + position_value(mark_price)` (`portfolio.py:519`).
- [ ] Verify `PortfolioV2` blocks pyramiding when `not position.is_flat()` (`portfolio.py:640, 692`).

#### 7.5 Broker Execution & Price Limits
- [ ] Verify limit order execution logic bounds fill prices within limit bounds (`broker.py:174-214`).
- [ ] Verify price improvement on Buy limit order opening below limit price (`broker.py:190`).
- [ ] Verify friction calculation includes both slippage and `spread_bps` (`broker.py:204-207`).
- [ ] Verify `execute_pending_orders_immediate` executes stop/TP orders on current bar (`broker.py:77-106`).

#### 7.6 Order Generation & Clamping
- [ ] Verify `_allocation_to_qty` divides by 100 if allocation > 1.0 (`order_generator.py:307-308`).
- [ ] Verify spot cash clamp prevents order notional exceeding available cash divided by `(1 + fee + slippage)` (`order_generator.py:321`).
- [ ] Verify lot-size rounding down via `(qty // lot_size) * lot_size` (`order_generator.py:344`).
- [ ] Verify max leverage clamping restricts exposure to `equity * max_leverage - existing_exposure` (`order_generator.py:360-366`).

#### 7.7 Risk Engine & Static Risk Exits
- [ ] Verify bar-by-bar stop-loss triggers on `low <= stop_price` with fill `min(open, stop_price)` (`risk.py:250-252`).
- [ ] Verify bar-by-bar take-profit triggers on `high >= take_profit` with fill `max(open, take_profit)` (`risk.py:281-283`).
- [ ] Verify stop-loss takes priority over take-profit on ambiguous bars (`risk.py:107-124`).
- [ ] Verify `update_position_risk_levels` prevents ratcheting once initialized (`risk.py:391-394`).
- [ ] Verify `compute_atr` filters data to `date <= current_date` (`risk.py:328`).

#### 7.8 Trigger Evaluator & State Mapping
- [ ] Verify `TriggerEvaluator` gates flat entry when `is_invalid` is True (`trigger_evaluator.py:164`).
- [ ] Verify `SETUP_INVALID_PATTERNS` regex matches English and Indonesian thesis breakdown phrases (`trigger_evaluator.py:21-33`).
- [ ] Verify `DecisionStateManager` normalizes all rating aliases (`overweight`, `hold`, `underweight`, `sell`) strictly to `Buy` or `WNS` (`decision_state_manager.py:23-30`).
- [ ] Verify long position + Buy rating maps to `PositionIntent.HOLD` and `OrderType.NO_ORDER` (`decision_state_manager.py:58`).

#### 7.9 Horizon Evaluation
- [ ] Verify `HorizonEvaluator.evaluate` enforces `stop_loss < entry_price < take_profit` (`horizon_evaluator.py:218-225`).
- [ ] Verify excursion trajectory records MFE, MAE, and close return percentages (`horizon_evaluator.py:269-272`).
- [ ] Verify timezone requirement for `entry_timestamp` (`horizon_evaluator.py:151-153`).

#### 7.10 Decision Parsing & Anti-Leakage
- [ ] Verify `MarkdownDecisionParser` marks decision invalid if `Decision Valid From` is missing (`markdown_parser.py:82-87`).
- [ ] Verify `DecisionCutoffValidator` enforces `max_news_time.date() < trade_date` and `max_sentiment_time.date() < trade_date` (`cutoff_validator.py:131, 172`).
- [ ] Verify `DecisionCutoffValidator` enforces `fundamental_available_date + buffer_days <= trade_date` (`cutoff_validator.py:151-158`).
- [ ] Verify `DecisionCutoffValidator` enforces `decision_valid_from > trade_date` (`cutoff_validator.py:183-190`).

#### 7.11 Performance Metrics & Reporting
- [ ] Verify Sortino ratio calculation uses downside semi-deviation across all N days (`metrics.py:98-102`).
- [ ] Verify FIFO trade matching in `_trade_stats` computes correct holding periods and win/loss streaks (`metrics.py:183-239`).
- [ ] Verify `BacktestReportGenerator` exports all required CSV, JSON, and Markdown artifacts (`reports.py:56-83`).

#### 7.12 Walk-Forward Execution & CLI Integration
- [ ] Verify `WalkForwardBacktestRunner` handles WNS reanalysis date skipping (`walk_forward_runner.py:401-417`).
- [ ] Verify remaining positions are force-closed at the final backtest date close (`walk_forward_runner.py:202-234`).
- [ ] Verify CLI `backtest.py:171` calls `BacktestEngine.from_dict` to preserve interactive prompt values.
- [ ] Verify background thread execution, watchdog heartbeat, and UI live rendering (`backtest.py:358-438`).
```

---

## 4. Handoff Summary
All 21 files comprising Subsystem 7 (`backtest`, `broker`, `portfolio`, `margin`, `risk`, `trigger`, `horizon`, `parser`, `validator`, `metrics`, `reports`, `runner`, `engine`, `cli`) have been mapped at line-level detail. All data structures, interfaces, contracts, formula alignments, and potential mismatch vectors are fully documented above and ready for insertion into `plan.md`.

---

## Subsystem 8: Backtest Walk-Forward Runners & Single-Shot Evaluators

# Subsystem 8: Backtest Runners & Evaluators — Exhaustive Architectural Mapping

## Executive Summary & Subsystem Boundary
Subsystem 8 manages point-in-time (PIT) historical simulation and forward causal evaluation across two primary execution modes:
1. **Multi-Day Walk-Forward Runner (`WalkForwardBacktestRunner` / `BacktestEngine`)**: Simulates day-by-day agent execution, snapshot generation, leakage validation, DSM rating translation, trigger-based gating, pending order queue processing, bar-by-bar risk enforcement (stop-loss, take-profit, time stops), daily mark-to-market settlement, and full accounting / audit metrics.
2. **Single-Shot Forward Horizon Evaluator (`HorizonEvaluator` / `evaluate-signal`)**: Runs a single decision at $T_0$ under strict PIT constraints and tracks causal future forward trajectories ($H \in [1, 252]$ trading days) bar-by-bar, recording MFE/MAE excursions, R:R realization, time-stops, and entry policies (`T1_OPEN`, `T1_LIMIT` via `ASSUMED_AI_ENTRY`).

---

## Mapped Modules & Fine-Grained Sections

```
Subsystem 8: Backtest Runners & Evaluators
├── Section 8.1: Config Resolution & Engine Entry Point
├── Section 8.2: Agent Execution & PIT Runtime Sandboxing
├── Section 8.3: Snapshot Provisioning & Rolling Data Windowing
├── Section 8.4: Anti-Data-Leakage Validation & Cutoff Guards
├── Section 8.5: Markdown Decision Parsing & Schema Normalization
├── Section 8.6: Decision State Manager (DSM) & Order Generation
├── Section 8.7: Trigger-Based Execution Agent (Gating Evaluator)
├── Section 8.8: Simulated Broker, Fill Routing & Execution Friction
├── Section 8.9: Portfolio State & Margin Accounting (PortfolioV2)
├── Section 8.10: Bar-by-Bar Risk Engine & Static Level Protection
├── Section 8.11: Walk-Forward Main Loop & State Transitions
├── Section 8.12: Performance & Margin Metrics Calculation
├── Section 8.13: Reporting, Artifact Persistence & Audit Export
├── Section 8.14: Single-Shot Forward Horizon Evaluator
├── Section 8.15: Single-Shot Result Bundling & Concurrency Locks
├── Section 8.16: CLI Backtest Command & Interactive Prompt Flow
└── Section 8.17: CLI Single-Shot Signal Evaluator & TUI Pipeline
```

---

### Section 8.1: Config Resolution & Engine Entry Point
- **Files & Line References**:
  - `tradingagents/backtesting/engine.py:28-250` (`BacktestEngine`, `from_yaml`, `from_dict`, `_build_config`, `ensure_ohlcv`, `snapshot_provider`, `run`)
  - `tradingagents/backtesting/config_resolver.py:1-120` (`resolve_backtest_config`, fallback priority cascades)
  - `tradingagents/backtesting/position.py:72-297` (`BacktestConfig`, `ExecutionConfig`, `MarginConfig`, `RiskConfig`, `DecisionMappingConfig`, `DataConfig`, `AgentConfig`, `LeakageGuardConfig`, `OutputConfig`)
  - `tradingagents/backtesting/decision_schema.py:1-87` (Legacy config re-exports and backwards compatibility shim)

- **Purpose & Core Contracts**:
  - Unifies YAML files, CLI interactive prompt dictionaries, environment variables, and constructor keyword overrides into strongly typed `BacktestConfig` dataclasses.
  - Exposes `.ensure_ohlcv(auto_fetch=bool)` to verify local data coverage up to `end_date` before entering simulation loops.
  - Instantiates `WalkForwardBacktestRunner` with injected callbacks and configurations.

- **Data Structures / Inputs & Outputs**:
  - Input: YAML path, raw dictionary with optional root key `backtest:`, CLI kwargs (`lookback_days`, `trigger`, `progress_callback`).
  - Output: `BacktestConfig` instance; `.run()` returns `dict[str, Any]` (`summary`, `leakage_audit`, `output_paths`, `margin_events`).

- **Cross-Check Inconsistency Vectors**:
  - `from_yaml` vs `from_dict`: `from_yaml` reads directly from disk, bypassing CLI prompted overrides (e.g. ticker, dates). CLI must call `from_dict` (`cli/commands/backtest.py:171`).
  - Dual Schema Re-exports: `decision_schema.py` re-exports `position.py` configs, but also defines legacy `ParsedDecision`, `Order`, `Trade`, `Position` with different field shapes.
  - Missing Margin Key: Default fallback sets `initial_margin_pct=1.0`, `maintenance_margin_pct=1.0`, `max_leverage=1.0` if `margin` key is missing (`engine.py:215-220`).

- **Checklist**:
  - [ ] `BacktestEngine.from_dict` preserves interactive CLI overrides without reloading stale YAML from disk (`engine.py:180-205`).
  - [ ] Default `margin` block configures spot cash behavior (`initial_margin_pct=1.0`, `maintenance_margin_pct=1.0`, `max_leverage=1.0`) (`engine.py:216-220`).
  - [ ] `ensure_ohlcv` validates date coverage against `end_date` before triggering yfinance fetch (`engine.py:101-104`).
  - [ ] `allowed_lookbacks` validation strictly enforces `ALLOWED_LOOKBACKS` constraint (`engine.py:108-110`).

---

### Section 8.2: Agent Execution & PIT Runtime Sandboxing
- **Files & Line References**:
  - `tradingagents/backtesting/agent_runner.py:67-264` (`TradingAgentsRunner`, `run`, `save_report`, `_validate_safe_agent_config`, `_safe_agent_runtime_config`, `_assert_runtime_config_is_safe`)
  - `tradingagents/backtesting/agent_runner.py:29-65` (`_derive_rating_from_text`, `_RATING_NORMALIZATION`, retry loop constants)
  - `tradingagents/backtesting/agent_callbacks.py:1-231` (`BacktestAgentCallback`, `KNOWN_NODES`, `on_chain_start`, `on_chain_end`, `on_chain_error`, `_emit`)

- **Purpose & Core Contracts**:
  - Executes the LangGraph agent team per backtest day under strict Point-In-Time guards.
  - Enforces hard assertions: `backtest_mode=True`, `memory_enabled=False`, `web_search_enabled=False`, `news_provider="snapshot"`, `asset_type="stock"`.
  - Retries failed LLM calls up to 5 times with exponential/linear back-off delay ($20\text{s} \times \text{attempt}$).
  - Emits real-time node duration callbacks (`BacktestAgentCallback`) filtering to `KNOWN_NODES`.

- **Data Structures / Inputs & Outputs**:
  - Input: `ticker: str`, `trade_date: str`, `snapshot: DataSnapshot`, `portfolio: Optional[Portfolio]`, `day_idx`, `n_days`.
  - Output: `str` Markdown report saved to `reports/<TICKER>/<DATE>/complete_report.md`.

- **Cross-Check Inconsistency Vectors**:
  - Portfolio State Injection: Injects `account_cash`, `account_equity`, `position_qty`, `position_avg_price`, `margin_available` into agent config at $T_0$ (`agent_runner.py:249-262`).
  - Rating Regex Extraction: Unlabeled prose defaults to `WNS` if explicit `Rating:` regex fails (`agent_runner.py:49-65`).
  - Error Propagation: If all 5 retries fail, raises exception to abort day and prevent silent fake-pass execution (`agent_runner.py:141-143`).

- **Checklist**:
  - [ ] PIT invariant assertions execute before LangGraph invocation (`agent_runner.py:93, 265-300`).
  - [ ] Node timing callbacks attach without breaking CLI when running headless or in test fixtures (`agent_runner.py:78-82`, `agent_callbacks.py:61-63`).
  - [ ] Report files are saved with UTF-8 encoding in `<reports_root>/<ticker>/<trade_date>/complete_report.md` (`agent_runner.py:150-153`).
  - [ ] Retry back-off sleep ($20\text{s} \times \text{attempt}$) logs explicit attempt failures and errors (`agent_runner.py:128-142`).

---

### Section 8.3: Snapshot Provisioning & Rolling Data Windowing
- **Files & Line References**:
  - `tradingagents/backtesting/snapshot_provider.py:57-71` (`DataSnapshot` dataclass)
  - `tradingagents/backtesting/snapshot_provider.py:73-200` (`SnapshotDataProvider`, `load_full_ohlcv`, `_get_previous_trading_day`, `_normalize_ohlcv_columns`)
  - `tradingagents/backtesting/snapshot_provider.py:201-682` (`create_snapshot`, `get_market_point`, `load_instrument_spec`)
  - `tradingagents/backtesting/data_window.py:29-112` (`ALLOWED_LOOKBACKS`, `WindowCutoffs`, `compute_window`, `DEFAULT_CALENDAR_BUFFER_DAYS`)
  - `tradingagents/backtesting/data_window.py:114-282` (`slice_ohlcv`, `slice_news`, `slice_fundamentals`, `slice_sentiment`, `slice_broker_activity`, `_filter_records_by_field`)

- **Purpose & Core Contracts**:
  - Loads flat snapshot assets (`ohlcv.csv`, `news.json`, `fundamentals.json`, `sentiment.json`, `broker_activity.json`) from `data/<TICKER>/`.
  - Applies rolling `lookback_days` cutoffs (e.g. 5, 10, 20, 40, 60, 80, 100, 120, 240, 512).
  - Hardens temporal cutoffs: `news`, `sentiment`, `broker_activity` truncated to `previous_day` trading close (or `trade_date 16:30:00`), `fundamentals` delayed by `fundamental_buffer_days` (default 3 days).

- **Data Structures / Inputs & Outputs**:
  - Input: `symbol`, `trade_date`, `config: BacktestConfig`.
  - Output: `DataSnapshot` with filtered DataFrames/lists and `SnapshotMetadata`.

- **Cross-Check Inconsistency Vectors**:
  - Lookback Allowlist: `lookback_days` must strictly be in `ALLOWED_LOOKBACKS = (None, 5, 10, 20, 40, 60, 80, 100, 120, 240, 512)` (`data_window.py:30-32`).
  - News Leakage Guard: `previous_day` strategy uses the previous trading day's date as the upper bound for news publication (`data_window.py:181-193`).
  - Fundamental Publication Buffer: `available_date + fundamental_buffer_days <= trade_date` enforced to prevent early leakage of quarterly reports (`data_window.py:200-240`).

- **Checklist**:
  - [ ] `SnapshotDataProvider` loads OHLCV, sorts ascending by date, and normalizes column headers (`snapshot_provider.py:177-198`).
  - [ ] `slice_ohlcv` takes exact tail $N$ rows where `date <= trade_date` (`data_window.py:114-134`).
  - [ ] News, sentiment, and broker activity respect `previous_day` upper bound timestamp (`data_window.py:176-194`).
  - [ ] `SnapshotMetadata` correctly stores `max_ohlcv_date`, `max_news_time`, and `max_fundamental_available_date` (`snapshot_provider.py:57-71`).

---

### Section 8.4: Anti-Data-Leakage Validation & Cutoff Guards
- **Files & Line References**:
  - `tradingagents/backtesting/cutoff_validator.py:19-39` (`LeakageValidationError`, `DecisionCutoffValidator`, `_fail`, `_pass`)
  - `tradingagents/backtesting/cutoff_validator.py:47-192` (`validate`, `_validate_required_fields`, `_validate_provider`, `_validate_ohlcv`, `_validate_news`, `_validate_fundamentals`, `_validate_sentiment`, `_validate_next_bar_execution`)

- **Purpose & Core Contracts**:
  - Validates zero future information leakage in decisions and snapshot payloads before order generation.
  - Enforces `fail_on_future_data=True` to immediately raise `LeakageValidationError` when any check fails.
  - Verifies:
    1. Provider mode is strictly `"snapshot"`.
    2. Snapshot `max_ohlcv_date <= trade_date` and `last_data_date <= trade_date`.
    3. `max_news_time.date() < trade_date` (for previous-day policy).
    4. `max_fundamental_available_date + buffer_days <= trade_date`.
    5. `max_sentiment_time.date() < trade_date`.
    6. `decision_valid_from > trade_date` (next-bar execution invariant).

- **Data Structures / Inputs & Outputs**:
  - Input: `decision: ParsedDecision`, `snapshot_metadata: SnapshotMetadata`.
  - Output: `dict[str, str]` audit mapping (e.g. `{"ohlcv_cutoff": "PASSED", ...}`).

- **Cross-Check Inconsistency Vectors**:
  - Fail-Closed vs Warning: If `fail_on_future_data=True`, any timestamp irregularity terminates the backtest (`cutoff_validator.py:35-36`).
  - Same-Day Execution Ban: `decision_valid_from <= trade_date` fails `next_bar_execution` check (`cutoff_validator.py:183-191`).

- **Checklist**:
  - [ ] `DecisionCutoffValidator` verifies all 7 critical PIT invariants (`cutoff_validator.py:56-64`).
  - [ ] `fail_on_future_data` raises `LeakageValidationError` when violation detected (`cutoff_validator.py:35-36`).
  - [ ] `next_bar_execution` rejects decisions attempting same-day execution (`cutoff_validator.py:183-191`).
  - [ ] Fundamental availability buffer (3 days) is added to `available_date` before comparing to `trade_date` (`cutoff_validator.py:140-160`).

---

### Section 8.5: Markdown Decision Parsing & Schema Normalization
- **Files & Line References**:
  - `tradingagents/backtesting/markdown_parser.py:18-97` (`MarkdownDecisionParser`, `parse_file`, `parse_text`, rating regex patterns)
  - `tradingagents/backtesting/markdown_parser.py:98-200` (`agent_rating`, `confidence`, `allocation_pct`, `reduce_pct`, `leverage`, `stop_price`, `take_profit`, `planned_entry_price`, `wns_trigger_price`, `wns_recheck_date`, `time_horizon_days`)
  - `tradingagents/backtesting/markdown_parser.py:201-411` (Regex extractors: `_extract_pct`, `_extract_float`, `_extract_int`, `_extract_text`, `_upper_bound_horizon_days`)
  - `tradingagents/backtesting/position.py:461-550` (`ExtendedDecision`, `ParsedDecision` PRD §9 dataclass)

- **Purpose & Core Contracts**:
  - Parses Markdown research reports from the PM agent into structured `ExtendedDecision` objects.
  - Normalizes multi-lingual ratings: `Buy`, `Overweight`, `Beli` $\to$ `BUY`; `Sell`, `Hold`, `WNS`, `Underweight`, `Tahan`, `Jual` $\to$ `WNS`.
  - Extracts numeric trade parameters: `stop_price`, `take_profit`, `planned_entry_price`, `wns_trigger_price`, `wns_recheck_date`, `confidence`, `allocation_pct`.
  - Adheres to PRD §9: parser does NOT infer action; it extracts raw parameters and delegates intent mapping to DSM.

- **Data Structures / Inputs & Outputs**:
  - Input: Report Markdown text string or file path, fallback metadata.
  - Output: `ExtendedDecision` instance (with `.valid=False` and `.invalid_reason` on failure).

- **Cross-Check Inconsistency Vectors**:
  - Percentage Normalization: `confidence` and `allocation_pct` can appear as `10%`, `0.10`, or `10`. `_extract_pct` normalizes values $> 1.0$ by dividing by 100.0 (`markdown_parser.py:101-118`).
  - Time Horizon String to Int: Labels like `"2-3 weeks"` or `"10 hari"` converted to integer trading days via `_upper_bound_horizon_days` (`markdown_parser.py:182-198`).
  - Indonesian Keyword Support: Regex supports bilingual tokens (`cut loss`, `alokasi`, `target harga`, `harga masuk`, `tesis rusak`).

- **Checklist**:
  - [ ] Parser extracts canonical ratings (`BUY` vs `WNS`) from Indonesian and English patterns (`markdown_parser.py:19-23, 89-100`).
  - [ ] Values $> 1.0$ for percentage fields (`confidence`, `allocation_pct`) are scaled to $[0.0, 1.0]$ (`markdown_parser.py:101-119`).
  - [ ] Unparseable reports generate `ExtendedDecision.invalid()` rather than raising uncaught exceptions (`markdown_parser.py:69-96`).
  - [ ] WNS price triggers (`wns_trigger_price`) and recheck dates (`wns_recheck_date`) are parsed into `ExtendedDecision` (`markdown_parser.py:164-181`).

---

### Section 8.6: Decision State Manager (DSM) & Order Generation
- **Files & Line References**:
  - `tradingagents/backtesting/decision_state_manager.py:1-111` (`DecisionStateManager`, `resolve`, `map`, `_canonical_rating`, `RATING_BUY`, `RATING_WNS`)
  - `tradingagents/backtesting/order_generator.py:38-103` (`OrderGenerator`, `decide`, new PRD-compliant entry point)
  - `tradingagents/backtesting/order_generator.py:104-251` (`generate`, `_generate_from_old_schema`, `_generate_from_new_schema`)
  - `tradingagents/backtesting/order_generator.py:252-411` (`_build_orders`, `_allocation_to_qty`, `_round_to_lot`, `_cap_by_leverage`)

- **Purpose & Core Contracts**:
  - `DecisionStateManager`: State machine mapping `(rating, current_position_side, allow_new_position)` $\to$ `(PositionIntent, target_side, OrderType)`.
    - `FLAT + BUY` $\to$ `(PositionIntent.OPEN, "LONG", OrderType.BUY_TO_OPEN)`
    - `LONG + BUY` $\to$ `(PositionIntent.HOLD, "LONG", OrderType.NO_ORDER)` (no pyramiding)
    - `* + WNS` $\to$ `(PositionIntent.HOLD, current_side, OrderType.NO_ORDER)` (static exits own all selling)
  - `OrderGenerator`: Translates `ExtendedDecision` into executable `Order` objects with lot rounding (`lot_size`), cash constraints (`max_entry_pct`), and leverage capping.

- **Data Structures / Inputs & Outputs**:
  - Input: `ExtendedDecision`, `Position`, `equity: float`, `cash: float`, `reference_price: float`.
  - Output: `list[Order]` (typically 1 `BUY_TO_OPEN` order or empty list `[]`).

- **Cross-Check Inconsistency Vectors**:
  - Pyramiding Block: When position is already `LONG`, `BUY` decision generates `NO_ORDER` (`decision_state_manager.py:55-58`).
  - Cash vs Equity Sizing: Uses `initial_entry_pct` of equity, constrained by `max_entry_pct` and available free cash (`order_generator.py:287-350`).
  - Lot Size Rounding: Quantities floored or rounded to integer multiples of `lot_size` (e.g. 100 shares for Indonesia IDX `.JK` stocks) (`order_generator.py:352-375`).

- **Checklist**:
  - [ ] DSM maps `FLAT + BUY` to `BUY_TO_OPEN` and `LONG + BUY` to `NO_ORDER` (`decision_state_manager.py:51-58`).
  - [ ] OrderGenerator rejects non-`BUY_TO_OPEN` orders emitted by agent ratings (`order_generator.py:67-73`).
  - [ ] Order quantity is capped by available cash, `max_entry_pct`, and leverage bounds (`order_generator.py:85-91, 380-410`).
  - [ ] Order execution date set to $T+1$ trading day (`order_generator.py:78-81`).

---

### Section 8.7: Trigger-Based Execution Agent (Gating Evaluator)
- **Files & Line References**:
  - `tradingagents/backtesting/trigger_evaluator.py:36-78` (`TriggerConfig`, `TriggerResult`, regex patterns `SETUP_INVALID_PATTERNS`)
  - `tradingagents/backtesting/trigger_evaluator.py:79-173` (`TriggerEvaluator`, `evaluate`, master switch, condition aggregation)
  - `tradingagents/backtesting/trigger_evaluator.py:174-342` (`_price_level_touched`, `_tp_sl_hit`, `_setup_invalid`, `_rr_deteriorated`, `_compute_real_rr_drop`, `_strong_exit_signal`, `_better_candidate`, `_entry_condition_changed`, `_rating_confirmed`)

- **Purpose & Core Contracts**:
  - Gates order emission: an agent rating produces an active order ONLY if at least one trigger condition evaluates to `True`.
  - If `triggered=False`, overrides decision action to `NO_ORDER` (logged as skipped order).
  - Conditions evaluated:
    1. `tp_sl_hit`: Risk engine hit stop-loss or take-profit.
    2. `setup_invalid`: Thesis invalidation regex matches $\ge$ keyword threshold (default 2).
    3. `rr_deteriorated`: Real R:R drop or confidence drop $\ge 30\%$.
    4. `better_candidate`: Confidence ratio $\ge 1.5\times$ prior day and $\ge 0.60$.
    5. `entry_condition_changed`: Rating changed from `WNS` to `BUY` while flat.
    6. `rating_confirmed`: `BUY` confirmed across consecutive sessions.
    7. `price_level_touched`: Current bar low $\le$ `planned_entry_price`.

- **Data Structures / Inputs & Outputs**:
  - Input: `today_decision: ExtendedDecision`, `prev_decision: Optional[ExtendedDecision]`, `current_position: Position`, `risk_order: Optional[Order]`, `current_equity`, `current_bar`.
  - Output: `TriggerResult(triggered: bool, reasons: list[str], details: dict[str, Any])`.

- **Cross-Check Inconsistency Vectors**:
  - Exit Trigger Invariance: Agent signals never trigger exits (`_strong_exit_signal` hardcoded to return `False` — static risk engine owns exits) (`trigger_evaluator.py:292-300`).
  - Invalidation Overrides Entry: For flat positions, `setup_invalid` or `rr_deteriorated` forces `triggered=False` even if entry conditions fired (`trigger_evaluator.py:149-165`).

- **Checklist**:
  - [ ] Trigger evaluator returns `triggered=True` with reason `"triggers_disabled"` when `enabled=False` (`trigger_evaluator.py:103-108`).
  - [ ] `_price_level_touched` verifies `bar.low <= planned_entry_price` for long entries (`trigger_evaluator.py:177-195`).
  - [ ] `_setup_invalid` checks bilingual invalidation keywords with threshold count (`trigger_evaluator.py:204-215`).
  - [ ] `_compute_real_rr_drop` computes distance between entry, stop, and take-profit (`trigger_evaluator.py:253-291`).

---

### Section 8.8: Simulated Broker, Fill Routing & Execution Friction
- **Files & Line References**:
  - `tradingagents/backtesting/broker.py:35-76` (`SimulatedBroker`, `add_pending_orders`, `cancel_orders_before`)
  - `tradingagents/backtesting/broker.py:77-106` (`execute_pending_orders_immediate` for immediate intraday risk fills)
  - `tradingagents/backtesting/broker.py:110-147` (`execute_pending_orders` for standard next-session open fills)
  - `tradingagents/backtesting/broker.py:148-262` (`_execute_order`, limit price improvement, percentage fees, slippage, spread friction)

- **Purpose & Core Contracts**:
  - Simulates exchange fill dynamics with realistic market frictions.
  - Supports two execution paths:
    1. **Immediate Execution (`execute_pending_orders_immediate`)**: Fills intraday risk orders (stop-loss, take-profit, forced liquidations) at the exact trigger price on the current bar.
    2. **Deferred Next-Session Execution (`execute_pending_orders`)**: Fills pending agent orders matching `execution_date == current_date` at market open.
  - Applies asymmetric transaction fees (`buy_fee=0.0015`, `sell_fee=0.0025`), percentage slippage (`slippage=0.001`), and half-spread (`spread_bps`).
  - Handles limit order price improvement: `fill_price = min(open_price, limit_price)` for buys when market opens lower.

- **Data Structures / Inputs & Outputs**:
  - Input: `date: str`, `market_point: MarketPoint`, `portfolio: Portfolio`, `spec: InstrumentSpec`.
  - Output: `list[Trade]` objects applied to portfolio state.

- **Cross-Check Inconsistency Vectors**:
  - Limit Price Bounds: A limit buy order is unfilled (`status="UNFILLED"`) if `market_point.low > limit_price` (`broker.py:177-186`).
  - Slippage Model: Supports both percentage slippage (`config.slippage`) and tick slippage (`config.tick_slippage * tick_size`) fallback (`broker.py:196-203`).
  - Asymmetric Fee Drag: Default buys charged 15 bps, sells charged 25 bps (standard Indonesia IDX equity broker/tax rates) (`broker.py:221-230`).

- **Checklist**:
  - [ ] Broker rejects non-`BUY_TO_OPEN` and non-`SELL_TO_CLOSE` orders with explicit rejection reasons (`broker.py:59-66, 159-165`).
  - [ ] Risk orders execute immediately on the current bar without deferral to next session (`broker.py:77-106`).
  - [ ] Limit buy fill price absorbs slippage up to `limit_price` without violating limit ceiling (`broker.py:210-215`).
  - [ ] Legacy tick fees fall back cleanly to percentage fees when configured (`broker.py:196-230`).

---

### Section 8.9: Portfolio State & Margin Accounting (PortfolioV2)
- **Files & Line References**:
  - `tradingagents/backtesting/portfolio.py:55-105` (`Portfolio`, `PortfolioV2`, init, position state, margin parameters)
  - `tradingagents/backtesting/portfolio.py:106-188` (`has_position`, `is_long`, `is_flat`, `account_equity`, `margin_used`, `margin_available`, `leverage`, `margin_util`)
  - `tradingagents/backtesting/portfolio.py:189-285` (`apply_trade`, `_apply_open`, `_apply_close`, daily settlement logic)
  - `tradingagents/backtesting/portfolio.py:286-817` (`mark_to_market`, `PortfolioSnapshot`, `PortfolioV2` PRD §7.3 state tracking)
  - `tradingagents/backtesting/margin.py:1-148` (`MarginAccount`, `MarginResult`, `initial_margin`, `maintenance_margin`, `liquidation_guard`, `margin_call`)
  - `tradingagents/backtesting/margin_engine.py:1-185` (`notional_value`, `initial_margin`, `maintenance_margin`, `excess_margin`, `is_margin_call`, `is_intraday_margin_breach`, `leverage`, `margin_utilization`)

- **Purpose & Core Contracts**:
  - Maintains strict double-entry ledger of cash, equity, position quantity, average entry price, realized PnL, fees, and margin posted.
  - Supports spot cash accounting (`initial_margin_pct=1.0`) and margin simulation (`initial_margin_pct=0.5`, `maintenance_margin_pct=0.35`).
  - `mark_to_market(date, close_price)` computes daily unrealized PnL, account equity, leverage, excess margin, and drawdown peak tracking.

- **Data Structures / Inputs & Outputs**:
  - Input: `Trade` objects on fill events; `close_price` on daily settlement.
  - Output: `PortfolioSnapshot` appended to `equity_curve`.

- **Cross-Check Inconsistency Vectors**:
  - Realized PnL Source of Truth: Realized PnL calculated exclusively inside `Portfolio._apply_close` via lot matching rather than estimated in the broker (`portfolio.py:245-285`).
  - Single Position Invariant: Attempting `_apply_open` while `quantity > 0` raises `ValueError` (no pyramiding allowed) (`portfolio.py:230-234`).
  - Zero/Negative Equity Handling: Peak equity tracking handles drawdowns $\ge 100\%$ without crashing or dividing by zero (`portfolio.py:320-360`).

- **Checklist**:
  - [ ] Cash balance correctly decremented by trade fees on every fill (`portfolio.py:213-214`).
  - [ ] `_apply_close` correctly releases posted margin and updates lifetime realized PnL (`portfolio.py:245-280`).
  - [ ] `account_equity` equals `cash + unrealized_pnl` across both flat and open position states (`portfolio.py:130-134`).
  - [ ] `is_intraday_margin_breach` evaluates worst-case intraday low against maintenance margin requirement (`margin_engine.py:111-142`).

---

### Section 8.10: Bar-by-Bar Risk Engine & Static Level Protection
- **Files & Line References**:
  - `tradingagents/backtesting/risk.py:22-31` (`RiskEvent` dataclass)
  - `tradingagents/backtesting/risk.py:32-128` (`RiskEngine`, `check_bar`, 8-step PRD §11 priority cascade)
  - `tradingagents/backtesting/risk.py:129-294` (`_check_hard_risk`, `_check_liquidation`, `_check_stop_loss`, `_check_take_profit`, `_force_close_position`)
  - `tradingagents/backtesting/risk.py:316-350` (`compute_atr`, rolling True Range calculation)
  - `tradingagents/backtesting/risk.py:352-425` (`update_position_risk_levels`, static invariant enforcement)

- **Purpose & Core Contracts**:
  - Evaluates intraday bar extremes ($O, H, L, C$) against position risk thresholds.
  - Priority Execution Order (PRD §11):
    1. Hard Risk (max single trade loss / max portfolio loss).
    2. Liquidation / Margin Call guard.
    3. Stop-Loss (evaluates $L \le \text{stop\_price}$; gap-down fill at $\min(O, \text{stop})$).
    4. Take-Profit (evaluates $H \ge \text{take\_profit}$; gap-up fill at $\max(O, \text{take\_profit})$).
  - Intraday Ambiguity Rule: If both stop-loss and take-profit are breached on the same bar, Stop-Loss always takes precedence (conservative simulation principle).
  - Static Level Protection: Once initialized at position entry, `stop_price` and `take_profit` are immutable invariants; subsequent daily reviews CANNOT widen or ratchet stops.

- **Data Structures / Inputs & Outputs**:
  - Input: `date: str`, `bar: dict`, `position: Position`, `equity: float`, margin parameters.
  - Output: `tuple[Optional[Order], list[RiskEvent]]`.

- **Cross-Check Inconsistency Vectors**:
  - ATR Stop Multiplier R:R Guarantee: Default ATR multipliers (`atr_stop_multiplier=1.5`, `atr_tp_multiplier=3.0`) guarantee planned $R:R \ge 2.0$ (`position.py:174-176`).
  - Gap Open Slippage: If bar open gaps below stop loss ($O \le \text{stop}$), fill price is the open price $O$, not the stop price (`risk.py:250-255`).
  - Lookahead Invariant in ATR: `compute_atr` filters OHLCV to `date <= current_date` to prevent future bar leakage (`risk.py:326-330`).

- **Checklist**:
  - [ ] Conservative ambiguity resolution: stop-loss executes before take-profit when both hit on the same bar (`risk.py:107-124`).
  - [ ] Gap-down executions fill at market open price when $O < \text{stop\_price}$ (`risk.py:251-253`).
  - [ ] `update_position_risk_levels` prevents ratcheting or widening of established stop levels (`risk.py:389-395`).
  - [ ] `compute_atr` slices data strictly at `current_date` to prevent forward data leakage (`risk.py:326-330`).

---

### Section 8.11: Walk-Forward Main Loop & State Transitions
- **Files & Line References**:
  - `tradingagents/backtesting/walk_forward_runner.py:63-177` (`WalkForwardBacktestRunner`, `__init__`, `_init_portfolio`)
  - `tradingagents/backtesting/walk_forward_runner.py:178-237` (`run`, main loop across trading days, force close at backtest end)
  - `tradingagents/backtesting/walk_forward_runner.py:266-417` (`_process_day`, Steps 0-4: market point, pending orders, risk, MTM, snapshot, WNS gating)
  - `tradingagents/backtesting/walk_forward_runner.py:418-602` (`_process_day`, Steps 5-9: agent run, parse, leakage check, DSM map, trigger eval, order generation)
  - `tradingagents/backtesting/walk_forward_runner.py:603-775` (`_check_risk_events`, `_time_stop_due`, `_build_force_close`, logging helpers)
  - `tradingagents/backtesting/walk_forward_runner.py:776-880` (`_load_initial_report_if_any`, `_update_risk_levels`, `_safe_next_trading_day`)

- **Purpose & Core Contracts**:
  - Orchestrates the 10-step daily walk-forward cycle:
    1. Execute pending orders at today's open ($T+1$ fills).
    2. Check bar-by-bar risk events (stops, targets, liquidations).
    3. Mark-to-market settlement at close.
    4. Construct daily snapshot up to current date.
    5. Evaluate WNS Price / Time Gate skips (conditional reanalysis).
    6. Run Daily Review Agent (LangGraph).
    7. Parse report into structured `ExtendedDecision`.
    8. Validate PIT leakage invariants.
    9. Run Trigger-Based Execution Agent.
    10. Generate orders for next session open.
  - End-of-backtest teardown: automatically force-closes open inventory on the final bar at close price.

- **Data Structures / Inputs & Outputs**:
  - Input: `BacktestConfig`, `TradingAgentsRunner`, `progress_callback`.
  - Output: `dict[str, Any]` (`summary`, `leakage_audit`, `output_paths`, `margin_events`).

- **Cross-Check Inconsistency Vectors**:
  - Open vs Close Risk Evaluation: Position marked to open before bar risk check to eliminate 1-day lag in unrealized PnL calculation (`walk_forward_runner.py:335-337`).
  - WNS Catalyst Skip: If `_next_reanalysis_date` is active and `wns_trigger_price` was not touched, skips agent execution to conserve LLM tokens while holding flat cash (`walk_forward_runner.py:401-417`).
  - Final Day Mark-to-Market: Pops duplicate equity curve record when closing final position on the last trading day (`walk_forward_runner.py:223-228`).

- **Checklist**:
  - [ ] Order execution occurs at bar open before daily agent analysis runs (`walk_forward_runner.py:288-296`).
  - [ ] Risk-triggered immediate fills clear pending WNS triggers and reanalysis gates (`walk_forward_runner.py:669-673`).
  - [ ] Final bar force-close flattens inventory and updates final equity curve record (`walk_forward_runner.py:202-234`).
  - [ ] Time-stop check forces position liquidation when `holding_days >= max_holding_days` (`walk_forward_runner.py:636-638, 695-707`).

---

### Section 8.12: Performance & Margin Metrics Calculation
- **Files & Line References**:
  - `tradingagents/backtesting/metrics.py:23-106` (`MetricsCalculator`, `calculate`, CAGR, Sharpe, Sortino, Downside Semi-Deviation, Drawdown)
  - `tradingagents/backtesting/metrics.py:107-168` (`MetricsCalculator`, margin metrics, alpha, exposure time, base metrics assembly)
  - `tradingagents/backtesting/metrics.py:169-180` (`_trigger_metrics_block`, trigger hit rate, rating distributions)
  - `tradingagents/backtesting/metrics.py:181-253` (`_trade_stats`, FIFO lot matching, win rate, profit factor, holding periods, consecutive streaks)
  - `tradingagents/backtesting/metrics.py:254-336` (`_empty_metrics`, `_augment_with_margin_metrics`)

- **Purpose & Core Contracts**:
  - Ingests `equity_curve` DataFrame, `trades` list, and `margin_events` to compute standard and institutional quant metrics.
  - Implements True Downside Semi-Deviation (root-mean-square of negative returns relative to 0 across all $N$ days) for Sortino Ratio.
  - Performs FIFO lot matching on trades to compute realized win rate, profit factor, average gain/loss, turnover, fee drag, and slippage drag.
  - Computes margin-specific analytics: average/max leverage, margin utilization %, margin calls, and liquidation counts.

- **Data Structures / Inputs & Outputs**:
  - Input: `equity_curve: pd.DataFrame`, `trades: list[Trade]`, `initial_cash: float`, `benchmark_curve: Optional[pd.DataFrame]`, `margin_events: list[MarginEvent]`, `trigger_stats: dict`.
  - Output: `dict[str, Any]` metrics dictionary.

- **Cross-Check Inconsistency Vectors**:
  - Downside Deviation Zero Floor: Downside deviation clips positive returns to 0.0 and calculates RMS over all days ($N$), avoiding biased sample subsets (`metrics.py:98-105`).
  - Infinity Handling: If total gross losses are 0, `profit_factor` returns `float("inf")` if gross profit $>0$, else `0.0` (`metrics.py:225-226`).
  - CAGR Edge Cases: Sets `cagr = -1.0` if `final_equity <= 0` (`metrics.py:84-86`).

- **Checklist**:
  - [ ] FIFO lot matching accurately calculates realized PnL for multi-lot trades (`metrics.py:181-220`).
  - [ ] Fee drag and slippage drag are expressed as percentages of initial cash (`metrics.py:160-163`).
  - [ ] True downside semi-deviation is calculated against target 0 across all trading days (`metrics.py:97-105`).
  - [ ] Alpha % correctly subtracts benchmark return from total strategy return (`metrics.py:110-116`).

---

### Section 8.13: Reporting, Artifact Persistence & Audit Export
- **Files & Line References**:
  - `tradingagents/backtesting/reports.py:35-84` (`BacktestReportGenerator`, `export_all`, output paths dictionary)
  - `tradingagents/backtesting/reports.py:85-172` (`export_trigger_log`, `export_summary`, `export_trade_log`, `export_equity_curve`, `export_decision_log`, `export_account_state`, `export_margin_events`, `export_leakage_audit`, `export_position_log`, `export_margin_log`, `export_config`)
  - `tradingagents/backtesting/reports.py:173-259` (`export_markdown_report`, Markdown summary rendering, margin table formatting)
  - `tradingagents/backtesting/reports.py:261-304` (`build_leakage_audit`, `build_trigger_stats` aggregator)
  - `tradingagents/backtesting/decision_store.py:1-41` (`DecisionStore`, `save`, `all`, `latest_before`)

- **Purpose & Core Contracts**:
  - Persists all backtest artifacts into `backtest_results/<TICKER>/`.
  - Generates 11 standardized artifact files:
    1. `summary.json`: Top-level performance and execution summary.
    2. `trade_log.csv`: Chronological fill records with fees and slippage.
    3. `equity_curve.csv`: Daily mark-to-market equity snapshots.
    4. `position_log.csv`: Daily position sizes, marks, unrealized PnL.
    5. `margin_log.csv`: Daily leverage and margin utilization.
    6. `decision_log.jsonl`: JSON Lines log of parsed decisions.
    7. `trigger_log.jsonl`: JSON Lines log of trigger evaluations.
    8. `leakage_audit.json`: Audit log of PIT validation checks.
    9. `account_state.csv`: Daily cash and equity accounting.
    10. `margin_events.csv`: Margin calls, liquidations, and risk events.
    11. `report.md`: Formatted Markdown report.
    12. `config.json`: Serialized backtest configuration.

- **Data Structures / Inputs & Outputs**:
  - Input: `summary`, `portfolio`, `decisions`, `leakage_audit`, `margin_events`, logs.
  - Output: `dict[str, str]` mapping artifact names to their absolute file paths.

- **Cross-Check Inconsistency Vectors**:
  - Atomic Directory Creation: Calls `ensure_dir` on output directories before writing (`reports.py:38-40`).
  - JSON Serialization: Uses `default=str` and `default=lambda o: getattr(o, "__dict__", str(o))` to prevent serialization crashes on date/enum objects (`reports.py:96, 162`).

- **Checklist**:
  - [ ] All 11 export methods write valid files when respective config toggles are enabled (`reports.py:56-83`).
  - [ ] `build_leakage_audit` marks status as `PASSED` only if all 7 critical checks passed (`reports.py:261-280`).
  - [ ] `build_trigger_stats` calculates trigger hit rate and rating distributions (`reports.py:282-304`).
  - [ ] Markdown report contains complete Performance Summary and Margin Activity tables (`reports.py:173-258`).

---

### Section 8.14: Single-Shot Forward Horizon Evaluator
- **Files & Line References**:
  - `tradingagents/backtesting/horizon_evaluator.py:16-24` (`EvaluationOutcome` enum: `HIT_TAKE_PROFIT`, `HIT_STOP_LOSS`, `HIT_TIME_STOP`, `EXPIRED`, `NO_ORDER`, `INSUFFICIENT_DATA`, `NO_FILL`)
  - `tradingagents/backtesting/horizon_evaluator.py:26-76` (`DailyExcursionBar`, `EvaluationResult` dataclasses)
  - `tradingagents/backtesting/horizon_evaluator.py:77-106` (`_no_order` helper)
  - `tradingagents/backtesting/horizon_evaluator.py:107-226` (`HorizonEvaluator.evaluate`, date validation, entry policies `T1_OPEN` vs `ASSUMED_AI_ENTRY`, limit fills)
  - `tradingagents/backtesting/horizon_evaluator.py:227-311` (`HorizonEvaluator.evaluate`, forward loop, MFE/MAE tracking, conservative same-bar collision, terminal result assembly)

- **Purpose & Core Contracts**:
  - Evaluates a single causal BUY signal generated at $T_0$ across the future $H$ trading days ($1 \le H \le 252$).
  - Simulates bar-by-bar forward price trajectory:
    - Calculates daily unrealized return, MFE (Max Favorable Excursion), MAE (Max Adverse Excursion).
    - Checks barrier hits: `HIT_TAKE_PROFIT` ($H_t \ge \text{TP}$), `HIT_STOP_LOSS` ($L_t \le \text{SL}$), `HIT_TIME_STOP` ($t \ge \text{max\_holding\_days}$), or `EXPIRED`.
  - Supports Entry Policies:
    - `T1_OPEN`: Fills at $T+1$ market open.
    - `ASSUMED_AI_ENTRY` (`T1_LIMIT`): Fills at `planned_entry_price` if $L_1 \le \text{planned} \le H_1$, else `NO_FILL`.
  - Enforces conservative ambiguity: if both TP and SL are hit on the same day, `HIT_STOP_LOSS` wins.

- **Data Structures / Inputs & Outputs**:
  - Input: `ticker`, `signal_date`, `side`, `take_profit`, `stop_loss`, `time_horizon_days`, `ohlcv_df`, optional entry prices/timestamps.
  - Output: `EvaluationResult` with full trajectory, realized return %, MFE/MAE %, MFE efficiency, and planned/realized R:R.

- **Cross-Check Inconsistency Vectors**:
  - Limit Fill on Gap-Down: If $O_1 < \text{planned\_price}$, fills at the better open price $O_1$ (`horizon_evaluator.py:205-206`).
  - Barrier Geometry Invariant: BUY signals strictly require $\text{stop\_loss} < \text{entry\_price} < \text{take\_profit}$ (`horizon_evaluator.py:218-225`).
  - Timezone-Aware Timestamps: Validates that `entry_timestamp` carries valid timezone information (`horizon_evaluator.py:150-153`).

- **Checklist**:
  - [ ] `WNS`, `FLAT`, `HOLD`, `SELL` signals return `EvaluationOutcome.NO_ORDER` with 0% realized return (`horizon_evaluator.py:133-137`).
  - [ ] Same-bar TP/SL collisions prioritize `HIT_STOP_LOSS` (`horizon_evaluator.py:252-258`).
  - [ ] Trajectory records daily unrealized returns, MFE %, and MAE % for all evaluated bars (`horizon_evaluator.py:266-285`).
  - [ ] Terminal outcomes compute realized $R:R$ ratio and MFE efficiency (`horizon_evaluator.py:293-296`).

---

### Section 8.15: Single-Shot Result Bundling & Concurrency Locks
- **Files & Line References**:
  - `cli/commands/evaluate_results.py:20-34` (Secret regex patterns, `_SOURCE_DATE_FIELDS`)
  - `cli/commands/evaluate_results.py:36-70` (`_parse_date`, `_validated_records` timestamp assertion)
  - `cli/commands/evaluate_results.py:71-136` (`_json_safe`, `_sanitize_url`, `sanitize_config` credential scrubbing)
  - `cli/commands/evaluate_results.py:157-257` (`write_result_bundle`, atomic locking with `fcntl.flock`, version incrementing `v1.0`, `v2.0`, schema files)

- **Purpose & Core Contracts**:
  - Persists an immutable evaluation bundle under `result_backtest/<TICKER>/<TRADE_DATE>/v<N>.0/`.
  - Uses `fcntl.flock` file locking to prevent race conditions during concurrent runs.
  - Validates that every record in `ohlcv`, `news`, `fundamentals`, `sentiment`, `broker_activity` has timestamps $\le T_0$.
  - Sanitizes sensitive credentials, API keys, bearer tokens, and passwords before exporting `config.json`.

- **Data Structures / Inputs & Outputs**:
  - Input: `root`, `ticker`, `trade_date`, `snapshot_data`, `agent_report`, `signal`, `evaluation`, `config`.
  - Output: `Path` to the created versioned bundle directory containing 7 files (`agent_report.md`, `signal.json`, `evaluation.json`, `execution_data.json`, `leakage_audit.json`, `config.json`, `summary.json`, and `snapshot/`).

- **Cross-Check Inconsistency Vectors**:
  - Strict PIT Exception: If any record timestamp $> \text{trade\_date}$, raises `ValueError` immediately (`evaluate_results.py:64-66, 185-187`).
  - API Key Masking: Masks strings matching `sk-[A-Za-z0-9...]` or bearer tokens with `[REDACTED]` (`evaluate_results.py:24, 119-135`).

- **Checklist**:
  - [ ] `write_result_bundle` acquires `fcntl.flock` before checking and creating version directories (`evaluate_results.py:200-202`).
  - [ ] Future timestamps in snapshot data cause immediate hard failure (`evaluate_results.py:64-66, 185-187`).
  - [ ] `sanitize_config` scrubs API keys and bearer tokens from exported JSON (`evaluate_results.py:119-135`).
  - [ ] Result bundle includes raw snapshot data in `snapshot/` subdirectory (`evaluate_results.py:209-215`).

---

### Section 8.16: CLI Backtest Command & Interactive Prompt Flow
- **Files & Line References**:
  - `cli/commands/backtest.py:69-165` (`backtest`, CLI arguments, YAML load, interactive prompt cascade, dry run, LLM agent config)
  - `cli/commands/backtest.py:166-242` (`backtest`, `BacktestEngine.from_dict`, OHLCV verification, market days count, UI initialization)
  - `cli/commands/backtest.py:243-498` (`backtest`, Rich Live display layout, progress callback handler, error handling)
  - `cli/commands/backtest_prompts.py:32-114` (`_prompt_for_missing_config`, prompt defaults, date validation)
  - `cli/commands/backtest_prompts.py:124-223` (`_select_lookback`, provider selection helpers)

- **Purpose & Core Contracts**:
  - Primary CLI user interface for launching multi-day walk-forward backtests (`tradingagents backtest`).
  - Always interactively prompts for `ticker`, `start_date`, and `end_date`, defaulting to YAML values if present.
  - Automatically offers to download missing OHLCV data from yfinance if local CSV is missing or stale.
  - Renders real-time Rich terminal UI with node execution timings, trade notifications, and progress bar.

- **Data Structures / Inputs & Outputs**:
  - Input: CLI options `--config`, `--lookback`, `--dry-run`.
  - Output: Interactive terminal session, stdout results table, and persisted backtest artifact files.

- **Cross-Check Inconsistency Vectors**:
  - Interactive Overrides via `from_dict`: CLI passes the mutated prompt dictionary into `BacktestEngine.from_dict` to prevent `from_yaml` from discarding typed inputs (`backtest.py:171-175`).
  - Accurate Progress Count: Counts actual market dates from the OHLCV file rather than rough $(5/7 \times \text{days})$ estimate (`backtest.py:217`).

- **Checklist**:
  - [ ] Interactive prompts default to YAML values on Enter (`backtest_prompts.py:48-100`).
  - [ ] Missing OHLCV triggers yfinance auto-fetch confirmation prompt (`backtest.py:182-209`).
  - [ ] Progress bar accurately reflects market trading days from OHLCV (`backtest.py:217`).
  - [ ] Results summary table prints final performance metrics upon completion (`backtest.py:480-498`).

---

### Section 8.17: CLI Single-Shot Signal Evaluator & TUI Pipeline
- **Files & Line References**:
  - `cli/commands/evaluate.py:37-53` (`signal_from_final_state`, SignalContract extraction and validation)
  - `cli/commands/evaluate.py:55-105` (`evaluate_signal_cmd`, date validation, TUI context manager)
  - `cli/commands/evaluate.py:107-169` (`_run_forward_evaluation`, side resolution, entry mode mapping)
  - `cli/commands/evaluate.py:171-344` (`_evaluate_signal_with_tui`, snapshot provisioning, graph propagation, result bundling)
  - `cli/commands/evaluate_tui.py:34-124` (`SingleShotTUI`, lifecycle, phase transitions, live updates)
  - `cli/commands/evaluate_tui.py:125-220` (`consume_graph_chunk`, report updating, agent status tracking)

- **Purpose & Core Contracts**:
  - Executes single-shot prediction at target date $T_0$ via `tradingagents evaluate-signal`.
  - Runs full 9-agent LangGraph team under strict PIT sandbox (only OHLCV $\le T_0$ visible).
  - Extracts typed `SignalContract` from Portfolio Manager final state.
  - Passes signal to `HorizonEvaluator` to compute forward trajectory across future trading days.
  - Persists bundle via `write_result_bundle`.

- **Data Structures / Inputs & Outputs**:
  - Input: `--ticker`, `--date`, `--provider`, `--kronos/--no-kronos`.
  - Output: Rich TUI execution, terminal summary cards, persisted result bundle in `result_backtest/`.

- **Cross-Check Inconsistency Vectors**:
  - Signal Validation: Confirms `signal.ticker == ticker` and `signal.signal_date == trade_date` (`evaluate.py:48-51`).
  - Limit Order Policy Mapping: `signal.entry_mode == EntryMode.T1_LIMIT` sets `entry_policy = "ASSUMED_AI_ENTRY"` (`evaluate.py:115-116`).

- **Checklist**:
  - [ ] Rejects future dates with clear error message (`evaluate.py:87-88`).
  - [ ] Sandboxes market data strictly to `date <= trade_date` (`evaluate.py:209-223`).
  - [ ] Extracts typed `SignalContract` directly from PM graph state (`evaluate.py:37-53`).
  - [ ] Forward evaluation runs and writes immutable result bundle (`evaluate.py:147-168, 300-340`).

---

## Master Cross-Check Matrix (Subsystem 8)

| Concern / Vector | Expected Contract | Actual Implementation & Location | Status |
| :--- | :--- | :--- | :--- |
| **Timeframe / Horizon** | Multi-horizon forward evaluation ($1 \le H \le 252$) | `HorizonEvaluator` validates $1 \le H \le 252$ (`horizon_evaluator.py:128-129`) | `ALIGNED` |
| **Rating Normalization** | Strict canonical `BUY` vs `WNS` | Normalized in `MarkdownDecisionParser` & `DSM` (`markdown_parser.py:98-100`, `decision_state_manager.py:47-61`) | `ALIGNED` |
| **Short Selling Ban** | Spot long-only execution (reject all shorts) | Short orders/positions rejected across Broker, DSM, Portfolio, Risk (`broker.py:59-66`, `portfolio.py:78-80`, `risk.py:84-85`) | `ALIGNED` |
| **Order Sizing** | Equity % capped by cash & leverage | Sized in `OrderGenerator._allocation_to_qty` (`order_generator.py:287-350`) | `ALIGNED` |
| **Stop / TP Collisions** | Stop-Loss always wins on same bar | Evaluated first in `RiskEngine` & `HorizonEvaluator` (`risk.py:107-124`, `horizon_evaluator.py:252-258`) | `ALIGNED` |
| **Static Level Protection**| Never ratchet/widen stops after entry | Enforced in `update_position_risk_levels` (`risk.py:389-395`) | `ALIGNED` |
| **Data Leakage Cutoffs** | OHLCV $\le T$, News/Sent $< T$, Fund $+3\text{d} \le T$ | Enforced in `DecisionCutoffValidator` & `data_window` (`cutoff_validator.py:98-180`, `data_window.py:176-240`) | `ALIGNED` |
| **Execution Timing** | Agent orders at $T+1$ Open; Risk immediate | Split execution in `SimulatedBroker` (`broker.py:77-106, 110-147`) | `ALIGNED` |
| **Downside Semi-Dev** | Target 0 over all $N$ trading days | Implemented in `MetricsCalculator.calculate` (`metrics.py:97-105`) | `ALIGNED` |
| **Credential Scrubbing** | Redact secrets in exported config | Sanitized in `evaluate_results.sanitize_config` (`evaluate_results.py:119-135`) | `ALIGNED` |

---

## Verification & Test Cross-References
- `tests/test_horizon_evaluator.py`: Verifies `T1_OPEN`, `ASSUMED_AI_ENTRY`, TP/SL barrier hits, and conservative same-bar collisions.
- `tests/test_trigger_evaluator.py`: Verifies gating conditions (`tp_sl_hit`, `setup_invalid`, `rr_deteriorated`, `better_candidate`).
- `tests/test_backtest_config_env.py`: Verifies configuration cascades, YAML loading, and environment variable overrides.
- `tests/test_evaluate_tui.py`: Verifies TUI rendering, LangGraph chunk consumption, and progress events.

---

## Subsystem 9: CLI Interface, Commands, Modular Handlers & TUI Renderers

# Subsystem 9: CLI Commands & TUI Interfaces — Crosscheck Matrix & Architectural Mapping

### Scope & Files Covered
- `cli/config.py` (Lines 1–6)
- `cli/models.py` (Lines 1–18)
- `cli/stats_handler.py` (Lines 1–76)
- `cli/announcements.py` (Lines 1–62)
- `cli/progress_contract.py` (Lines 1–97)
- `cli/message_buffer.py` (Lines 1–186)
- `cli/display.py` (Lines 1–236)
- `cli/report_io.py` (Lines 1–182)
- `cli/utils.py` (Lines 1–571)
- `cli/selections.py` (Lines 1–270)
- `cli/main.py` (Lines 1–633)
- `cli/static/welcome.txt`
- `cli/commands/__init__.py` (Lines 1–1)
- `cli/commands/backtest.py` (Lines 1–498)
- `cli/commands/backtest_prompts.py` (Lines 1–223)
- `cli/commands/backtest_report.py` (Lines 1–326)
- `cli/commands/backtest_tui.py` (Lines 1–281)
- `cli/commands/evaluate.py` (Lines 1–344)
- `cli/commands/evaluate_results.py` (Lines 1–257)
- `cli/commands/evaluate_tui.py` (Lines 1–318)

---

## 1. CLI Core Entry Points & Application Dispatcher

- **Exact Paths & Line Numbers**: `cli/main.py:54–60`, `cli/main.py:426–594`, `cli/main.py:596–633`, `cli/static/welcome.txt:1–13`
- **Purpose & Core Contracts**:
  - `app = typer.Typer(...)` declares CLI entry point with shell completion (`add_completion=True`).
  - `main_menu(ctx)`: Callback when CLI invoked without subcommands (`cli/main.py:596–631`), renders interactive Questionary select with 4 choices: Live Analysis (`live`), Single-Shot Evaluator (`evaluate`), Walk-Forward Backtest (`backtest`), and Exit (`exit`).
  - Commands registered: `analyze` (`cli/main.py:426–515`), `backtest` (`cli/main.py:517–550`), `evaluate-signal` (`cli/main.py:552–594`).
- **Data Structures, Inputs & Outputs**:
  - Input: CLI arguments / options or questionary interactive choices.
  - Output: `run_analysis(...)`, `evaluate_signal_cmd(...)`, or `backtest(...)` invocation, returning state dict or exiting with code 0/1/130.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Flag normalization: `--date`/`-d` vs config `trade_date`/`curr_date`/`analysis_date`.
  - `[ ]` Checkpoint clearing: `--clear-checkpoints` runs `clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])` prior to analysis (`cli/main.py:489–493`).
  - `[ ]` Headless auto-detection: `is_headless = headless or (ticker is not None)` defaults ticker to `"SPY"` if `--headless` passed without `--ticker` (`cli/main.py:495–497`).

---

## 2. Interactive Selection Flow & Headless Configuration Builder

- **Exact Paths & Line Numbers**: `cli/selections.py:1–270`, `cli/utils.py:28–363`, `cli/utils.py:476–553`
- **Purpose & Core Contracts**:
  - `get_user_selections()`: Prompts user step-by-step (Ticker, Date, Language, Analysts, Depth, Provider, Region, API Key, Thinking Engines, Provider-specific reasoning configs).
  - `build_headless_selections()`: Constructs configuration dictionary without Questionary prompts for automated scripts/cron (`cli/selections.py:213–270`).
  - `detect_asset_type()`: Checks ticker suffix against `CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")` to set `AssetType.CRYPTO` vs `AssetType.STOCK` (`cli/utils.py:25, 54–59`).
  - `normalize_ticker_symbol()`: Calls `safe_ticker_component(cleaned)` (`cli/utils.py:48–52`).
- **Data Structures, Inputs & Outputs**:
  - Returns dictionary: `{"ticker": str, "asset_type": "stock"|"crypto", "analysis_date": str, "analysts": list, "research_depth": int, "llm_provider": str, "backend_url": str|None, "shallow_thinker": str, "deep_thinker": str, "google_thinking_level": str|None, "openai_reasoning_effort": str|None, "anthropic_effort": str|None, "output_language": str}`.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Hardcoded provider model defaults in `build_headless_selections` (`cli/selections.py:228–248`): Defaults `gpt-5.4-mini`/`gpt-5.4`, `claude-haiku-4-5`/`claude-sonnet-4-6`, `gemini-2.5-flash`/`gemini-2.5-pro`, `deepseek-chat`/`deepseek-reasoner`, `llama3.2`/`llama3.3`. Check consistency with `tradingagents/llm_clients/model_catalog.py`.
  - `[ ]` Analyst enum vs string value serialization: `analysts` contains `[AnalystType.MARKET, ...]` in interactive mode vs raw enum values in headless mode. `cli/main.py:143–146` handles normalization via `getattr(analyst, "value", str(analyst))`.
  - `[ ]` Regional provider endpoint splits: DashScope Qwen (`dashscope-intl.aliyuncs.com` vs `dashscope.aliyuncs.com`), MiniMax (`api.minimax.io` vs `api.minimaxi.com`), GLM (`api.z.ai` vs `open.bigmodel.cn`) have non-interchangeable keys (`cli/utils.py:365–443`).

---

## 3. Live Analysis Execution & Point-in-Time Fail-Closed Guard

- **Exact Paths & Line Numbers**: `cli/main.py:61–424`
- **Purpose & Core Contracts**:
  - `run_analysis()` executes live or historical graph run within Rich `Live` UI.
  - Historical Point-in-Time sandbox: If `parsed_analysis_date < datetime.date.today()`, sets `point_in_time_mode=True`, `backtest_mode=True`, `memory_enabled=False`, `web_search_enabled=False`.
  - PIT Fail-Closed guard: Sets vendor data sources to `"snapshot"`. Instantiates `SnapshotDataProvider(fetch_from_api=False)`. If snapshot data missing, aborts immediately (`cli/main.py:120–129`).
  - Graph concurrency and analyst execution plan: `build_analyst_execution_plan(selected_analyst_keys, concurrency_limit=config["analyst_concurrency_limit"])` and `AnalystWallTimeTracker` (`cli/main.py:147–153`).
- **Data Structures, Inputs & Outputs**:
  - Input: `checkpoint: bool`, `selections: dict`, `output_dir: Path|None`, `headless: bool`, `kronos: bool`.
  - Output: `final_state: dict` containing reports, debate states, and signal contracts.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Future date guard: Throws `typer.BadParameter("analysis_date cannot be in the future")` (`cli/main.py:78–79`).
  - `[ ]` Chunk message deduplication: Uses `message_buffer._processed_message_ids` on `message.id` (`cli/main.py:245–249`).
  - `[ ]` Streaming state merge: Chunks are merged via `final_state.update(chunk)` over trace (`cli/main.py:347–349`).
  - `[ ]` Memory persistence: Stores decision in `memory_log.store_decision` only if live run (`cli/main.py:366–379`).

---

## 4. Rich CLI Display & Message Streaming Buffer

- **Exact Paths & Line Numbers**: `cli/display.py:1–236`, `cli/message_buffer.py:1–186`, `cli/progress_contract.py:1–97`, `cli/utils.py:554–571`
- **Purpose & Core Contracts**:
  - `create_cli_layout()`: Header (size 3), main (split into upper ratio 3 [progress ratio 2, messages ratio 3] and analysis ratio 5), footer (size 3).
  - `MessageBuffer`: Thread-safe deque (maxlen=100) storing `(timestamp, message_type, content)` and tool calls `(timestamp, tool_name, args)`. Dispatches real-time writes to `log_file` and `report_dir`.
  - `classify_message_type()`: Maps LangChain `HumanMessage` -> `Control`/`User`, `ToolMessage` -> `Data`, `AIMessage` -> `Agent`, fallback -> `System` (`cli/progress_contract.py:81–93`).
  - `update_analyst_statuses()`: Synchronizes status transitions (pending -> in_progress -> completed) using `AnalystWallTimeTracker` and accumulated `report_sections` (`cli/display.py:194–236`).
- **Data Structures, Inputs & Outputs**:
  - `ALL_TEAMS`: Analyst (`Market`, `Sentiment`, `News`, `Fundamentals`), Research (`Bull Researcher`, `Bear Researcher`, `Research Manager`), Trading (`Trader`), Risk (`Aggressive Analyst`, `Conservative Analyst`, `Neutral Analyst`), Portfolio (`Portfolio Manager`).
  - `REPORT_SECTIONS`: Mapped to finalized agent outputs (`cli/progress_contract.py:20–28`).
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Sentiment Analyst wire key vs display name: `social` wire key in `ANALYST_MAPPING` vs display label `Sentiment Analyst` (`cli/progress_contract.py:13–18`, `cli/models.py:7–14`).
  - `[ ]` Report section count: `get_completed_reports_count()` only increments when the finalizing agent is marked `completed` (`cli/message_buffer.py:71–82`).
  - `[ ]` Status color/icon alignment: `completed` (green, `✓`), `in_progress` (cyan, `⟳`), `pending` (dim, `·`), `error` (red, `✗`) (`cli/display.py:44–45`).

---

## 5. Report Persistence, Sanitization & Formatted CLI Output

- **Exact Paths & Line Numbers**: `cli/report_io.py:1–182`, `cli/commands/evaluate_results.py:20–34, 119–135`
- **Purpose & Core Contracts**:
  - `save_report_to_disk()`: Saves analysis artifacts under `save_path`:
    - `1_analysts/` (`market.md`, `sentiment.md`, `news.md`, `fundamentals.md`)
    - `2_research/` (`bull.md`, `bear.md`, `manager.md`)
    - `3_trading/` (`trader.md`)
    - `4_risk/` (`aggressive.md`, `conservative.md`, `neutral.md`)
    - `5_portfolio/` (`decision.md`)
    - `signal.json` (Serialized `SignalContract`)
    - `config.json` (Sanitized configuration dictionary)
    - `complete_report.md` (Consolidated markdown)
  - `display_complete_report()`: Formats complete report sequentially to console without truncation (`cli/report_io.py:124–182`).
  - `sanitize_config()`: Strips credentials/secrets matching `_SECRET_KEY` regex and redacts bearer tokens (`cli/commands/evaluate_results.py:119–135`).
- **Data Structures, Inputs & Outputs**:
  - Inputs: `final_state: dict`, `ticker: str`, `save_path: Path`, `config: dict|None`.
  - Outputs: Created directory structure and `complete_report.md` Path.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Secret leakage in `config.json`: Verified via `_SECRET_KEY` / `_SECRET_VALUE` patterns redacting URLs, tokens, and keys (`cli/commands/evaluate_results.py:20–24, 119–135`).
  - `[ ]` SignalContract serialization: Uses `signal_contract.model_dump(mode="json")` if Pydantic model (`cli/report_io.py:101–105`).

---

## 6. Callback Telemetry & Token Tracking

- **Exact Paths & Line Numbers**: `cli/stats_handler.py:1–76`
- **Purpose & Core Contracts**:
  - `StatsCallbackHandler(BaseCallbackHandler)`: LangChain thread-safe callback handler tracking `llm_calls`, `tool_calls`, `tokens_in`, `tokens_out`.
  - Hook implementations: `on_llm_start`, `on_chat_model_start`, `on_llm_end` (extracts `usage_metadata.get("input_tokens")` & `output_tokens` from `AIMessage`), `on_tool_start`.
- **Data Structures, Inputs & Outputs**:
  - `get_stats() -> dict`: `{"llm_calls": int, "tool_calls": int, "tokens_in": int, "tokens_out": int}`.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Token extraction fallback: If LLM provider does not populate `AIMessage.usage_metadata`, `tokens_in`/`tokens_out` remain 0 and display renders `Tokens: --` (`cli/display.py:166–170`).
  - `[ ]` Thread lock concurrency: `self._lock` protects all counters against concurrent node executions (`cli/stats_handler.py:14, 27, 37, 54, 65, 70`).

---

## 7. Announcement Service & Network Polling

- **Exact Paths & Line Numbers**: `cli/announcements.py:1–62`, `cli/config.py:1–6`
- **Purpose & Core Contracts**:
  - `fetch_announcements()`: GET request to `https://api.tauric.ai/v1/announcements` with 1.0s timeout.
  - Fallback handling: On any network failure / timeout, returns fallback dict pointing to `https://github.com/TauricResearch`.
  - `display_announcements()`: Renders announcement panel. If `require_attention=True` and in TTY, blocks with `getpass.getpass("Press Enter to continue...")` (`cli/announcements.py:56–60`).
- **Data Structures, Inputs & Outputs**:
  - Output: `{"announcements": list[str], "require_attention": bool}`.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Headless blocking risk: `sys.stdin.isatty()` guard prevents prompt hanging in non-interactive / cron environments (`cli/announcements.py:56`).

---

## 8. Walk-Forward Backtest Command & Lifecycle Orchestrator

- **Exact Paths & Line Numbers**: `cli/commands/backtest.py:1–498`
- **Purpose & Core Contracts**:
  - `backtest()`: CLI entry point for walk-forward portfolio simulation and margin evaluation.
  - Dual-window architecture:
    * Outer window: `start_date` -> `end_date` (loops over actual OHLCV market days).
    * Inner window: `lookback_days` (N trading days of history visible per snapshot).
  - Validation: Lookback restricted to `ALLOWED_LOOKBACKS = (5, 10, 20, 40, 60, 80, 100, 120, 240)` (`cli/commands/backtest.py:136–141`).
  - `BacktestEngine.from_dict()`: Instantiates backtest engine with interactive prompt overrides (`cli/commands/backtest.py:171–175`).
  - Auto-fetch guard: `engine.ensure_ohlcv(auto_fetch=False)` prompts user to download OHLCV from yfinance if missing or stale (`cli/commands/backtest.py:182–212`).
  - Multithreaded execution: Background worker thread executes `engine.run()`, while watchdog monitor checks for thread idle/stuck states (>30s warning, >60s error) (`cli/commands/backtest.py:358–424`).
- **Data Structures, Inputs & Outputs**:
  - Input: `config_path: Path`, `lookback: int|None`, `dry_run: bool`.
  - Output: Summary dict, results table, leakage audit report, output paths (`cli/commands/backtest.py:449–460`).
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Outer window trading days calculation: `_count_market_days()` uses snapshot provider market dates rather than calendar math (`cli/commands/backtest.py:476–495`).
  - `[ ]` UI Thread synchronization: Live progress bar refreshed at 4 Hz from runner phase callbacks (`day_start`, `agent_node`, `trigger`, `execute`, `risk`, `day_complete`) (`cli/commands/backtest.py:265–355`).
  - `[ ]` Signal/Action mapping in trade logs: `action_map = {"BUY_TO_OPEN": "Buy entry", "SELL_TO_CLOSE": "Static exit"}` (`cli/commands/backtest.py:306–309`).

---

## 9. Backtest Prompts, Date Validation & LLM Configuration Bridge

- **Exact Paths & Line Numbers**: `cli/commands/backtest_prompts.py:1–223`
- **Purpose & Core Contracts**:
  - `_prompt_for_missing_config()`: Prompts for `ticker`, `start_date`, `end_date`, `initial_cash` (if missing), validating `start_date <= end_date` and date format `YYYY-MM-DD` (`cli/commands/backtest_prompts.py:32–114`).
  - `_build_llm_agent_config()`: Interactively configures LLM provider, reasoning settings, and exports `TRADINGAGENTS_*` environment variables (`TRADINGAGENTS_LLM_PROVIDER`, `TRADINGAGENTS_QUICK_THINK_LLM`, `TRADINGAGENTS_DEEP_THINK_LLM`, `TRADINGAGENTS_LLM_BACKEND_URL`, etc.).
  - Hot reload: Calls `importlib.reload(tradingagents.default_config)` and `importlib.reload(agent_runner)` so newly exported env vars take effect immediately in runner deepcopies (`cli/commands/backtest_prompts.py:210–220`).
- **Data Structures, Inputs & Outputs**:
  - Returns `AgentConfig(**(agent_overrides or {}))`.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Currency consistency: Cash prompt defaults to IDR integer (`100_000_000` IDR) (`cli/commands/backtest_prompts.py:104–112`).
  - `[ ]` Lookback choice mapping: `ALLOWED_LOOKBACK_CHOICES` filters out `None` from `ALLOWED_LOOKBACKS` (`cli/commands/backtest_prompts.py:29`).

---

## 10. Backtest Live TUI & Panel Renderers

- **Exact Paths & Line Numbers**: `cli/commands/backtest_tui.py:1–281`
- **Purpose & Core Contracts**:
  - `_BacktestUI`: State store tracking `n_days`, `current_day`, `current_phase`, `current_node`, `node_durations`, `trade_log`, `trigger_triggered`, `trigger_skipped`.
  - Panel Renderers:
    - `_render_header()`: Displays Ticker, Lookback window, and total days (`cli/commands/backtest_tui.py:118–131`).
    - `_render_progress()`: Renders ASCII progress bar (`████░░░░`), percentage, current phase, active node execution timer, and trigger count (`cli/commands/backtest_tui.py:134–183`).
    - `_render_log()`: Displays last 15 activity log events (`cli/commands/backtest_tui.py:185–203`).
    - `_render_trading()`: Formats recent 15 trade fills (`Date`, `Action`, `Price`, `Qty`, `PnL`) (`cli/commands/backtest_tui.py:205–231`).
    - `_render_footer()`: Renders completion summary (Final equity IDR, Total return %, Trades, Trigger hit rate, Lookback) (`cli/commands/backtest_tui.py:233–272`).
- **Data Structures, Inputs & Outputs**:
  - `PHASE_ICONS` / `PHASE_LABELS`: 14 distinct phases (`start`, `day_start`, `day_complete`, `execute`, `risk`, `snapshot`, `agent`, `agent_node`, `parse`, `trigger`, `orders`, `wait`, `done`, `complete`, `error`).
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Node duration history cap: Capped at 200 items to prevent memory growth (`cli/commands/backtest_tui.py:97–98`).
  - `[ ]` Percentage formatting: PnL formatted as `+,.2f%` with color coding (green >= 0, red < 0) (`cli/commands/backtest_tui.py:247–251`).

---

## 11. Backtest Static Reports, Explainer & Dry-Run Plan

- **Exact Paths & Line Numbers**: `cli/commands/backtest_report.py:1–326`
- **Purpose & Core Contracts**:
  - `_print_results_table()`: Rich table printing 16 summary metrics (Ticker, Period, Lookback, Decision mapping, Initial cash, Final equity, Total return, Max drawdown, Sharpe ratio, Total trades, Win rate, Trigger hit rate, Trigger triggered ratio, Avg hold days, Margin calls, Liquidations), Leakage audit status, Rating distribution, Trigger reasons fired (`cli/commands/backtest_report.py:15–84`).
  - `_print_dry_run_plan()`: Renders parameter table, resolved JSON configuration, and window explainer without running backtest (`cli/commands/backtest_report.py:86–276`).
  - `_print_window_explainer()`: Explains difference between outer window (`start_date`/`end_date`) and inner window (`lookback_days`) (`cli/commands/backtest_report.py:278–326`).
- **Data Structures, Inputs & Outputs**:
  - Formats summary dictionaries and leakage audit objects into terminal output.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Unit and scale formatting: Margin and fee percentages displayed in basis points/percentages (`.4%` for fees, `.0%` for initial margin) (`cli/commands/backtest_report.py:156–178`).
  - `[ ]` IDR vs USD currency representation: Hardcoded formatting uses `IDR` label (`cli/commands/backtest_report.py:36, 119`).

---

## 12. Single-Shot Horizon Evaluator Command & Entry Policy Contract

- **Exact Paths & Line Numbers**: `cli/commands/evaluate.py:1–344`
- **Purpose & Core Contracts**:
  - `evaluate_signal_cli()` / `evaluate_signal_cmd()`: Executes T0 single-shot agentic prediction and forward horizon evaluation (1–252 trading days).
  - Entry Policy contract (`_run_forward_evaluation`):
    * Limit order (`signal.entry_mode == EntryMode.T1_LIMIT` and `planned_entry_price` set): policy = `ASSUMED_AI_ENTRY`.
    * Market order: policy = `T1_OPEN`.
    * Side mapping: `BUY` -> `LONG`; `WNS`/`HOLD`/`SELL` -> `FLAT` (evaluates as `EvaluationOutcome.NO_ORDER` with 0.0% realized return) (`cli/commands/evaluate.py:115–145`).
  - `signal_from_final_state()`: Extracts PM `SignalContract` and verifies `signal.ticker == ticker` and `signal.signal_date == trade_date` (`cli/commands/evaluate.py:37–53`).
  - Auto-provisioning: `SnapshotDataProvider(fetch_from_api=True)` automatically downloads OHLCV from yfinance if missing (`cli/commands/evaluate.py:183–189`).
  - Reference price pinning: Signals with missing reference prices are patched using the close price at or before T0 cutoff (`cli/commands/evaluate.py:262–282`).
- **Data Structures, Inputs & Outputs**:
  - Inputs: `ticker: str`, `trade_date: str`, `llm_provider: str|None`, `kronos_enabled: bool`.
  - Outputs: `EvaluationResult` containing outcome, actual entry price/date, exit price/date, realized return %, MFE, MAE, holding days.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Date validation: Fails closed on future dates or non-ISO format (`cli/commands/evaluate.py:83–88`).
  - `[ ]` Horizon boundary: Clamped to `1 <= effective_horizon <= 252` trading days (`cli/commands/evaluate.py:256–257`).
  - `[ ]` Rating enum aliases: Normalizes `HOLD`, `SELL`, `UNDERWEIGHT`, `OVERWEIGHT` to canonical `BUY`/`WNS` contract (`tradingagents/agents/schemas.py:53–86, 705–712, 771–780`).

---

## 13. Single-Shot Live TUI & Realization Summarizer

- **Exact Paths & Line Numbers**: `cli/commands/evaluate_tui.py:1–318`
- **Purpose & Core Contracts**:
  - `SingleShotTUI`: Context manager wrapping Rich `Live` display for single-shot prediction and evaluation phases (`Market Data`, `Analyst Team`, `Research Team`, `Trader`, `Risk Team`, `Portfolio Manager`, `Forward Evaluator`, `Result Bundle`).
  - `consume_graph_chunk()`: Consumes streamed LangGraph chunks, extracts message deltas, updates reports, and updates agent statuses without altering graph execution (`cli/commands/evaluate_tui.py:117–200`).
  - `render_signal_summary()`: Summarizes Agent Rating, Action, Planned Entry, Take Profit, Stop Loss, Entry Mode, Reference Price, Horizon (`cli/commands/evaluate_tui.py:283–296`).
  - `render_evaluation_summary()`: Summarizes Horizon Realization outcome, Entry/Exit fills, Holding Period, Realized Return %, MFE %, MAE %, Planned/Realized R:R, and MFE Efficiency (`cli/commands/evaluate_tui.py:298–318`).
- **Data Structures, Inputs & Outputs**:
  - Inputs: LangGraph values chunks, `SignalContract`, `EvaluationResult`.
  - Outputs: Terminal TUI panels and formatted markdown summaries.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Formatting of optional metrics: Handles `planned_rr_ratio`, `realized_rr_ratio`, and `mfe_efficiency` conditionally if present on `result` (`cli/commands/evaluate_tui.py:312–317`).
  - `[ ]` Message deduplication in TUI: Tracks processed message IDs using `_processed_message_ids` (`cli/commands/evaluate_tui.py:131–137`).

---

## 14. Single-Shot Result Bundle Persistence & Leakage Audit

- **Exact Paths & Line Numbers**: `cli/commands/evaluate_results.py:1–257`
- **Purpose & Core Contracts**:
  - `write_result_bundle()`: Persists immutable evaluation bundle under `root / safe_ticker / trade_date / v{version}.0`:
    - `snapshot/ohlcv.csv`
    - `snapshot/{news, fundamentals, sentiment, broker_activity}.json`
    - `agent_report.md`
    - `signal.json`
    - `evaluation.json`
    - `execution_data.json`
    - `leakage_audit.json`
    - `config.json` (Sanitized)
    - `summary.json`
  - Atomic writing & locking: Uses `fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)` on `.bundle.lock` and writes to temporary folder before atomic directory rename (`os.replace`) (`cli/commands/evaluate_results.py:199–256`).
  - Strict Point-in-Time Cutoff validation: Validates all OHLCV rows and source records (`news`, `fundamentals`, `sentiment`, `broker_activity`) against `trade_date`. If any item contains a date > `trade_date`, raises `ValueError` immediately (`cli/commands/evaluate_results.py:58–66, 184–187`).
- **Data Structures, Inputs & Outputs**:
  - Inputs: `root`, `ticker`, `trade_date`, `snapshot_data`, `agent_report`, `signal`, `evaluation`, `config`, `execution_data`.
  - Output: `Path` to created `v{version}.0` directory.
- **Potential Mismatch / Inconsistency Vectors**:
  - `[ ]` Publication date key extraction: Inspects candidate keys across data types (`published_at`, `available_date`, `filingDate`, `timestamp`, `date`) (`cli/commands/evaluate_results.py:25–33, 58–61`).
  - `[ ]` NaN/Inf JSON serialization: Handles `np.bool_`, `np.integer`, `np.floating`, `np.ndarray`, `pd.Timestamp`, `datetime`, and Pydantic `model_dump()` safely (`cli/commands/evaluate_results.py:71–90`).
  - `[ ]` URL Credential Scrubbing: `_sanitize_url()` parses userinfo/query strings and replaces passwords and secret query params with `[REDACTED]` (`cli/commands/evaluate_results.py:93–117`).

---

## 15. Cross-Subsystem Interface Crosscheck Matrix Checklist

- `[ ]` **Rating & Signal Enum Normalization**:
  - Verify CLI commands, schemas, and reports only produce or consume canonical `PortfolioRating` (`BUY`, `WNS`), while gracefully mapping legacy aliases (`HOLD`, `SELL`, `UNDERWEIGHT`, `OVERWEIGHT`) (`tradingagents/agents/schemas.py:53–86, 705–712`, `cli/commands/evaluate.py:118–123`).
- `[ ]` **Order Geometry Consistency**:
  - Ensure BUY signal contracts satisfy `stop_loss < planned_entry_price < take_profit` or `stop_loss < take_profit` for `T1_OPEN` orders (`tradingagents/agents/schemas.py:288–298, 450–454, 818–827`).
- `[ ]` **Time Horizon Scale**:
  - Check that text representations (e.g. `"1-3 months"`, `"2 weeks"`) convert via multiplier to exact trading days (`1 month = 21 days`, `1 week = 5 days`, `1 year = 252 days`) clamped to `1 <= days <= 252` (`tradingagents/agents/schemas.py:456–484`).
- `[ ]` **Point-in-Time Fail-Closed Integrity**:
  - Historical live runs (`cli/main.py:98–138`) and single-shot evaluation (`cli/commands/evaluate.py:202–223`) must set `point_in_time_mode=True`, `backtest_mode=True`, `memory_enabled=False`, `web_search_enabled=False`, and reject missing local snapshots.
- `[ ]` **Config vs Environment Variable Hierarchy**:
  - Verify CLI interactive selections and flags correctly export to `os.environ` and trigger `importlib.reload(tradingagents.default_config)` to update `DEFAULT_CONFIG` (`cli/commands/backtest_prompts.py:197–220`, `tradingagents/default_config.py:10–50`).
- `[ ]` **Currency & Unit Consistency**:
  - Currency metrics displayed in `IDR` for Indonesian equities (`.JK` suffix) vs default numerical units (`cli/commands/backtest_report.py:36, 119`, `cli/commands/backtest_tui.py:242`).
- `[ ]` **Credential Redaction in Report Artifacts**:
  - Ensure `save_report_to_disk` and `write_result_bundle` execute `sanitize_config` before persisting `config.json` (`cli/report_io.py:111–116`, `cli/commands/evaluate_results.py:119–135, 241`).

---

## Subsystem 10: Configuration Resolvers, Environment Binding & Master Test Matrix

# Master Architectural Crosscheck: Subsystem 10 - Config Resolvers & Test Matrix

---

## 1. Environment-Based Config Overrides & Default Config Engine

### Exact Locations
- `tradingagents/default_config.py:1-146`
- `tests/test_env_overrides.py:1-98`
- `tests/test_audit_fixes_10subagents.py:14-25`

### Purpose, Contracts, Data Structures, Inputs/Outputs
- **Central Default State**: `DEFAULT_CONFIG` dictionary initialized at module load (`default_config.py:52-146`).
- **Override Registry**: `_ENV_OVERRIDES` (`default_config.py:10-28`) maps `TRADINGAGENTS_*` env vars to canonical internal config keys.
- **Dynamic Type Coercion**: `_coerce(value, reference)` (`default_config.py:31-39`) dynamically coerces env strings based on runtime types (`bool`, `int`, `float`, `str`).
- **In-Place Mutation**: `_apply_env_overrides(config)` (`default_config.py:42-50`) mutates input dict and returns reference.

### Potential Mismatch / Inconsistency Vectors
- [ ] `default_config.py:10-28`: `_ENV_OVERRIDES` key synchronization with new CLI features (e.g. `TRADINGAGENTS_KRONOS_MODEL_TIER` maps to `kronos_model_tier`, verify all 17 keys match downstream expectations).
- [ ] `default_config.py:33-34`: Boolean coercion parses `"true", "1", "yes", "on"` case-insensitively; check if any consumer parses raw `"0"` or `"False"` without `_coerce`.
- [ ] `default_config.py:35-36`: Integer coercion strictly excludes `bool` (since `isinstance(True, int)` is True in Python); verify no float strings like `"1.0"` passed to int fields (will throw `ValueError`).
- [ ] `default_config.py:70-74`: `backend_url`, `google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort` default to `None`; verify clients handle `None` vs empty string `""` without crashing.
- [ ] `default_config.py:108-117`: `data_vendors` nested dict structure — env overrides only cover top-level keys; category overrides require explicit dict merges.

---

## 2. Dataflows Config Context Manager & Thread Safety

### Exact Locations
- `tradingagents/dataflows/config.py:1-53`
- `tests/test_dataflows_config.py:1-61`
- `tests/test_snapshot_enforcement.py:1-66`

### Purpose, Contracts, Data Structures, Inputs/Outputs
- **Global Thread-Safe Storage**: `_config: Optional[Dict]` protected by `_lock = threading.RLock()` (`dataflows/config.py:8-9`).
- **Initialization**: `initialize_config()` deepcopies `DEFAULT_CONFIG` on first access (`dataflows/config.py:12-18`).
- **Partial Deep Merge**: `set_config(config)` merges dict-valued keys (e.g. `data_vendors`, `tool_vendors`) one level deep while overwriting scalar keys (`dataflows/config.py:20-36`).
- **Isolated Retrieval**: `get_config()` returns a fresh `deepcopy(_config)` preventing caller side-effects (`dataflows/config.py:38-44`).
- **PIT Gate**: `is_point_in_time_mode(config)` checks `point_in_time_mode` or `backtest_mode` flags (`dataflows/config.py:46-50`).

### Potential Mismatch / Inconsistency Vectors
- [ ] `dataflows/config.py:23-36`: Nested merge depth is strictly 1-level deep (`dict.update`); verify multi-nested dicts (if added) do not shallow-reference sub-dicts.
- [ ] `dataflows/config.py:46-50`: `is_point_in_time_mode` accepts optional `config` argument; check that all dataflow tools pass their localized config or fall back safely to `get_config()`.
- [ ] `dataflows/config.py:1-53`: Module-level global state can leak across unit tests unless reset in `setUp`/`tearDown` (locked in `test_dataflows_config.py:13-16`).

---

## 3. Backtest Config Resolver & API Key Validation Engine

### Exact Locations
- `tradingagents/backtesting/config_resolver.py:1-141`
- `tradingagents/llm_clients/api_key_env.py:1-48`
- `tests/test_backtest_config_env.py:1-104`
- `tests/test_api_key_env.py:1-148`

### Purpose, Contracts, Data Structures, Inputs/Outputs
- **Resolution Precedence**: YAML dict value > `TRADINGAGENTS_LLM_*` env vars > `ConfigError` (`config_resolver.py:8-12, 25-98`).
- **Model Fallback Chain**: `yaml.agent.model` -> `TRADINGAGENTS_LLM_MODEL` -> `TRADINGAGENTS_DEEP_THINK_LLM` -> `ConfigError` (`config_resolver.py:57-62`).
- **Provider API Key Check**: `_validate_api_key(provider, env)` (`config_resolver.py:101-112`) queries `get_api_key_env(provider)` (`api_key_env.py:42-48`).
- **BacktestConfig Adapter**: `validate_backtest_config(config)` converts dataclass to dict, resolves envs, and calls `config.validate()` (`config_resolver.py:114-141`).

### Potential Mismatch / Inconsistency Vectors
- [ ] `config_resolver.py:52-62`: Whitespace stripping (`.strip()`) handles empty string vs unset; verify empty YAML fields `""` correctly trigger env var fallback.
- [ ] `config_resolver.py:101-112`: Ollama returns `None` for API key (`api_key_env.py:38`), bypassing validation; verify other local/custom endpoints behave consistently without failing validation.
- [ ] `api_key_env.py:25-30`: Dual-region providers (`qwen` vs `qwen-cn`, `glm` vs `glm-cn`, `minimax` vs `minimax-cn`) enforce distinct API key env vars (`DASHSCOPE_API_KEY` vs `DASHSCOPE_CN_API_KEY`); check that config resolver validates exact region-specific key.

---

## 4. Dataclass Configuration Hierarchy & Validation Engine

### Exact Locations
- `tradingagents/backtesting/position.py:72-296, 474-563`
- `tradingagents/backtesting/decision_schema.py:1-246`
- `tradingagents/backtesting/engine.py:28-250`
- `tests/test_position.py:1-120`

### Purpose, Contracts, Data Structures, Inputs/Outputs
- **Modular Config Dataclasses**:
  - `ExecutionConfig` (`position.py:72-122`): Percentage fees (`buy_fee=0.0015`, `sell_fee=0.0025`, `slippage=0.001`, `spread_bps=5.0`).
  - `MarginConfig` (`position.py:124-162`): Sizing and margin thresholds (`initial_margin_pct`, `maintenance_margin_pct`, `max_leverage`).
  - `RiskConfig` (`position.py:164-177`): ATR & static TP/SL (`default_stop_pct=0.08`, `atr_stop_multiplier=1.5`, `atr_tp_multiplier=3.0` enforcing R:R >= 2.0).
  - `DecisionMappingConfig` (`position.py:179-215`): Enforces `mode="spot_long_only"` and `short_allowed=False`.
  - `DataConfig` (`position.py:217-237`): Snapshot file boundaries and paths.
  - `AgentConfig` (`position.py:239-260`): Runtime agent knobs and Kronos foundation model parameters.
  - `LeakageGuardConfig` (`position.py:262-273`): PIT cutoff and buffer enforcement.
  - `OutputConfig` (`position.py:275-295`): CSV/JSON output filenames and reporting destinations.
- **Top-Level Orchestrator**: `BacktestConfig` (`position.py:474-563`) aggregates all sub-configs with strict validation invariants (`validate()`).
- **Legacy Compatibility Shim**: `decision_schema.py:1-246` re-exports canonical classes from `position.py` while isolating first-gen legacy shapes (`ParsedDecision`, `Order`, `Trade`, `Position`).
- **Engine Factory**: `BacktestEngine.from_yaml` and `BacktestEngine.from_dict` (`engine.py:156-250`).

### Potential Mismatch / Inconsistency Vectors
- [ ] `position.py:89-90, 116-117`: `initial_position_side` must be `PositionSide.FLAT`; non-flat initial side raises `ValueError`.
- [ ] `position.py:206-214`: `DecisionMappingConfig.__post_init__` forces `self.mode = "spot_long_only"` and `self.spot_mode = True`, rejecting short flags (`allow_short_on_underweight`, `allow_short_on_sell`, `short_allowed`).
- [ ] `position.py:494-500`: `BacktestConfig.margin` defaults to cash-only (`initial_margin_pct=1.0`, `maintenance_margin_pct=1.0`, `max_leverage=1.0`), overriding `MarginConfig` default leverage of 2.0.
- [ ] `position.py:513-562`: `BacktestConfig.validate()` fail-closed assertions:
  - `backtest_mode == True`
  - `asset_class == "stock"`
  - `data.provider == "snapshot"`, `data.fetch_from_api == False`, `data.news_provider == "snapshot"`
  - `disable_live_news == True`, `disable_live_web_search == True`, `disable_live_fundamentals == True`
  - `agent.memory_enabled == False`, `agent.web_search_enabled == False`, `agent.news_provider == "snapshot"`
  - `lookback_days` in `ALLOWED_LOOKBACKS` (`(None, 5, 10, 20, 40, 60, 80, 100, 120, 240)`).
- [ ] `engine.py:180-206`: `BacktestEngine.from_dict` vs `from_yaml` divergence — CLI interactive prompts must use `from_dict` to prevent re-reading stale disk YAML.

---

## 5. CLI Interactive & Headless Prompt Configuration Ingestion

### Exact Locations
- `cli/commands/backtest.py:69-242`
- `cli/commands/backtest_prompts.py:1-223`
- `cli/config.py:1-6`
- `cli/main.py:1-120`
- `tests/test_cli_main_headless.py:1-117`
- `tests/test_cli_module_split.py:1-64`

### Purpose, Contracts, Data Structures, Inputs/Outputs
- **Interactive Backtest Flow**: `cli/commands/backtest.py` coordinates YAML parsing, interactive prompt overrides, dry-run evaluation, engine instantiation, and TUI progress rendering.
- **Runtime Prompt Engine**: `_prompt_for_missing_config(raw_config)` (`backtest_prompts.py:32-114`) interactively prompts for `ticker`, `start_date`, `end_date`, and `initial_cash`.
- **Environment Exporter**: `_build_llm_agent_config(agent_overrides)` (`backtest_prompts.py:157-223`) interactive wizard sets `TRADINGAGENTS_*` in `os.environ` and triggers `importlib.reload` on `default_config` and `agent_runner`.
- **Headless Mode Contract**: `build_headless_selections` (`cli/main.py`, tested in `test_cli_main_headless.py:14-28`) constructs non-interactive config payloads.
- **Sanitized Persistence**: `save_report_to_disk` (`cli/report_io.py`, tested in `test_cli_main_headless.py:29-82`) redacts API keys and secret URLs from exported `config.json`.

### Potential Mismatch / Inconsistency Vectors
- [ ] `backtest_prompts.py:94-98`: Date range validation enforces `start_date <= end_date` using ISO-8601 string comparison.
- [ ] `backtest_prompts.py:197-219`: `importlib.reload(tradingagents.default_config)` and `importlib.reload(agent_runner)` are required after setting environment variables so runtime deepcopies pick up interactive choices.
- [ ] `backtest.py:171-176`: Explicitly uses `BacktestEngine.from_dict(raw)` to ensure prompt overrides take precedence over YAML.
- [ ] `cli/config.py:1-6`: Contains announcement endpoint defaults (`announcements_url`, `announcements_timeout`); isolated from trading configuration.

---

## 6. Model Validation & Provider Capability Testing Matrix

### Exact Locations
- `tradingagents/llm_clients/model_catalog.py:1-120`
- `tradingagents/llm_clients/validators.py:1-80`
- `tests/test_model_validation.py:1-55`
- `tests/test_ollama_base_url.py:1-60`
- `tests/test_minimax.py:1-70`
- `tests/test_bluesmind_provider.py:1-65`
- `tests/test_deepseek_reasoning.py:1-50`
- `tests/test_anthropic_effort.py:1-55`
- `tests/test_google_api_key.py:1-50`

### Purpose, Contracts, Data Structures, Inputs/Outputs
- **Catalog Verification**: `get_known_models()` returns approved models per provider; tested against `validate_model` (`test_model_validation.py:26-34`).
- **Strict vs Permissive Validation**: Strict providers emit `UserWarning` on unknown models; `openrouter` and `ollama` accept arbitrary custom model names without warning (`test_model_validation.py:46-55`).
- **Provider Parameter Integrations**:
  - `openai`: `reasoning_effort` (`"low"`, `"medium"`, `"high"`).
  - `anthropic`: `thinking` budget / `effort` level.
  - `google`: `thinking_level` (`"high"`, `"minimal"`, etc.).
  - `deepseek`: Reasoning extraction and temperature handling.
  - `bluesmind` / `sumopod` / `tokenrouter`: Custom OpenAI-compatible endpoints.

### Potential Mismatch / Inconsistency Vectors
- [ ] `model_catalog.py` vs CLI interactive choices: Provider catalog additions must stay in sync with `select_shallow_thinking_agent` and `select_deep_thinking_agent`.
- [ ] API Key Env mapping in `api_key_env.py` must include every provider defined in `model_catalog.py` (`test_api_key_env.py:15-28`).
- [ ] Provider URL resolution: `backend_url` must not leak across provider instances (e.g. OpenAI endpoint forwarded to Gemini client).

---

## 7. Quant Invariants, Risk Engine Units, & Execution Horizon Verification Matrix

### Exact Locations
- `tests/test_audit_invariants.py:1-237`
- `tests/test_risk_unit_fix.py:1-138`
- `tests/test_risk.py:1-180`
- `tests/test_strict_5tier.py:1-80`
- `tests/test_stock_order_generator.py:1-160`
- `tests/test_trigger_evaluator.py:1-257`
- `tests/test_single_shot_results.py:1-284`
- `tests/test_data_window.py:1-278`
- `tests/test_anti_leakage_hardening.py:1-200`
- `tests/test_safe_ticker_component.py:1-52`
- `tests/test_ticker_symbol_handling.py:1-21`

### Purpose, Contracts, Data Structures, Inputs/Outputs
- **Quant Horizon Clamping**: `portfolio_decision_to_signal_contract` clamps horizons > 63 days down to max allowable holding window (63 bars) (`test_audit_invariants.py:85-93`).
- **Limit Order Fill Invariants**:
  - Gap-up open on Limit Buy capped exactly at limit price (`test_audit_invariants.py:131-137`).
  - Gap-down open on Limit Buy fills at improved open price (`test_audit_invariants.py:138-144`).
  - Untouched limit remains `UNFILLED` (`test_audit_invariants.py:145-150`).
- **Market Force-Close Execution**: Liquidation orders execute at open through gap-downs rather than stranding at stale limit prices (`test_audit_invariants.py:152-194`).
- **Unit & Scale Normalization**:
  - Risk parameters stored strictly as decimals/fractions (`0.05` for 5%, not `5.0`) (`test_risk_unit_fix.py:1-138`, `test_snapshot_enforcement.py:67-82`).
  - Position unrealized PnL vs equity evaluated as decimal fraction vs decimal fraction (`test_risk_unit_fix.py:19-50`).
- **Decision & Signal Mapping Normalization**:
  - Ratings normalize to canonical 2-tier: `BUY`, `WNS` (`test_strict_5tier.py:38-43`).
  - Legacy ratings (`HOLD`, `SELL`, `UNDERWEIGHT`) normalize to `WNS` (`test_strict_5tier.py:57-61`).
  - `OVERWEIGHT` normalizes to `BUY` (`test_strict_5tier.py:39`).
  - Spot long-only execution strictly rejects short positions, pyramid additions, and reverse orders (`test_strict_5tier.py:51-76`, `test_stock_order_generator.py:83-120`, `test_audit_fixes_10subagents.py:51-64`).
- **Security & PIT Cutoff Guards**:
  - Directory traversal blocking on ticker paths (`test_safe_ticker_component.py:1-52`).
  - Multi-exchange ticker suffix preservation (`test_ticker_symbol_handling.py:1-21`).
  - Data window slicing strictly enforces `trade_date` cutoff (`test_data_window.py:1-278`).
  - Single-shot result bundle leakage audit verification (`test_single_shot_results.py:1-90`).

### Potential Mismatch / Inconsistency Vectors
- [ ] Horizon bounds: PM model outputs (e.g. 126 or 252 days) vs backtester engine execution bounds (capped at 63 trading days).
- [ ] Sizing formulas: Target position percentage (`target_position_pct`) vs fixed dollar sizing vs margin-capped share allocations.
- [ ] ATR lookback calculations: Pre-filtering OHLCV to `current_date` to prevent lookahead bias when computing 14-period ATR (`test_snapshot_enforcement.py:83-110`).
- [ ] Safe ticker validation: Regex rejects path separators `/`, `\`, null bytes `\x00`, and traversal strings `..` before filesystem joins.

---

## 8. Complete Architectural Verification Checklist

```markdown
### Subsystem 10: Config Resolvers & Test Matrix Crosscheck

- [ ] 10.1 Environment Overrides (`default_config.py:10-50`)
  - [ ] `_ENV_OVERRIDES` registry covers all 17 active `TRADINGAGENTS_*` environment variables.
  - [ ] Dynamic type coercion correctly handles boolean strings (`"true"`, `"1"`, `"yes"`, `"on"`).
  - [ ] Int coercion excludes booleans and raises `ValueError` on malformed numeric strings.
  - [ ] `None` defaults for LLM reasoning effort parameters preserve provider-specific defaults.

- [ ] 10.2 Dataflows Config Isolation (`dataflows/config.py:1-53`)
  - [ ] `initialize_config()` creates independent `deepcopy` of `DEFAULT_CONFIG`.
  - [ ] `get_config()` returns isolated `deepcopy`, preventing mutation of global state.
  - [ ] `set_config()` merges nested dicts (`data_vendors`, `tool_vendors`) one level deep.
  - [ ] `is_point_in_time_mode()` fail-closed check enforces snapshot routing when `point_in_time_mode` or `backtest_mode` is True.

- [ ] 10.3 Backtest Config Resolver & Key Validation (`config_resolver.py:1-141`, `api_key_env.py:1-48`)
  - [ ] Resolution order strictly respects: YAML value > Env var > ConfigError.
  - [ ] Model fallback checks `TRADINGAGENTS_LLM_MODEL` then `TRADINGAGENTS_DEEP_THINK_LLM`.
  - [ ] API key validator checks `api_key_env.py` mapping for all non-local providers.
  - [ ] Local providers (`ollama`) bypass API key validation cleanly.
  - [ ] Dual-region providers (`qwen-cn`, `glm-cn`, `minimax-cn`) enforce region-specific API keys.

- [ ] 10.4 Config Dataclass Hierarchy (`position.py:72-296, 474-563`)
  - [ ] `ExecutionConfig` validates `lot_size >= 1`, `contract_multiplier > 0`, non-negative fees, and `initial_position_side == PositionSide.FLAT`.
  - [ ] `MarginConfig` validates `maintenance_margin_pct <= initial_margin_pct` and `max_leverage >= 1.0`.
  - [ ] `RiskConfig` enforces ATR R:R >= 2.0 (`atr_tp_multiplier=3.0`, `atr_stop_multiplier=1.5`).
  - [ ] `DecisionMappingConfig` locks `mode = "spot_long_only"` and rejects all short-selling flags.
  - [ ] `BacktestConfig.validate()` fail-closed assertions enforce snapshot-only data/news providers and disabled live web/memory search.
  - [ ] Lookback days restricted strictly to `ALLOWED_LOOKBACKS` `(None, 5, 10, 20, 40, 60, 80, 100, 120, 240)`.

- [ ] 10.5 CLI Interactive & Headless Ingestion (`cli/commands/backtest.py:69-242`, `cli/commands/backtest_prompts.py:1-223`)
  - [ ] Date prompts validate ISO-8601 format and enforce `start_date <= end_date`.
  - [ ] Interactive prompts export to `os.environ` and trigger `importlib.reload` on config modules.
  - [ ] `BacktestEngine.from_dict` used instead of `from_yaml` to preserve prompted values.
  - [ ] Headless runner serializes sanitized config with API keys and passwords redacted (`[REDACTED]`).

- [ ] 10.6 Model Validation Matrix (`test_model_validation.py:1-55`)
  - [ ] Catalog models pass `validate_model` across all providers.
  - [ ] Strict providers raise warnings on unknown model names.
  - [ ] `ollama` and `openrouter` allow arbitrary custom model strings without warnings.

- [ ] 10.7 Quant Invariants & Unit Scaling Matrix (`test_audit_invariants.py:1-237`, `test_risk_unit_fix.py:1-138`)
  - [ ] Signal contract holding horizon clamped to `le=63` trading days.
  - [ ] Limit orders guarantee price improvement on gap-downs and cap execution price on gap-ups.
  - [ ] Force-close liquidations execute as market orders at open through gap-downs.
  - [ ] Risk thresholds and unrealized PnL consistently scaled as decimal fractions (`0.05` = 5%).
  - [ ] 5-tier to 2-tier rating mapping converts `BUY`/`OVERWEIGHT` to `BUY`, and all other ratings to `WNS`.
  - [ ] Safe ticker sanitization blocks directory traversal across all filesystem joins.
```

---

