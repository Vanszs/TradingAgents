# Backtest Report - DEWA.JK (Stock)

## Period

- Asset Class: `stock`
- Ticker: `DEWA.JK`
- Start Date: `2026-05-25`
- End Date: `2026-05-29`
- Initial Cash: `100,000,000.00`
- Initial Margin %: `50.00%`
- Maintenance Margin %: `35.00%`

## Performance Summary

- Final Equity: `100,000,000.00`
- Total Return: `0.00%`
- CAGR: `0.00%`
- Max Drawdown: `0.00%`
- Sharpe Ratio: `0.0000`
- Sortino Ratio: `0.0000`
- Win Rate: `0.00%`
- Profit Factor: `0.0`
- Number of Trades: `0`
- Exposure Time: `0.00%`
- Benchmark Return: `None`
- Alpha: `None`

## Margin-Specific Metrics

- Average Leverage: `0.00x`
- Max Leverage: `0.00x`
- Average Margin Utilization: `0.00%`
- Max Margin Utilization: `0.00%`
- Margin Calls: `0`
- Liquidations: `0`
- Reverse Count: `0`
- Avg Holding Period (days): `0.00`
- Max Consecutive Wins: `0`
- Max Consecutive Losses: `0`
- Long Win Rate: `0.00%`
- Short Win Rate: `0.00%`
- Long Realized PnL: `0.00`
- Short Realized PnL: `0.00`
- Calmar Ratio: `0.0000`
- Fee Drag: `0.0000%`
- Slippage Drag: `0.0000%`
- Total Fees: `0.00`
- Total Slippage: `0.00`
- Turnover Notional: `0.00`

## Margin Activity

No margin events.

## Leakage Audit

Status: **PASSED**

- `memory_isolation`: **PASSED**
- `live_provider_disabled`: **PASSED**
- `lookback_window_respected`: **PASSED**
- `required_decision_fields`: **PASSED**
- `snapshot_metadata_required`: **PASSED**
- `ohlcv_cutoff`: **PASSED**
- `last_data_date_cutoff`: **PASSED**
- `news_cutoff`: **PASSED**
- `fundamental_available_date`: **PASSED**
- `sentiment_cutoff`: **PASSED**
- `next_bar_execution`: **PASSED**

## Notes

- Asset class: **stock**. Margin, long+short, daily mark-to-market settlement, auto-liquidation.
- Decision pada tanggal `t` hanya dieksekusi pada trading day berikutnya.
- Harga eksekusi memakai `next session open` dengan ``tick_slippage`` ticks.
- Live provider wajib disabled saat backtest historis.
- ``account_state.csv`` berisi snapshot harian (cash, position, margin, leverage).
