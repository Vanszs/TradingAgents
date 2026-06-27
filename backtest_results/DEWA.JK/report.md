# Backtest Report - DEWA.JK (Stock)

## Period

- Asset Class: `stock`
- Ticker: `DEWA.JK`
- Start Date: `2025-01-25`
- End Date: `2025-05-02`
- Initial Cash: `100,000,000.00`
- Initial Margin %: `50.00%`
- Maintenance Margin %: `35.00%`

## Performance Summary

- Final Equity: `86,229,466.25`
- Total Return: `-13.77%`
- CAGR: `-44.47%`
- Max Drawdown: `-55.12%`
- Sharpe Ratio: `1.4318`
- Sortino Ratio: `2.4168`
- Win Rate: `29.41%`
- Profit Factor: `0.2394151256235316`
- Number of Trades: `33`
- Exposure Time: `49.15%`
- Benchmark Return: `None`
- Alpha: `None`

## Margin-Specific Metrics

- Average Leverage: `0.14x`
- Max Leverage: `0.44x`
- Average Margin Utilization: `7.10%`
- Max Margin Utilization: `22.32%`
- Margin Calls: `0`
- Liquidations: `0`
- Reverse Count: `0`
- Avg Holding Period (days): `0.00`
- Max Consecutive Wins: `1`
- Max Consecutive Losses: `3`
- Long Win Rate: `33.33%`
- Short Win Rate: `27.27%`
- Long Realized PnL: `-5,089,259.70`
- Short Realized PnL: `-7,023,834.10`
- Calmar Ratio: `-80.6744`
- Fee Drag: `1.8095%`
- Slippage Drag: `0.0000%`
- Total Fees: `1,809,506.23`
- Total Slippage: `0.00`
- Turnover Notional: `915,589,128.50`

## Margin Activity

| Date | Kind | Mark | Deficit | Account Equity | Action |
|------|------|------|---------|----------------|--------|
| 2025-02-03 | hard_risk | 110.89 | 0.00 | 129,587,110.08 | Max single trade risk exceeded: 0.65% |
| 2025-02-06 | hard_risk | 118.88 | 0.00 | 131,567,817.21 | Max single trade risk exceeded: 1.14% |
| 2025-02-11 | hard_risk | 109.89 | 0.00 | 130,807,841.98 | Max single trade risk exceeded: 1.03% |
| 2025-02-14 | hard_risk | 112.89 | 0.00 | 128,758,959.69 | Max single trade risk exceeded: 1.51% |
| 2025-03-10 | hard_risk | 129.13 | 0.00 | 68,569,870.06 | Max single trade risk exceeded: 0.62% |
| 2025-03-13 | hard_risk | 128.13 | 0.00 | 66,298,011.78 | Max single trade risk exceeded: 1.79% |
| 2025-03-18 | hard_risk | 118.12 | 0.00 | 64,773,520.48 | Max single trade risk exceeded: 1.46% |
| 2025-03-21 | hard_risk | 109.11 | 0.00 | 63,760,382.90 | Max single trade risk exceeded: 0.86% |
| 2025-03-26 | hard_risk | 97.10 | 0.00 | 66,002,782.69 | Max single trade risk exceeded: 1.68% |
| 2025-04-14 | hard_risk | 93.63 | 0.00 | 119,065,694.20 | Max single trade risk exceeded: 2.05% |
| 2025-04-16 | hard_risk | 111.62 | 0.00 | 92,064,519.98 | Max single trade risk exceeded: 0.10% |
| 2025-04-24 | hard_risk | 118.12 | 0.00 | 59,820,674.14 | Max single trade risk exceeded: 0.80% |
| 2025-04-29 | stop | 113.00 | 0.00 | 108,687,322.19 | Short stop triggered: high 129.0 >= stop 113.0 |

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
- `long_short_pnl_rules`: **PASSED**
- `reverse_fee_rule`: **PASSED**
- `margin_rule`: **PASSED**
- `leverage_limit`: **PASSED**

## Notes

- Asset class: **stock**. Margin, long+short, daily mark-to-market settlement, auto-liquidation.
- Decision pada tanggal `t` hanya dieksekusi pada trading day berikutnya.
- Harga eksekusi memakai `next session open` dengan ``tick_slippage`` ticks.
- Live provider wajib disabled saat backtest historis.
- ``account_state.csv`` berisi snapshot harian (cash, position, margin, leverage).
