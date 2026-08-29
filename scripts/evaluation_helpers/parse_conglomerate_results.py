import json
import os
import pandas as pd
import numpy as np

tests = [
    ("BREN.JK", "2023-11-03", "Barito / Prajogo Pangestu", "BUY (Momentum)", "Strong Momentum IPO Breakout Stage 2"),
    ("TPIA.JK", "2023-12-01", "Barito / Prajogo Pangestu", "BUY (Breakout)", "Multi-Month Base Breakout Rp 3.000"),
    ("BUMI.JK", "2022-08-12", "Bakrie & Salim Group", "BUY (Supercycle)", "Commodity Supercycle Breakout Rp 133"),
    ("BRPT.JK", "2023-11-24", "Barito / Prajogo Pangestu", "BUY (Reclaim)", "MA 50/200 Reclaim Expansion"),
    ("BBCA.JK", "2023-11-10", "Djarum Group / Hartono", "BUY (Pullback)", "Bull Trend Pullback Support Rp 8.950"),
    ("ASII.JK", "2024-05-02", "Astra International", "WNS (Falling Knife)", "Sub-200 SMA Breakdown / Liquidity Drain"),
    ("ADRO.JK", "2022-02-18", "Boy Thohir / Saratoga", "BUY (Breakout)", "Energy Commodity Multi-Month Breakout"),
    ("UNTR.JK", "2023-08-25", "Astra Group", "BUY (Expansion)", "Mining Heavy Equip Trend Continuation"),
    ("MDKA.JK", "2023-10-20", "Saratoga / Thohir Group", "WNS (Downtrend Risk)", "Downtrend Channel Knife (High-Risk Base)"),
    ("INDF.JK", "2024-02-15", "Salim Group", "BUY (Consolidation)", "FMCG Value Consolidation Breakout")
]

results = []
for ticker, trade_date, group, expected_action, desc in tests:
    eval_f = f"result_backtest/{ticker}/{trade_date}/v1.0/evaluation.json"
    sig_f = f"result_backtest/{ticker}/{trade_date}/v1.0/signal.json"
    rep_f = f"result_backtest/{ticker}/{trade_date}/v1.0/agent_report.md"
    
    if os.path.exists(eval_f) and os.path.exists(sig_f):
        with open(eval_f) as f:
            e = json.load(f)
        with open(sig_f) as f:
            s = json.load(f)
        
        # Load OHLCV for forward actual stats
        df = pd.read_csv(f"data/{ticker}/ohlcv.csv")
        idx = df[df['date'] == trade_date].index[0]
        p0 = df.iloc[idx]['close']
        fwd = df.iloc[idx+1:idx+31]
        fwd_max = (fwd['high'].max() - p0) / p0 * 100
        fwd_min = (fwd['low'].min() - p0) / p0 * 100
        
        results.append({
            'ticker': ticker,
            'date': trade_date,
            'group': group,
            'expected_action': expected_action,
            'desc': desc,
            'p0': p0,
            'fwd_max': fwd_max,
            'fwd_min': fwd_min,
            'ai_action': s.get('action'),
            'entry_mode': s.get('entry_mode'),
            'planned_entry': s.get('planned_entry_price'),
            'actual_entry': e.get('actual_entry_price'),
            'stop_loss': s.get('stop_loss'),
            'take_profit': s.get('take_profit'),
            'planned_rr': e.get('planned_rr_ratio'),
            'realized_return': e.get('realized_return_pct', 0.0),
            'outcome': e.get('outcome'),
            'holding_days': e.get('actual_holding_days', 0),
            'mae': e.get('max_adverse_excursion_pct', 0.0),
            'mfe': e.get('max_favorable_excursion_pct', 0.0),
            'thesis': s.get('thesis_summary', '')
        })
    else:
        results.append({
            'ticker': ticker,
            'date': trade_date,
            'group': group,
            'expected_action': expected_action,
            'desc': desc,
            'status': 'PENDING / NOT FINISHED'
        })

print(json.dumps(results, indent=2))
