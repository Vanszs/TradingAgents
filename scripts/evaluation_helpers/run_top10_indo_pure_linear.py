import subprocess
import json
import os
import sys

top10_idx = [
    ("BBCA.JK", "2024-02-16", "Bank Central Asia (ATH Consolidation Breakout)", "Expected: BUY / WNS (Tight ATH R:R)"),
    ("BBRI.JK", "2023-11-10", "Bank Rakyat Indonesia (MA 50 Pullback Support)", "Expected: BUY (Dip in Uptrend)"),
    ("BMRI.JK", "2024-01-12", "Bank Mandiri (Uptrend ATH Breakout)", "Expected: BUY (Breakout Continuation)"),
    ("BBNI.JK", "2023-12-08", "Bank Negara Indonesia (Post-Stock Split Breakout)", "Expected: BUY (Momentum Reclaim)"),
    ("ASII.JK", "2023-05-19", "Astra International (Dividend Recovery Rebound)", "Expected: BUY (Value Recovery)"),
    ("TLKM.JK", "2023-11-03", "Telkom Indonesia (Support Base Reclaim)", "Expected: BUY (Base Reversal)"),
    ("AMMN.JK", "2023-10-13", "Amman Mineral (Copper Stage-2 Expansion)", "Expected: BUY (Momentum Expansion)"),
    ("BREN.JK", "2023-11-17", "Barito Renewables (Momentum Base Expansion)", "Expected: BUY (IPO Stage-2 Discovery)"),
    ("TPIA.JK", "2023-12-08", "Chandra Asri (Breakout Continuation)", "Expected: BUY (Momentum Acceleration)"),
    ("ADRO.JK", "2022-09-02", "Adaro Energy (Coal Supercycle Acceleration)", "Expected: BUY (Commodity Supercycle)")
]

print("=== Starting Pure Linear Execution (1 by 1) for Top 10 Indonesian Stocks ===")

for idx, (ticker, trade_date, company_desc, expected_desc) in enumerate(top10_idx, start=1):
    print(f"\n========================================================")
    print(f"[{idx}/10] EXECUTING: {ticker} on {trade_date}")
    print(f"Description: {company_desc}")
    print(f"Expectation: {expected_desc}")
    print(f"========================================================")
    sys.stdout.flush()
    
    # Clean old checkpoints for pristine linear test
    os.system(f"rm -f /home/vanszs/.tradingagents/cache/checkpoints/{ticker}.db*")
    
    cmd = [
        "python3", "-m", "cli.main", "evaluate-signal",
        "-t", ticker,
        "-d", trade_date,
        "--no-kronos"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    eval_f = f"result_backtest/{ticker}/{trade_date}/v1.0/evaluation.json"
    sig_f = f"result_backtest/{ticker}/{trade_date}/v1.0/signal.json"
    
    if os.path.exists(eval_f) and os.path.exists(sig_f):
        with open(eval_f) as fp:
            e = json.load(fp)
        with open(sig_f) as fp:
            s = json.load(fp)
        ret = e.get('realized_return_pct')
        ret_str = f"{ret:.2f}%" if ret is not None else "0.00%"
        print(f"--> SUCCESS: Action={s.get('action')} | Mode={s.get('entry_mode')} | Return={ret_str} | Outcome={e.get('outcome')}")
        print(f"--> Thesis: {s.get('thesis_summary')[:120]}...")
    else:
        print(f"--> FAIL: {res.stderr[-250:]}")
    sys.stdout.flush()

print("\n=== All 10 Pure Linear Top IDX Stock Backtests Completed ===")
