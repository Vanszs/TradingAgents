import subprocess
import json
import os
import concurrent.futures

remaining_tests = [
    ("ADRO.JK", "2022-02-18", "Boy Thohir / Saratoga", "Energy Commodity Breakout"),
    ("UNTR.JK", "2023-08-25", "Astra Group", "Mining Heavy Equip Expansion"),
    ("MDKA.JK", "2023-10-20", "Saratoga / Thohir Group", "Downtrend Channel Knife (Expected: WNS / High-Risk)"),
    ("INDF.JK", "2024-02-15", "Salim Group", "FMCG Base Consolidation Breakout")
]

def run_single(item):
    ticker, trade_date, group, thesis_type = item
    eval_f = f"result_backtest/{ticker}/{trade_date}/v1.0/evaluation.json"
    sig_f = f"result_backtest/{ticker}/{trade_date}/v1.0/signal.json"
    
    if os.path.exists(eval_f) and os.path.exists(sig_f):
        print(f"[ALREADY DONE] {ticker} ({trade_date})")
        return (ticker, trade_date, True)
        
    print(f"[START] {ticker} ({trade_date}) - {group} - {thesis_type}")
    cmd = [
        "python3", "-m", "cli.main", "evaluate-signal",
        "-t", ticker,
        "-d", trade_date,
        "--no-kronos"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] {ticker} ({trade_date}): {res.stderr[-200:]}")
        return (ticker, trade_date, False)
    else:
        if os.path.exists(eval_f) and os.path.exists(sig_f):
            with open(eval_f) as f:
                e = json.load(f)
            with open(sig_f) as f:
                s = json.load(f)
            print(f"[DONE] {ticker} ({trade_date}): Decision={s.get('action')} | Ret={e.get('realized_return_pct')}% | Outcome={e.get('outcome')}")
            return (ticker, trade_date, True)
        else:
            print(f"[WARN] {ticker} ({trade_date}) completed but output not found.")
            return (ticker, trade_date, False)

print("Starting remaining 4 tasks...")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_single, remaining_tests))

print("All remaining tasks finished:", results)
