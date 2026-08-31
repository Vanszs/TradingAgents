import subprocess
import json
import os

wns_tests = [
    ("INCO.JK", "2023-11-24", "Vale Indonesia (Nickel Crash Sub-200 SMA)"),
    ("UNVR.JK", "2024-01-26", "Unilever Indonesia (FMCG Secular Breakdown)"),
    ("PYFA.JK", "2024-03-22", "Pyridam Farma (Penny Stock Dilution Freefall)"),
    ("INTC", "2024-04-26", "Intel Corp (Post-Earnings Breakdown)"),
    ("NKE", "2024-06-28", "Nike Inc (Earnings Freefall Gap Down)")
]

print("=== Starting Sequential (Linear) Execution for 5 WNS Setups ===")

for idx, (ticker, trade_date, desc) in enumerate(wns_tests, start=1):
    print(f"\n[{idx}/5] RUNNING: {ticker} on {trade_date} ({desc})...")
    # Clean old checkpoints
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
        print(f"  --> DONE {ticker}: Action={s.get('action')} | Outcome={e.get('outcome')} | Return={ret_str} | Thesis={s.get('thesis_summary')[:100]}...")
    else:
        print(f"  --> FAIL {ticker}: {res.stderr[-200:]}")

print("\n=== All 5 Sequential WNS Setups Completed ===")
