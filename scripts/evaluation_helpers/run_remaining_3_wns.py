import subprocess
import json
import os
import concurrent.futures

remaining_wns = [
    ("PYFA.JK", "2024-03-22", "Pyridam Farma (Penny Stock Dilution Freefall)"),
    ("INTC", "2024-04-26", "Intel Corp (Post-Earnings Breakdown)"),
    ("NKE", "2024-06-28", "Nike Inc (Earnings Freefall Gap Down)")
]

def run_single(item):
    ticker, trade_date, desc = item
    eval_f = f"result_backtest/{ticker}/{trade_date}/v1.0/evaluation.json"
    sig_f = f"result_backtest/{ticker}/{trade_date}/v1.0/signal.json"
    
    if os.path.exists(eval_f) and os.path.exists(sig_f):
        print(f"[ALREADY DONE] {ticker} ({trade_date})")
        return (ticker, True)
        
    print(f"[START] {ticker} ({trade_date}) - {desc}")
    os.system(f"rm -f /home/vanszs/.tradingagents/cache/checkpoints/{ticker}.db*")
    
    cmd = [
        "python3", "-m", "cli.main", "evaluate-signal",
        "-t", ticker,
        "-d", trade_date,
        "--no-kronos"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if os.path.exists(eval_f) and os.path.exists(sig_f):
        with open(eval_f) as fp:
            e = json.load(fp)
        with open(sig_f) as fp:
            s = json.load(fp)
        ret = e.get('realized_return_pct')
        ret_str = f"{ret:.2f}%" if ret is not None else "0.00%"
        print(f"[DONE] {ticker} ({trade_date}): Action={s.get('action')} | Outcome={e.get('outcome')} | Return={ret_str}")
        return (ticker, True)
    else:
        print(f"[FAIL] {ticker}: {res.stderr[-200:]}")
        return (ticker, False)

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run_single, remaining_wns))
    print("Remaining 3 finished:", results)
