import subprocess
import json
import os
import concurrent.futures

wns_tests = [
    ("ARTO.JK", "2022-06-17", "Bank Jago (Tech Banking Valuation Crash)", "Expected: WNS (Sub-200 SMA Freefall)"),
    ("BUKA.JK", "2022-03-11", "Bukalapak (Post-IPO Tech Freefall)", "Expected: WNS (Falling Knife below all MAs)"),
    ("WIKA.JK", "2023-05-12", "Wijaya Karya (SOE Debt Crisis / Default Risk)", "Expected: WNS (Fundamental Solvency Distress)"),
    ("BA", "2024-01-12", "Boeing Company (737-MAX Quality Failure)", "Expected: WNS (Regulatory / Quality Breakdown)"),
    ("PYPL", "2022-02-04", "PayPal Holdings (Post-Earnings -25% Freefall)", "Expected: WNS (Sub-200 SMA Liquidation Gap)")
]

def run_single(item):
    ticker, trade_date, company_desc, expected_desc = item
    eval_f = f"result_backtest/{ticker}/{trade_date}/v1.0/evaluation.json"
    sig_f = f"result_backtest/{ticker}/{trade_date}/v1.0/signal.json"
    
    if os.path.exists(eval_f) and os.path.exists(sig_f):
        print(f"[ALREADY DONE] {ticker} ({trade_date})")
        return (ticker, True)
        
    print(f"[START] {ticker:8} ({trade_date}) - {company_desc}")
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
        print(f"[DONE] {ticker:8} ({trade_date}): Signal={s.get('action'):4} | Mode={str(s.get('entry_mode')):7} | Ret={ret_str:8} | Outcome={e.get('outcome')}")
        return (ticker, True)
    else:
        print(f"[FAIL] {ticker:8}: {res.stderr[-200:]}")
        return (ticker, False)

if __name__ == '__main__':
    print("=== Launching 5 New WNS Setups (5 Parallel Workers) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(run_single, wns_tests))
    print("\nAll 5 New WNS Setups Completed:", results)
