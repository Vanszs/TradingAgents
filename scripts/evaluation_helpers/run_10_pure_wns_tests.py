import subprocess
import json
import os
import concurrent.futures

ten_wns = [
    ("GIAA.JK", "2023-01-06", "Garuda Indonesia", "Expected: WNS (Dilution Crash / Sub-200 SMA)"),
    ("ERAA.JK", "2023-10-27", "Erajaya Swasembada", "Expected: WNS (Margin Downtrend Breakdown)"),
    ("ANTM.JK", "2024-01-19", "Aneka Tambang", "Expected: WNS (Nickel Supply Glut / Trend Breakdown)"),
    ("KLBF.JK", "2023-11-03", "Kalbe Farma", "Expected: WNS (Pharma Margin Squeeze Freefall)"),
    ("PGEO.JK", "2023-10-06", "Pertamina Geothermal", "Expected: WNS (Parabolic Exhaustion / -25% Drop)"),
    ("DIS", "2023-08-25", "Walt Disney Co", "Expected: WNS (Streaming Margin Loss / 9Y Low)"),
    ("SBUX", "2024-05-03", "Starbucks Corp", "Expected: WNS (Guidance Slash / Boycott Shock)"),
    ("BABA", "2022-01-28", "Alibaba Group", "Expected: WNS (China Macro Freefall / -33% Drop)"),
    ("PFE", "2023-12-15", "Pfizer Inc", "Expected: WNS (Vaccine Cliff Multi-Year Low)"),
    ("SNOW", "2024-03-01", "Snowflake Inc", "Expected: WNS (CEO Departure / -19% Drop)")
]

def run_single(item):
    ticker, trade_date, company, expected_desc = item
    eval_f = f"result_backtest/{ticker}/{trade_date}/v1.0/evaluation.json"
    sig_f = f"result_backtest/{ticker}/{trade_date}/v1.0/signal.json"
    
    if os.path.exists(eval_f) and os.path.exists(sig_f):
        print(f"[ALREADY DONE] {ticker:8} ({trade_date})")
        return (ticker, True)
        
    print(f"[START] {ticker:8} ({trade_date}) - {company}")
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
        print(f"[DONE] {ticker:8} ({trade_date}): Action={s.get('action'):4} | Return={ret_str:6} | Outcome={e.get('outcome')}")
        return (ticker, True)
    else:
        print(f"[FAIL] {ticker:8}: {res.stderr[-200:]}")
        return (ticker, False)

if __name__ == '__main__':
    print("=== Launching 10 Pure WNS Setups Evaluation (5 Parallel Workers) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(run_single, ten_wns))
    print("\nAll 10 Pure WNS Setups Finished:", results)
