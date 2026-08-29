import subprocess
import json
import os
import concurrent.futures

global_tests = [
    ("NVDA", "2024-05-17", "AI Momentum High Breakout"),
    ("MSFT", "2023-10-26", "Bullish Pullback Support"),
    ("AMZN", "2024-11-22", "Trend Continuation"),
    ("TSM", "2024-01-11", "Secular Semiconductor Breakout"),
    ("AMD", "2024-01-05", "Data Center Momentum Expansion")
]

def run_single(item):
    ticker, trade_date, thesis_type = item
    print(f"[START GLOBAL] {ticker} ({trade_date}) - {thesis_type}")
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
        with open(eval_f) as f:
            e = json.load(f)
        with open(sig_f) as f:
            s = json.load(f)
        print(f"[DONE GLOBAL] {ticker}: Sinyal={s.get('action')} | Mode={s.get('entry_mode')} | Ret={e.get('realized_return_pct')}% | Outcome={e.get('outcome')}")
        return (ticker, True)
    else:
        print(f"[FAIL GLOBAL] {ticker}: {res.stderr[-200:]}")
        return (ticker, False)

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(run_single, global_tests))
    print("Completed 5 Global Re-runs:", results)
