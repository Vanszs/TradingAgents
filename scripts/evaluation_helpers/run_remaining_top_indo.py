import subprocess
import json
import os
import concurrent.futures

remaining_top = [
    ("BMRI.JK", "2024-01-12", "Bank Mandiri (Uptrend ATH Breakout)", "Expected: BUY (Breakout Continuation)"),
    ("BBNI.JK", "2023-12-08", "Bank Negara Indonesia (Post-Stock Split Breakout)", "Expected: BUY (Momentum Reclaim)"),
    ("ASII.JK", "2023-05-19", "Astra International (Dividend Recovery Rebound)", "Expected: BUY (Value Recovery)"),
    ("TLKM.JK", "2023-11-03", "Telkom Indonesia (Support Base Reclaim)", "Expected: BUY (Base Reversal)"),
    ("AMMN.JK", "2023-10-13", "Amman Mineral (Copper Stage-2 Expansion)", "Expected: BUY (Momentum Expansion)"),
    ("BREN.JK", "2023-11-17", "Barito Renewables (Momentum Base Expansion)", "Expected: BUY (IPO Stage-2 Discovery)"),
    ("TPIA.JK", "2023-12-08", "Chandra Asri (Breakout Continuation)", "Expected: BUY (Momentum Acceleration)"),
    ("ADRO.JK", "2022-09-02", "Adaro Energy (Coal Supercycle Acceleration)", "Expected: BUY (Commodity Supercycle)")
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
    print("=== Running Remaining 8 Top IDX Stocks (4 Parallel Workers) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_single, remaining_top))
    print("\nAll Remaining Top IDX Stocks Completed:", results)
