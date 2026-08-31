import subprocess
import json
import os
import concurrent.futures

tests = [
    # 5 Saham Indonesia
    ("TLKM.JK", "2024-04-19", "Telkom Indonesia (Value Dip / Sub-200 SMA)", "Expected: WNS (Falling Knife Protection)"),
    ("PGAS.JK", "2024-05-03", "Perusahaan Gas Negara (MA Reversal)", "Expected: BUY / WNS (Volatility Filter)"),
    ("ICBP.JK", "2023-05-19", "Indofood CBP (Consumer Trend Breakout)", "Expected: BUY (Defensive Expansion)"),
    ("MEDC.JK", "2023-08-25", "Medco Energi (Oil Commodity Momentum)", "Expected: BUY (Momentum Breakout)"),
    ("GOTO.JK", "2023-11-17", "GoTo Gojek Tokopedia (Tech Downchannel)", "Expected: WNS (Sub-200 SMA Falling Knife)"),
    # 5 Saham Global
    ("TSLA", "2024-07-01", "Tesla Inc. (High-Beta EV Breakout)", "Expected: BUY (Momentum Acceleration)"),
    ("NFLX", "2024-04-26", "Netflix Inc. (Post-Earnings Consolidation)", "Expected: BUY / WNS (Trend Support)"),
    ("AVGO", "2024-06-14", "Broadcom Inc. (AI Chip High-Vol Breakout)", "Expected: BUY / WNS (Overbought Gate)"),
    ("ASML", "2024-01-19", "ASML Holding (Semiconductor Expansion)", "Expected: BUY (Cycle Continuation)"),
    ("COST", "2023-12-15", "Costco Wholesale (Retail ATH Breakout)", "Expected: BUY (Dividend Gap Breakout)")
]

def run_single(item):
    ticker, trade_date, group, expected_desc = item
    print(f"[START] {ticker:8} ({trade_date}) - {group}")
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
        ret = e.get('realized_return_pct')
        ret_str = f"{ret:.2f}%" if ret is not None else "0.00%"
        print(f"[DONE] {ticker:8} ({trade_date}): Signal={s.get('action'):4} | Mode={str(s.get('entry_mode')):7} | Ret={ret_str:8} | Outcome={e.get('outcome')}")
        return (ticker, True)
    else:
        print(f"[FAIL] {ticker:8}: {res.stderr[-200:]}")
        return (ticker, False)

if __name__ == '__main__':
    print("=== Launching 10 Random Stocks Multi-Agent Evaluation (5 Workers) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(run_single, tests))
    print("\nAll 10 Random Tests Completed:", results)
