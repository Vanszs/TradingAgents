import subprocess
import json
import os
import sys

conglomerate_tests = [
    ("BREN.JK", "2023-11-03", "Barito / Prajogo Pangestu", "Strong Momentum IPO Breakout"),
    ("TPIA.JK", "2023-12-01", "Barito / Prajogo Pangestu", "Multi-Month Base Breakout"),
    ("BUMI.JK", "2022-08-12", "Bakrie & Salim Group", "Commodity Supercycle Breakout"),
    ("BRPT.JK", "2023-11-24", "Barito / Prajogo Pangestu", "MA 50/200 Reclaim Expansion"),
    ("BBCA.JK", "2023-11-10", "Djarum Group / Hartono", "Bull Trend Pullback Support"),
    ("ASII.JK", "2024-05-02", "Astra International", "Sub-200 SMA Breakdown (Expected: WNS)"),
    ("ADRO.JK", "2022-02-18", "Boy Thohir / Saratoga", "Energy Commodity Breakout"),
    ("UNTR.JK", "2023-08-25", "Astra Group", "Mining Heavy Equip Expansion"),
    ("MDKA.JK", "2023-10-20", "Saratoga / Thohir Group", "Downtrend Channel Knife (Expected: WNS / High-Risk)"),
    ("INDF.JK", "2024-02-15", "Salim Group", "FMCG Base Consolidation Breakout")
]

print(f"=== Starting Execution of {len(conglomerate_tests)} Indonesian Conglomerate Stock Backtests ===")

for idx, (ticker, trade_date, group, thesis_type) in enumerate(conglomerate_tests, start=1):
    print(f"\n[{idx}/10] Running AI Backtest for {ticker} on {trade_date} ({group} - {thesis_type})...")
    cmd = [
        "python3", "-m", "cli.main", "evaluate-signal",
        "-t", ticker,
        "-d", trade_date,
        "--no-kronos"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ERROR executing {ticker}: {res.stderr[-300:]}")
        print(f"  STDOUT: {res.stdout[-300:]}")
    else:
        eval_f = f"result_backtest/{ticker}/{trade_date}/v1.0/evaluation.json"
        sig_f = f"result_backtest/{ticker}/{trade_date}/v1.0/signal.json"
        if os.path.exists(eval_f) and os.path.exists(sig_f):
            with open(eval_f) as f:
                e = json.load(f)
            with open(sig_f) as f:
                s = json.load(f)
            print(f"  SUCCESS {ticker}: Decision={s.get('action')} | EntryMode={s.get('entry_mode')} | Return={e.get('realized_return_pct')}% | Outcome={e.get('outcome')}")
        else:
            print(f"  WARNING: Output files not found for {ticker}")

print("\n=== All 10 Conglomerate Stock Backtests Finished ===")
