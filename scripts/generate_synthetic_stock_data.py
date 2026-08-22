"""
Generate synthetic stock data for smoke tests.
Creates data/AAPL/ with a deterministic random-walk OHLCV.
"""
import math
import os
import random
import sys


def generate_aapl_data(data_dir="data", ticker="AAPL", days=30, seed=42):
    random.seed(seed)
    os.makedirs(os.path.join(data_dir, ticker), exist_ok=True)

    base_price = 150.0
    prices = []
    price = base_price
    for i in range(days):
        # Weekday-only (skip weekends)
        day_offset = i + (i // 5) * 2  # approximate weekday spacing
        pct = random.uniform(-0.02, 0.025)
        price = price * (1 + pct)
        price = max(price, 10.0)
        o = round(price * random.uniform(0.998, 1.002), 2)
        h = round(max(o, price) * random.uniform(1.001, 1.015), 2)
        low_val = round(min(o, price) * random.uniform(0.985, 0.999), 2)
        c = round(price, 2)
        v = random.randint(50_000_000, 200_000_000)
        prices.append((o, h, low_val, c, v))

    # Write OHLCV
    ohlcv_path = os.path.join(data_dir, ticker, "ohlcv.csv")
    with open(ohlcv_path, "w") as f:
        f.write("date,open,high,low,close,volume\n")
        for i, (o, h, low_val, c, v) in enumerate(prices):
            # Generate trading dates (skip weekends)
            from datetime import date, timedelta
            d = date(2024, 1, 2) + timedelta(days=i + (i // 5) * 2)
            while d.weekday() >= 5:
                d += timedelta(days=1)
            f.write(f"{d.isoformat()},{o},{h},{low_val},{c},{v}\n")

    # Empty sidecar JSON files
    for name in ("news.json", "fundamentals.json", "sentiment.json", "broker_activity.json"):
        path = os.path.join(data_dir, ticker, name)
        with open(path, "w") as f:
            f.write("[]")

    print(f"Generated {days} days of {ticker} data in {data_dir}/{ticker}/")
    return ohlcv_path


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    generate_aapl_data(data_dir=data_dir, days=days)
