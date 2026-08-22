from pathlib import Path

import pandas as pd
import yfinance as yf


def download_ohlcv(ticker: str, start_date: str, end_date: str, output_root: str = "data"):
    output_dir = Path(output_root) / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"[SKIP] No data for {ticker}")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    required_cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[required_cols]
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df.dropna(subset=["open", "high", "low", "close"])

    df.to_csv(output_dir / "ohlcv.csv", index=False)

    for filename in [
        "news.json",
        "fundamentals.json",
        "sentiment.json",
        "broker_activity.json",
    ]:
        path = output_dir / filename
        if not path.exists():
            path.write_text("[]", encoding="utf-8")

    print(f"[OK] Saved {ticker}")


def main():
    tickers = [
        "BUMI.JK",
        # "BBCA.JK",
        # "BBRI.JK",
        # "TLKM.JK",
        # "ASII.JK",
    ]

    for ticker in tickers:
        download_ohlcv(
            ticker=ticker,
            start_date="2021-01-01",
            end_date="2026-06-05",
        )


if __name__ == "__main__":
    main()