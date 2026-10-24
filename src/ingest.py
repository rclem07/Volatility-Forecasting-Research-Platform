
import os

import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values


# 1. Define which tickers to download
TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
    "TSLA", "JPM", "TLT", "GLD", "^VIX"
]

START_DATE = "2010-01-01"

COLUMNS = [
    "Open", "High", "Low", "Close",
    "Adj Close", "Volume"
]


# 2. Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="postgres",
    user="postgres",
    password=os.environ["PGPASSWORD"]
)


# 3. Download and store each ticker
try:
    for ticker in TICKERS:
        print(f"Downloading {ticker}...")

        try:
            data = yf.download(
                tickers=ticker,
                start=START_DATE,
                interval="1d",
                auto_adjust=False,
                multi_level_index=False,
                progress=False
            )

            if data is None or data.empty:
                print(f"No data found for {ticker}")
                continue

            # 4. Check for missing data
            print(f"Missing values for {ticker}:")

            for column in COLUMNS:
                if column in data.columns:
                    missing = data[column].isna().sum()
                    print(f"  {column}: {missing}")
                else:
                    print(f"  {column}: unavailable")

            rows = []
            skipped = 0

            # 5. Prepare each day's data for PostgreSQL
            for date, values in data.iterrows():

                def clean(column):
                    if column not in data.columns:
                        return None

                    value = values[column]

                    return None if pd.isna(value) else float(value)

                # Skip rows without a closing price
                if clean("Close") is None:
                    skipped += 1
                    continue

                volume = clean("Volume")

                rows.append((
                    ticker,
                    date.date(),
                    clean("Open"),
                    clean("High"),
                    clean("Low"),
                    clean("Close"),
                    clean("Adj Close"),
                    int(volume) if volume is not None else None
                ))

            if not rows:
                print(f"No usable rows for {ticker}")
                continue

            # 6. Insert records into prices table
            with conn.cursor() as cursor:
                execute_values(
                    cursor,
                    """
                    INSERT INTO prices (
                        ticker, date, open, high, low,
                        close, adj_close, volume
                    )
                    VALUES %s
                    ON CONFLICT (ticker, date)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        adj_close = EXCLUDED.adj_close,
                        volume = EXCLUDED.volume;
                    """,
                    rows
                )

            conn.commit()

            print(f"Saved {len(rows)} rows for {ticker}")
            print(f"Skipped {skipped} rows without Close data")

        except Exception as error:
            conn.rollback()
            print(f"Error with {ticker}: {error}")

finally:
    conn.close()

print("Ingestion finished.")