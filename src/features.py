
import os

import pandas as pd
import pandas_market_calendars as mcal
import psycopg2


# 1. Connect to PostgreSQL
conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="postgres",
    user="postgres",
    password=os.environ["PGPASSWORD"]
)

# 2. Retrieve historical prices
query = """
    SELECT ticker, date, open, high, low,
           close, adj_close, volume
    FROM prices
    ORDER BY ticker, date;
"""

# 3. Load the results into a pandas DataFrame
with conn.cursor() as cursor:
    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    sortedData = cursor.fetchall()

df = pd.DataFrame(sortedData, columns=columns)

# 4. Close the database connection
conn.close()

# 5. Display the first five rows
print(df.head())

# 6. Check for missing values
print("\nMissing values by column:")
print(df.isna().sum())


# 7. Check for invalid prices
price_columns = ["open", "high", "low", "close", "adj_close"]

for column in price_columns:
    invalid = (df[column] <= 0).sum()
    print(f"{column}: {invalid} zero or negative values")



# 8. Check for missing trading dates using the NYSE calendar
print("\nChecking missing trading dates:")

nyse = mcal.get_calendar("NYSE")

for ticker, group in df.groupby("ticker"):
    dates = pd.to_datetime(group["date"])

    expected = nyse.valid_days(
        start_date=dates.min(),
        end_date=dates.max()
    )

    # Convert the dates into sets for comparison
    expected_dates = set(expected.date)
    actual_dates = set(dates.dt.date)

    missing = sorted(expected_dates - actual_dates)

    print(f"{ticker}: {len(missing)} missing trading days")

    if missing:
        print(f"  First missing dates: {missing[:10]}")


# 9. Check for unusually large daily returns
print("\nChecking unusually large daily returns:")

for ticker, group in df.groupby("ticker"):
    prices = pd.to_numeric(group["adj_close"])
    daily_returns = prices.pct_change(fill_method=None)

    unusual = (daily_returns.abs() > 0.20).sum()

    print(f"{ticker}: {unusual} days with changes over 20%")


# 10. Investigate the top 10 largest daily returns above 20% per ticker
print("\nInvestigating large price changes:")

threshold = 20
max_results = 10

for ticker, group in df.groupby("ticker"):
    tickerData = group.sort_values("date").copy()

    prices = pd.to_numeric(tickerData["adj_close"])

    tickerData["daily_return"] = (
        prices.pct_change(fill_method=None) * 100
    )

    # Find returns greater than 20% in magnitude
    unusual_days = tickerData[
        tickerData["daily_return"].abs() > threshold
    ].copy()

    # Skip tickers without unusual returns
    if unusual_days.empty:
        continue

    # Sort by absolute percentage change
    unusual_days["absolute_return"] = (
        unusual_days["daily_return"].abs()
    )

    unusual_days = unusual_days.sort_values(
        "absolute_return",
        ascending=False
    ).head(max_results)

    print(f"\n{ticker} | Threshold: {threshold}%")
    print(f"Top {len(unusual_days)} largest changes:")

    for _, row in unusual_days.iterrows():
        print(
            f"{row['date']} | "
            f"{row['daily_return']:+.2f}%"
        )

