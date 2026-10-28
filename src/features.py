
import os

import pandas as pd
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


# 8. Check for missing trading dates
print("\nChecking missing trading dates:")

for ticker, group in df.groupby("ticker"):
    dates = pd.DatetimeIndex(pd.to_datetime(group["date"]))

    # Business days exclude weekends, but not market holidays
    expected = pd.bdate_range(
        start=dates.min(),
        end=dates.max()
    )

    missing = expected.difference(dates)

    print(f"{ticker}: {len(missing)} weekday gaps")


# 9. Check for unusually large daily returns
print("\nChecking unusually large daily returns:")

for ticker, group in df.groupby("ticker"):
    prices = pd.to_numeric(group["adj_close"])
    daily_returns = prices.pct_change(fill_method=None)

    unusual = (daily_returns.abs() > 0.20).sum()

    print(f"{ticker}: {unusual} days with changes over 20%")
