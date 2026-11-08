
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import psycopg2
from pathlib import Path


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


# 11. Calculate log returns and realized volatility
print("\nCalculating returns and volatility:")

featureData = []

for ticker, group in df.groupby("ticker"):
    tickerData = group.sort_values("date").copy()

    # Convert database prices into numeric values
    tickerData["adj_close"] = pd.to_numeric(
        tickerData["adj_close"]
    )

    # Calculate daily logarithmic returns
    tickerData["log_return"] = np.log(
        tickerData["adj_close"] /
        tickerData["adj_close"].shift(1)
    )

    # Square each daily log return
    squared_returns = tickerData["log_return"] ** 2

    # Calculate historical realized volatility
    tickerData["vol_1d"] = np.sqrt(squared_returns)

    tickerData["vol_5d"] = np.sqrt(
        squared_returns.rolling(window=5).mean()
    )

    tickerData["vol_22d"] = np.sqrt(
        squared_returns.rolling(window=22).mean()
    )

    # Calculate volatility over the NEXT 5 trading days
    tickerData["target_vol_next_5d"] = (
        tickerData["vol_5d"].shift(-5)
    )

    featureData.append(tickerData)

# Combine all ticker DataFrames
features = pd.concat(featureData, ignore_index=True)

# Display the first 10 rows for SPY
spy = features[features["ticker"] == "SPY"]

print(
    spy[
        [
            "date",
            "adj_close",
            "log_return",
            "vol_1d",
            "vol_5d",
            "vol_22d",
            "target_vol_next_5d"
        ]
    ].head(30).to_string(index=False)
)

# 12. Add VIX and trading volume features
print("\nAdding VIX and trading volume features:")

# Get the VIX closing price for each date
vixData = features[
    features["ticker"] == "^VIX"
][["date", "close"]].copy()

# Rename the VIX closing price column
vixData = vixData.rename(
    columns={"close": "vix_level"}
)

# Match VIX levels to other assets using their dates
features = features.merge(
    vixData,
    on="date",
    how="left",
    validate="many_to_one"
)

# Convert volume to numeric values
features["volume"] = pd.to_numeric(
    features["volume"],
    errors="coerce"
)

# Calculate daily volume changes separately for each ticker
features["volume_change"] = (
    features.groupby("ticker")["volume"]
    .pct_change(fill_method=None)
)

# VIX is an index, not a stock with ordinary share volume
features.loc[
    features["ticker"] == "^VIX",
    "volume_change"
] = np.nan

# Avoid infinite values caused by zero previous-day volume
features["volume_change"] = features[
    "volume_change"
].replace([np.inf, -np.inf], np.nan)

# Display sample results
print(
    features[
        features["ticker"] == "SPY"
    ][
        ["date", "ticker", "volume_change", "vix_level"]
    ].head(10).to_string(index=False)
)

# 13. Export calculated features to CSV

outputFolder = Path("data")
outputFolder.mkdir(exist_ok=True)

features.to_csv(
    outputFolder / "historical_features.csv",
    index=False
)

print("Features successfully exported to CSV.")

# 14. Plot SPY's historical realized volatility
print("\nCreating SPY volatility chart:")

# Select SPY observations
spyData = features[
    features["ticker"] == "SPY"
].copy()

# Convert dates into datetime objects
spyData["date"] = pd.to_datetime(spyData["date"])

# Convert daily volatility into annualized percentages
spyData["annualized_vol_22d"] = (
    spyData["vol_22d"] * np.sqrt(252) * 100
)

# Create the graph
plt.figure(figsize=(12, 6))

plt.plot(
    spyData["date"],
    spyData["annualized_vol_22d"],
    label="SPY 22-Day Realized Volatility"
)

# Add chart labels
plt.title("SPY 22-Day Annualized Realized Volatility (2010–Present)")
plt.xlabel("Date")
plt.ylabel("Annualized Volatility (%)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the graph to a separate folder
chartFolder = Path("charts")
chartFolder.mkdir(exist_ok=True)

plt.savefig(
    chartFolder / "spy_volatility.png",
    dpi=300
)

plt.show()

print("SPY volatility chart successfully saved.")
