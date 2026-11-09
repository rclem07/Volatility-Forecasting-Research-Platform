
import pandas as pd
import pandas_market_calendars as mcal


def validate_prices(df):
    # Check for missing values
    print("\nMissing values by column:")
    print(df.isna().sum())

    # Check for invalid prices
    price_columns = [
        "open", "high", "low", "close", "adj_close"
    ]

    for column in price_columns:
        invalid = (df[column] <= 0).sum()
        print(f"{column}: {invalid} zero or negative values")

    # Check for missing trading dates
    print("\nChecking missing trading dates:")

    nyse = mcal.get_calendar("NYSE")

    for ticker, group in df.groupby("ticker"):
        dates = pd.to_datetime(group["date"])

        expected = nyse.valid_days(
            start_date=dates.min(),
            end_date=dates.max()
        )

        expected_dates = set(expected.date)
        actual_dates = set(dates.dt.date)

        missing = sorted(expected_dates - actual_dates)

        print(f"{ticker}: {len(missing)} missing trading days")

        if missing:
            print(f"  First missing dates: {missing[:10]}")

    # Check for unusually large daily returns
    print("\nChecking unusually large daily returns:")

    for ticker, group in df.groupby("ticker"):
        tickerData = group.sort_values("date")

        prices = pd.to_numeric(tickerData["adj_close"])
        daily_returns = prices.pct_change(fill_method=None)

        unusual = (daily_returns.abs() > 0.20).sum()

        print(f"{ticker}: {unusual} days with changes over 20%")

    # Investigate the top 10 changes above 20% per ticker
    print("\nInvestigating large price changes:")

    threshold = 20
    max_results = 10

    for ticker, group in df.groupby("ticker"):
        tickerData = group.sort_values("date").copy()

        prices = pd.to_numeric(tickerData["adj_close"])

        tickerData["daily_return"] = (
            prices.pct_change(fill_method=None) * 100
        )

        unusual_days = tickerData[
            tickerData["daily_return"].abs() > threshold
        ].copy()

        if unusual_days.empty:
            continue

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


def validate_features(features):
    # Verify the final feature DataFrames
    print("\nVerifying feature data for each ticker:")

    feature_columns = [
        "log_return",
        "vol_1d",
        "vol_5d",
        "vol_22d",
        "target_vol_next_5d",
        "vix_level",
        "volume_change"
    ]

    for ticker, group in features.groupby("ticker"):
        tickerData = group.sort_values("date").copy()

        # Check for duplicate dates
        duplicates = tickerData["date"].duplicated().sum()

        # Check missing values in calculated features
        missing = tickerData[feature_columns].isna().sum()

        print(f"\nTicker: {ticker}")
        print(f"Total rows: {len(tickerData)}")
        print(f"Duplicate dates: {duplicates}")
        print("Missing feature values:")
        print(missing.to_string())
