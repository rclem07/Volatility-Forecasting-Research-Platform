
from pathlib import Path

import numpy as np
import pandas as pd

from calculations import (
    calculate_log_returns,
    calculate_volatility,
    calculate_future_volatility
)
from database import load_prices
from validation import validate_prices, validate_features
from visualization import plot_spy_volatility


# 1. Calculate returns and realized volatility
def build_features(df):
    print("\nCalculating returns and volatility:")

    featureData = []

    for ticker, group in df.groupby("ticker"):
        tickerData = group.sort_values("date").copy()

        # Convert database prices into numeric values
        tickerData["adj_close"] = pd.to_numeric(
            tickerData["adj_close"]
        )

        # Calculate daily logarithmic returns
        tickerData["log_return"] = calculate_log_returns(
            tickerData["adj_close"]
        )

        # Calculate historical realized volatility
        tickerData["vol_1d"] = calculate_volatility(
            tickerData["log_return"], 1
        )

        tickerData["vol_5d"] = calculate_volatility(
            tickerData["log_return"], 5
        )

        tickerData["vol_22d"] = calculate_volatility(
            tickerData["log_return"], 22
        )

        # Calculate volatility over the NEXT 5 trading days
        tickerData["target_vol_next_5d"] = (
            calculate_future_volatility(
                tickerData["log_return"], 5
            )
        )

        featureData.append(tickerData)

    # Combine all ticker DataFrames
    features = pd.concat(featureData, ignore_index=True)

    # Display the first 30 rows for SPY
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

    return features


# 2. Add VIX and trading volume features
def add_market_features(features):
    print("\nAdding VIX and trading volume features:")

    # Get the VIX closing price for each date
    vixData = features[
        features["ticker"] == "^VIX"
    ][["date", "close"]].copy()

    # Rename the VIX closing price column
    vixData = vixData.rename(
        columns={"close": "vix_level"}
    )

    # Match VIX levels to assets by date
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

    # Calculate daily volume changes for each ticker
    features["volume_change"] = (
        features.groupby("ticker")["volume"]
        .pct_change(fill_method=None)
    )

    # VIX does not have ordinary stock trading volume
    features.loc[
        features["ticker"] == "^VIX",
        "volume_change"
    ] = np.nan

    # Remove infinite values caused by zero previous volume
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

    return features


# 3. Export feature DataFrames to CSV
def export_features(features):
    # Export the combined dataset
    outputFolder = Path("data")
    outputFolder.mkdir(exist_ok=True)

    features.to_csv(
        outputFolder / "historical_features.csv",
        index=False
    )

    print("Features successfully exported to CSV.")

    # Export individual ticker datasets
    tickerFolder = Path("data/by_ticker")
    tickerFolder.mkdir(parents=True, exist_ok=True)

    for ticker, group in features.groupby("ticker"):
        tickerData = group.sort_values("date").copy()

        fileName = ticker.replace("^", "")

        tickerData.to_csv(
            tickerFolder / f"{fileName}.csv",
            index=False
        )

        print(f"Exported {ticker} data to {fileName}.csv")


# 4. Run the feature engineering pipeline
def main():
    # Load historical prices from PostgreSQL
    df = load_prices()
    print(df.head())

    # Validate original financial data
    validate_prices(df)

    # Calculate historical volatility features
    features = build_features(df)

    # Add market indicators
    features = add_market_features(features)

    # Export combined and per-ticker datasets
    export_features(features)

    # Generate SPY volatility chart
    plot_spy_volatility(features)

    # Verify final data quality
    validate_features(features)


if __name__ == "__main__":
    main()
