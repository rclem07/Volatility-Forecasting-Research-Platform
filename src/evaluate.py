
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_models import (
    predict_naive,
    predict_har,
    predict_garch
)


# 1. Create walk-forward splits for one ticker
def create_walk_forward_splits(tickerData):

    tickerData = tickerData.sort_values("date").dropna(
        subset=[
            "log_return",
            "vol_1d",
            "vol_5d",
            "vol_22d",
            "target_vol_next_5d"
        ]
    ).reset_index(drop=True)

    firstTestYear = tickerData["date"].dt.year.min() + 5
    lastTestYear = tickerData["date"].dt.year.max()

    for year in range(firstTestYear, lastTestYear + 1):

        trainData = tickerData[
            tickerData["date"].dt.year < year
        ].copy()

        # Prevent future targets from crossing into testing
        trainData = trainData.iloc[:-5].copy()

        testData = tickerData[
            tickerData["date"].dt.year == year
        ].copy()

        if trainData.empty or testData.empty:
            continue

        yield year, trainData, testData


# 2. Evaluate models across all tickers
def main():

    df = pd.read_csv(
        "data/historical_features.csv",
        parse_dates=["date"]
    )

    allPredictions = []

    for ticker, group in df.groupby("ticker"):

        if ticker == "^VIX":
            continue

        print(f"\nEvaluating {ticker}:")

        for year, trainData, testData in create_walk_forward_splits(group):

            # Actual future volatility
            actual = testData["target_vol_next_5d"].to_numpy()

            # Generate predictions using both models
            models = {
                "Naive": predict_naive(testData),
                "HAR": predict_har(trainData, testData),
                "GARCH(1,1)": predict_garch(group, testData)
            }

            # Save each model's predictions
            for modelName, predictions in models.items():

                for date, actualValue, predictedValue in zip(
                    testData["date"], actual, predictions
                ):
                    allPredictions.append({
                        "ticker": ticker,
                        "date": date,
                        "test_year": year,
                        "model": modelName,
                        "actual": actualValue,
                        "prediction": predictedValue
                    })

            print(f"  {year}: {len(testData)} test observations")

    # 3. Combine all predictions into one DataFrame
    predictionsDF = pd.DataFrame(allPredictions)

    # Calculate prediction errors
    predictionsDF["error"] = (
        predictionsDF["prediction"] - predictionsDF["actual"]
    )

    # 4. Calculate overall metrics for each ticker and model
    results = predictionsDF.groupby(
        ["ticker", "model"]
    ).agg(
        observations=("error", "size"),
        mae_pp=("error", lambda x: x.abs().mean() * 100),
        rmse_pp=("error", lambda x: np.sqrt((x ** 2).mean()) * 100)
    ).reset_index()

    results = results.rename(columns={
        "mae_pp": "MAE (pp)",
        "rmse_pp": "RMSE (pp)"
    })

    results = results.sort_values(
        ["ticker", "RMSE (pp)"]
    )

    # 5. Calculate metrics separately for every test year
    yearlyResults = predictionsDF.groupby(
        ["ticker", "test_year", "model"]
    ).agg(
        observations=("error", "size"),
        mae_pp=("error", lambda x: x.abs().mean() * 100),
        rmse_pp=("error", lambda x: np.sqrt((x ** 2).mean()) * 100)
    ).reset_index()

    yearlyResults = yearlyResults.rename(columns={
        "mae_pp": "MAE (pp)",
        "rmse_pp": "RMSE (pp)"
    })

    yearlyResults = yearlyResults.sort_values(
        ["ticker", "test_year", "RMSE (pp)"]
    )

    # 6. Display the overall comparison table
    print("\nWalk-Forward Model Comparison:")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # 7. Display SPY's yearly comparison
    print("\nSPY Yearly Walk-Forward Results:")

    print(
        yearlyResults[
            yearlyResults["ticker"] == "SPY"
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # 8. Export predictions and results
    outputFolder = Path("data")
    outputFolder.mkdir(exist_ok=True)

    predictionsDF.to_csv(
        outputFolder / "walk_forward_predictions.csv",
        index=False
    )

    results.to_csv(
        outputFolder / "walk_forward_summary.csv",
        index=False
    )

    yearlyResults.to_csv(
        outputFolder / "walk_forward_yearly_summary.csv",
        index=False
    )

    print("\nWalk-forward results successfully exported.")

    
    # 9. Target-shuffling sanity check for HAR
    print("\nRunning HAR target-shuffling check:")

    rng = np.random.default_rng(42)

    spyData = df[df["ticker"] == "SPY"]

    shuffledPredictions = []
    actualValues = []

    for year, trainData, testData in create_walk_forward_splits(spyData):

        # Make a copy so the original training data stays unchanged
        shuffledTrain = trainData.copy()

        # Randomize the historical answers
        shuffledTrain["target_vol_next_5d"] = rng.permutation(
            shuffledTrain["target_vol_next_5d"].to_numpy()
        )

        # Train HAR on shuffled answers
        predictions = predict_har(shuffledTrain, testData)

        shuffledPredictions.extend(predictions)
        actualValues.extend(
            testData["target_vol_next_5d"].to_numpy()
        )

    # Calculate error from shuffled-target training
    shuffledErrors = (
        np.array(shuffledPredictions) - np.array(actualValues)
    )

    shuffledRMSE = np.sqrt(
        (shuffledErrors ** 2).mean()
    ) * 100

    # Retrieve the original HAR results
    originalHAR = predictionsDF[
        (predictionsDF["ticker"] == "SPY") &
        (predictionsDF["model"] == "HAR")
    ]

    originalRMSE = np.sqrt(
        (originalHAR["error"] ** 2).mean()
    ) * 100

    print(f"Original HAR RMSE: {originalRMSE:.4f} pp")
    print(f"Shuffled HAR RMSE: {shuffledRMSE:.4f} pp")



if __name__ == "__main__":
    main()
