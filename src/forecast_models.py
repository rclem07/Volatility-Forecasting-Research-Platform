
import numpy as np
import pandas as pd

from arch import arch_model
from sklearn.linear_model import LinearRegression


HAR_FEATURES = ["vol_1d", "vol_5d", "vol_22d"]


# 1. Naive volatility forecast
def predict_naive(testData):
    return testData["vol_5d"].to_numpy()


# 2. HAR volatility forecast
def predict_har(trainData, testData):
    model = LinearRegression()

    model.fit(
        trainData[HAR_FEATURES],
        trainData["target_vol_next_5d"]
    )

    return model.predict(testData[HAR_FEATURES])


# 3. GARCH(1,1) volatility forecast
def predict_garch(tickerData, testData):

    # Only include returns available through this test year
    lastTestDate = testData["date"].max()

    garchData = tickerData[
        tickerData["date"] <= lastTestDate
    ].sort_values("date").dropna(
        subset=["log_return"]
    ).copy()

    # Convert daily log returns into percentage units
    garchReturns = (
        garchData.set_index("date")["log_return"]
        .astype(float) * 100
    )

    # Find where the test year begins
    firstTestDate = testData["date"].min()
    splitIndex = garchReturns.index.get_loc(firstTestDate)

    # Create the GARCH(1,1) model
    model = arch_model(
        garchReturns,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    # Estimate parameters using returns before the test year
    fittedModel = model.fit(
        last_obs=splitIndex,
        disp="off"
    )

    # Forecast the next five trading days from each test date
    forecasts = fittedModel.forecast(
        horizon=5,
        start=splitIndex,
        align="origin",
        reindex=False
    )

    # Match forecasts to the exact test dates
    testDates = pd.DatetimeIndex(testData["date"])

    forecastVariances = forecasts.variance.reindex(
        testDates
    )

    # Each date must have five valid variance forecasts
    if forecastVariances.isna().any().any():
        raise ValueError("Missing GARCH variance forecasts")

    # Convert five daily variance forecasts into
    # one predicted daily-scale five-day volatility
    predictions = (
        np.sqrt(
            forecastVariances.mean(axis=1).to_numpy()
        ) / 100
    )

    if not np.isfinite(predictions).all():
        raise ValueError("Invalid GARCH predictions")

    return predictions
