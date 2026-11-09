
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import LinearRegression

# 1. Load the processed financial data
df = pd.read_csv(
    "data/historical_features.csv",
    parse_dates=["date"]
)

# 2. Select SPY and sort chronologically
spy = df[df["ticker"] == "SPY"].copy()
spy = spy.sort_values("date")

# 3. Remove rows missing required values
spy = spy.dropna(
    subset=["vol_5d", "target_vol_next_5d"]
)

# 4. Reserve the most recent 20% for evaluation
split = int(len(spy) * 0.80)
testData = spy.iloc[split:].copy()

# 5. Make naive forecasts
testData["prediction"] = testData["vol_5d"]

# 6. Compare forecasts with actual future volatility
errors = (
    testData["prediction"] -
    testData["target_vol_next_5d"]
)

mae = errors.abs().mean()
rmse = np.sqrt((errors ** 2).mean())

# 7. Display evaluation results
print("SPY Naive Volatility Forecast")
print(f"Test observations: {len(testData)}")
print(f"MAE: {mae * 100:.4f} percentage points")
print(f"RMSE: {rmse * 100:.4f} percentage points")

# 8. HAR volatility forecasting model

# Select the three historical volatility features
harFeatures = ["vol_1d", "vol_5d", "vol_22d"]

# Prepare training data using earlier observations
# Exclude the last 5 training rows to prevent future leakage
trainData = spy.iloc[:split - 5].dropna(
    subset=harFeatures
).copy()

# Create the linear regression model
harModel = LinearRegression()

# Train the model on historical volatility data
harModel.fit(
    trainData[harFeatures],
    trainData["target_vol_next_5d"]
)

# Predict future volatility on the same test observations
testData["har_prediction"] = harModel.predict(
    testData[harFeatures]
)

# Calculate HAR prediction errors
harErrors = (
    testData["har_prediction"] -
    testData["target_vol_next_5d"]
)

# Calculate HAR accuracy metrics
harMAE = harErrors.abs().mean()
harRMSE = np.sqrt((harErrors ** 2).mean())

# Display HAR results
print("\nSPY HAR Volatility Forecast")
print(f"Training observations: {len(trainData)}")
print(f"Test observations: {len(testData)}")
print(f"MAE: {harMAE * 100:.4f} percentage points")
print(f"RMSE: {harRMSE * 100:.4f} percentage points")

# 9. Examine HAR model coefficients
print("\nHAR Model Coefficients:")

for feature, coefficient in zip(harFeatures, harModel.coef_):
    print(f"{feature}: {coefficient:.4f}")

print(f"Intercept: {harModel.intercept_:.6f}")

# 10. GARCH(1,1) volatility forecasting model
print("\nTraining GARCH model:")

# Convert logarithmic returns to percentage units
garchReturns = spy["log_return"].astype(float) * 100

# Create a GARCH(1,1) model
garchModel = arch_model(
    garchReturns,
    mean="Zero",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

# Fit parameters using only the training period
garchResult = garchModel.fit(
    last_obs=split,
    disp="off"
)

# Generate five-day forecasts for each test date
garchForecasts = garchResult.forecast(
    horizon=5,
    start=split,
    align="origin",
    reindex=False
)

# Convert the five daily variance forecasts into
# one daily-scale volatility prediction per test date
garchPredictions = (
    np.sqrt(garchForecasts.variance.mean(axis=1).to_numpy())
    / 100
)

# Verify that forecasts match the test observations
assert len(garchPredictions) == len(testData)
assert np.isfinite(garchPredictions).all()

testData["garch_prediction"] = garchPredictions

# Calculate GARCH prediction errors
garchErrors = (
    testData["garch_prediction"] -
    testData["target_vol_next_5d"]
)

garchMAE = garchErrors.abs().mean()
garchRMSE = np.sqrt((garchErrors ** 2).mean())

# Display GARCH results
print("\nSPY GARCH(1,1) Volatility Forecast")
print(f"Test observations: {len(testData)}")
print(f"MAE: {garchMAE * 100:.4f} percentage points")
print(f"RMSE: {garchRMSE * 100:.4f} percentage points")

# Display learned GARCH parameters
print("\nGARCH Model Parameters:")
print(garchResult.params)


# 11. Compare all three baseline models

results = pd.DataFrame({
    "Model": ["Naive", "HAR", "GARCH(1,1)"],
    "MAE (pp)": [mae * 100, harMAE * 100, garchMAE * 100],
    "RMSE (pp)": [rmse * 100, harRMSE * 100, garchRMSE * 100]
})

# Sort models by lowest RMSE
results = results.sort_values("RMSE (pp)")

print("\nBaseline Model Comparison:")
print(results.to_string(index=False, float_format="%.4f"))
