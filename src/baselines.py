
import numpy as np
import pandas as pd


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

