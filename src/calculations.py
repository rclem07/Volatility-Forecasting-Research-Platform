import numpy as np


# Calculate daily logarithmic returns
def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1))


# Calculate realized volatility over a rolling window
def calculate_volatility(log_returns, window):
    squared_returns = log_returns ** 2

    return np.sqrt(
        squared_returns.rolling(window=window).mean()
    )


# Calculate future realized volatility targets
def calculate_future_volatility(log_returns, window=5):
    volatility = calculate_volatility(log_returns, window)

    return volatility.shift(-window)
