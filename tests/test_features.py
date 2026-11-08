import numpy as np
import pandas as pd
import pytest

from src.calculations import (
    calculate_log_returns,
    calculate_volatility,
    calculate_future_volatility
)


# Test 1: Constant prices should produce zero volatility
def test_constant_prices():
    prices = pd.Series([100, 100, 100, 100, 100, 100])

    returns = calculate_log_returns(prices)
    volatility = calculate_volatility(returns, 5)

    assert (returns.iloc[1:] == 0).all()
    assert volatility.iloc[-1] == 0


# Test 2: Verify daily logarithmic returns
def test_log_returns():
    prices = pd.Series([100, 110, 121])

    actual = calculate_log_returns(prices)
    expected = np.log(1.10)

    assert actual.iloc[1] == pytest.approx(expected)
    assert actual.iloc[2] == pytest.approx(expected)


# Test 3: Verify five-day realized volatility
def test_realized_volatility():
    returns = pd.Series([0.10, -0.20, 0.10, -0.20, 0.10])

    actual = calculate_volatility(returns, 5)

    expected = np.sqrt(
        (0.10**2 + (-0.20)**2 + 0.10**2
         + (-0.20)**2 + 0.10**2) / 5
    )

    assert actual.iloc[-1] == pytest.approx(expected)

# Test 4: Verify future volatility target alignment
def test_future_volatility():
    returns = pd.Series([
        0.01, 0.02, 0.03, 0.04, 0.05,
        0.06, 0.07, 0.08, 0.09, 0.10
    ])

    actual = calculate_future_volatility(returns, 5)

    # Expected volatility from the NEXT five returns
    expected = np.sqrt(
        (0.02**2 + 0.03**2 + 0.04**2 +
         0.05**2 + 0.06**2) / 5
    )

    assert actual.iloc[0] == pytest.approx(expected)

    # The final five dates have no future five-day target
    assert actual.iloc[-5:].isna().all()
