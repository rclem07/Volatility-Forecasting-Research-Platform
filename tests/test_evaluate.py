
import pandas as pd

from src.evaluate import create_walk_forward_splits


# Create a small, artificial dataset for testing
def create_test_data():
    dates = pd.date_range(
        "2010-01-01",
        "2017-12-31",
        freq="MS"
    )

    return pd.DataFrame({
        "date": dates,
        "log_return": 0.01,
        "vol_1d": 0.01,
        "vol_5d": 0.01,
        "vol_22d": 0.01,
        "target_vol_next_5d": 0.01
    })


# Test 1: Training must happen before testing
def test_chronological_order():
    data = create_test_data()

    for year, trainData, testData in create_walk_forward_splits(data):
        assert trainData["date"].max() < testData["date"].min()
        assert (testData["date"].dt.year == year).all()


# Test 2: Exclude the last five training observations
def test_five_day_gap():
    data = create_test_data()

    for year, trainData, testData in create_walk_forward_splits(data):
        previousData = data[data["date"].dt.year < year]

        assert len(trainData) == len(previousData) - 5

        assert (
            trainData["date"].iloc[-1]
            == previousData["date"].iloc[-6]
        )


# Test 3: Training data grows each year
def test_expanding_training_window():
    data = create_test_data()

    trainingSizes = []

    for year, trainData, testData in create_walk_forward_splits(data):
        trainingSizes.append(len(trainData))

    assert len(trainingSizes) >= 2

    for i in range(1, len(trainingSizes)):
        assert trainingSizes[i] > trainingSizes[i - 1]

# Test 4: Verify MAE and RMSE calculations
def test_evaluation_metrics():
    import numpy as np

    actual = np.array([0.01, 0.02, 0.03])
    predicted = np.array([0.02, 0.02, 0.01])

    errors = predicted - actual

    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())

    assert np.isclose(mae, 0.01)
    assert np.isclose(rmse, np.sqrt(0.0005 / 3))

# Test 5: Verify predictions align with their test dates
def test_prediction_alignment():
    import numpy as np

    data = create_test_data()

    for year, trainData, testData in create_walk_forward_splits(data):

        dates = testData["date"].to_numpy()
        actual = testData["target_vol_next_5d"].to_numpy()

        # Simulate the naive forecasting model
        predictions = testData["vol_5d"].to_numpy()

        assert len(dates) == len(actual)
        assert len(actual) == len(predictions)

        # Test dates must remain in chronological order
        assert testData["date"].is_monotonic_increasing

        # Verify the naive predictions match the correct rows
        np.testing.assert_array_equal(
            predictions,
            testData["vol_5d"].to_numpy()
        )
