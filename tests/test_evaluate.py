
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
