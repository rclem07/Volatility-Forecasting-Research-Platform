
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from neural_models import VolatilityMLP
import matplotlib.pyplot as plt
from pathlib import Path


# 1. Select the financial features for the MLP
FEATURE_COLUMNS = [
    "vol_1d",
    "vol_5d",
    "vol_22d",
    "vix_level",
    "volume_change"
]


# 2. Load and prepare SPY data
df = pd.read_csv(
    "data/historical_features.csv",
    parse_dates=["date"]
)

spy = df[df["ticker"] == "SPY"].copy()

spy = spy.sort_values("date").dropna(
    subset=FEATURE_COLUMNS + ["target_vol_next_5d"]
).reset_index(drop=True)

# 3. Split chronologically into training, validation, and testing

# Reserve the last 20% for final testing
split = int(len(spy) * 0.80)

developmentData = spy.iloc[:split - 5].copy()
testData = spy.iloc[split:].copy()

# Reserve the last 20% of development data for validation
validationSplit = int(len(developmentData) * 0.80)

# Remove the last 5 training observations to prevent
# their future targets from overlapping validation
trainData = developmentData.iloc[:validationSplit - 5].copy()

validationData = developmentData.iloc[validationSplit:].copy()


# 4. Fit the scaler using training data ONLY
scaler = StandardScaler()

trainFeatures = scaler.fit_transform(
    trainData[FEATURE_COLUMNS]
)

validationFeatures = scaler.transform(
    validationData[FEATURE_COLUMNS]
)

testFeatures = scaler.transform(
    testData[FEATURE_COLUMNS]
)

# 5. Display the results
print("Training observations:", len(trainData))
print("Test observations:", len(testData))

print("\nOriginal training features:")
print(trainData[FEATURE_COLUMNS].head())

print("\nNormalized training features:")
print(trainFeatures[:5])

print("\nTraining feature means:")
print(np.round(trainFeatures.mean(axis=0), 4))

print("\nTraining feature standard deviations:")
print(np.round(trainFeatures.std(axis=0), 4))


# 6. Convert normalized features into PyTorch tensors

trainX = torch.tensor(
    trainFeatures,
    dtype=torch.float32
)

trainY = torch.tensor(
    trainData["target_vol_next_5d"].to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)

testX = torch.tensor(
    testFeatures,
    dtype=torch.float32
)

testY = torch.tensor(
    testData["target_vol_next_5d"].to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)

validationX = torch.tensor(
    validationFeatures,
    dtype=torch.float32
)

validationY = torch.tensor(
    validationData["target_vol_next_5d"].to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)


# 7. Create a Dataset and DataLoader

trainDataset = TensorDataset(trainX, trainY)

trainLoader = DataLoader(
    trainDataset,
    batch_size=32,
    shuffle=True
)

validationDataset = TensorDataset(validationX, validationY)

validationLoader = DataLoader(
    validationDataset,
    batch_size=32,
    shuffle=False
)


# 8. Verify tensor and batch shapes

print("\nTraining tensor shape:", trainX.shape)
print("Training target shape:", trainY.shape)

batchX, batchY = next(iter(trainLoader))

print("\nFirst batch features:", batchX.shape)
print("First batch targets:", batchY.shape)


# 9. Verify dataset sizes and chronological boundaries

print("\nFinal dataset sizes:")
print("Training:", len(trainData))
print("Validation:", len(validationData))
print("Testing:", len(testData))

print("\nValidation feature shape:", validationX.shape)
print("Validation target shape:", validationY.shape)

print("\nChronological boundaries:")
print("Training ends:", trainData["date"].max())
print("Validation starts:", validationData["date"].min())
print("Validation ends:", validationData["date"].max())
print("Testing starts:", testData["date"].min())


# 10. Train the MLP neural network

# Set a random seed so results are reproducible
torch.manual_seed(42)

# Create the model
model = VolatilityMLP(input_size=5)

# Define the loss function and optimizer
lossFunction = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training settings
epochs = 100
patience = 10

bestValidationLoss = float("inf")
epochsWithoutImprovement = 0

bestModelWeights = None

trainingLosses = []
validationLosses = []

for epoch in range(epochs):

    # Enable training mode
    model.train()

    totalTrainingLoss = 0

    for batchX, batchY in trainLoader:

        # Clear previous gradients
        optimizer.zero_grad()

        # Generate predictions
        predictions = model(batchX)

        # Calculate loss
        loss = lossFunction(predictions, batchY)

        # Calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        totalTrainingLoss += loss.item() * len(batchX)

    averageTrainingLoss = totalTrainingLoss / len(trainDataset)

    # Evaluate on validation data
    model.eval()

    totalValidationLoss = 0

    with torch.no_grad():

        for batchX, batchY in validationLoader:

            predictions = model(batchX)

            loss = lossFunction(predictions, batchY)

            totalValidationLoss += loss.item() * len(batchX)

    averageValidationLoss = (
        totalValidationLoss / len(validationDataset)
    )

    trainingLosses.append(averageTrainingLoss)
    validationLosses.append(averageValidationLoss)

    # Save the best-performing model weights
    if averageValidationLoss < bestValidationLoss:

        bestValidationLoss = averageValidationLoss
        epochsWithoutImprovement = 0

        bestModelWeights = {
            key: value.clone()
            for key, value in model.state_dict().items()
        }

    else:
        epochsWithoutImprovement += 1

    print(
        f"Epoch {epoch + 1}: "
        f"Training Loss = {averageTrainingLoss:.8f}, "
        f"Validation Loss = {averageValidationLoss:.8f}"
    )

    # Stop when validation loss stops improving
    if epochsWithoutImprovement >= patience:
        print("\nEarly stopping triggered.")
        break

# Restore the best-performing model
model.load_state_dict(bestModelWeights)

print("\nMLP training completed.")
print(f"Best validation loss: {bestValidationLoss:.8f}")


# 11. Visualize MLP training and validation loss

# Identify the epoch with the lowest validation loss
bestEpoch = np.argmin(validationLosses) + 1

# Create the graph
plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(trainingLosses) + 1),
    trainingLosses,
    label="Training Loss"
)

plt.plot(
    range(1, len(validationLosses) + 1),
    validationLosses,
    label="Validation Loss"
)

# Mark the best epoch
plt.axvline(
    bestEpoch,
    linestyle="--",
    label=f"Best Epoch ({bestEpoch})"
)

plt.title("MLP Training vs. Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the graph
chartFolder = Path("charts")
chartFolder.mkdir(exist_ok=True)

plt.savefig(
    chartFolder / "mlp_training_loss.png",
    dpi=300
)

plt.show()
plt.close()

print("MLP training loss chart successfully saved.")


# 12. Evaluate the trained MLP on unseen test data

# Switch the model into evaluation mode
model.eval()

# Generate predictions without calculating gradients
with torch.no_grad():
    testPredictions = model(testX)

# Convert PyTorch tensors into NumPy arrays
predictions = testPredictions.numpy().flatten()
actual = testY.numpy().flatten()

# Calculate prediction errors
errors = predictions - actual

# Calculate MAE and RMSE
testMAE = np.abs(errors).mean()
testRMSE = np.sqrt((errors ** 2).mean())

# Display results
print("\nSPY MLP Test Results:")
print(f"Test observations: {len(actual)}")
print(f"MAE: {testMAE * 100:.4f} percentage points")
print(f"RMSE: {testRMSE * 100:.4f} percentage points")

# Check for negative volatility predictions
negativePredictions = (predictions < 0).sum()
print(f"Negative predictions: {negativePredictions}")
