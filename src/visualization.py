
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spy_volatility(features):
    print("\nCreating SPY volatility chart:")

    # Select SPY observations
    spyData = features[
        features["ticker"] == "SPY"
    ].copy()

    # Convert dates into datetime objects
    spyData["date"] = pd.to_datetime(spyData["date"])

    # Convert daily volatility into annualized percentages
    spyData["annualized_vol_22d"] = (
        spyData["vol_22d"] * np.sqrt(252) * 100
    )

    # Create the graph
    plt.figure(figsize=(12, 6))

    plt.plot(
        spyData["date"],
        spyData["annualized_vol_22d"],
        label="SPY 22-Day Realized Volatility"
    )

    # Add chart labels
    plt.title(
        "SPY 22-Day Annualized Realized Volatility (2010–Present)"
    )
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility (%)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the graph
    chartFolder = Path("charts")
    chartFolder.mkdir(exist_ok=True)

    plt.savefig(
        chartFolder / "spy_volatility.png",
        dpi=300
    )

    plt.show()
    plt.close()

    print("SPY volatility chart successfully saved.")
