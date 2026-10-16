# Volatility Forecasting Pipeline

A research pipeline for forecasting short-term market volatility and
comparing models with honest evaluation.

**Status: In progress** (started October 2026)

## Goal
Forecast the next 5-day realized volatility for a small set of assets and
test whether a neural network beats simple, well-known baselines.

## Why volatility?
Predicting price direction is mostly noise. Volatility clusters (calm
periods follow calm periods, turbulent follow turbulent), so it is
forecastable. Strong simple baselines also make it a good test of whether a
complex model adds real value.

## Planned scope
- [ ] Ingest 15+ years of daily prices for ~10 assets into PostgreSQL
- [ ] Build features (realized volatility, VIX, volume) with pandas and NumPy
- [ ] Baselines: naive, HAR, and GARCH
- [ ] Neural models in PyTorch: MLP and LSTM
- [ ] Walk-forward validation with a data leakage check
- [ ] Backtest a volatility-targeting strategy vs. buy-and-hold
- [ ] Log experiments to DynamoDB and store raw data in AWS S3
- [ ] Unit tests for feature and evaluation logic

## Tech stack (planned)
Python, pandas, NumPy, PyTorch, scikit-learn, arch, PostgreSQL, Docker,
AWS (S3, DynamoDB)

## Results
N/A

## Limitations
- Uses free, daily-frequency data (yfinance), which is unofficial and may
  contain gaps or errors
- Small set of assets; results may not generalize to other markets
- This is a research project, not a trading system. Backtests ignore many
  real-world factors such as slippage, liquidity, and taxes
- Past performance of any model does not predict future results

## Setup
Instructions coming as the project develops.
