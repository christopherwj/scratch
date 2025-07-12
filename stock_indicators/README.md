# Stock Indicator Analysis and Backtesting Project

This project aims to analyze stock performance by generating buy/sell signals from a combination of technical indicators. It includes a backtesting engine to simulate trading strategies and an optimization module to find the most profitable parameters for the indicators.

## Methodology

The core methodology is based on combining trend-following and momentum indicators to create a robust trading signal.

### 1. Indicators
We will start with two of the most popular and effective technical indicators:
- **Moving Average Convergence Divergence (MACD):** A trend-following momentum indicator that shows the relationship between two moving averages of a stock's price.
- **Relative Strength Index (RSI):** A momentum oscillator that measures the speed and change of price movements to identify overbought or oversold conditions.

### 2. Signal Generation
A weighted signal is generated from the indicators:
`Signal = (w_macd * Normalized_MACD) + (w_rsi * Normalized_RSI)`

Trading decisions (buy/sell) are made based on whether this signal crosses predefined thresholds.

### 3. Backtesting and Optimization
To find the optimal parameters (indicator periods, weights, thresholds), we will use an exhaustive backtesting approach on historical data.

- **Data Splitting:** The historical data is split into a training set (for optimization) and a test set (for validation) to prevent overfitting.
- **Optimization:** A Grid Search algorithm will be used to test numerous combinations of parameters against the training data.
- **Performance Metric:** The **Sharpe Ratio** will be the primary metric to identify the best parameter set, as it balances returns against risk.
- **Rolling-Window Optimization:** To keep the strategy adaptive, we will implement a rolling-window optimization to periodically update the parameters.

## Project Structure

```
.
├── data/
│   └── (raw stock data files)
├── notebooks/
│   └── (Jupyter notebooks for analysis and visualization)
├── src/
│   ├── data_loader.py
│   ├── indicators.py
│   ├── strategy.py
│   ├── backtester.py
│   ├── optimizer.py
│   └── main.py
├── requirements.txt
└── README.md
``` 