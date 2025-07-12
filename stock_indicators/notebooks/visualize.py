import matplotlib.pyplot as plt
import sys
import pandas as pd

# Add project root to path to allow imports from src
sys.path.append('../')

from src.data_loader import fetch_data
from src.optimizer import Optimizer
from src.strategy import Strategy
from src.backtester import Backtester

def visualize_performance():
    """
    Runs the full workflow and visualizes the backtest performance against a benchmark.
    """
    # --- 1. Parameters ---
    TICKER = 'AAPL'
    BENCHMARK_TICKER = 'SPY' # S&P 500 ETF
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    TRAIN_TEST_SPLIT_DATE = pd.to_datetime('2023-01-01')
    
    # --- 2. Fetch and Split Data ---
    print(f"Fetching data for {TICKER}...")
    full_data = fetch_data(TICKER, START_DATE, END_DATE)
    if full_data is None:
        return
    
    train_data = full_data[full_data.index < TRAIN_TEST_SPLIT_DATE]
    test_data = full_data[full_data.index >= TRAIN_TEST_SPLIT_DATE]

    # --- 3. Optimize on Training Data ---
    param_grid = {
        'rsi_period': [14, 21, 28],
        'macd_fast': [12],
        'macd_slow': [26],
        'macd_signal': [9],
        'weights': [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)],
        'buy_threshold': [0.5, 0.7],
        'sell_threshold': [-0.5, -0.7]
    }
    print("Running optimization...")
    optimizer = Optimizer(data=train_data, param_grid=param_grid)
    optimizer.run_optimization()
    best_params = optimizer.get_best_params()
    print("\nBest Parameters Found:", best_params)

    # --- 4. Backtest on Test Data ---
    print("\nRunning final backtest on test data...")
    strategy_test = Strategy(
        data=test_data.copy(), # Use a copy to avoid SettingWithCopyWarning
        weights={'rsi': best_params['rsi_weight'], 'macd': best_params['macd_weight']},
        rsi_period=int(best_params['rsi_period']),
        macd_fast=int(best_params['macd_fast']),
        macd_slow=int(best_params['macd_slow']),
        macd_signal=int(best_params['macd_signal'])
    )
    signal_df_test = strategy_test.generate_signals()
    backtester_test = Backtester(
        data=signal_df_test,
        buy_threshold=best_params['buy_threshold'],
        sell_threshold=best_params['sell_threshold']
    )
    backtester_test.run()
    
    # --- 5. Fetch Benchmark Data ---
    print(f"\nFetching benchmark data for {BENCHMARK_TICKER}...")
    benchmark_data = fetch_data(BENCHMARK_TICKER, test_data.index.min().strftime('%Y-%m-%d'), test_data.index.max().strftime('%Y-%m-%d'))
    if benchmark_data is None:
        print("Could not fetch benchmark data. Skipping comparison.")
        return

    # --- 6. Visualize Results ---
    print("\nGenerating plot...")
    results_df = backtester_test.results
    
    plt.figure(figsize=(15, 8))
    
    # Plot Strategy Performance
    plt.plot(results_df.index, results_df['total'], label='Strategy Portfolio Value', color='royalblue', linewidth=2)
    
    # Plot AAPL Buy and Hold
    aapl_buy_hold = (test_data['Close'] / test_data['Close'].iloc[0]) * backtester_test.initial_cash
    plt.plot(aapl_buy_hold.index, aapl_buy_hold, label=f'Buy and Hold {TICKER}', color='darkorange', linestyle='--')

    # Plot SPY Buy and Hold
    spy_buy_hold = (benchmark_data['Close'] / benchmark_data['Close'].iloc[0]) * backtester_test.initial_cash
    plt.plot(spy_buy_hold.index, spy_buy_hold, label=f'Buy and Hold {BENCHMARK_TICKER}', color='green', linestyle='--')
    
    plt.title(f'Optimized Strategy vs. Buy and Hold: {TICKER} vs. {BENCHMARK_TICKER}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plot_filename = f'{TICKER}_vs_{BENCHMARK_TICKER}_performance.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    
    plt.show()

if __name__ == '__main__':
    visualize_performance() 