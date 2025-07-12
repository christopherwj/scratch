import pandas as pd
from src.data_loader import fetch_data
from src.optimizer import Optimizer
from src.strategy import Strategy
from src.backtester import Backtester

def main():
    """
    Main function to run the full backtesting and optimization workflow.
    """
    # --- 1. Parameters ---
    TICKER = 'AAPL'
    # These dates are now for reference; the loader uses the full sample file
    START_DATE = '2022-01-01'
    END_DATE = '2022-12-31'
    TRAIN_TEST_SPLIT_DATE = '2022-03-01' # Split within the sample data
    
    # --- 2. Fetch Data ---
    print(f"Fetching data for {TICKER}...")
    full_data = fetch_data(TICKER, START_DATE, END_DATE)
    if full_data is None:
        print("Failed to fetch data. Exiting.")
        return
        
    # --- 3. Split Data ---
    train_data = full_data[full_data.index < TRAIN_TEST_SPLIT_DATE]
    test_data = full_data[full_data.index >= TRAIN_TEST_SPLIT_DATE]
    print(f"Data split into {len(train_data)} training rows and {len(test_data)} testing rows.")

    # --- 4. Define Parameter Grid for Optimization ---
    # A smaller grid for a quicker demonstration
    param_grid = {
        'rsi_period': [14, 21, 28],
        'macd_fast': [12],
        'macd_slow': [26],
        'macd_signal': [9],
        'weights': [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)],
        'buy_threshold': [0.4, 0.6, 0.8],
        'sell_threshold': [-0.4, -0.6, -0.8]
    }
    
    # --- 5. Run Optimizer on Training Data ---
    print("\nRunning optimization on training data...")
    optimizer = Optimizer(data=train_data, param_grid=param_grid)
    optimizer.run_optimization()
    best_params = optimizer.get_best_params()
    
    print("\n--- Best Parameters Found ---")
    for key, val in best_params.items():
        print(f"{key}: {val}")

    # --- 6. Run Final Backtest on Test Data ---
    print("\nRunning final backtest on test data with best parameters...")
    
    # Re-create the strategy with the best parameters on the test data
    strategy_test = Strategy(
        data=test_data,
        weights={'rsi': best_params['rsi_weight'], 'macd': best_params['macd_weight']},
        rsi_period=int(best_params['rsi_period']),
        macd_fast=int(best_params['macd_fast']),
        macd_slow=int(best_params['macd_slow']),
        macd_signal=int(best_params['macd_signal'])
    )
    signal_df_test = strategy_test.generate_signals()

    # Run the backtester
    backtester_test = Backtester(
        data=signal_df_test,
        buy_threshold=best_params['buy_threshold'],
        sell_threshold=best_params['sell_threshold']
    )
    backtester_test.run()
    
    # --- 7. Report Final Performance ---
    performance_test = backtester_test.get_performance()
    print("\n--- Final Performance on Test Data ---")
    for metric, value in performance_test.items():
        print(f"{metric}: {value:.2f}")

if __name__ == '__main__':
    main() 