import pandas as pd
import numpy as np
from src.optimizer_gpu import OptimizerGPU
from src.strategy import Strategy
from src.backtester import Backtester
from dateutil.relativedelta import relativedelta
import time

class RollingOptimizerGPU:
    def __init__(self, data: pd.DataFrame, train_window: relativedelta, test_window: relativedelta, param_grid: dict, initial_cash: float = 10000.0, num_cores: int = 1, batch_size: int = 100):
        """
        Initializes the GPU-accelerated RollingOptimizer.
        """
        self.data = data
        self.train_window = train_window
        self.test_window = test_window
        self.param_grid = param_grid
        self.initial_cash = initial_cash
        self.num_cores = num_cores
        self.batch_size = batch_size

    def run(self):
        """
        Runs the rolling window optimization and backtest with GPU acceleration.
        Returns:
            (pd.DataFrame, pd.Series, list): A tuple of all trades, the final portfolio value series, and the history of best parameters.
        """
        start_date = self.data.index.min()
        end_date = self.data.index.max()
        
        current_date = start_date + self.train_window
        
        all_trades = []
        all_portfolio_values = []
        best_params_history = []
        
        last_portfolio_value = self.initial_cash
        
        window_count = 0
        total_windows = 0
        
        # Calculate total number of windows
        temp_date = start_date + self.train_window
        while temp_date + self.test_window <= end_date:
            total_windows += 1
            temp_date += self.test_window

        print(f"Total optimization windows: {total_windows}")
        start_time = time.time()

        while current_date + self.test_window <= end_date:
            window_count += 1
            window_start_time = time.time()
            
            train_start = current_date - self.train_window
            train_end = current_date
            test_start = current_date
            test_end = current_date + self.test_window
            
            print(f"\n--- Window {window_count}/{total_windows}: Train {train_start.date()} to {train_end.date()} | Test {test_start.date()} to {test_end.date()} ---")

            # 1. Select data for the current window
            train_data = self.data.loc[train_start:train_end]
            test_data = self.data.loc[test_start:test_end]

            # 2. Find best parameters on the training data using GPU acceleration
            print(f"Running GPU-accelerated optimization for window {window_count}...")
            optimizer = OptimizerGPU(
                data=train_data, 
                param_grid=self.param_grid, 
                num_cores=self.num_cores,
                batch_size=self.batch_size
            )
            optimizer.run_optimization()
            best_params = optimizer.get_best_params()
            best_params_history.append({'start': test_start, 'end': test_end, 'params': best_params})
            
            window_opt_time = time.time() - window_start_time
            print(f"Optimization completed in {window_opt_time:.2f} seconds")
            print(f"Best params for window: RSI Period={best_params['rsi_period']}, MACD=({int(best_params['macd_fast_period'])},{int(best_params['macd_slow_period'])},{int(best_params['macd_signal_period'])})")

            # 3. Backtest on the test data using these parameters
            print(f"Running backtest for window {window_count}...")
            strategy = Strategy(
                data=test_data,
                rsi_period=best_params['rsi_period'],
                macd_fast_period=best_params['macd_fast_period'],
                macd_slow_period=best_params['macd_slow_period'],
                macd_signal_period=best_params['macd_signal_period']
            )
            
            # Start this window's backtest with cash from the end of the last one
            backtester = Backtester(strategy, initial_cash=last_portfolio_value)
            portfolio_value, trades = backtester.run()
            
            # 4. Store results and update cash for next window
            if not portfolio_value.empty:
                all_portfolio_values.append(portfolio_value)
                last_portfolio_value = portfolio_value.iloc[-1]
            if not trades.empty:
                all_trades.append(trades)

            # 5. Move to the next window
            current_date += self.test_window
            
            # Print progress
            elapsed_time = time.time() - start_time
            avg_time_per_window = elapsed_time / window_count
            remaining_windows = total_windows - window_count
            estimated_remaining_time = remaining_windows * avg_time_per_window
            
            print(f"Window {window_count} completed in {window_opt_time:.2f}s. "
                  f"Elapsed: {elapsed_time/60:.1f}min, "
                  f"Estimated remaining: {estimated_remaining_time/60:.1f}min")
        
        # Combine results from all windows
        final_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
        final_portfolio_value = pd.concat(all_portfolio_values) if all_portfolio_values else pd.Series()
        
        total_time = time.time() - start_time
        print(f"\n=== GPU-Accelerated Rolling Optimization Completed ===")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Average time per window: {total_time/total_windows:.2f} seconds")
        print(f"Windows processed: {total_windows}")
            
        return final_trades, final_portfolio_value, best_params_history

if __name__ == '__main__':
    # Test GPU rolling optimizer
    print("Testing GPU-accelerated rolling optimizer...")
    
    # Create dummy data
    dummy_index = pd.to_datetime(pd.date_range(start='2020-01-01', periods=1000))
    dummy_data = {'Close': [100 + i*0.1 + np.random.uniform(-5, 5) for i in range(1000)]}
    df = pd.DataFrame(dummy_data, index=dummy_index)
    
    param_grid = {
        'rsi_period': [14, 21],
        'macd_fast_period': [12, 24],
        'macd_slow_period': [26, 52],
        'macd_signal_period': [9, 12]
    }
    
    train_window = relativedelta(months=6)
    test_window = relativedelta(months=2)
    
    rolling_optimizer = RollingOptimizerGPU(
        data=df,
        train_window=train_window,
        test_window=test_window,
        param_grid=param_grid,
        num_cores=4,
        batch_size=50
    )
    
    trades, portfolio_value, params_history = rolling_optimizer.run()
    
    print(f"\nResults:")
    print(f"Total trades: {len(trades)}")
    print(f"Final portfolio value: ${portfolio_value.iloc[-1]:.2f}")
    print(f"Parameter changes: {len(params_history)}") 