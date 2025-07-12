import pandas as pd
import numpy as np
from src.optimizer import Optimizer
from src.strategy import Strategy
from src.backtester import Backtester
from dateutil.relativedelta import relativedelta

class RollingOptimizer:
    def __init__(self, data: pd.DataFrame, train_window: relativedelta, test_window: relativedelta, param_grid: dict, initial_cash: float = 10000.0, num_cores: int = 1):
        """
        Initializes the RollingOptimizer.
        """
        self.data = data
        self.train_window = train_window
        self.test_window = test_window
        self.param_grid = param_grid
        self.initial_cash = initial_cash
        self.num_cores = num_cores

    def run(self):
        """
        Runs the rolling window optimization and backtest.
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

        while current_date + self.test_window <= end_date:
            train_start = current_date - self.train_window
            train_end = current_date
            test_start = current_date
            test_end = current_date + self.test_window
            
            print(f"--- Processing window: Train {train_start.date()} to {train_end.date()} | Test {test_start.date()} to {test_end.date()} ---")

            # 1. Select data for the current window
            train_data = self.data.loc[train_start:train_end]
            test_data = self.data.loc[test_start:test_end]

            # 2. Find best parameters on the training data
            optimizer = Optimizer(data=train_data, param_grid=self.param_grid, num_cores=self.num_cores)
            optimizer.run_optimization()
            best_params = optimizer.get_best_params()
            best_params_history.append({'start': test_start, 'end': test_end, 'params': best_params})
            print(f"Best params for window: RSI Period={best_params['rsi_period']}, MACD=({int(best_params['macd_fast_period'])},{int(best_params['macd_slow_period'])},{int(best_params['macd_signal_period'])})")

            # 3. Backtest on the test data using these parameters
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
        
        # Combine results from all windows
        final_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
        final_portfolio_value = pd.concat(all_portfolio_values) if all_portfolio_values else pd.Series()
            
        return final_trades, final_portfolio_value, best_params_history 