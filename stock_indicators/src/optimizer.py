import pandas as pd
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.strategy import Strategy
from src.backtester import Backtester

def _evaluate_params(params):
    """
    Helper function to evaluate a single set of parameters, designed to be executed in a separate process.
    This function must be at the top level of the module to be picklable.
    """
    data, param_dict = params  # Unpack data and parameters

    # Constraint: macd_slow must be greater than macd_fast
    if param_dict['macd_slow_period'] <= param_dict['macd_fast_period']:
        return None

    try:
        strategy = Strategy(
            data=data,
            rsi_period=param_dict['rsi_period'],
            macd_fast_period=param_dict['macd_fast_period'],
            macd_slow_period=param_dict['macd_slow_period'],
            macd_signal_period=param_dict['macd_signal_period']
        )
        
        backtester = Backtester(strategy)
        backtester.run()
        performance = backtester.get_performance()
        
        return {**param_dict, **performance}
    except Exception as e:
        # Log the error and the parameters that caused it, then continue
        print(f"Error evaluating params {param_dict}: {e}")
        return None


class Optimizer:
    def __init__(self, data: pd.DataFrame, param_grid: dict, num_cores: int = 1):
        """
        Initializes the Optimizer.
        """
        self.data = data
        self.param_grid = param_grid
        self.results = None
        self.num_cores = num_cores

    def run_optimization(self):
        """
        Runs the grid search optimization in parallel.
        """
        param_names = list(self.param_grid.keys())
        param_combinations = list(product(*self.param_grid.values()))
        
        tasks = []
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            tasks.append((self.data, param_dict))

        results_list = []

        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            # Submitting all tasks to the executor
            future_to_params = {executor.submit(_evaluate_params, task): task for task in tasks}
            
            # Processing results as they are completed
            for future in as_completed(future_to_params):
                result = future.result()
                if result is not None:
                    results_list.append(result)
            
        self.results = pd.DataFrame(results_list)
        return self.results

    def get_best_params(self, metric='sharpe_ratio'):
        """
        Finds the best set of parameters based on a given performance metric.
        """
        if self.results is None or self.results.empty:
            # Return a default parameter set if no valid results were found
            print("Warning: No valid optimization results. Returning default parameters.")
            return {
                'rsi_period': 14,
                'macd_fast_period': 12,
                'macd_slow_period': 26,
                'macd_signal_period': 9,
                'total_return_pct': 0,
                'sharpe_ratio': 0
            }
        
        best_params = self.results.loc[self.results[metric].idxmax()]
        return best_params.to_dict()

if __name__ == '__main__':
    # Create a dummy dataframe
    dummy_data = {'Close': [i + np.random.uniform(-5, 5) for i in range(100, 500)]}
    dummy_index = pd.to_datetime(pd.date_range(start='2022-01-01', periods=400))
    df = pd.DataFrame(dummy_data, index=dummy_index)
    
    # Define a smaller parameter grid for the example
    param_grid = {
        'rsi_period': [14, 21, 28],
        'macd_fast_period': [12, 24],
        'macd_slow_period': [26, 52],
        'macd_signal_period': [9, 12]
    }
    
    # Use 4 cores for the example
    optimizer = Optimizer(data=df, param_grid=param_grid, num_cores=4)
    optimization_results = optimizer.run_optimization()
    
    print("Optimization Results:")
    print(optimization_results)
    
    print("\nBest Parameters (based on Sharpe Ratio):")
    best_params = optimizer.get_best_params()
    print(best_params) 