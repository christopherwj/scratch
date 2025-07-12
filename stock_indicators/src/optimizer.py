import pandas as pd
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.strategy import Strategy
from src.backtester import Backtester

def _run_single_backtest(data, params):
    """
    Runs a single backtest for a given set of parameters.
    Returns the performance dictionary.
    """
    # Constraint: macd_slow must be greater than macd_fast
    if params['macd_slow_period'] <= params['macd_fast_period']:
        return None

    try:
        strategy = Strategy(
            data=data,
            rsi_period=params['rsi_period'],
            macd_fast_period=params['macd_fast_period'],
            macd_slow_period=params['macd_slow_period'],
            macd_signal_period=params['macd_signal_period']
        )
        backtester = Backtester(strategy)
        backtester.run()
        return backtester.get_performance()
    except Exception:
        return None

def _evaluate_params_with_stability(task_params):
    """
    Evaluates a single parameter set along with its neighbors to assess stability.
    This function must be at the top level of the module to be picklable.
    """
    data, main_params = task_params

    # 1. Evaluate the main parameter set
    main_performance = _run_single_backtest(data, main_params)
    if main_performance is None:
        return None

    # 2. Define and evaluate neighbors
    neighbor_performances = []
    # Perturb each parameter by a small step to create neighbors
    for param_name in main_params.keys():
        # Define a reasonable step size for each parameter
        step = max(1, int(main_params[param_name] * 0.1)) # 10% step, at least 1
        
        # Neighbor 1 (param - step)
        neighbor1_params = main_params.copy()
        neighbor1_params[param_name] = max(1, neighbor1_params[param_name] - step) # ensure param > 0
        perf1 = _run_single_backtest(data, neighbor1_params)
        if perf1: neighbor_performances.append(perf1)

        # Neighbor 2 (param + step)
        neighbor2_params = main_params.copy()
        neighbor2_params[param_name] += step
        perf2 = _run_single_backtest(data, neighbor2_params)
        if perf2: neighbor_performances.append(perf2)

    if not neighbor_performances:
        return {**main_params, **main_performance, 'stability_score': 0, 'adjusted_sharpe': 0}

    # 3. Calculate stability score
    main_sharpe = main_performance['sharpe_ratio']
    avg_neighbor_sharpe = np.mean([p['sharpe_ratio'] for p in neighbor_performances])

    # Avoid division by zero and handle cases where neighbor performance is negative
    if abs(main_sharpe) < 1e-6:
        stability_score = 0
    else:
        # Stability is the ratio of average neighbor performance to main performance.
        # It's capped at 1, as we don't want to reward neighbors outperforming the main params.
        stability_score = min(1.0, avg_neighbor_sharpe / main_sharpe) if main_sharpe > 0 else 0

    adjusted_sharpe = main_sharpe * stability_score

    return {
        **main_params,
        **main_performance,
        'stability_score': stability_score,
        'adjusted_sharpe': adjusted_sharpe
    }

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
        Runs the grid search optimization with stability analysis in parallel.
        """
        param_names = list(self.param_grid.keys())
        param_combinations = list(product(*self.param_grid.values()))
        
        tasks = []
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            # Basic validation before creating a task
            if param_dict['macd_slow_period'] > param_dict['macd_fast_period']:
                tasks.append((self.data, param_dict))

        results_list = []
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            future_to_params = {executor.submit(_evaluate_params_with_stability, task): task for task in tasks}
            for future in as_completed(future_to_params):
                result = future.result()
                if result is not None:
                    results_list.append(result)
            
        self.results = pd.DataFrame(results_list)
        return self.results

    def get_best_params(self, metric='adjusted_sharpe'):
        """
        Finds the best set of parameters based on the adjusted Sharpe ratio.
        """
        if self.results is None or self.results.empty:
            print("Warning: No valid optimization results. Returning default parameters.")
            return {
                'rsi_period': 14, 'macd_fast_period': 12, 'macd_slow_period': 26, 'macd_signal_period': 9,
                'total_return_pct': 0, 'sharpe_ratio': 0, 'stability_score': 0, 'adjusted_sharpe': 0
            }
        
        best_params = self.results.loc[self.results[metric].idxmax()]
        return best_params.to_dict()

if __name__ == '__main__':
    dummy_index = pd.to_datetime(pd.date_range(start='2022-01-01', periods=400))
    dummy_data = {'Close': [i + np.random.uniform(-5, 5) for i in range(100, 500)]}
    df = pd.DataFrame(dummy_data, index=dummy_index)
    
    param_grid = {
        'rsi_period': [14, 21],
        'macd_fast_period': [12, 24],
        'macd_slow_period': [26, 52],
        'macd_signal_period': [9, 12]
    }
    
    optimizer = Optimizer(data=df, param_grid=param_grid, num_cores=4)
    optimization_results = optimizer.run_optimization()
    
    print("Optimization Results (Top 5 by Adjusted Sharpe):")
    if not optimization_results.empty:
        print(optimization_results.sort_values(by='adjusted_sharpe', ascending=False).head())
    
    print("\nBest Parameters (based on Adjusted Sharpe Ratio):")
    best_params = optimizer.get_best_params()
    print(best_params) 