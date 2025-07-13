import pandas as pd
import numpy as np
import cupy as cp
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from src.strategy import Strategy
from src.backtester import Backtester
from src.indicators_gpu import calculate_rsi, calculate_macd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def _run_single_backtest_gpu(data, params):
    """
    Runs a single backtest for a given set of parameters using GPU-accelerated indicators.
    Always returns a dict with parameter values included.
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
        performance = backtester.get_performance()
        if performance is not None:
            performance.update({
                'rsi_period': params['rsi_period'],
                'macd_fast_period': params['macd_fast_period'],
                'macd_slow_period': params['macd_slow_period'],
                'macd_signal_period': params['macd_signal_period']
            })
        return performance
    except Exception:
        return None

def _batch_backtest_gpu(data, param_batch):
    """
    Runs multiple backtests in a batch using GPU acceleration.
    """
    results = []
    
    # Convert data to GPU arrays for batch processing
    try:
        close_prices_gpu = cp.asarray(data['Close'].values, dtype=cp.float32)
        
        for params in param_batch:
            if params['macd_slow_period'] <= params['macd_fast_period']:
                results.append(None)
                continue
                
            try:
                # Use GPU-accelerated indicators
                rsi = calculate_rsi(data['Close'], params['rsi_period'])
                macd_line, signal_line, _ = calculate_macd(
                    data['Close'], 
                    params['macd_fast_period'], 
                    params['macd_slow_period'], 
                    params['macd_signal_period']
                )
                
                # Create strategy data with GPU-calculated indicators
                strategy_data = data.copy()
                strategy_data['rsi'] = rsi
                strategy_data['macd_line'] = macd_line
                strategy_data['macd_signal'] = signal_line
                strategy_data.dropna(inplace=True)
                
                # Generate signals
                buy_conditions = (
                    (strategy_data['macd_line'] > strategy_data['macd_signal']) &
                    (strategy_data['macd_line'].shift(1) <= strategy_data['macd_signal'].shift(1)) &
                    (strategy_data['rsi'] < 70)
                )
                
                sell_conditions = (
                    (strategy_data['macd_line'] < strategy_data['macd_signal']) &
                    (strategy_data['macd_line'].shift(1) >= strategy_data['macd_signal'].shift(1)) &
                    (strategy_data['rsi'] > 30)
                )
                
                strategy_data['signal'] = 0
                strategy_data.loc[buy_conditions, 'signal'] = 1
                strategy_data.loc[sell_conditions, 'signal'] = -1
                
                # Run backtest
                backtester = Backtester(Strategy(data=strategy_data))
                backtester.run()
                performance = backtester.get_performance()
                
                # Ensure parameter values are included in the result
                if performance is not None:
                    performance.update({
                        'rsi_period': params['rsi_period'],
                        'macd_fast_period': params['macd_fast_period'],
                        'macd_slow_period': params['macd_slow_period'],
                        'macd_signal_period': params['macd_signal_period']
                    })
                results.append(performance)
                
            except Exception as e:
                print(f"Error in batch backtest: {e}")
                results.append(None)
                
    except Exception as e:
        print(f"GPU batch processing failed, falling back to CPU: {e}")
        # Fallback to individual CPU processing
        for params in param_batch:
            results.append(_run_single_backtest_gpu(data, params))
    
    return results

def _evaluate_params_with_stability_gpu(task_params):
    """
    Evaluates a single parameter set along with its neighbors to assess stability using GPU acceleration.
    Always returns a dict with parameter values included.
    """
    data, main_params = task_params

    # 1. Evaluate the main parameter set
    main_performance = _run_single_backtest_gpu(data, main_params)
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
        perf1 = _run_single_backtest_gpu(data, neighbor1_params)
        if perf1: neighbor_performances.append(perf1)

        # Neighbor 2 (param + step)
        neighbor2_params = main_params.copy()
        neighbor2_params[param_name] += step
        perf2 = _run_single_backtest_gpu(data, neighbor2_params)
        if perf2: neighbor_performances.append(perf2)

    if not neighbor_performances:
        # Always include parameter values in the result
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

    # Always include parameter values in the result
    return {
        **main_params,
        **main_performance,
        'stability_score': stability_score,
        'adjusted_sharpe': adjusted_sharpe
    }

class OptimizerGPU:
    def __init__(self, data: pd.DataFrame, param_grid: dict, num_cores: int = 1, batch_size: int = 100):
        """
        Initializes the GPU-accelerated Optimizer.
        """
        self.data = data
        self.param_grid = param_grid
        self.results = None
        self.num_cores = num_cores
        self.batch_size = batch_size

    def run_optimization(self):
        """
        Runs the grid search optimization with GPU acceleration and stability analysis.
        """
        param_names = list(self.param_grid.keys())
        param_combinations = list(product(*self.param_grid.values()))
        
        # Filter valid combinations
        valid_combinations = []
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            if param_dict['macd_slow_period'] > param_dict['macd_fast_period']:
                valid_combinations.append(param_dict)
        
        print(f"Processing {len(valid_combinations)} valid parameter combinations...")
        
        # Process in batches for GPU efficiency
        results_list = []
        
        # Use GPU batch processing for the main parameter evaluation
        for i in range(0, len(valid_combinations), self.batch_size):
            batch = valid_combinations[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}/{(len(valid_combinations) + self.batch_size - 1)//self.batch_size}")
            
            batch_results = _batch_backtest_gpu(self.data, batch)
            results_list.extend([r for r in batch_results if r is not None])
        
        # Now run stability analysis in parallel using CPU cores
        print("Running stability analysis...")
        stability_tasks = [(self.data, params) for params in valid_combinations]
        
        stability_results = {}
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            future_to_params = {executor.submit(_evaluate_params_with_stability_gpu, task): task for task in stability_tasks}
            
            for future in as_completed(future_to_params):
                result = future.result()
                if result is not None:
                    # Create a key for the parameter combination
                    key = (result['rsi_period'], result['macd_fast_period'], result['macd_slow_period'], result['macd_signal_period'])
                    stability_results[key] = {
                        'stability_score': result['stability_score'],
                        'adjusted_sharpe': result['adjusted_sharpe']
                    }
        
        # Merge stability results with main results
        for result in results_list:
            key = (result['rsi_period'], result['macd_fast_period'], result['macd_slow_period'], result['macd_signal_period'])
            if key in stability_results:
                result.update(stability_results[key])
            else:
                result['stability_score'] = 0
                result['adjusted_sharpe'] = result['sharpe_ratio']
            
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
        
        # Ensure the metric column exists
        if metric not in self.results.columns:
            print(f"Warning: {metric} column not found. Using sharpe_ratio instead.")
            metric = 'sharpe_ratio'
        
        best_params = self.results.loc[self.results[metric].idxmax()]
        return best_params.to_dict()

if __name__ == '__main__':
    # Test GPU optimization
    print("Testing GPU-accelerated optimization...")
    
    # Create dummy data
    dummy_index = pd.to_datetime(pd.date_range(start='2022-01-01', periods=400))
    dummy_data = {'Close': [i + np.random.uniform(-5, 5) for i in range(100, 500)]}
    df = pd.DataFrame(dummy_data, index=dummy_index)
    
    param_grid = {
        'rsi_period': [14, 21],
        'macd_fast_period': [12, 24],
        'macd_slow_period': [26, 52],
        'macd_signal_period': [9, 12]
    }
    
    optimizer = OptimizerGPU(data=df, param_grid=param_grid, num_cores=4, batch_size=50)
    optimization_results = optimizer.run_optimization()
    
    print("Optimization Results (Top 5 by Adjusted Sharpe):")
    if not optimization_results.empty:
        print(optimization_results.sort_values(by='adjusted_sharpe', ascending=False).head())
    
    print("\nBest Parameters (based on Adjusted Sharpe Ratio):")
    best_params = optimizer.get_best_params()
    print(best_params) 