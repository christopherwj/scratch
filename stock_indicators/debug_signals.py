import pandas as pd
import numpy as np
from src.strategy import Strategy
from src.backtester import Backtester
from src.data_loader import fetch_data, load_aapl_split_adjusted
from src.indicators import calculate_rsi as cpu_rsi, calculate_macd as cpu_macd
try:
    from src.indicators_gpu import calculate_rsi as gpu_rsi, calculate_macd as gpu_macd
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def debug_signals():
    print("=== DEBUGGING SIGNAL GENERATION ===")
    data = load_aapl_split_adjusted()
    # Filter to 2020 data for debugging
    data = data[data.index.year == 2020]
    if data is None:
        print("Could not fetch data")
        return
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Compare CPU and GPU indicators
    print("\n=== INDICATOR COMPARISON (CPU vs GPU) ===")
    rsi_cpu = cpu_rsi(data['Close'], 14)
    macd_cpu, signal_cpu, _ = cpu_macd(data['Close'], 12, 26, 9)
    if GPU_AVAILABLE:
        rsi_gpu = gpu_rsi(data['Close'], 14)
        macd_gpu, signal_gpu, _ = gpu_macd(data['Close'], 12, 26, 9)
        compare = pd.DataFrame({
            'RSI_CPU': rsi_cpu,
            'RSI_GPU': rsi_gpu,
            'MACD_CPU': macd_cpu,
            'MACD_GPU': macd_gpu,
            'SIGNAL_CPU': signal_cpu,
            'SIGNAL_GPU': signal_gpu
        })
        compare['RSI_DIFF'] = (compare['RSI_CPU'] - compare['RSI_GPU']).abs()
        compare['MACD_DIFF'] = (compare['MACD_CPU'] - compare['MACD_GPU']).abs()
        compare['SIGNAL_DIFF'] = (compare['SIGNAL_CPU'] - compare['SIGNAL_GPU']).abs()
        print(compare.tail(10))
        print(f"\nMax RSI diff: {compare['RSI_DIFF'].max():.6f}")
        print(f"Max MACD diff: {compare['MACD_DIFF'].max():.6f}")
        print(f"Max SIGNAL diff: {compare['SIGNAL_DIFF'].max():.6f}")
    else:
        print("GPU indicators not available for comparison.")

if __name__ == '__main__':
    debug_signals() 