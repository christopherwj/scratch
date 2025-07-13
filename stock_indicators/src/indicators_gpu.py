import pandas as pd
import numpy as np
import cupy as cp
from numba import cuda, jit
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@cuda.jit
def _rsi_kernel(prices, gains, losses, rsi, period):
    """
    CUDA kernel for RSI calculation.
    """
    idx = cuda.grid(1)
    if idx >= prices.shape[0]:
        return
    
    if idx < period:
        rsi[idx] = 50.0  # Default value for insufficient data
        return
    
    # Calculate gains and losses
    if idx > 0:
        diff = prices[idx] - prices[idx - 1]
        if diff > 0:
            gains[idx] = diff
            losses[idx] = 0.0
        else:
            gains[idx] = 0.0
            losses[idx] = -diff
    
    # Calculate RSI using exponential moving average
    if idx >= period:
        avg_gain = 0.0
        avg_loss = 0.0
        
        # Initial average
        for i in range(period):
            avg_gain += gains[idx - i]
            avg_loss += losses[idx - i]
        
        avg_gain /= period
        avg_loss /= period
        
        # Apply smoothing
        for i in range(period, idx + 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0.0:
            rsi[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[idx] = 100.0 - (100.0 / (1.0 + rs))

def calculate_rsi_gpu(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    GPU-accelerated RSI calculation using CuPy and Numba CUDA.
    """
    # Convert to numpy array and move to GPU
    prices_gpu = cp.asarray(close_prices.values, dtype=cp.float32)
    gains_gpu = cp.zeros_like(prices_gpu)
    losses_gpu = cp.zeros_like(prices_gpu)
    rsi_gpu = cp.zeros_like(prices_gpu)
    
    # Configure CUDA grid
    threadsperblock = 256
    blockspergrid = (prices_gpu.size + (threadsperblock - 1)) // threadsperblock
    
    # Launch kernel
    _rsi_kernel[blockspergrid, threadsperblock](prices_gpu, gains_gpu, losses_gpu, rsi_gpu, period)
    
    # Copy result back to CPU
    rsi_cpu = cp.asnumpy(rsi_gpu)
    
    # Create pandas Series with original index
    return pd.Series(rsi_cpu, index=close_prices.index, name='RSI')

def calculate_macd_gpu(close_prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """
    Calculates MACD using pandas' reliable .ewm() to ensure correctness.
    The custom CUDA kernel for EMA was incorrect for a sequential operation.
    This function remains in the GPU workflow for consistency.
    """
    # EMA is a sequential calculation, so we use pandas' correct implementation.
    fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) with GPU acceleration.
    Falls back to CPU if GPU is not available.
    """
    try:
        return calculate_rsi_gpu(close_prices, period)
    except Exception as e:
        print(f"GPU RSI calculation failed, falling back to CPU: {e}")
        # Fallback to CPU implementation
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def calculate_macd(close_prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """
    Calculates the Moving Average Convergence Divergence (MACD) with GPU acceleration.
    Falls back to CPU if GPU is not available.
    """
    try:
        return calculate_macd_gpu(close_prices, fast_period, slow_period, signal_period)
    except Exception as e:
        print(f"GPU MACD calculation failed, falling back to CPU: {e}")
        # Fallback to CPU implementation
        fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

if __name__ == '__main__':
    # Test GPU acceleration
    print("Testing GPU-accelerated indicators...")
    
    # Create test data
    data = {'Close': [i + (i*0.1) * (-1)**i for i in range(100, 200)]}
    df = pd.DataFrame(data)
    
    # Test RSI
    print("Calculating RSI...")
    df['RSI'] = calculate_rsi(df['Close'])
    print("RSI Results:")
    print(df[['Close', 'RSI']].tail())
    
    # Test MACD
    print("\nCalculating MACD...")
    df['MACD_line'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['Close'])
    print("MACD Results:")
    print(df[['Close', 'MACD_line', 'MACD_signal', 'MACD_hist']].tail()) 