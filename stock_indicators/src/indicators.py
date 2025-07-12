import pandas as pd

def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        close_prices (pd.Series): A pandas Series of closing stock prices.
        period (int): The time period to use for RSI calculation.

    Returns:
        pd.Series: A pandas Series containing the RSI values.
    """
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(close_prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """
    Calculates the Moving Average Convergence Divergence (MACD).

    Args:
        close_prices (pd.Series): A pandas Series of closing stock prices.
        fast_period (int): The time period for the fast EMA.
        slow_period (int): The time period for the slow EMA.
        signal_period (int): The time period for the signal line EMA.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: A tuple containing the MACD line, signal line, and histogram.
    """
    fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

if __name__ == '__main__':
    # Create a dummy dataframe for testing
    data = {'Close': [i + (i*0.1) * (-1)**i for i in range(100, 200)]}
    df = pd.DataFrame(data)

    # Test RSI
    df['RSI'] = calculate_rsi(df['Close'])
    print("RSI Results:")
    print(df[['Close', 'RSI']].tail())
    
    # Test MACD
    df['MACD_line'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['Close'])
    print("\nMACD Results:")
    print(df[['Close', 'MACD_line', 'MACD_signal', 'MACD_hist']].tail()) 