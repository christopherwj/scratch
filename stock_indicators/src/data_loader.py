import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_data(ticker: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
    """
    Fetches historical stock data, using a local CSV file as a cache.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the OHLC data, or None if download fails.
    """
    data_path = Path('data')
    data_path.mkdir(exist_ok=True)
    file_path = data_path / f"{ticker}_{start_date}_{end_date}.csv"

    # Check if cached file exists
    if file_path.exists():
        print(f"Loading cached data for {ticker} from {file_path}")
        try:
            # Correctly parse dates and set index when loading from cache
            data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            return data
        except Exception as e:
            print(f"Could not read cache file {file_path}. Refetching. Error: {e}")

    # If no cache, fetch from yfinance
    print(f"No cache found for {ticker}. Fetching from yfinance...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            print(f"No data found for ticker {ticker} from {start_date} to {end_date}. It might be delisted or an invalid ticker.")
            return None
        
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Make index timezone-naive before saving to prevent issues when loading from cache
        data.index = data.index.tz_localize(None)

        # Save the data to cache
        data.to_csv(file_path)
        print(f"Data for {ticker} saved to cache at {file_path}")
        
        return data
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")
        return None

if __name__ == '__main__':
    # Demonstrate caching functionality
    print("--- First Run (should fetch and cache) ---")
    aapl_data_first = fetch_data('MSFT', '2021-01-01', '2021-12-31')
    if aapl_data_first is not None:
        print("\nMSFT Data:")
        print(aapl_data_first.head())

    print("\n--- Second Run (should load from cache) ---")
    aapl_data_second = fetch_data('MSFT', '2021-01-01', '2021-12-31')
    if aapl_data_second is not None:
        print("\nMSFT Data (from cache):")
        print(aapl_data_second.head()) 