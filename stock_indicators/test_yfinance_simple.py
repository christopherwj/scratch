import yfinance as yf
import pandas as pd

print("Testing yfinance...")
print(f"Version: {yf.__version__}")

# Test 1: Try recent data
print("\n1. Testing recent data...")
try:
    data = yf.download('AAPL', start='2024-01-01', end='2024-01-31', progress=False)
    print(f"Recent data: {len(data)} rows" if not data.empty else "No recent data")
except Exception as e:
    print(f"Error with recent data: {e}")

# Test 2: Try with different ticker
print("\n2. Testing different ticker (MSFT)...")
try:
    data = yf.download('MSFT', start='2023-01-01', end='2023-01-31', progress=False)
    print(f"MSFT data: {len(data)} rows" if not data.empty else "No MSFT data")
except Exception as e:
    print(f"Error with MSFT: {e}")

# Test 3: Try with period
print("\n3. Testing with period...")
try:
    ticker = yf.Ticker('AAPL')
    data = ticker.history(period='1mo')
    print(f"Period data: {len(data)} rows" if not data.empty else "No period data")
except Exception as e:
    print(f"Error with period: {e}")

# Test 4: Try with info
print("\n4. Testing ticker info...")
try:
    ticker = yf.Ticker('AAPL')
    info = ticker.info
    print(f"Info available: {len(info)} fields")
    print(f"Symbol: {info.get('symbol', 'N/A')}")
except Exception as e:
    print(f"Error with info: {e}") 