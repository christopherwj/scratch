#!/usr/bin/env python3
"""
Debug script to test yfinance with different date ranges for AAPL.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_yfinance_dates():
    """Test different date ranges to find the issue."""
    print("🔍 Testing yfinance with different date ranges for AAPL...")
    
    # Test different date ranges
    test_ranges = [
        ('2023-01-01', '2023-12-31'),  # Recent data
        ('2020-01-01', '2020-12-31'),  # COVID year
        ('2018-01-01', '2018-12-31'),  # Older data
        ('2010-01-01', '2010-12-31'),  # Much older
        ('2005-01-01', '2005-12-31'),  # 2005 only
        ('2007-01-01', '2023-12-31'),  # Start from 2007
        ('2005-01-01', '2023-12-31'),  # Full range
    ]
    
    ticker = 'AAPL'
    
    for start_date, end_date in test_ranges:
        print(f"\n📅 Testing {ticker} from {start_date} to {end_date}...")
        
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"   ❌ No data returned")
            else:
                print(f"   ✅ Success: {len(data)} days")
                print(f"   📊 Date range: {data.index.min().date()} to {data.index.max().date()}")
                print(f"   💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_manual_fetch():
    """Test manual yfinance fetch with different approaches."""
    print(f"\n🔧 Testing manual yfinance approaches...")
    
    ticker = 'AAPL'
    
    # Approach 1: Standard history call
    print(f"\n1. Standard history() call:")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start='2005-01-01', end='2023-12-31')
        print(f"   Result: {len(data) if not data.empty else 'Empty'} rows")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Approach 2: With period parameter
    print(f"\n2. Using period parameter:")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='max')
        print(f"   Result: {len(data)} rows")
        print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Approach 3: Download function
    print(f"\n3. Using yf.download():")
    try:
        data = yf.download(ticker, start='2005-01-01', end='2023-12-31')
        print(f"   Result: {len(data) if not data.empty else 'Empty'} rows")
        if not data.empty:
            print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
    except Exception as e:
        print(f"   Error: {e}")

def fetch_extended_aapl_manual():
    """Manually fetch extended AAPL data using the best approach."""
    print(f"\n🎯 Fetching extended AAPL data manually...")
    
    try:
        # Try with period='max' first
        print("Trying with period='max'...")
        stock = yf.Ticker('AAPL')
        data = stock.history(period='max')
        
        if data.empty:
            print("❌ No data with period='max'")
            
            # Try with yf.download
            print("Trying with yf.download()...")
            data = yf.download('AAPL', start='2005-01-01', end='2023-12-31')
            
        if not data.empty:
            print(f"✅ Success! Got {len(data)} days of data")
            print(f"📅 Date range: {data.index.min().date()} to {data.index.max().date()}")
            
            # Filter to our desired range
            data = data[data.index >= '2005-01-01']
            
            # Clean up the data
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.index = data.index.tz_localize(None)
            
            # Save to file
            output_file = 'data/aapl_split_adjusted_extended.csv'
            data.to_csv(output_file)
            print(f"💾 Saved to {output_file}")
            
            # Also update the original file
            original_file = 'data/aapl_split_adjusted.csv'
            data_subset = data[data.index >= '2018-01-01']
            data_subset.to_csv(original_file)
            print(f"💾 Updated {original_file}")
            
            return data
        else:
            print("❌ Failed to get data with any method")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == '__main__':
    test_yfinance_dates()
    test_manual_fetch()
    
    # Try to fetch the data
    extended_data = fetch_extended_aapl_manual()
    
    if extended_data is not None:
        print(f"\n🎉 Successfully fetched extended AAPL data!")
        print(f"📊 Total samples: {len(extended_data):,}")
        print(f"📅 Date range: {extended_data.index.min().date()} to {extended_data.index.max().date()}")
        print(f"💰 Price range: ${extended_data['Close'].min():.2f} - ${extended_data['Close'].max():.2f}")
    else:
        print(f"\n❌ Failed to fetch extended data") 