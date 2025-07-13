#!/usr/bin/env python3
"""
Script to fetch extended AAPL data from 2005 to 2023 and save it to the data directory.
This will replace the existing AAPL data with a much longer historical dataset.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src directory to path to import data_loader
sys.path.append('src')
from data_loader import fetch_data

def fetch_extended_aapl_data():
    """Fetch extended AAPL data from 2005 to 2023."""
    print("="*80)
    print("📊 FETCHING EXTENDED AAPL DATA (2005-2023)")
    print("🎯 Expanding historical dataset for improved ML training")
    print("="*80)
    
    # Define the extended date range
    start_date = '2005-01-01'
    end_date = '2023-12-31'
    ticker = 'AAPL'
    
    print(f"\n1. 📈 Fetching {ticker} data from {start_date} to {end_date}...")
    
    # Fetch the data using the existing data_loader
    extended_data = fetch_data(ticker, start_date, end_date)
    
    if extended_data is None:
        print("❌ Failed to fetch extended AAPL data")
        return None
    
    # Display data summary
    print(f"\n2. 📊 Extended AAPL Data Summary:")
    print(f"   Date range: {extended_data.index.min().date()} to {extended_data.index.max().date()}")
    print(f"   Total trading days: {len(extended_data):,}")
    print(f"   Data shape: {extended_data.shape}")
    print(f"   Price range: ${extended_data['Close'].min():.2f} - ${extended_data['Close'].max():.2f}")
    
    # Show first and last few rows
    print(f"\n3. 📋 First 5 rows:")
    print(extended_data.head())
    
    print(f"\n4. 📋 Last 5 rows:")
    print(extended_data.tail())
    
    # Calculate some basic statistics
    print(f"\n5. 📈 Basic Statistics:")
    total_return = (extended_data['Close'].iloc[-1] / extended_data['Close'].iloc[0] - 1) * 100
    annual_volatility = extended_data['Close'].pct_change().std() * (252 ** 0.5) * 100
    
    print(f"   Total return (2005-2023): {total_return:+.2f}%")
    print(f"   Annualized volatility: {annual_volatility:.2f}%")
    print(f"   Average daily volume: {extended_data['Volume'].mean():,.0f}")
    
    # Show yearly returns
    print(f"\n6. 📅 Yearly Returns:")
    yearly_returns = []
    for year in range(2005, 2024):
        year_data = extended_data[extended_data.index.year == year]
        if len(year_data) > 0:
            year_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0] - 1) * 100
            yearly_returns.append((year, year_return))
            print(f"   {year}: {year_return:+7.2f}%")
    
    # Identify major market events
    print(f"\n7. 🔍 Major Market Events Covered:")
    events = [
        ("2008-2009", "Financial Crisis"),
        ("2020", "COVID-19 Pandemic"),
        ("2022", "Tech Selloff"),
        ("2023", "AI Boom & Rate Concerns")
    ]
    
    for period, event in events:
        print(f"   ✅ {period}: {event}")
    
    # Compare with original data
    original_file = Path('data/aapl_split_adjusted.csv')
    if original_file.exists():
        original_data = pd.read_csv(original_file, index_col='Date', parse_dates=True)
        print(f"\n8. 📊 Comparison with Original Data:")
        print(f"   Original data: {len(original_data):,} days ({original_data.index.min().date()} to {original_data.index.max().date()})")
        print(f"   Extended data: {len(extended_data):,} days ({extended_data.index.min().date()} to {extended_data.index.max().date()})")
        print(f"   Additional data: {len(extended_data) - len(original_data):,} days ({len(extended_data) / len(original_data) - 1:.1%} more)")
    
    # Save the extended data with the standard filename
    output_file = Path('data/aapl_split_adjusted_extended.csv')
    extended_data.to_csv(output_file)
    print(f"\n9. 💾 Extended data saved to: {output_file}")
    
    # Also create a copy with the original filename for compatibility
    original_filename = Path('data/aapl_split_adjusted.csv')
    extended_data_subset = extended_data[extended_data.index >= '2018-01-01']
    extended_data_subset.to_csv(original_filename)
    print(f"   Compatibility copy saved to: {original_filename}")
    
    print(f"\n✅ Extended AAPL data fetch complete!")
    print(f"🎯 ML models can now train on {len(extended_data):,} days of historical data")
    print(f"📈 This includes {len(extended_data) - len(extended_data[extended_data.index >= '2018-01-01']):,} additional days of pre-2018 data")
    
    return extended_data

def analyze_extended_data_benefits():
    """Analyze the benefits of extended data for ML training."""
    print(f"\n" + "="*80)
    print("🧠 EXTENDED DATA BENEFITS FOR ML TRAINING")
    print("="*80)
    
    # Load the extended data
    extended_file = Path('data/aapl_split_adjusted_extended.csv')
    if not extended_file.exists():
        print("❌ Extended data file not found. Please run fetch_extended_aapl_data() first.")
        return
    
    extended_data = pd.read_csv(extended_file, index_col='Date', parse_dates=True)
    
    print(f"\n1. 📊 Training Data Expansion:")
    print(f"   Total samples: {len(extended_data):,}")
    print(f"   Training years: {extended_data.index.max().year - extended_data.index.min().year + 1}")
    
    # Analyze market regimes
    extended_data['returns'] = extended_data['Close'].pct_change()
    extended_data['volatility'] = extended_data['returns'].rolling(20).std()
    
    # Define different market periods
    periods = [
        ('2005-2007', 'Pre-Crisis Bull Market'),
        ('2008-2009', 'Financial Crisis'),
        ('2010-2015', 'Recovery & Growth'),
        ('2016-2019', 'Tech Boom'),
        ('2020-2021', 'Pandemic & Stimulus'),
        ('2022-2023', 'Rate Hikes & AI Boom')
    ]
    
    print(f"\n2. 🔍 Market Regime Diversity:")
    for period, description in periods:
        start_year, end_year = period.split('-')
        period_data = extended_data[
            (extended_data.index.year >= int(start_year)) & 
            (extended_data.index.year <= int(end_year))
        ]
        
        if len(period_data) > 0:
            period_return = (period_data['Close'].iloc[-1] / period_data['Close'].iloc[0] - 1) * 100
            period_vol = period_data['volatility'].mean() * 100 * (252 ** 0.5)
            print(f"   {period}: {description}")
            print(f"     Return: {period_return:+7.2f}%, Volatility: {period_vol:.1f}%")
    
    print(f"\n3. 💡 ML Training Advantages:")
    print(f"   ✅ More diverse market conditions for robust training")
    print(f"   ✅ Better representation of rare events (crashes, booms)")
    print(f"   ✅ Improved feature stability across different regimes")
    print(f"   ✅ Better generalization to unseen market conditions")
    print(f"   ✅ Reduced overfitting to recent market behavior")
    
    # Volatility analysis
    high_vol_days = len(extended_data[extended_data['volatility'] > extended_data['volatility'].quantile(0.9)])
    low_vol_days = len(extended_data[extended_data['volatility'] < extended_data['volatility'].quantile(0.1)])
    
    print(f"\n4. 📈 Volatility Distribution:")
    print(f"   High volatility days (>90th percentile): {high_vol_days:,}")
    print(f"   Low volatility days (<10th percentile): {low_vol_days:,}")
    print(f"   Normal volatility days: {len(extended_data) - high_vol_days - low_vol_days:,}")
    
    print(f"\n5. 🎯 Specific Benefits for July-Sep 2023 Issues:")
    print(f"   ✅ More examples of volatile periods (2008, 2020, 2022)")
    print(f"   ✅ Better understanding of market stress patterns")
    print(f"   ✅ Improved regime detection with longer history")
    print(f"   ✅ More robust volatility-based thresholds")

if __name__ == '__main__':
    # Fetch extended AAPL data
    extended_data = fetch_extended_aapl_data()
    
    if extended_data is not None:
        # Analyze the benefits
        analyze_extended_data_benefits()
    else:
        print("❌ Failed to fetch extended data") 