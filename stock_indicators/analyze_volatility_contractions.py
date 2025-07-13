import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from volatility_contraction_detector import VolatilityContractionDetector

def analyze_recent_contractions():
    """Analyze recent volatility contraction patterns and their outcomes"""
    print("Analyzing Recent Volatility Contraction Patterns")
    print("=" * 55)
    
    # Initialize detector and run analysis
    detector = VolatilityContractionDetector()
    data = detector.run_analysis()
    
    # Focus on last 5 years of data
    five_years_ago = data['Date'].max() - timedelta(days=5*365)
    recent_data = data[data['Date'] >= five_years_ago].copy()
    
    print(f"\nAnalyzing {len(recent_data)} days from {recent_data['Date'].min().strftime('%Y-%m-%d')} to {recent_data['Date'].max().strftime('%Y-%m-%d')}")
    
    # Find all contraction periods
    contractions = recent_data[recent_data['strong_contraction'] == 1].copy()
    extreme_contractions = recent_data[recent_data['extreme_contraction'] == 1].copy()
    
    print(f"\nFound {len(contractions)} strong contraction periods")
    print(f"Found {len(extreme_contractions)} extreme contraction periods")
    
    if len(contractions) == 0:
        print("No strong contraction patterns found in the recent period")
        return
    
    # Analyze each contraction period
    results = []
    
    for idx, contraction in contractions.iterrows():
        # Look ahead 30 days to see what happened (longer timeframe without take profit)
        future_data = data.iloc[idx:idx+31]  # Current day + 30 days ahead
        
        if len(future_data) < 2:
            continue
            
        # Calculate outcomes
        entry_price = contraction['Close']
        max_price = future_data['High'].max()
        min_price = future_data['Low'].min()
        end_price = future_data.iloc[-1]['Close']
        
        max_gain = (max_price - entry_price) / entry_price * 100
        max_loss = (min_price - entry_price) / entry_price * 100
        final_return = (end_price - entry_price) / entry_price * 100
        
        # Check for breakouts
        breakout_occurred = (future_data['breakout_signal'].sum() > 0)
        breakdown_occurred = (future_data['breakdown_signal'].sum() > 0)
        
        results.append({
            'Date': contraction['Date'],
            'Entry_Price': entry_price,
            'Contraction_Strength': contraction['contraction_strength'],
            'Contraction_Duration': contraction['contraction_duration'],
            'Max_Gain_30d': max_gain,
            'Max_Loss_30d': max_loss,
            'Final_Return_30d': final_return,
            'Breakout_Occurred': breakout_occurred,
            'Breakdown_Occurred': breakdown_occurred,
            'Volume_Ratio': contraction['volume_ratio'],
            'BB_Position': contraction['bb_position_20'],
            'Price_vs_SMA20': contraction['price_vs_sma_20']
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\nContraction Pattern Analysis Summary (30-day timeframe, no take profit):")
    print("-" * 60)
    print(f"Average Contraction Strength: {results_df['Contraction_Strength'].mean():.3f}")
    print(f"Average Contraction Duration: {results_df['Contraction_Duration'].mean():.1f} days")
    print(f"Average Max Gain (30 days): {results_df['Max_Gain_30d'].mean():.2f}%")
    print(f"Average Max Loss (30 days): {results_df['Max_Loss_30d'].mean():.2f}%")
    print(f"Average Final Return (30 days): {results_df['Final_Return_30d'].mean():.2f}%")
    print(f"Breakout Rate: {results_df['Breakout_Occurred'].sum()}/{len(results_df)} ({results_df['Breakout_Occurred'].mean()*100:.1f}%)")
    print(f"Breakdown Rate: {results_df['Breakdown_Occurred'].sum()}/{len(results_df)} ({results_df['Breakdown_Occurred'].mean()*100:.1f}%)")
    
    # Show recent contractions
    print(f"\nMost Recent Contraction Patterns:")
    print("-" * 40)
    recent_results = results_df.tail(10)
    
    for _, row in recent_results.iterrows():
        print(f"Date: {row['Date'].strftime('%Y-%m-%d')}")
        print(f"  Price: ${row['Entry_Price']:.2f}")
        print(f"  Strength: {row['Contraction_Strength']:.3f}, Duration: {row['Contraction_Duration']} days")
        print(f"  30-day Max Gain: {row['Max_Gain_30d']:.2f}%, Max Loss: {row['Max_Loss_30d']:.2f}%")
        print(f"  Final Return: {row['Final_Return_30d']:.2f}%")
        print(f"  Breakout: {'Yes' if row['Breakout_Occurred'] else 'No'}, Breakdown: {'Yes' if row['Breakdown_Occurred'] else 'No'}")
        print()
    
    # Create visualization
    create_contraction_analysis_chart(recent_data, results_df)
    
    return results_df

def create_contraction_analysis_chart(data, results):
    """Create a comprehensive chart showing contraction patterns and outcomes"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('AAPL Volatility Contraction Pattern Analysis (Last 5 Years)', fontsize=16, fontweight='bold')
    
    # 1. Price chart with contraction signals
    ax1 = axes[0]
    ax1.plot(data['Date'], data['Close'], label='AAPL Close', color='black', linewidth=1)
    
    # Add Bollinger Bands
    ax1.plot(data['Date'], data['bb_upper_20'], label='BB Upper', color='red', alpha=0.5)
    ax1.plot(data['Date'], data['bb_lower_20'], label='BB Lower', color='red', alpha=0.5)
    
    # Highlight contraction periods
    contractions = data[data['strong_contraction'] == 1]
    ax1.scatter(contractions['Date'], contractions['Close'], 
               color='orange', s=40, alpha=0.8, label='Strong Contraction')
    
    # Highlight extreme contractions
    extreme = data[data['extreme_contraction'] == 1]
    ax1.scatter(extreme['Date'], extreme['Close'], 
               color='red', s=60, alpha=0.9, label='Extreme Contraction')
    
    ax1.set_title('Price Action with Contraction Signals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Volatility indicators
    ax2 = axes[1]
    ax2.plot(data['Date'], data['volatility_20d'], label='20-day Volatility', color='blue')
    ax2.plot(data['Date'], data['volatility_50d'], label='50-day Volatility', color='red')
    
    # Highlight low volatility periods
    low_vol = data[data['contraction_signal'] == 1]
    ax2.scatter(low_vol['Date'], low_vol['volatility_20d'], 
               color='orange', s=20, alpha=0.7, label='Contraction Periods')
    
    ax2.set_title('Volatility Indicators')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Contraction outcomes
    ax3 = axes[2]
    
    # Plot returns for each contraction
    for _, row in results.iterrows():
        color = 'green' if row['Final_Return_30d'] > 0 else 'red'
        ax3.bar(row['Date'], row['Final_Return_30d'], 
               color=color, alpha=0.7, width=timedelta(days=5))
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title('30-Day Returns After Contraction Signals (No Take Profit)')
    ax3.set_ylabel('Return (%)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('recent_volatility_contractions.png', dpi=300, bbox_inches='tight')
    print("Chart saved as 'recent_volatility_contractions.png'")
    plt.show()

def generate_trading_strategy():
    """Generate a trading strategy based on volatility contraction patterns"""
    print("\n" + "="*60)
    print("VOLATILITY CONTRACTION TRADING STRATEGY")
    print("="*60)
    
    detector = VolatilityContractionDetector()
    data = detector.run_analysis()
    
    # Strategy parameters
    min_strength = 0.3
    min_duration = 3
    volume_threshold = 1.2
    
    # Generate signals
    long_signals = data[
        (data['strong_contraction'] == 1) &
        (data['contraction_strength'] >= min_strength) &
        (data['contraction_duration'] >= min_duration) &
        (data['price_vs_sma_20'] > 0) &  # Above 20-day SMA
        (data['bb_position_20'] > 0.5) &  # Upper half of Bollinger Bands
        (data['volume_ratio'] > volume_threshold)
    ].copy()
    
    short_signals = data[
        (data['strong_contraction'] == 1) &
        (data['contraction_strength'] >= min_strength) &
        (data['contraction_duration'] >= min_duration) &
        (data['price_vs_sma_20'] < 0) &  # Below 20-day SMA
        (data['bb_position_20'] < 0.5) &  # Lower half of Bollinger Bands
        (data['volume_ratio'] > volume_threshold)
    ].copy()
    
    print(f"Strategy Parameters:")
    print(f"  Minimum Contraction Strength: {min_strength}")
    print(f"  Minimum Contraction Duration: {min_duration} days")
    print(f"  Volume Threshold: {volume_threshold}x average")
    print()
    
    print(f"Signal Counts:")
    print(f"  Long Signals: {len(long_signals)}")
    print(f"  Short Signals: {len(short_signals)}")
    print(f"  Total Signals: {len(long_signals) + len(short_signals)}")
    print()
    
    # Show recent signals
    print("Recent Trading Signals:")
    print("-" * 30)
    
    recent_signals = pd.concat([long_signals, short_signals]).sort_values('Date').tail(10)
    
    for _, signal in recent_signals.iterrows():
        signal_type = "LONG" if signal['price_vs_sma_20'] > 0 else "SHORT"
        print(f"{signal['Date'].strftime('%Y-%m-%d')} - {signal_type}")
        print(f"  Price: ${signal['Close']:.2f}")
        print(f"  Strength: {signal['contraction_strength']:.3f}, Duration: {signal['contraction_duration']} days")
        print(f"  Volume Ratio: {signal['volume_ratio']:.2f}")
        print()
    
    return long_signals, short_signals

if __name__ == "__main__":
    # Run the analysis
    results = analyze_recent_contractions()
    
    # Generate trading strategy
    long_signals, short_signals = generate_trading_strategy()
    
    print("\nAnalysis Complete!")
    print("Files generated:")
    print("- recent_volatility_contractions.png")
    print("- volatility_contraction_signals.csv (if exported)") 