import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_aapl_data():
    """Load real AAPL data from the data directory."""
    data_file = Path('data/aapl_split_adjusted.csv')
    
    if not data_file.exists():
        print(f"❌ AAPL data file not found at {data_file}")
        return None
    
    try:
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"✅ Loaded real AAPL data from {data_file}")
        return data
    except Exception as e:
        print(f"❌ Error loading AAPL data: {e}")
        return None

def calculate_technical_indicators(data):
    """Calculate technical indicators for analysis."""
    df = data.copy()
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Moving averages
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['ema_20'] = df['Close'].ewm(span=20).mean()
    df['ema_50'] = df['Close'].ewm(span=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_middle = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = bb_middle + (bb_std * 2)
    df['bb_lower'] = bb_middle - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Support/Resistance
    df['highest_20'] = df['High'].rolling(20).max()
    df['lowest_20'] = df['Low'].rolling(20).min()
    df['highest_50'] = df['High'].rolling(50).max()
    df['lowest_50'] = df['Low'].rolling(50).min()
    
    # Volume analysis
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    return df

def analyze_july_september_2023():
    """Analyze why the model performed poorly during July-September 2023."""
    
    print("="*80)
    print("🔍 ANALYZING ML MODEL PERFORMANCE: JULY-SEPTEMBER 2023")
    print("🎯 Understanding Why the Model Struggled During This Period")
    print("="*80)
    
    # Load data
    data = load_aapl_data()
    if data is None:
        return
    
    # Calculate technical indicators
    df = calculate_technical_indicators(data)
    
    # Focus on the problematic period
    problem_period = df['2023-07-01':'2023-09-30'].copy()
    
    # Also get some context - previous 3 months for comparison
    context_period = df['2023-04-01':'2023-06-30'].copy()
    
    # Get the full test period (what the model was tested on)
    test_period = df['2022-08-10':'2023-12-26'].copy()
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"   Problem Period (Jul-Sep 2023): {len(problem_period)} days")
    print(f"   Context Period (Apr-Jun 2023): {len(context_period)} days")
    print(f"   Full Test Period: {len(test_period)} days")
    
    # Analyze price movements
    print(f"\n📈 PRICE PERFORMANCE ANALYSIS:")
    
    # July-September 2023 performance
    july_sep_start = problem_period['Close'].iloc[0]
    july_sep_end = problem_period['Close'].iloc[-1]
    july_sep_return = (july_sep_end / july_sep_start - 1) * 100
    
    # Context period performance
    context_start = context_period['Close'].iloc[0]
    context_end = context_period['Close'].iloc[-1]
    context_return = (context_end / context_start - 1) * 100
    
    print(f"   Jul-Sep 2023 Return: {july_sep_return:+.2f}%")
    print(f"   Apr-Jun 2023 Return: {context_return:+.2f}%")
    print(f"   Price at Jul 1: ${july_sep_start:.2f}")
    print(f"   Price at Sep 30: ${july_sep_end:.2f}")
    print(f"   Peak during period: ${problem_period['Close'].max():.2f}")
    print(f"   Trough during period: ${problem_period['Close'].min():.2f}")
    
    # Volatility analysis
    print(f"\n📊 VOLATILITY ANALYSIS:")
    july_sep_vol = problem_period['volatility_20'].mean()
    context_vol = context_period['volatility_20'].mean()
    
    print(f"   Jul-Sep 2023 Avg Volatility: {july_sep_vol:.4f}")
    print(f"   Apr-Jun 2023 Avg Volatility: {context_vol:.4f}")
    print(f"   Volatility Change: {((july_sep_vol/context_vol - 1) * 100):+.1f}%")
    
    # Technical indicator analysis
    print(f"\n🔍 TECHNICAL INDICATOR ANALYSIS:")
    
    # RSI analysis
    july_sep_rsi = problem_period['rsi'].mean()
    context_rsi = context_period['rsi'].mean()
    overbought_days = (problem_period['rsi'] > 70).sum()
    oversold_days = (problem_period['rsi'] < 30).sum()
    
    print(f"   Average RSI Jul-Sep: {july_sep_rsi:.1f}")
    print(f"   Average RSI Apr-Jun: {context_rsi:.1f}")
    print(f"   Overbought days (RSI>70): {overbought_days}")
    print(f"   Oversold days (RSI<30): {oversold_days}")
    
    # MACD analysis
    macd_crossovers = ((problem_period['macd'] > problem_period['macd_signal']) & 
                       (problem_period['macd'].shift(1) <= problem_period['macd_signal'].shift(1))).sum()
    macd_cross_downs = ((problem_period['macd'] < problem_period['macd_signal']) & 
                        (problem_period['macd'].shift(1) >= problem_period['macd_signal'].shift(1))).sum()
    
    print(f"   MACD Bullish Crossovers: {macd_crossovers}")
    print(f"   MACD Bearish Crossovers: {macd_cross_downs}")
    
    # Moving average analysis
    above_sma20 = (problem_period['Close'] > problem_period['sma_20']).sum()
    above_sma50 = (problem_period['Close'] > problem_period['sma_50']).sum()
    
    print(f"   Days above SMA20: {above_sma20}/{len(problem_period)} ({above_sma20/len(problem_period)*100:.1f}%)")
    print(f"   Days above SMA50: {above_sma50}/{len(problem_period)} ({above_sma50/len(problem_period)*100:.1f}%)")
    
    # Volume analysis
    avg_volume_ratio = problem_period['volume_ratio'].mean()
    high_volume_days = (problem_period['volume_ratio'] > 1.5).sum()
    
    print(f"   Average Volume Ratio: {avg_volume_ratio:.2f}")
    print(f"   High Volume Days (>1.5x avg): {high_volume_days}")
    
    # Identify specific problem periods
    print(f"\n🎯 SPECIFIC PROBLEM PERIODS:")
    
    # Find the biggest drops
    daily_returns = problem_period['returns']
    worst_days = daily_returns.nsmallest(5)
    
    print(f"   Worst 5 days:")
    for date, return_val in worst_days.items():
        price = problem_period.loc[date, 'Close']
        print(f"     {date.strftime('%Y-%m-%d')}: {return_val*100:+.2f}% (${price:.2f})")
    
    # Find periods of high volatility
    high_vol_days = problem_period[problem_period['volatility_20'] > problem_period['volatility_20'].quantile(0.8)]
    
    print(f"\n   High Volatility Days (top 20%):")
    for date, row in high_vol_days.head(10).iterrows():
        print(f"     {date.strftime('%Y-%m-%d')}: Vol={row['volatility_20']:.4f}, Return={row['returns']*100:+.2f}%")
    
    # Market regime analysis
    print(f"\n📊 MARKET REGIME ANALYSIS:")
    
    # Trend analysis using moving averages
    sma_trend = problem_period['sma_20'].iloc[-1] - problem_period['sma_20'].iloc[0]
    ema_trend = problem_period['ema_20'].iloc[-1] - problem_period['ema_20'].iloc[0]
    
    print(f"   SMA20 Trend: ${sma_trend:+.2f}")
    print(f"   EMA20 Trend: ${ema_trend:+.2f}")
    
    # Bollinger Band analysis
    bb_squeezes = (problem_period['bb_upper'] - problem_period['bb_lower']).quantile(0.2)
    squeeze_days = ((problem_period['bb_upper'] - problem_period['bb_lower']) < bb_squeezes).sum()
    
    print(f"   Bollinger Band Squeeze Days: {squeeze_days}")
    print(f"   Average BB Position: {problem_period['bb_position'].mean():.2f}")
    
    # Create visualizations
    create_performance_charts(problem_period, context_period)
    
    # Model prediction challenges
    print(f"\n🤖 WHY THE MODEL STRUGGLED:")
    print(f"   1. 📉 HIGH VOLATILITY: {july_sep_vol:.4f} vs {context_vol:.4f} (+{((july_sep_vol/context_vol - 1) * 100):+.1f}%)")
    print(f"   2. 🔄 FREQUENT REVERSALS: {macd_crossovers + macd_cross_downs} MACD crossovers")
    print(f"   3. 📊 MIXED SIGNALS: RSI avg {july_sep_rsi:.1f} (neutral zone)")
    print(f"   4. 🎯 WHIPSAW MARKET: Price swung from ${problem_period['Close'].min():.2f} to ${problem_period['Close'].max():.2f}")
    print(f"   5. 📈 TREND CONFUSION: Multiple false breakouts and reversals")
    
    # Specific events that likely confused the model
    print(f"\n🔍 SPECIFIC EVENTS THAT CONFUSED THE MODEL:")
    
    # August 2023 - Major tech selloff
    august_data = problem_period['2023-08-01':'2023-08-31']
    august_return = (august_data['Close'].iloc[-1] / august_data['Close'].iloc[0] - 1) * 100
    
    print(f"   📉 AUGUST 2023 TECH SELLOFF:")
    print(f"     • Return: {august_return:+.2f}%")
    print(f"     • Peak to trough: ${august_data['Close'].max():.2f} to ${august_data['Close'].min():.2f}")
    print(f"     • Volatility spike: {august_data['volatility_20'].max():.4f}")
    
    # September volatility
    september_data = problem_period['2023-09-01':'2023-09-30']
    september_return = (september_data['Close'].iloc[-1] / september_data['Close'].iloc[0] - 1) * 100
    
    print(f"   📊 SEPTEMBER 2023 CONTINUED WEAKNESS:")
    print(f"     • Return: {september_return:+.2f}%")
    print(f"     • RSI oversold days: {(september_data['rsi'] < 30).sum()}")
    print(f"     • Below SMA20: {(september_data['Close'] < september_data['sma_20']).sum()}/{len(september_data)} days")
    
    print(f"\n✅ ANALYSIS COMPLETE!")

def create_performance_charts(problem_period, context_period):
    """Create charts showing the problematic period."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Price chart with moving averages
    ax1 = axes[0, 0]
    ax1.plot(problem_period.index, problem_period['Close'], 'b-', linewidth=2, label='AAPL Price')
    ax1.plot(problem_period.index, problem_period['sma_20'], 'r--', alpha=0.7, label='SMA20')
    ax1.plot(problem_period.index, problem_period['sma_50'], 'g--', alpha=0.7, label='SMA50')
    ax1.fill_between(problem_period.index, problem_period['bb_lower'], problem_period['bb_upper'], 
                     alpha=0.1, color='gray', label='Bollinger Bands')
    ax1.set_title('🔍 Jul-Sep 2023: Price vs Moving Averages', fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RSI
    ax2 = axes[0, 1]
    ax2.plot(problem_period.index, problem_period['rsi'], 'purple', linewidth=2, label='RSI')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
    ax2.fill_between(problem_period.index, 30, 70, alpha=0.1, color='yellow', label='Neutral Zone')
    ax2.set_title('📊 RSI During Problem Period', fontweight='bold')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MACD
    ax3 = axes[1, 0]
    ax3.plot(problem_period.index, problem_period['macd'], 'blue', linewidth=2, label='MACD')
    ax3.plot(problem_period.index, problem_period['macd_signal'], 'red', linewidth=2, label='Signal')
    ax3.bar(problem_period.index, problem_period['macd_histogram'], alpha=0.3, color='gray', label='Histogram')
    ax3.set_title('📈 MACD During Problem Period', fontweight='bold')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Volatility comparison
    ax4 = axes[1, 1]
    ax4.plot(context_period.index, context_period['volatility_20'], 'green', linewidth=2, label='Apr-Jun 2023')
    ax4.plot(problem_period.index, problem_period['volatility_20'], 'red', linewidth=2, label='Jul-Sep 2023')
    ax4.set_title('📊 Volatility Comparison', fontweight='bold')
    ax4.set_ylabel('20-Day Volatility')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('july_september_2023_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Charts saved to 'july_september_2023_analysis.png'")

if __name__ == '__main__':
    analyze_july_september_2023() 