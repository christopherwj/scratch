"""
Ideal VCP Pattern Chart Generator
Based on the Config class parameters, this generates a perfect VCP pattern
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from config import Config

def generate_ideal_vcp_data():
    """
    Generate ideal VCP pattern data based on Config parameters
    """
    # Total days for the pattern
    total_days = 252  # 1 year of data
    dates = pd.date_range(start='2023-01-01', periods=total_days, freq='D')
    
    # Initialize arrays
    prices = np.zeros(total_days)
    volumes = np.zeros(total_days)
    
    # Phase 1: Uptrend (first 100 days)
    uptrend_days = 100
    base_price = 100
    uptrend_slope = 0.005  # 0.5% daily growth
    uptrend_volatility = 0.02  # 2% daily volatility
    
    for i in range(uptrend_days):
        trend_component = base_price * (1 + uptrend_slope) ** i
        noise = np.random.normal(0, uptrend_volatility * trend_component)
        prices[i] = trend_component + noise
        volumes[i] = np.random.uniform(1000000, 2000000)  # High volume during uptrend
    
    # Phase 2: First consolidation (days 100-130)
    consolidation_start = uptrend_days
    consolidation_end = consolidation_start + 30  # 30 days consolidation
    resistance_level = prices[consolidation_start - 1] * 1.05  # 5% above last uptrend price
    support_level = prices[consolidation_start - 1] * 0.95  # 5% below
    
    for i in range(consolidation_start, consolidation_end):
        # Oscillating pattern within range
        cycle = np.sin((i - consolidation_start) * 2 * np.pi / 15) * 0.02
        prices[i] = (resistance_level + support_level) / 2 + cycle * (resistance_level - support_level) / 2
        volumes[i] = np.random.uniform(800000, 1200000)  # Lower volume during consolidation
    
    # Phase 3: Second uptrend (days 130-160)
    second_uptrend_start = consolidation_end
    second_uptrend_end = second_uptrend_start + 30
    second_base = prices[second_uptrend_start - 1]
    
    for i in range(second_uptrend_start, second_uptrend_end):
        trend_component = second_base * (1 + uptrend_slope) ** (i - second_uptrend_start)
        noise = np.random.normal(0, uptrend_volatility * trend_component)
        prices[i] = trend_component + noise
        volumes[i] = np.random.uniform(1000000, 2000000)
    
    # Phase 4: Second consolidation (days 160-190) - Volatility Contraction
    second_consolidation_start = second_uptrend_end
    second_consolidation_end = second_consolidation_start + 30
    second_resistance = prices[second_consolidation_start - 1] * 1.03  # Tighter range
    second_support = prices[second_consolidation_start - 1] * 0.97
    
    for i in range(second_consolidation_start, second_consolidation_end):
        # Much tighter oscillation - volatility contraction
        cycle = np.sin((i - second_consolidation_start) * 2 * np.pi / 20) * 0.01
        prices[i] = (second_resistance + second_support) / 2 + cycle * (second_resistance - second_support) / 2
        volumes[i] = np.random.uniform(600000, 900000)  # Further volume decline
    
    # Phase 5: Final consolidation (days 190-220) - Maximum contraction
    final_consolidation_start = second_consolidation_end
    final_consolidation_end = final_consolidation_start + 30
    final_resistance = prices[final_consolidation_start - 1] * 1.015  # Very tight range
    final_support = prices[final_consolidation_start - 1] * 0.985
    
    for i in range(final_consolidation_start, final_consolidation_end):
        # Minimal oscillation - maximum volatility contraction
        cycle = np.sin((i - final_consolidation_start) * 2 * np.pi / 25) * 0.005
        prices[i] = (final_resistance + final_support) / 2 + cycle * (final_resistance - final_support) / 2
        volumes[i] = np.random.uniform(400000, 700000)  # Minimum volume
    
    # Phase 6: Breakout (days 220-252)
    breakout_start = final_consolidation_end
    breakout_resistance = final_resistance
    
    for i in range(breakout_start, total_days):
        # Strong breakout above resistance
        breakout_strength = 0.02  # 2% daily growth during breakout
        days_since_breakout = i - breakout_start
        prices[i] = breakout_resistance * (1 + breakout_strength) ** days_since_breakout
        volumes[i] = np.random.uniform(2000000, 4000000)  # High volume breakout
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volumes
    })
    
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    df['SMA'] = df['Close'].rolling(window=period).mean()
    df['BB_Upper'] = df['SMA'] + (df['Close'].rolling(window=period).std() * std_dev)
    df['BB_Lower'] = df['SMA'] - (df['Close'].rolling(window=period).std() * std_dev)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA']
    return df

def calculate_rsi(df, period=14):
    """Calculate RSI"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def create_ideal_vcp_chart():
    """Create the ideal VCP pattern chart"""
    
    # Generate data
    df = generate_ideal_vcp_data()
    df = calculate_bollinger_bands(df, Config.BOLLINGER_BANDS_PERIOD, Config.BOLLINGER_BANDS_STD)
    df = calculate_rsi(df, Config.RSI_PERIOD)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), 
                                            gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Plot 1: Price and Bollinger Bands
    ax1.plot(df['Date'], df['Close'], label='Price', linewidth=2, color='#1f77b4')
    ax1.plot(df['Date'], df['BB_Upper'], '--', label='Bollinger Upper', alpha=0.7, color='#ff7f0e')
    ax1.plot(df['Date'], df['BB_Lower'], '--', label='Bollinger Lower', alpha=0.7, color='#ff7f0e')
    ax1.plot(df['Date'], df['SMA'], '--', label='20 SMA', alpha=0.7, color='#2ca02c')
    
    # Highlight VCP phases
    phases = [
        (0, 100, 'Phase 1: Uptrend', '#2ecc71'),
        (100, 130, 'Phase 2: First Consolidation', '#f39c12'),
        (130, 160, 'Phase 3: Second Uptrend', '#2ecc71'),
        (160, 190, 'Phase 4: Volatility Contraction', '#e74c3c'),
        (190, 220, 'Phase 5: Final Consolidation', '#9b59b6'),
        (220, 252, 'Phase 6: Breakout', '#3498db')
    ]
    
    for start, end, label, color in phases:
        ax1.axvspan(df['Date'].iloc[start], df['Date'].iloc[end-1], 
                   alpha=0.2, color=color, label=label)
    
    ax1.set_title('Ideal VCP Pattern - Perfect Volatility Contraction Pattern', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volume
    ax2.bar(df['Date'], df['Volume'], alpha=0.7, color='#34495e')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_title('Volume Pattern - Declining During Consolidation, Spike on Breakout', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bollinger Band Width (Volatility)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['Date'], df['BB_Width'], color='red', linewidth=2, label='BB Width')
    ax2_twin.set_ylabel('BB Width (Volatility)', color='red', fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # Plot 4: RSI
    ax3.plot(df['Date'], df['RSI'], color='#e74c3c', linewidth=2)
    ax3.axhline(y=Config.RSI_OVERBOUGHT, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax3.axhline(y=Config.RSI_OVERSOLD, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Neutral')
    ax3.set_ylabel('RSI', fontsize=12)
    ax3.set_title('RSI - Momentum Indicator', fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 5: Key Metrics
    # Calculate key VCP metrics
    consolidation_start = 160
    consolidation_end = 220
    pre_consolidation_volatility = df['BB_Width'].iloc[consolidation_start-20:consolidation_start].mean()
    consolidation_volatility = df['BB_Width'].iloc[consolidation_start:consolidation_end].mean()
    volatility_contraction = (pre_consolidation_volatility - consolidation_volatility) / pre_consolidation_volatility
    
    pre_consolidation_volume = df['Volume'].iloc[consolidation_start-20:consolidation_start].mean()
    consolidation_volume = df['Volume'].iloc[consolidation_start:consolidation_end].mean()
    volume_decline = (pre_consolidation_volume - consolidation_volume) / pre_consolidation_volume
    
    # Create metrics text
    metrics_text = f"""
    VCP Pattern Metrics:
    • Consolidation Days: {consolidation_end - consolidation_start} days
    • Volatility Contraction: {volatility_contraction:.1%} (Target: >{Config.VOLATILITY_CONTRACTION_THRESHOLD:.1%})
    • Volume Decline: {volume_decline:.1%} (Target: >{Config.VOLUME_DECLINE_THRESHOLD:.1%})
    • Breakout Percentage: {Config.BREAKOUT_PERCENTAGE:.1%}
    • Volume Multiplier: {Config.BREAKOUT_VOLUME_MULTIPLIER}x
    • Stop Loss: {Config.STOP_LOSS_PERCENTAGE:.1%}
    • Profit Target: {Config.PROFIT_TARGET_MULTIPLIER:.1f}:1 R:R
    """
    
    ax4.text(0.05, 0.5, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", 
                                                 facecolor="lightblue", alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('VCP Detection Parameters', fontsize=12, fontweight='bold')
    
    # Format x-axis
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df

if __name__ == "__main__":
    print("Generating ideal VCP pattern chart based on Config parameters...")
    df = create_ideal_vcp_chart()
    print("Chart generated successfully!")
    print(f"Pattern shows {Config.VCP_MIN_CONSOLIDATION_DAYS}-{Config.VCP_MAX_CONSOLIDATION_DAYS} day consolidation")
    print(f"with {Config.VOLATILITY_CONTRACTION_THRESHOLD:.1%} volatility contraction and")
    print(f"{Config.VOLUME_DECLINE_THRESHOLD:.1%} volume decline threshold.") 