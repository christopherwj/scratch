import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VolatilityContractionDetector:
    """
    Advanced Volatility Contraction Pattern Detector
    
    Identifies periods where volatility is decreasing across multiple timeframes,
    which often precede significant price movements (breakouts or breakdowns).
    """
    
    def __init__(self, data_path='data/aapl_split_adjusted.csv'):
        self.data_path = data_path
        self.data = None
        self.features = None
        self.contraction_signals = None
        
    def load_data(self):
        """Load and prepare the stock data"""
        print("Loading AAPL data...")
        self.data = pd.read_csv(self.data_path)
        if self.data is None or len(self.data) == 0:
            raise ValueError("Failed to load data or data is empty")
            
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Calculate returns
        self.data['returns'] = self.data['Close'].pct_change()
        self.data['log_returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        print(f"Loaded {len(self.data)} days of data from {self.data['Date'].min()} to {self.data['Date'].max()}")
        return self.data
    
    def calculate_volatility_indicators(self):
        """Calculate multiple volatility indicators across different timeframes"""
        print("Calculating volatility indicators...")
        
        # 1. Rolling Standard Deviation (multiple periods)
        for period in [5, 10, 20, 30, 50]:
            self.data[f'volatility_{period}d'] = self.data['returns'].rolling(period).std() * np.sqrt(252)
            self.data[f'log_volatility_{period}d'] = self.data['log_returns'].rolling(period).std() * np.sqrt(252)
        
        # 2. True Range and Average True Range
        self.data['high_low'] = self.data['High'] - self.data['Low']
        self.data['high_close'] = np.abs(self.data['High'] - self.data['Close'].shift(1))
        self.data['low_close'] = np.abs(self.data['Low'] - self.data['Close'].shift(1))
        self.data['true_range'] = np.maximum(self.data['high_low'], 
                                           np.maximum(self.data['high_close'], self.data['low_close']))
        
        for period in [14, 20, 30]:
            self.data[f'atr_{period}'] = self.data['true_range'].rolling(period).mean()
            self.data[f'atr_pct_{period}'] = self.data[f'atr_{period}'] / self.data['Close'] * 100
        
        # 3. Bollinger Band Width (volatility measure)
        for period in [20, 30, 50]:
            bb_middle = self.data['Close'].rolling(period).mean()
            bb_std = self.data['Close'].rolling(period).std()
            self.data[f'bb_width_{period}'] = (bb_middle + 2*bb_std - (bb_middle - 2*bb_std)) / bb_middle
        
        # 4. Historical Volatility (annualized)
        for period in [10, 20, 30, 60]:
            self.data[f'hist_vol_{period}'] = self.data['log_returns'].rolling(period).std() * np.sqrt(252) * 100
        
        # 5. Parkinson Volatility (uses high-low range)
        for period in [10, 20, 30]:
            self.data[f'parkinson_vol_{period}'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                ((np.log(self.data['High'] / self.data['Low']) ** 2).rolling(period).mean())
            ) * np.sqrt(252) * 100
        
        # 6. Garman-Klass Volatility
        for period in [10, 20, 30]:
            self.data[f'gk_vol_{period}'] = np.sqrt(
                (0.5 * (np.log(self.data['High'] / self.data['Low']) ** 2) - 
                 (2*np.log(2) - 1) * (np.log(self.data['Close'] / self.data['Open']) ** 2)).rolling(period).mean()
            ) * np.sqrt(252) * 100
        
        return self.data
    
    def detect_contraction_patterns(self):
        """Detect volatility contraction patterns using multiple criteria"""
        print("Detecting volatility contraction patterns...")
        
        # Initialize contraction signals
        self.data['contraction_signal'] = 0
        self.data['contraction_strength'] = 0.0
        self.data['contraction_duration'] = 0
        
        # 1. Short-term volatility contraction (5-20 days)
        self.data['vol_5_20_ratio'] = self.data['volatility_5d'] / self.data['volatility_20d']
        self.data['vol_10_30_ratio'] = self.data['volatility_10d'] / self.data['volatility_30d']
        
        # 2. ATR contraction
        self.data['atr_14_30_ratio'] = self.data['atr_14'] / self.data['atr_30']
        
        # 3. Bollinger Band width contraction
        self.data['bb_width_20_50_ratio'] = self.data['bb_width_20'] / self.data['bb_width_50']
        
        # 4. Historical volatility contraction
        self.data['hist_vol_10_30_ratio'] = self.data['hist_vol_10'] / self.data['hist_vol_30']
        
        # 5. Parkinson volatility contraction
        self.data['park_vol_10_30_ratio'] = self.data['parkinson_vol_10'] / self.data['parkinson_vol_30']
        
        # Define contraction thresholds
        contraction_thresholds = {
            'vol_5_20_ratio': 0.8,      # Short-term vol < 80% of medium-term
            'vol_10_30_ratio': 0.85,    # Medium-term vol < 85% of long-term
            'atr_14_30_ratio': 0.8,     # ATR contraction
            'bb_width_20_50_ratio': 0.85, # BB width contraction
            'hist_vol_10_30_ratio': 0.8, # Historical vol contraction
            'park_vol_10_30_ratio': 0.8  # Parkinson vol contraction
        }
        
        # Calculate contraction strength for each indicator
        contraction_indicators = []
        
        for indicator, threshold in contraction_thresholds.items():
            # Create binary signal
            signal = (self.data[indicator] < threshold).astype(int)
            self.data[f'{indicator}_signal'] = signal
            
            # Calculate strength (how much below threshold)
            strength = np.maximum(0, (threshold - self.data[indicator]) / threshold)
            self.data[f'{indicator}_strength'] = strength
            
            contraction_indicators.append(indicator)
        
        # Combined contraction signal
        signal_columns = [f'{ind}_signal' for ind in contraction_indicators]
        strength_columns = [f'{ind}_strength' for ind in contraction_indicators]
        
        # Require at least 3 indicators to show contraction
        self.data['contraction_signal'] = (self.data[signal_columns].sum(axis=1) >= 3).astype(int)
        
        # Average strength of all indicators
        self.data['contraction_strength'] = self.data[strength_columns].mean(axis=1)
        
        # Calculate contraction duration
        self.data['contraction_duration'] = 0
        duration = 0
        for i in range(1, len(self.data)):
            if self.data.loc[i, 'contraction_signal'] == 1:
                if self.data.loc[i-1, 'contraction_signal'] == 1:
                    duration += 1
                else:
                    duration = 1
            else:
                duration = 0
            self.data.loc[i, 'contraction_duration'] = duration
        
        # Enhanced signals for strong contractions
        self.data['strong_contraction'] = (
            (self.data['contraction_signal'] == 1) & 
            (self.data['contraction_strength'] > 0.3) &
            (self.data['contraction_duration'] >= 3)
        ).astype(int)
        
        # Extreme contraction signals
        self.data['extreme_contraction'] = (
            (self.data['contraction_signal'] == 1) & 
            (self.data['contraction_strength'] > 0.5) &
            (self.data['contraction_duration'] >= 5)
        ).astype(int)
        
        return self.data
    
    def analyze_breakout_potential(self):
        """Analyze potential breakout scenarios after contraction"""
        print("Analyzing breakout potential...")
        
        # Price position relative to moving averages
        for period in [20, 50, 100, 200]:
            self.data[f'sma_{period}'] = self.data['Close'].rolling(period).mean()
            self.data[f'price_vs_sma_{period}'] = self.data['Close'] / self.data[f'sma_{period}'] - 1
        
        # Bollinger Band position
        for period in [20, 30]:
            bb_middle = self.data['Close'].rolling(period).mean()
            bb_std = self.data['Close'].rolling(period).std()
            self.data[f'bb_upper_{period}'] = bb_middle + 2*bb_std
            self.data[f'bb_lower_{period}'] = bb_middle - 2*bb_std
            self.data[f'bb_position_{period}'] = (self.data['Close'] - self.data[f'bb_lower_{period}']) / \
                                                (self.data[f'bb_upper_{period}'] - self.data[f'bb_lower_{period}'])
        
        # Volume analysis during contraction
        self.data['volume_sma_20'] = self.data['Volume'].rolling(20).mean()
        self.data['volume_ratio'] = self.data['Volume'] / self.data['volume_sma_20']
        
        # Breakout signals
        self.data['breakout_signal'] = 0
        self.data['breakdown_signal'] = 0
        
        # Look for breakouts after contraction periods
        for i in range(5, len(self.data)):
            if self.data.loc[i-1, 'strong_contraction'] == 1:
                # Check for upward breakout
                if (self.data.loc[i, 'Close'] > self.data.loc[i, 'bb_upper_20'] and
                    self.data.loc[i, 'volume_ratio'] > 1.5):
                    self.data.loc[i, 'breakout_signal'] = 1
                
                # Check for downward breakout
                elif (self.data.loc[i, 'Close'] < self.data.loc[i, 'bb_lower_20'] and
                      self.data.loc[i, 'volume_ratio'] > 1.5):
                    self.data.loc[i, 'breakdown_signal'] = 1
        
        return self.data
    
    def generate_trading_signals(self):
        """Generate trading signals based on volatility contraction patterns"""
        print("Generating trading signals...")
        
        # Entry signals
        self.data['long_entry'] = 0
        self.data['short_entry'] = 0
        self.data['exit_signal'] = 0
        
        # Long entry: Strong contraction + bullish setup
        self.data['long_entry'] = (
            (self.data['strong_contraction'] == 1) &
            (self.data['price_vs_sma_20'] > 0) &
            (self.data['bb_position_20'] > 0.5) &
            (self.data['volume_ratio'] > 1.2)
        ).astype(int)
        
        # Short entry: Strong contraction + bearish setup
        self.data['short_entry'] = (
            (self.data['strong_contraction'] == 1) &
            (self.data['price_vs_sma_20'] < 0) &
            (self.data['bb_position_20'] < 0.5) &
            (self.data['volume_ratio'] > 1.2)
        ).astype(int)
        
        # Exit signals: Only volatility expansion or trend reversal (no take profit)
        self.data['exit_signal'] = (
            (self.data['contraction_signal'] == 0) |
            (self.data['breakout_signal'] == 1) |
            (self.data['breakdown_signal'] == 1)
        ).astype(int)
        
        return self.data
    
    def run_analysis(self):
        """Run complete volatility contraction analysis"""
        print("Starting Volatility Contraction Analysis...")
        
        # Load and process data
        self.load_data()
        self.calculate_volatility_indicators()
        self.detect_contraction_patterns()
        self.analyze_breakout_potential()
        self.generate_trading_signals()
        
        # Clean up NaN values
        self.data = self.data.dropna()
        
        print("Analysis complete!")
        return self.data
    
    def get_statistics(self):
        """Get summary statistics of the analysis"""
        if self.data is None:
            print("Please run analysis first")
            return
        
        total_days = len(self.data)
        contraction_days = self.data['contraction_signal'].sum()
        strong_contraction_days = self.data['strong_contraction'].sum()
        extreme_contraction_days = self.data['extreme_contraction'].sum()
        breakout_signals = self.data['breakout_signal'].sum()
        breakdown_signals = self.data['breakdown_signal'].sum()
        long_entries = self.data['long_entry'].sum()
        short_entries = self.data['short_entry'].sum()
        
        stats = {
            'Total Trading Days': total_days,
            'Contraction Days': contraction_days,
            'Contraction Percentage': f"{contraction_days/total_days*100:.1f}%",
            'Strong Contraction Days': strong_contraction_days,
            'Strong Contraction Percentage': f"{strong_contraction_days/total_days*100:.1f}%",
            'Extreme Contraction Days': extreme_contraction_days,
            'Extreme Contraction Percentage': f"{extreme_contraction_days/total_days*100:.1f}%",
            'Breakout Signals': breakout_signals,
            'Breakdown Signals': breakdown_signals,
            'Long Entry Signals': long_entries,
            'Short Entry Signals': short_entries,
            'Average Contraction Strength': f"{self.data['contraction_strength'].mean():.3f}",
            'Average Contraction Duration': f"{self.data['contraction_duration'].mean():.1f} days"
        }
        
        return stats
    
    def plot_analysis(self, start_date=None, end_date=None, save_path=None):
        """Create comprehensive visualization of volatility contraction analysis"""
        if self.data is None:
            print("Please run analysis first")
            return
        
        # Filter data by date range if specified
        plot_data = self.data.copy()
        if start_date:
            plot_data = plot_data[plot_data['Date'] >= start_date]
        if end_date:
            plot_data = plot_data[plot_data['Date'] <= end_date]
        
        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle('AAPL Volatility Contraction Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price and Bollinger Bands
        ax1 = axes[0]
        ax1.plot(plot_data['Date'], plot_data['Close'], label='AAPL Close', color='black', linewidth=1)
        ax1.plot(plot_data['Date'], plot_data['bb_upper_20'], label='BB Upper (20)', color='red', alpha=0.7)
        ax1.plot(plot_data['Date'], plot_data['bb_lower_20'], label='BB Lower (20)', color='red', alpha=0.7)
        ax1.plot(plot_data['Date'], plot_data['sma_20'], label='SMA 20', color='blue', alpha=0.7)
        
        # Highlight contraction periods
        contraction_periods = plot_data[plot_data['strong_contraction'] == 1]
        ax1.scatter(contraction_periods['Date'], contraction_periods['Close'], 
                   color='orange', s=30, alpha=0.7, label='Strong Contraction')
        
        # Highlight extreme contractions
        extreme_periods = plot_data[plot_data['extreme_contraction'] == 1]
        ax1.scatter(extreme_periods['Date'], extreme_periods['Close'], 
                   color='red', s=50, alpha=0.8, label='Extreme Contraction')
        
        ax1.set_title('Price Action with Contraction Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Volatility Indicators
        ax2 = axes[1]
        ax2.plot(plot_data['Date'], plot_data['volatility_20d'], label='20-day Volatility', color='blue')
        ax2.plot(plot_data['Date'], plot_data['volatility_50d'], label='50-day Volatility', color='red')
        ax2.plot(plot_data['Date'], plot_data['atr_pct_20'], label='ATR % (20)', color='green')
        
        # Highlight low volatility periods
        low_vol = plot_data[plot_data['contraction_signal'] == 1]
        ax2.scatter(low_vol['Date'], low_vol['volatility_20d'], 
                   color='orange', s=20, alpha=0.7, label='Contraction Periods')
        
        ax2.set_title('Volatility Indicators')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Contraction Strength and Duration
        ax3 = axes[2]
        ax3.plot(plot_data['Date'], plot_data['contraction_strength'], 
                label='Contraction Strength', color='purple', linewidth=2)
        ax3.plot(plot_data['Date'], plot_data['contraction_duration'] / 10, 
                label='Contraction Duration (scaled)', color='brown', alpha=0.7)
        
        # Add threshold lines
        ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Strong Threshold')
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Extreme Threshold')
        
        ax3.set_title('Contraction Strength and Duration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Volume and Breakout Signals
        ax4 = axes[3]
        ax4.plot(plot_data['Date'], plot_data['volume_ratio'], 
                label='Volume Ratio', color='gray', alpha=0.7)
        
        # Highlight breakout signals
        breakouts = plot_data[plot_data['breakout_signal'] == 1]
        breakdowns = plot_data[plot_data['breakdown_signal'] == 1]
        
        ax4.scatter(breakouts['Date'], breakouts['volume_ratio'], 
                   color='green', s=50, alpha=0.8, label='Breakout Signal')
        ax4.scatter(breakdowns['Date'], breakdowns['volume_ratio'], 
                   color='red', s=50, alpha=0.8, label='Breakdown Signal')
        
        ax4.axhline(y=1.5, color='black', linestyle='--', alpha=0.5, label='High Volume Threshold')
        ax4.set_title('Volume Analysis and Breakout Signals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def export_signals(self, output_path='volatility_contraction_signals.csv'):
        """Export trading signals to CSV"""
        if self.data is None:
            print("Please run analysis first")
            return
        
        # Select relevant columns for export
        export_columns = [
            'Date', 'Close', 'Volume', 'returns',
            'contraction_signal', 'contraction_strength', 'contraction_duration',
            'strong_contraction', 'extreme_contraction',
            'breakout_signal', 'breakdown_signal',
            'long_entry', 'short_entry', 'exit_signal',
            'volatility_20d', 'volatility_50d', 'atr_pct_20',
            'bb_position_20', 'volume_ratio'
        ]
        
        export_data = self.data[export_columns].copy()
        export_data.to_csv(output_path, index=False)
        print(f"Signals exported to {output_path}")
        
        return export_data

def main():
    """Main function to run the volatility contraction analysis"""
    print("AAPL Volatility Contraction Pattern Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = VolatilityContractionDetector()
    
    # Run analysis
    data = detector.run_analysis()
    
    # Get statistics
    stats = detector.get_statistics()
    print("\nAnalysis Statistics:")
    print("-" * 30)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Plot recent data (last 2 years)
    recent_start = data['Date'].max() - timedelta(days=730)
    detector.plot_analysis(start_date=recent_start, save_path='volatility_contraction_analysis.png')
    
    # Export signals
    detector.export_signals()
    
    # Show recent signals
    recent_signals = data[data['Date'] >= recent_start].copy()
    recent_contractions = recent_signals[recent_signals['strong_contraction'] == 1]
    
    print(f"\nRecent Strong Contraction Signals (last 2 years):")
    print("-" * 50)
    if len(recent_contractions) > 0:
        for _, row in recent_contractions.tail(10).iterrows():
            print(f"Date: {row['Date'].strftime('%Y-%m-%d')}, "
                  f"Price: ${row['Close']:.2f}, "
                  f"Strength: {row['contraction_strength']:.3f}, "
                  f"Duration: {row['contraction_duration']} days")
    else:
        print("No strong contraction signals in the recent period")
    
    return detector

if __name__ == "__main__":
    detector = main() 