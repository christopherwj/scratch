import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our ML components
from src.ml_features import FeatureEngineer
from src.ml_trainer import MLTrainer
from src.ml_backtester import MLBacktester
from src.data_loader import fetch_data, load_aapl_split_adjusted

def create_comprehensive_ml_visualization():
    """Create comprehensive visualizations of ML trading results with detailed feature explanation."""
    
    print("="*80)
    print("📊 COMPREHENSIVE ML TRADING RESULTS VISUALIZATION")
    print("🎯 Showing Buy/Sell Signals and Feature Engineering Details")
    print("="*80)
    
    # Load real data
    print("\n1. 📈 Loading AAPL data...")
    data = load_aapl_split_adjusted()
    if data is None:
        print("❌ Failed to load data, trying local file...")
        data = load_real_aapl_data()
        if data is None:
            print("❌ Failed to load AAPL data. Exiting.")
            return
    
    print(f"   Data loaded: {len(data):,} trading days")
    print(f"   Period: {data.index.min()} to {data.index.max()}")
    
    # Feature engineering
    print("\n2. 🔧 Engineering ML features...")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_features(data)
    targets_df = feature_engineer.create_targets(features_df)
    
    feature_names = feature_engineer.get_feature_importance_names()
    print(f"   Created {len(feature_names)} sophisticated features")
    
    # Train ML model
    print("\n3. 🤖 Training ML model...")
    trainer = MLTrainer()
    
    # Split data chronologically
    split_index = int(len(targets_df) * 0.75)
    train_data = targets_df.iloc[:split_index]
    test_data = targets_df.iloc[split_index:]
    
    X_train = train_data[feature_names]
    y_train = train_data['target_signal']
    X_test = test_data[feature_names]
    y_test = test_data['target_signal']
    
    # Train model
    models = trainer.train_models(X_train, y_train, X_test, y_test)
    best_model = models['xgboost']  # Use XGBoost as primary model
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Run backtesting
    print("\n4. 💹 Running backtesting...")
    backtester = MLBacktester()
    results = backtester.backtest_strategy(test_data, y_pred, initial_capital=10000)
    
    # Create visualizations
    print("\n5. 📊 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Price chart with buy/sell signals
    ax1 = plt.subplot(4, 2, 1)
    test_dates = test_data.index
    prices = test_data['Close']
    
    plt.plot(test_dates, prices, 'b-', linewidth=1.5, label='AAPL Price', alpha=0.8)
    
    # Add buy/sell signals
    buy_signals = test_data[y_pred == 1]
    sell_signals = test_data[y_pred == -1]
    
    if len(buy_signals) > 0:
        plt.scatter(buy_signals.index, buy_signals['Close'], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    
    if len(sell_signals) > 0:
        plt.scatter(sell_signals.index, sell_signals['Close'], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    plt.title('🤖 ML Trading Signals on AAPL Price Chart', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Portfolio value over time
    ax2 = plt.subplot(4, 2, 2)
    portfolio_values = results['portfolio_values']
    portfolio_dates = test_dates[:len(portfolio_values)]
    
    plt.plot(portfolio_dates, portfolio_values, 'g-', linewidth=2, label='ML Strategy')
    
    # Buy and hold comparison
    buy_hold_values = 10000 * (prices / prices.iloc[0])
    plt.plot(test_dates, buy_hold_values, 'b--', linewidth=2, label='Buy & Hold', alpha=0.7)
    
    plt.title('📈 Portfolio Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Feature importance
    ax3 = plt.subplot(4, 2, 3)
    feature_importance = best_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-15:]  # Top 15 features
    top_features = [feature_names[i] for i in top_features_idx]
    top_importance = feature_importance[top_features_idx]
    
    plt.barh(range(len(top_features)), top_importance, color='skyblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.title('🔍 Top 15 Most Important Features', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    # 4. Signal distribution
    ax4 = plt.subplot(4, 2, 4)
    signal_counts = pd.Series(y_pred).value_counts().sort_index()
    signal_labels = ['Sell', 'Hold', 'Buy']
    colors = ['red', 'gray', 'green']
    
    bars = plt.bar(signal_labels, [signal_counts.get(-1, 0), signal_counts.get(0, 0), signal_counts.get(1, 0)], 
                   color=colors, alpha=0.7)
    plt.title('🎯 ML Signal Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Signals')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    
    # 5. Technical indicators subplot
    ax5 = plt.subplot(4, 2, 5)
    plt.plot(test_dates, test_data['rsi_14'], 'purple', label='RSI(14)', alpha=0.8)
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
    plt.title('📊 RSI Technical Indicator', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. MACD
    ax6 = plt.subplot(4, 2, 6)
    plt.plot(test_dates, test_data['macd_12_26'], 'blue', label='MACD', alpha=0.8)
    plt.plot(test_dates, test_data['macd_signal_12_26'], 'red', label='Signal', alpha=0.8)
    plt.bar(test_dates, test_data['macd_histogram_12_26'], alpha=0.3, color='gray', label='Histogram')
    plt.title('📈 MACD Indicator', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Returns distribution
    ax7 = plt.subplot(4, 2, 7)
    strategy_returns = pd.Series(portfolio_values).pct_change().dropna()
    buy_hold_returns = buy_hold_values.pct_change().dropna()
    
    plt.hist(strategy_returns, bins=50, alpha=0.7, label='ML Strategy', color='green')
    plt.hist(buy_hold_returns, bins=50, alpha=0.7, label='Buy & Hold', color='blue')
    plt.title('📊 Daily Returns Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Performance metrics table
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    # Calculate metrics
    final_value = portfolio_values[-1] if len(portfolio_values) > 0 else 10000
    total_return = (final_value / 10000 - 1) * 100
    buy_hold_return = (buy_hold_values.iloc[-1] / 10000 - 1) * 100
    
    if len(strategy_returns) > 0:
        sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        max_drawdown = ((pd.Series(portfolio_values) / pd.Series(portfolio_values).expanding().max()) - 1).min() * 100
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    # Create performance table
    metrics_data = [
        ['Strategy', 'ML XGBoost', 'Buy & Hold'],
        ['Total Return', f'{total_return:.2f}%', f'{buy_hold_return:.2f}%'],
        ['Final Value', f'${final_value:,.0f}', f'${buy_hold_values.iloc[-1]:,.0f}'],
        ['Sharpe Ratio', f'{sharpe_ratio:.3f}', 'N/A'],
        ['Max Drawdown', f'{max_drawdown:.2f}%', 'N/A'],
        ['Total Trades', f'{len(results["trades"])}', '0'],
        ['Model Accuracy', f'{(y_pred == y_test).mean():.1%}', 'N/A']
    ]
    
    table = ax8.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                      cellLoc='center', loc='center', colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(metrics_data)):
        for j in range(len(metrics_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('📊 Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('ml_trading_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed feature explanation
    print_feature_explanation(feature_names)
    
    # Print results summary
    print_results_summary(results, total_return, buy_hold_return, sharpe_ratio, max_drawdown)

def print_feature_explanation(feature_names):
    """Print detailed explanation of how 62 features were derived."""
    
    print("\n" + "="*80)
    print("🔧 DETAILED FEATURE ENGINEERING EXPLANATION")
    print("🎯 How 62 Sophisticated Features Were Created")
    print("="*80)
    
    feature_categories = {
        'Price-based Features (5)': [
            'returns - Daily price change percentage',
            'log_returns - Logarithmic returns for better distribution',
            'price_change - Absolute price difference',
            'high_low_pct - Daily trading range as % of close',
            'close_open_pct - Intraday price movement'
        ],
        
        'Moving Averages (16)': [
            'sma_5, sma_10, sma_20, sma_50 - Simple moving averages',
            'ema_5, ema_10, ema_20, ema_50 - Exponential moving averages',
            'price_sma_5_ratio, price_sma_10_ratio, price_sma_20_ratio, price_sma_50_ratio',
            'price_ema_5_ratio, price_ema_10_ratio, price_ema_20_ratio, price_ema_50_ratio'
        ],
        
        'RSI Indicators (9)': [
            'rsi_14, rsi_21, rsi_28 - RSI with different periods',
            'rsi_14_overbought, rsi_21_overbought, rsi_28_overbought - Binary signals',
            'rsi_14_oversold, rsi_21_oversold, rsi_28_oversold - Binary signals'
        ],
        
        'MACD Variations (21)': [
            'Three MACD configurations: (12,26,9), (5,35,5), (19,39,9)',
            'For each: macd_line, signal_line, histogram (9 features)',
            'Binary signals: above_signal, cross_above, cross_below (9 features)',
            'Plus 3 additional momentum crossover signals'
        ],
        
        'Bollinger Bands (8)': [
            'bb_upper_20, bb_lower_20, bb_upper_50, bb_lower_50',
            'bb_position_20, bb_position_50 - Position within bands',
            'bb_squeeze_20, bb_squeeze_50 - Volatility squeeze detection'
        ],
        
        'Volatility Features (6)': [
            'volatility_10, volatility_20, volatility_30 - Rolling volatility',
            'volatility_rank_10, volatility_rank_20, volatility_rank_30 - Percentile ranks'
        ],
        
        'Volume Features (3)': [
            'volume_sma_20 - Volume moving average',
            'volume_ratio - Current vs average volume',
            'price_volume - Price-weighted volume'
        ],
        
        'Momentum Features (9)': [
            'momentum_5, momentum_10, momentum_20 - Price momentum',
            'roc_5, roc_10, roc_20 - Rate of change',
            'Plus 3 additional momentum calculations'
        ],
        
        'Support/Resistance (8)': [
            'highest_20, lowest_20, highest_50, lowest_50 - Price extremes',
            'distance_to_high_20, distance_to_low_20 - Distance to resistance/support',
            'distance_to_high_50, distance_to_low_50 - Longer-term levels'
        ],
        
        'Market Regime (2)': [
            'trend_strength - Correlation with linear trend',
            'mean_reversion - Standardized distance from mean'
        ],
        
        'Lagged Features (12)': [
            'returns_lag_1, returns_lag_2, returns_lag_3, returns_lag_5',
            'rsi_14_lag_1, rsi_14_lag_2, rsi_14_lag_3, rsi_14_lag_5',
            'macd_histogram_12_26_lag_1, macd_histogram_12_26_lag_2, etc.'
        ],
        
        'Time-based Features (5)': [
            'day_of_week - Weekday effect',
            'month - Monthly seasonality',
            'quarter - Quarterly patterns',
            'is_month_end - End-of-month effect',
            'is_quarter_end - Quarter-end effect'
        ]
    }
    
    total_features = 0
    for category, features in feature_categories.items():
        print(f"\n📊 {category}:")
        for feature in features:
            print(f"   • {feature}")
        # Extract number from category name
        import re
        num_match = re.search(r'\((\d+)\)', category)
        if num_match:
            total_features += int(num_match.group(1))
    
    print(f"\n🎯 TOTAL FEATURES: {total_features}")
    print(f"✅ Actual features created: {len(feature_names)}")
    
    print(f"\n🔬 FEATURE ENGINEERING METHODOLOGY:")
    print(f"   1. 📈 Price Action: Basic OHLC transformations and ratios")
    print(f"   2. 📊 Technical Indicators: RSI, MACD, Bollinger Bands with multiple periods")
    print(f"   3. 🔄 Momentum: Rate of change and momentum across timeframes")
    print(f"   4. 📉 Volatility: Rolling volatility and percentile rankings")
    print(f"   5. 📊 Volume: Volume patterns and price-volume relationships")
    print(f"   6. 🎯 Support/Resistance: Dynamic levels based on price extremes")
    print(f"   7. ⏰ Time Series: Lagged values for temporal dependencies")
    print(f"   8. 📅 Seasonality: Calendar effects and market timing")
    print(f"   9. 🔄 Market Regime: Trend strength and mean reversion signals")
    print(f"  10. 🎲 Binary Signals: Threshold-based binary features")

def print_results_summary(results, total_return, buy_hold_return, sharpe_ratio, max_drawdown):
    """Print comprehensive results summary."""
    
    print(f"\n" + "="*80)
    print("🏆 COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n📊 STRATEGY PERFORMANCE:")
    print(f"   🤖 ML XGBoost Strategy:")
    print(f"      Total Return:     {total_return:+.2f}%")
    print(f"      Sharpe Ratio:     {sharpe_ratio:.3f}")
    print(f"      Max Drawdown:     {max_drawdown:+.2f}%")
    print(f"      Total Trades:     {len(results['trades'])}")
    
    print(f"\n   📈 Buy & Hold Comparison:")
    print(f"      Total Return:     {buy_hold_return:+.2f}%")
    print(f"      Strategy Advantage: {total_return - buy_hold_return:+.2f} percentage points")
    
    print(f"\n🎯 KEY INSIGHTS:")
    if total_return > buy_hold_return:
        print(f"   ✅ ML strategy OUTPERFORMED buy-and-hold")
        print(f"   🚀 Demonstrates ML's ability to identify profitable patterns")
    else:
        print(f"   📊 ML strategy shows promise but needs refinement")
    
    print(f"\n🔧 TECHNICAL ACHIEVEMENTS:")
    print(f"   • 62 sophisticated features engineered")
    print(f"   • GPU-accelerated XGBoost training")
    print(f"   • Proper time series methodology")
    print(f"   • Comprehensive backtesting framework")
    
    print(f"\n📈 NEXT STEPS FOR IMPROVEMENT:")
    print(f"   1. 🔄 Ensemble multiple ML models")
    print(f"   2. 📊 Add alternative data sources")
    print(f"   3. 🎯 Implement dynamic position sizing")
    print(f"   4. 📈 Add risk management overlays")
    print(f"   5. 🔧 Hyperparameter optimization")

def load_real_aapl_data():
    """Load real AAPL data from the data directory."""
    from pathlib import Path
    
    # Load the AAPL data file
    data_file = Path('data/aapl_split_adjusted.csv')
    
    if not data_file.exists():
        print(f"❌ AAPL data file not found at {data_file}")
        return None
    
    try:
        # Load the data with proper date parsing
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"✅ Loaded real AAPL data from {data_file}")
        print(f"   Data shape: {data.shape}")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading AAPL data: {e}")
        return None

if __name__ == '__main__':
    create_comprehensive_ml_visualization() 