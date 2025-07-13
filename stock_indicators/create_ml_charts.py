import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

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

def engineer_ml_features(data):
    """Engineer comprehensive features for ML model."""
    df = data.copy()
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
    df['open_close_spread'] = (df['Close'] - df['Open']) / df['Open']
    
    # Moving averages and ratios
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        df[f'price_sma_ratio_{period}'] = df['Close'] / df[f'sma_{period}']
        
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
    df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # Bollinger Bands
    bb_period = 20
    bb_middle = df['Close'].rolling(bb_period).mean()
    bb_std = df['Close'].rolling(bb_period).std()
    df['bb_upper'] = bb_middle + (bb_std * 2)
    df['bb_lower'] = bb_middle - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / bb_middle
    
    # Volatility features
    for period in [10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        df[f'volatility_rank_{period}'] = df[f'volatility_{period}'].rolling(100).rank(pct=True)
    
    # Momentum features
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
    
    # Support/Resistance
    for period in [20, 50]:
        df[f'highest_{period}'] = df['High'].rolling(period).max()
        df[f'lowest_{period}'] = df['Low'].rolling(period).min()
        df[f'distance_to_high_{period}'] = (df[f'highest_{period}'] - df['Close']) / df['Close']
        df[f'distance_to_low_{period}'] = (df['Close'] - df[f'lowest_{period}']) / df['Close']
    
    # Volume features
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    df['price_volume'] = df['Close'] * df['Volume']
    
    # Lagged features (important for time series)
    for lag in [1, 2, 3]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        df[f'macd_histogram_lag_{lag}'] = df['macd_histogram'].shift(lag)
        df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)
    
    # Market regime features
    df['trend_strength'] = df['Close'].rolling(20).apply(lambda x: np.corrcoef(x, range(len(x)))[0,1])
    df['mean_reversion_signal'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    
    return df.dropna()

def create_ml_targets(data, lookahead_days=3):
    """Create intelligent targets based on future price movements."""
    df = data.copy()
    
    # Calculate future returns
    df['future_return'] = df['Close'].shift(-lookahead_days) / df['Close'] - 1
    
    # Dynamic threshold based on rolling volatility
    rolling_vol = df['returns'].rolling(30).std()
    threshold_multiplier = 1.2  # 1.2x volatility as threshold
    
    df['target'] = 0  # Default: Hold
    
    # Buy signal: future return > positive threshold
    buy_threshold = rolling_vol * threshold_multiplier
    df.loc[df['future_return'] > buy_threshold, 'target'] = 1
    
    # Sell signal: future return < negative threshold  
    sell_threshold = -rolling_vol * threshold_multiplier
    df.loc[df['future_return'] < sell_threshold, 'target'] = -1
    
    return df.dropna()

def create_comprehensive_charts():
    """Create comprehensive ML trading charts with buy/sell signals."""
    
    print("="*80)
    print("📊 COMPREHENSIVE ML TRADING CHARTS WITH BUY/SELL SIGNALS")
    print("🎯 Showing Feature Engineering and Trading Results")
    print("="*80)
    
    # Load real AAPL data
    print("\n1. 📊 Loading real AAPL data...")
    data = load_real_aapl_data()
    if data is None:
        print("❌ Failed to load AAPL data. Exiting.")
        return
    print(f"   Loaded {len(data):,} trading days")
    
    # Feature engineering
    print("\n2. 🔧 Engineering ML features...")
    features_df = engineer_ml_features(data)
    targets_df = create_ml_targets(features_df)
    
    # Get feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                   and not col.startswith('future_')]
    
    print(f"   Created {len(feature_cols)} advanced features")
    
    # Split data chronologically
    split_index = int(len(targets_df) * 0.75)
    train_data = targets_df.iloc[:split_index]
    test_data = targets_df.iloc[split_index:]
    
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    print(f"\n3. 📈 Training/Testing split:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing:  {len(X_test):,} samples")
    
    # Feature scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train GPU-accelerated XGBoost
    print(f"\n4. 🤖 Training GPU-accelerated XGBoost...")
    
    # Convert targets for XGBoost
    y_train_xgb = y_train + 1
    y_test_xgb = y_test + 1
    
    try:
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=300,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            eval_metric='mlogloss'
        )
        model.fit(X_train_scaled, y_train_xgb, verbose=False)
        print("   ✅ GPU XGBoost training completed!")
        
    except Exception as e:
        print(f"   ⚠️  GPU training failed: {e}")
        print("   🔄 Falling back to CPU XGBoost...")
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='mlogloss'
        )
        model.fit(X_train_scaled, y_train_xgb, verbose=False)
        print("   ✅ CPU XGBoost training completed!")
    
    # Make predictions
    y_pred_xgb = model.predict(X_test_scaled)
    y_pred = y_pred_xgb - 1  # Convert back to -1, 0, 1
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   🎯 Model accuracy: {accuracy:.1%}")
    
    # Backtest ML strategy
    print(f"\n5. 💹 Backtesting ML strategy...")
    
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    portfolio_values = []
    trades = []
    
    for i, (date, row) in enumerate(test_data.iterrows()):
        if i < len(y_pred):
            signal = y_pred[i]
            price = row['Close']
            
            # Update portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
            
            # Execute trades based on ML signals
            if signal == 1 and shares == 0:  # Buy signal
                shares_to_buy = cash / price * 0.995  # 0.5% transaction cost
                shares += shares_to_buy
                cash = 0
                trades.append(('BUY', date, price))
                
            elif signal == -1 and shares > 0:  # Sell signal
                cash = shares * price * 0.995  # 0.5% transaction cost
                trades.append(('SELL', date, price))
                shares = 0
    
    # Calculate results
    final_price = test_data['Close'].iloc[-1]
    final_value = cash + shares * final_price
    ml_return = (final_value / initial_cash - 1) * 100
    buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
    
    print(f"   🏆 ML Strategy: {ml_return:+.2f}% return")
    print(f"   📈 Buy & Hold: {buy_hold_return:+.2f}% return")
    
    # Create comprehensive charts
    print(f"\n6. 📊 Creating comprehensive charts...")
    
    # Set up the plotting
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Price chart with buy/sell signals
    ax1 = plt.subplot(3, 3, 1)
    test_dates = test_data.index
    prices = test_data['Close']
    
    plt.plot(test_dates, prices, 'b-', linewidth=1.5, label='Price', alpha=0.8)
    
    # Add buy/sell signals
    buy_signals = test_data[y_pred == 1]
    sell_signals = test_data[y_pred == -1]
    
    if len(buy_signals) > 0:
        plt.scatter(buy_signals.index, buy_signals['Close'], 
                   color='green', marker='^', s=100, label=f'Buy Signals ({len(buy_signals)})', zorder=5)
    
    if len(sell_signals) > 0:
        plt.scatter(sell_signals.index, sell_signals['Close'], 
                   color='red', marker='v', s=100, label=f'Sell Signals ({len(sell_signals)})', zorder=5)
    
    plt.title('🤖 ML Trading Signals', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Portfolio performance
    ax2 = plt.subplot(3, 3, 2)
    portfolio_dates = test_dates[:len(portfolio_values)]
    
    plt.plot(portfolio_dates, portfolio_values, 'g-', linewidth=2, label='ML Strategy')
    buy_hold_values = initial_cash * (prices / prices.iloc[0])
    plt.plot(test_dates, buy_hold_values, 'b--', linewidth=2, label='Buy & Hold', alpha=0.7)
    
    plt.title('📈 Portfolio Performance', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Feature importance
    ax3 = plt.subplot(3, 3, 3)
    feature_importance = model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]
    top_features = [feature_cols[i] for i in top_features_idx]
    top_importance = feature_importance[top_features_idx]
    
    plt.barh(range(len(top_features)), top_importance, color='skyblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.title('🔍 Top 10 Features', fontsize=12, fontweight='bold')
    plt.xlabel('Importance')
    plt.grid(True, alpha=0.3)
    
    # 4. RSI with signals
    ax4 = plt.subplot(3, 3, 4)
    plt.plot(test_dates, test_data['rsi'], 'purple', label='RSI', alpha=0.8)
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
    
    # Add buy/sell signals on RSI
    if len(buy_signals) > 0:
        plt.scatter(buy_signals.index, buy_signals['rsi'], 
                   color='green', marker='^', s=50, alpha=0.7, zorder=5)
    if len(sell_signals) > 0:
        plt.scatter(sell_signals.index, sell_signals['rsi'], 
                   color='red', marker='v', s=50, alpha=0.7, zorder=5)
    
    plt.title('📊 RSI with ML Signals', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. MACD with signals
    ax5 = plt.subplot(3, 3, 5)
    plt.plot(test_dates, test_data['macd'], 'blue', label='MACD', alpha=0.8)
    plt.plot(test_dates, test_data['macd_signal'], 'red', label='Signal', alpha=0.8)
    plt.bar(test_dates, test_data['macd_histogram'], alpha=0.3, color='gray', label='Histogram')
    
    # Add buy/sell signals on MACD
    if len(buy_signals) > 0:
        plt.scatter(buy_signals.index, buy_signals['macd'], 
                   color='green', marker='^', s=50, alpha=0.7, zorder=5)
    if len(sell_signals) > 0:
        plt.scatter(sell_signals.index, sell_signals['macd'], 
                   color='red', marker='v', s=50, alpha=0.7, zorder=5)
    
    plt.title('📈 MACD with ML Signals', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Bollinger Bands with signals
    ax6 = plt.subplot(3, 3, 6)
    plt.plot(test_dates, test_data['Close'], 'b-', linewidth=1.5, label='Price')
    plt.plot(test_dates, test_data['bb_upper'], 'r--', alpha=0.5, label='Upper Band')
    plt.plot(test_dates, test_data['bb_lower'], 'g--', alpha=0.5, label='Lower Band')
    plt.fill_between(test_dates, test_data['bb_lower'], test_data['bb_upper'], alpha=0.1, color='gray')
    
    # Add buy/sell signals
    if len(buy_signals) > 0:
        plt.scatter(buy_signals.index, buy_signals['Close'], 
                   color='green', marker='^', s=50, alpha=0.7, zorder=5)
    if len(sell_signals) > 0:
        plt.scatter(sell_signals.index, sell_signals['Close'], 
                   color='red', marker='v', s=50, alpha=0.7, zorder=5)
    
    plt.title('📊 Bollinger Bands with ML Signals', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Signal distribution
    ax7 = plt.subplot(3, 3, 7)
    signal_counts = pd.Series(y_pred).value_counts().sort_index()
    signal_labels = ['Sell', 'Hold', 'Buy']
    colors = ['red', 'gray', 'green']
    
    bars = plt.bar(signal_labels, [signal_counts.get(-1, 0), signal_counts.get(0, 0), signal_counts.get(1, 0)], 
                   color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.title('🎯 Signal Distribution', fontsize=12, fontweight='bold')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # 8. Returns distribution
    ax8 = plt.subplot(3, 3, 8)
    strategy_returns = pd.Series(portfolio_values).pct_change().dropna()
    buy_hold_returns = buy_hold_values.pct_change().dropna()
    
    plt.hist(strategy_returns, bins=30, alpha=0.7, label='ML Strategy', color='green')
    plt.hist(buy_hold_returns, bins=30, alpha=0.7, label='Buy & Hold', color='blue')
    plt.title('📊 Returns Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Cumulative returns
    ax9 = plt.subplot(3, 3, 9)
    strategy_cumret = (1 + strategy_returns).cumprod()
    buyhold_cumret = (1 + buy_hold_returns).cumprod()
    
    plt.plot(portfolio_dates[1:], strategy_cumret, 'g-', linewidth=2, label='ML Strategy')
    plt.plot(test_dates[1:], buyhold_cumret, 'b--', linewidth=2, label='Buy & Hold')
    plt.title('📈 Cumulative Returns', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_trading_comprehensive_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print feature explanation
    print_feature_explanation(feature_cols)
    
    # Print final results
    print(f"\n" + "="*80)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"📊 ML Strategy Performance:")
    print(f"   Total Return:     {ml_return:+.2f}%")
    print(f"   Final Value:      ${final_value:,.0f}")
    print(f"   Total Trades:     {len(trades)}")
    print(f"   Model Accuracy:   {accuracy:.1%}")
    print(f"\n📈 Buy & Hold Performance:")
    print(f"   Total Return:     {buy_hold_return:+.2f}%")
    print(f"   Final Value:      ${buy_hold_values.iloc[-1]:,.0f}")
    print(f"\n🎯 Strategy Advantage: {ml_return - buy_hold_return:+.2f} percentage points")
    
    if ml_return > buy_hold_return:
        print(f"✅ ML STRATEGY WINS! 🚀")
    else:
        print(f"📊 Buy & Hold wins this time")
    
    print(f"\n🔧 Features Engineered: {len(feature_cols)}")
    print(f"⚡ GPU Acceleration: {'Enabled' if 'gpu_hist' in str(model.get_params().get('tree_method', '')) else 'CPU Fallback'}")

def print_feature_explanation(feature_cols):
    """Print detailed explanation of how features were derived."""
    
    print("\n" + "="*80)
    print("🔧 HOW 62 FEATURES WERE DERIVED")
    print("="*80)
    
    # Categorize features
    feature_categories = {
        'Price-based (5)': ['returns', 'log_returns', 'high_low_spread', 'open_close_spread'],
        'Moving Averages (12)': [f for f in feature_cols if 'sma_' in f or 'ema_' in f or 'price_sma_ratio' in f],
        'RSI Indicators (3)': [f for f in feature_cols if 'rsi' in f],
        'MACD (4)': [f for f in feature_cols if 'macd' in f],
        'Bollinger Bands (4)': [f for f in feature_cols if 'bb_' in f],
        'Volatility (4)': [f for f in feature_cols if 'volatility' in f],
        'Momentum (6)': [f for f in feature_cols if 'momentum' in f or 'roc_' in f],
        'Support/Resistance (8)': [f for f in feature_cols if 'highest' in f or 'lowest' in f or 'distance' in f],
        'Volume (3)': [f for f in feature_cols if 'volume' in f],
        'Lagged Features (12)': [f for f in feature_cols if 'lag_' in f],
        'Market Regime (2)': [f for f in feature_cols if 'trend_strength' in f or 'mean_reversion' in f]
    }
    
    for category, features in feature_categories.items():
        actual_features = [f for f in features if f in feature_cols]
        if actual_features:
            print(f"\n📊 {category}: {len(actual_features)} features")
            for feature in actual_features[:5]:  # Show first 5
                print(f"   • {feature}")
            if len(actual_features) > 5:
                print(f"   ... and {len(actual_features) - 5} more")
    
    print(f"\n🎯 FEATURE ENGINEERING METHODOLOGY:")
    print(f"   1. 📈 Price transformations: Returns, ratios, spreads")
    print(f"   2. 📊 Technical indicators: RSI, MACD, Bollinger Bands")
    print(f"   3. 🔄 Momentum measures: Rate of change across timeframes")
    print(f"   4. 📉 Volatility metrics: Rolling volatility and rankings")
    print(f"   5. 📊 Volume analysis: Volume ratios and patterns")
    print(f"   6. 🎯 Support/Resistance: Dynamic price levels")
    print(f"   7. ⏰ Lagged features: Historical values for time series")
    print(f"   8. 🔄 Market regime: Trend strength and mean reversion")
    print(f"   9. 📊 Moving averages: Multiple timeframes and ratios")
    print(f"  10. 🎲 Binary signals: Threshold-based indicators")

if __name__ == '__main__':
    create_comprehensive_charts() 