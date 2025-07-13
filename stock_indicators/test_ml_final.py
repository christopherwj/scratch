import pandas as pd
import numpy as np
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

def run_ml_trading_demo():
    """Demonstrate ML trading strategy beating classical approaches."""
    print("="*80)
    print("🚀 MACHINE LEARNING TRADING STRATEGY DEMONSTRATION")
    print("🎯 Using RTX 3090 Ti GPU Acceleration with XGBoost")
    print("="*80)
    
    # Load real AAPL data
    print("\n1. 📊 Loading real AAPL data...")
    data = load_real_aapl_data()
    if data is None:
        print("❌ Failed to load AAPL data. Exiting.")
        return
    print(f"   Loaded {len(data):,} trading days")
    print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Feature engineering
    print("\n2. 🔧 Engineering ML features...")
    features_df = engineer_ml_features(data)
    targets_df = create_ml_targets(features_df)
    
    # Get feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                   and not col.startswith('future_')]
    
    print(f"   Created {len(feature_cols)} advanced features")
    print(f"   Features include: technical indicators, momentum, volatility, volume, lagged values")
    
    # Split data chronologically (no random shuffling for time series)
    split_index = int(len(targets_df) * 0.75)  # 75% train, 25% test
    
    train_data = targets_df.iloc[:split_index]
    test_data = targets_df.iloc[split_index:]
    
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    print(f"\n3. 📈 Data split:")
    print(f"   Training: {len(X_train):,} samples ({train_data.index.min().date()} to {train_data.index.max().date()})")
    print(f"   Testing:  {len(X_test):,} samples ({test_data.index.min().date()} to {test_data.index.max().date()})")
    
    # Check target distribution
    train_dist = y_train.value_counts().sort_index()
    test_dist = y_test.value_counts().sort_index()
    print(f"   Training signals: Sell={train_dist.get(-1,0)}, Hold={train_dist.get(0,0)}, Buy={train_dist.get(1,0)}")
    print(f"   Test signals:     Sell={test_dist.get(-1,0)}, Hold={test_dist.get(0,0)}, Buy={test_dist.get(1,0)}")
    
    # Feature scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train GPU-accelerated XGBoost
    print(f"\n4. 🎯 Training GPU-accelerated XGBoost model...")
    
    # Convert targets for XGBoost (0, 1, 2 instead of -1, 0, 1)
    y_train_xgb = y_train + 1
    y_test_xgb = y_test + 1
    
    try:
        # GPU XGBoost
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
        print("   ✅ GPU XGBoost training completed successfully!")
        
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
    
    # Feature importance analysis
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n5. 🔍 Top 10 most predictive features:")
    for i, (feature, importance) in enumerate(top_features, 1):
        print(f"   {i:2d}. {feature:<30}: {importance:.4f}")
    
    # Backtest ML strategy
    print(f"\n6. 💹 Backtesting ML strategy...")
    
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
    
    # Final results
    final_price = test_data['Close'].iloc[-1]
    final_value = cash + shares * final_price
    ml_return = (final_value / initial_cash - 1) * 100
    
    # Buy and hold comparison
    buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
    
    # Classical MACD strategy for comparison
    test_data_copy = test_data.copy()
    classical_signals = []
    for i in range(len(test_data_copy)):
        if i > 0:
            macd_curr = test_data_copy['macd'].iloc[i]
            macd_signal_curr = test_data_copy['macd_signal'].iloc[i]
            macd_prev = test_data_copy['macd'].iloc[i-1]
            macd_signal_prev = test_data_copy['macd_signal'].iloc[i-1]
            rsi = test_data_copy['rsi'].iloc[i]
            
            # MACD crossover with RSI filter
            if macd_curr > macd_signal_curr and macd_prev <= macd_signal_prev and rsi < 70:
                classical_signals.append(1)  # Buy
            elif macd_curr < macd_signal_curr and macd_prev >= macd_signal_prev and rsi > 30:
                classical_signals.append(-1)  # Sell
            else:
                classical_signals.append(0)  # Hold
        else:
            classical_signals.append(0)
    
    # Backtest classical strategy
    classical_cash = initial_cash
    classical_shares = 0
    classical_trades = []
    
    for i, signal in enumerate(classical_signals):
        if i < len(test_data):
            price = test_data['Close'].iloc[i]
            
            if signal == 1 and classical_shares == 0:  # Buy
                classical_shares = classical_cash / price * 0.995
                classical_cash = 0
                classical_trades.append(('BUY', price))
                
            elif signal == -1 and classical_shares > 0:  # Sell
                classical_cash = classical_shares * price * 0.995
                classical_trades.append(('SELL', price))
                classical_shares = 0
    
    classical_final_value = classical_cash + classical_shares * final_price
    classical_return = (classical_final_value / initial_cash - 1) * 100
    
    # Performance metrics
    if len(portfolio_values) > 1:
        portfolio_series = pd.Series(portfolio_values)
        daily_returns = portfolio_series.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            max_drawdown = ((portfolio_series / portfolio_series.expanding().max()) - 1).min() * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    # Results summary
    print(f"\n7. 🏆 PERFORMANCE COMPARISON:")
    print(f"   {'Strategy':<25} | {'Return':>12} | {'Final Value':>12} | {'Trades':>8}")
    print(f"   {'-'*25} | {'-'*12} | {'-'*12} | {'-'*8}")
    print(f"   {'🤖 ML XGBoost':<25} | {ml_return:+11.2f}% | ${final_value:11,.0f} | {len(trades):7d}")
    print(f"   {'📊 Classical MACD':<25} | {classical_return:+11.2f}% | ${classical_final_value:11,.0f} | {len(classical_trades):7d}")
    print(f"   {'📈 Buy & Hold':<25} | {buy_hold_return:+11.2f}% | ${initial_cash * (1 + buy_hold_return/100):11,.0f} | {0:7d}")
    
    print(f"\n8. 📊 ML STRATEGY METRICS:")
    print(f"   Sharpe Ratio:     {sharpe_ratio:.3f}")
    print(f"   Max Drawdown:     {max_drawdown:+.2f}%")
    print(f"   Total Trades:     {len(trades)}")
    print(f"   Model Accuracy:   {accuracy:.1%}")
    
    # Determine winner
    print(f"\n9. 🎉 RESULTS:")
    if ml_return > max(classical_return, buy_hold_return):
        best_alternative = max(classical_return, buy_hold_return)
        improvement = ml_return - best_alternative
        print(f"   🥇 ML STRATEGY WINS!")
        print(f"   🚀 Outperformed best alternative by {improvement:+.2f} percentage points")
        print(f"   💡 This demonstrates ML's ability to find complex patterns")
        print(f"      that traditional technical analysis might miss!")
        
    elif classical_return > buy_hold_return:
        print(f"   🥈 Classical MACD strategy wins")
        print(f"   📈 ML shows promise but needs refinement")
        
    else:
        print(f"   🥉 Buy & Hold wins")
        print(f"   💭 Both active strategies need improvement")
    
    print(f"\n✅ GPU-Accelerated ML Trading Analysis Complete!")
    print(f"🔧 Features engineered: {len(feature_cols)}")
    print(f"🎯 Model accuracy: {accuracy:.1%}")
    print(f"⚡ GPU acceleration: {'Enabled' if 'gpu_hist' in str(model.get_params().get('tree_method', '')) else 'CPU Fallback'}")

if __name__ == '__main__':
    run_ml_trading_demo() 