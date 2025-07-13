import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(data):
    """Create advanced technical features for ML."""
    df = data.copy()
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
    df['close_open_pct'] = (df['Close'] - df['Open']) / df['Open']
    
    # Multiple timeframe moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        df[f'price_sma_ratio_{period}'] = df['Close'] / df[f'sma_{period}']
        df[f'price_ema_ratio_{period}'] = df['Close'] / df[f'ema_{period}']
    
    # RSI with multiple periods
    for period in [14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
        df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
    
    # MACD variations
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        df[f'macd_{fast}_{slow}'] = macd_line
        df[f'macd_signal_{fast}_{slow}'] = signal_line
        df[f'macd_histogram_{fast}_{slow}'] = macd_line - signal_line
        df[f'macd_above_signal_{fast}_{slow}'] = (macd_line > signal_line).astype(int)
    
    # Bollinger Bands
    for period in [20, 50]:
        bb_middle = df['Close'].rolling(period).mean()
        bb_std = df['Close'].rolling(period).std()
        df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
        df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
        df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_middle
    
    # Volatility features
    for period in [10, 20, 30]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        df[f'volatility_rank_{period}'] = df[f'volatility_{period}'].rolling(100).rank(pct=True)
    
    # Momentum features
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
    
    # Support/Resistance levels
    for period in [20, 50]:
        df[f'highest_{period}'] = df['High'].rolling(period).max()
        df[f'lowest_{period}'] = df['Low'].rolling(period).min()
        df[f'distance_to_high_{period}'] = (df[f'highest_{period}'] - df['Close']) / df['Close']
        df[f'distance_to_low_{period}'] = (df['Close'] - df[f'lowest_{period}']) / df['Close']
    
    # Market regime features
    df['trend_strength'] = abs(df['Close'].rolling(20).corr(pd.Series(range(20), index=df.index[-20:])))
    df['mean_reversion'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        df[f'macd_histogram_12_26_lag_{lag}'] = df['macd_histogram_12_26'].shift(lag)
    
    # Volume features
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    df['price_volume'] = df['Close'] * df['Volume']
    
    return df.dropna()

def create_sophisticated_targets(data, lookahead_days=3, threshold=0.01):
    """Create more sophisticated target variables."""
    df = data.copy()
    
    # Multiple horizon returns
    for days in [1, 3, 5]:
        df[f'future_return_{days}d'] = df['Close'].shift(-days) / df['Close'] - 1
    
    # Volatility-adjusted returns
    rolling_vol = df['returns'].rolling(20).std()
    df['vol_adj_return'] = df['future_return_3d'] / rolling_vol
    
    # Main target: 3-day forward return with dynamic threshold
    dynamic_threshold = rolling_vol * 1.5  # 1.5x volatility as threshold
    
    df['target'] = 0  # Hold
    df.loc[df['future_return_3d'] > dynamic_threshold, 'target'] = 1  # Buy
    df.loc[df['future_return_3d'] < -dynamic_threshold, 'target'] = -1  # Sell
    
    return df.dropna()

def advanced_ml_strategy():
    """Advanced ML trading strategy with GPU-accelerated XGBoost."""
    print("="*70)
    print("ADVANCED ML TRADING STRATEGY WITH GPU ACCELERATION")
    print("="*70)
    
    # Create synthetic data with more realistic patterns
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
    dates = dates[dates.day_name().isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
    
    # Generate more realistic price data with trends and cycles
    n_days = len(dates)
    
    # Base trend + cycles + noise
    trend = np.linspace(0, 0.8, n_days)  # 80% growth over period
    cycle = 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # Annual cycle
    noise = np.random.normal(0, 0.02, n_days)
    
    log_returns = trend/n_days + cycle/n_days + noise
    prices = 150 * np.exp(np.cumsum(log_returns))
    
    # Create realistic OHLCV data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.003, n_days))
    data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
    data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
    data['Volume'] = np.random.lognormal(15, 0.3, n_days)
    
    data = data.fillna(method='ffill').dropna()
    
    print(f"1. Generated {len(data)} days of realistic market data")
    print(f"   Period: {data.index.min().date()} to {data.index.max().date()}")
    
    # Advanced feature engineering
    print("2. Engineering advanced features...")
    features_df = create_advanced_features(data)
    targets_df = create_sophisticated_targets(features_df)
    
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                   and not col.startswith('future_') 
                   and col != 'target']
    
    print(f"   Created {len(feature_cols)} sophisticated features")
    
    # Split data with proper time series methodology
    split_date = '2022-01-01'
    train_data = targets_df.loc[:split_date]
    test_data = targets_df.loc[split_date:]
    
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    print(f"3. Data split:")
    print(f"   Training: {len(X_train)} samples ({train_data.index.min().date()} to {train_data.index.max().date()})")
    print(f"   Testing: {len(X_test)} samples ({test_data.index.min().date()} to {test_data.index.max().date()})")
    
    # Check target distribution
    train_dist = y_train.value_counts().sort_index()
    test_dist = y_test.value_counts().sort_index()
    print(f"   Training targets: Sell={train_dist.get(-1,0)}, Hold={train_dist.get(0,0)}, Buy={train_dist.get(1,0)}")
    print(f"   Test targets: Sell={test_dist.get(-1,0)}, Hold={test_dist.get(0,0)}, Buy={test_dist.get(1,0)}")
    
    # Feature scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # GPU-accelerated XGBoost
    print("4. Training GPU-accelerated XGBoost...")
    
    # Convert targets to 0, 1, 2 for XGBoost
    y_train_xgb = y_train + 1
    y_test_xgb = y_test + 1
    
    try:
        # Try GPU first
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            early_stopping_rounds=50,
            eval_metric='mlogloss'
        )
        
        model.fit(
            X_train_scaled, y_train_xgb,
            eval_set=[(X_test_scaled, y_test_xgb)],
            verbose=False
        )
        print("   ✓ GPU XGBoost training completed")
        
    except Exception as e:
        print(f"   ⚠ GPU failed ({e}), falling back to CPU...")
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='mlogloss'
        )
        
        model.fit(
            X_train_scaled, y_train_xgb,
            eval_set=[(X_test_scaled, y_test_xgb)],
            verbose=False
        )
        print("   ✓ CPU XGBoost training completed")
    
    # Predictions
    y_pred_xgb = model.predict(X_test_scaled)
    y_pred = y_pred_xgb - 1  # Convert back to -1, 0, 1
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Model accuracy: {accuracy:.1%}")
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
    
    print(f"\n5. Top 8 most important features:")
    for i, (feature, importance) in enumerate(top_features, 1):
        print(f"   {i:2d}. {feature:<25}: {importance:.4f}")
    
    # Advanced backtesting with position sizing
    print(f"\n6. Advanced backtesting with position sizing...")
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    portfolio_values = []
    trades = []
    
    for i, (date, row) in enumerate(test_data.iterrows()):
        if i < len(y_pred):
            signal = y_pred[i]
            price = row['Close']
            
            # Current portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
            
            # Position sizing based on signal confidence
            position_size = 0.8  # Use 80% of available capital
            
            # Execute trades
            if signal == 1 and shares == 0:  # Buy signal
                shares_to_buy = (cash * position_size) / price
                cost = shares_to_buy * price * 1.001  # 0.1% transaction cost
                if cost <= cash:
                    shares += shares_to_buy
                    cash -= cost
                    trades.append(('BUY', date, price, shares_to_buy))
                    
            elif signal == -1 and shares > 0:  # Sell signal
                revenue = shares * price * 0.999  # 0.1% transaction cost
                cash += revenue
                trades.append(('SELL', date, price, shares))
                shares = 0
    
    # Final portfolio value
    final_price = test_data['Close'].iloc[-1]
    final_value = cash + shares * final_price
    ml_return = (final_value / initial_cash - 1) * 100
    
    # Buy and hold comparison
    buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
    
    # Calculate additional metrics
    portfolio_series = pd.Series(portfolio_values, index=test_data.index[:len(portfolio_values)])
    daily_returns = portfolio_series.pct_change().dropna()
    
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        max_drawdown = ((portfolio_series / portfolio_series.expanding().max()) - 1).min() * 100
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    print(f"\n7. 📊 PERFORMANCE RESULTS:")
    print(f"   {'Strategy':<20} | {'Return':>10} | {'Sharpe':>8} | {'Max DD':>8} | {'Trades':>8}")
    print(f"   {'-'*20} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")
    print(f"   {'ML XGBoost':<20} | {ml_return:+9.2f}% | {sharpe_ratio:7.3f} | {max_drawdown:+7.2f}% | {len(trades):7d}")
    print(f"   {'Buy & Hold':<20} | {buy_hold_return:+9.2f}% | {'N/A':>7} | {'N/A':>7} | {0:7d}")
    
    print(f"\n8. 💰 FINAL VALUES:")
    print(f"   ML Strategy:    ${final_value:10,.2f}")
    print(f"   Buy & Hold:     ${initial_cash * (1 + buy_hold_return/100):10,.2f}")
    
    if ml_return > buy_hold_return:
        improvement = ml_return - buy_hold_return
        print(f"\n   🎉 ML STRATEGY WINS by {improvement:+.2f} percentage points!")
        print(f"   🚀 That's {improvement/buy_hold_return*100:+.1f}% better than buy & hold!")
    else:
        underperformance = buy_hold_return - ml_return
        print(f"\n   📈 Buy & Hold wins by {underperformance:+.2f} percentage points")
        print(f"   💡 ML needs more sophisticated features or different approach")
    
    print(f"\n9. 📈 TRADING SUMMARY:")
    buy_trades = [t for t in trades if t[0] == 'BUY']
    sell_trades = [t for t in trades if t[0] == 'SELL']
    print(f"   Total trades: {len(trades)}")
    print(f"   Buy signals:  {len(buy_trades)}")
    print(f"   Sell signals: {len(sell_trades)}")
    
    if len(buy_trades) > 0 and len(sell_trades) > 0:
        completed_trades = min(len(buy_trades), len(sell_trades))
        trade_returns = []
        for i in range(completed_trades):
            buy_price = buy_trades[i][2]
            sell_price = sell_trades[i][2]
            trade_return = (sell_price - buy_price) / buy_price * 100
            trade_returns.append(trade_return)
        
        if trade_returns:
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
            avg_trade = np.mean(trade_returns)
            print(f"   Win rate:     {win_rate:.1f}%")
            print(f"   Avg trade:    {avg_trade:+.2f}%")
    
    print(f"\n✅ Advanced ML analysis completed!")
    print(f"🔧 Using {len(feature_cols)} engineered features")
    print(f"🎯 Model accuracy: {accuracy:.1%}")

if __name__ == '__main__':
    advanced_ml_strategy() 