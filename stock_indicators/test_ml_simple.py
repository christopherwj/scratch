import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Create synthetic AAPL data for testing ML system
def create_test_data():
    """Creates synthetic AAPL-like data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    # Generate realistic price data
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    
    # Start price
    price = 150.0
    prices = []
    
    for ret in returns:
        price = price * (1 + ret)
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, n_days))
    data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    data['Volume'] = np.random.lognormal(15, 0.5, n_days)
    
    # Forward fill NaN values
    data = data.fillna(method='ffill').dropna()
    
    return data

def simple_feature_engineering(data):
    """Simple feature engineering without external dependencies."""
    df = data.copy()
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['price_change'] = df['Close'] - df['Close'].shift(1)
    df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
    
    # Moving averages
    for period in [5, 10, 20]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
    
    # Simple RSI approximation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Simple MACD approximation
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Momentum
    for period in [5, 10]:
        df['momentum_{}'.format(period)] = df['Close'] / df['Close'].shift(period) - 1
    
    # Volume features
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    
    # Remove NaN values
    df = df.dropna()
    
    return df

def create_targets(data, lookahead_days=3, threshold=0.015):
    """Create target variables for ML."""
    df = data.copy()
    
    # Future returns
    df['future_return'] = df['Close'].shift(-lookahead_days) / df['Close'] - 1
    
    # Classification targets (-1: sell, 0: hold, 1: buy)
    df['target'] = 0
    df.loc[df['future_return'] > threshold, 'target'] = 1
    df.loc[df['future_return'] < -threshold, 'target'] = -1
    
    return df.dropna()

def simple_ml_strategy():
    """Test ML strategy with simple features."""
    print("="*60)
    print("SIMPLE ML TRADING STRATEGY TEST")
    print("="*60)
    
    # Create test data
    print("1. Creating synthetic AAPL data...")
    data = create_test_data()
    print(f"   Generated {len(data)} days of data")
    
    # Engineer features
    print("2. Engineering features...")
    features_df = simple_feature_engineering(data)
    targets_df = create_targets(features_df)
    print(f"   Created {features_df.shape[1]} features")
    
    # Prepare data for ML
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Split data
    split_idx = int(len(targets_df) * 0.7)
    train_data = targets_df.iloc[:split_idx]
    test_data = targets_df.iloc[split_idx:]
    
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    print(f"3. Training data: {len(X_train)} samples")
    print(f"   Test data: {len(X_test)} samples")
    print(f"   Features: {len(feature_cols)}")
    
    # Simple ML model (Random Forest)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("4. Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   Model accuracy: {accuracy:.1%}")
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\n5. Top 5 features:")
    for i, (feature, imp) in enumerate(top_features, 1):
        print(f"   {i}. {feature}: {imp:.4f}")
    
    # Simple backtest
    print("\n6. Backtesting...")
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    portfolio_values = []
    
    for i, (date, row) in enumerate(test_data.iterrows()):
        if i < len(y_pred):
            signal = y_pred[i]
            price = row['Close']
            
            # Current portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
            
            # Execute trades
            if signal == 1 and shares == 0:  # Buy
                shares = cash / price * 0.99  # 1% transaction cost
                cash = 0
            elif signal == -1 and shares > 0:  # Sell
                cash = shares * price * 0.99  # 1% transaction cost
                shares = 0
    
    # Final portfolio value
    final_value = cash + shares * test_data['Close'].iloc[-1]
    total_return = (final_value / initial_cash - 1) * 100
    
    # Buy and hold return
    buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
    
    print(f"\n7. Results:")
    print(f"   ML Strategy return: {total_return:+.2f}%")
    print(f"   Buy & Hold return:  {buy_hold_return:+.2f}%")
    print(f"   Final portfolio:    ${final_value:,.2f}")
    
    if total_return > buy_hold_return:
        improvement = total_return - buy_hold_return
        print(f"   🎉 ML beats Buy & Hold by {improvement:+.2f}%!")
    else:
        print(f"   📈 Buy & Hold wins this time")
    
    print(f"\n✅ Simple ML test completed!")

if __name__ == '__main__':
    simple_ml_strategy() 