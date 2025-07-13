import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

def load_real_aapl_data():
    """Load real AAPL data from the data directory."""
    from pathlib import Path
    
    data_file = Path('data/aapl_split_adjusted.csv')
    
    if not data_file.exists():
        print(f"❌ AAPL data file not found at {data_file}")
        return None
    
    try:
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"✅ Loaded real AAPL data from {data_file}")
        print(f"   Data shape: {data.shape}")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        return data
        
    except Exception as e:
        print(f"❌ Error loading AAPL data: {e}")
        return None

def engineer_aggressive_features(data):
    """Engineer features optimized for generating trading signals."""
    df = data.copy()
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    df['open_close_range'] = (df['Close'] - df['Open']) / df['Open']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Key moving averages
    periods = [5, 10, 20, 50]
    for period in periods:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        df[f'price_sma_ratio_{period}'] = df['Close'] / df[f'sma_{period}']
        df[f'price_ema_ratio_{period}'] = df['Close'] / df[f'ema_{period}']
        df[f'sma_slope_{period}'] = df[f'sma_{period}'].pct_change(3)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_momentum'] = df['rsi'].diff()
    df['rsi_oversold'] = (df['rsi'] < 35).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 65).astype(int)
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
    df['macd_momentum'] = df['macd_histogram'].diff()
    
    # Bollinger Bands
    bb_period = 20
    bb_middle = df['Close'].rolling(bb_period).mean()
    bb_std = df['Close'].rolling(bb_period).std()
    df['bb_upper'] = bb_middle + (bb_std * 2)
    df['bb_lower'] = bb_middle - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / bb_middle
    
    # Stochastic
    lowest_low = df['Low'].rolling(14).min()
    highest_high = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    df['volatility_rank'] = df['volatility'].rolling(100).rank(pct=True)
    
    # Momentum
    for period in [3, 5, 10]:
        df[f'momentum_{period}'] = df['Close'].pct_change(period)
        df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
    
    # Volume
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma'] + 1e-8)
    df['volume_momentum'] = df['Volume'].pct_change()
    
    # Support/Resistance
    df['highest_20'] = df['High'].rolling(20).max()
    df['lowest_20'] = df['Low'].rolling(20).min()
    df['distance_to_high'] = (df['highest_20'] - df['Close']) / df['Close']
    df['distance_to_low'] = (df['Close'] - df['lowest_20']) / df['Close']
    
    # Trend
    df['trend_strength'] = df['Close'].rolling(20).apply(lambda x: np.corrcoef(x, range(len(x)))[0,1] if len(x) > 1 else 0)
    df['mean_reversion'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    
    # Lagged features
    for lag in [1, 2, 3]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        df[f'macd_histogram_lag_{lag}'] = df['macd_histogram'].shift(lag)
    
    # Time features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()

def create_aggressive_targets(data, lookahead_days=3):
    """Create more aggressive targets that generate more trading signals."""
    df = data.copy()
    
    # Calculate future returns
    df['future_return'] = df['Close'].shift(-lookahead_days) / df['Close'] - 1
    
    # Use more aggressive percentiles for more signals
    returns = df['future_return'].dropna()
    
    # Use 40th and 60th percentiles instead of 25th and 75th
    buy_threshold = returns.quantile(0.60)   # Top 40%
    sell_threshold = returns.quantile(0.40)  # Bottom 40%
    
    print(f"   Aggressive thresholds: Sell < {sell_threshold:.3f}, Hold = middle 20%, Buy > {buy_threshold:.3f}")
    
    # Create targets
    df['target'] = 0  # Hold
    df.loc[df['future_return'] > buy_threshold, 'target'] = 1  # Buy
    df.loc[df['future_return'] < sell_threshold, 'target'] = -1  # Sell
    
    return df.dropna()

def create_aggressive_models(X_train, y_train, X_test, y_test):
    """Create models optimized for generating signals."""
    print("   🎯 Training aggressive models...")
    
    y_train_xgb = y_train + 1
    y_test_xgb = y_test + 1
    
    models = {}
    predictions = {}
    prediction_probabilities = {}
    
    # XGBoost optimized for signals
    try:
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        try:
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = 0
            models['xgboost'] = xgb.XGBClassifier(**xgb_params)
        except:
            xgb_params['tree_method'] = 'hist'
            models['xgboost'] = xgb.XGBClassifier(**xgb_params)
        
        models['xgboost'].fit(X_train, y_train_xgb)
        predictions['xgboost'] = models['xgboost'].predict(X_test) - 1
        prediction_probabilities['xgboost'] = models['xgboost'].predict_proba(X_test)
        print("   ✅ XGBoost model trained")
        
    except Exception as e:
        print(f"   ❌ XGBoost failed: {e}")
    
    # Random Forest optimized for signals
    try:
        rf_params = {
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        models['random_forest'] = RandomForestClassifier(**rf_params)
        models['random_forest'].fit(X_train, y_train)
        predictions['random_forest'] = models['random_forest'].predict(X_test)
        prediction_probabilities['random_forest'] = models['random_forest'].predict_proba(X_test)
        print("   ✅ Random Forest model trained")
        
    except Exception as e:
        print(f"   ❌ Random Forest failed: {e}")
    
    # Simple ensemble
    if len(predictions) > 1:
        ensemble_pred = np.zeros(len(X_test))
        for pred in predictions.values():
            ensemble_pred += pred
        ensemble_pred = np.round(ensemble_pred / len(predictions)).astype(int)
        predictions['ensemble'] = ensemble_pred
        print("   ✅ Ensemble created")
    
    return models, predictions, prediction_probabilities

def backtest_aggressive_strategy(test_data, predictions, probabilities=None):
    """Backtest with aggressive trading parameters."""
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    portfolio_values = []
    trades = []
    
    # AGGRESSIVE PARAMETERS - Much more active trading
    max_position_size = 0.95
    min_confidence = 0.35      # Reduced from 0.4 to 0.35
    stop_loss_pct = 0.08       # Increased from 0.05 to 0.08 (wider stops)
    take_profit_pct = 0.12     # Increased from 0.10 to 0.12 (wider targets)
    
    entry_price = 0
    
    for i, (date, row) in enumerate(test_data.iterrows()):
        if i < len(predictions):
            signal = predictions[i]
            price = row['Close']
            
            # Get confidence
            confidence = 1.0
            if probabilities is not None:
                confidence = max(probabilities[i])
            
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
            
            # Less restrictive risk management
            if shares > 0 and entry_price > 0:
                # Stop loss
                if price <= entry_price * (1 - stop_loss_pct):
                    cash = shares * price * 0.995
                    trades.append(('STOP_LOSS', date, price))
                    shares = 0
                    entry_price = 0
                    continue
                
                # Take profit
                if price >= entry_price * (1 + take_profit_pct):
                    cash = shares * price * 0.995
                    trades.append(('TAKE_PROFIT', date, price))
                    shares = 0
                    entry_price = 0
                    continue
            
            # More aggressive position sizing
            position_size = min(max_position_size, max(0.5, confidence * 1.2))
            
            # AGGRESSIVE TRADING - Lower confidence thresholds
            if signal == 1 and shares == 0 and confidence > min_confidence:  # Buy
                shares_to_buy = (cash * position_size) / price * 0.995
                shares += shares_to_buy
                cash -= shares_to_buy * price * 1.005
                trades.append(('BUY', date, price))
                entry_price = price
                
            elif signal == -1 and shares > 0 and confidence > min_confidence:  # Sell
                cash += shares * price * 0.995
                trades.append(('SELL', date, price))
                shares = 0
                entry_price = 0
    
    # Final value
    final_price = test_data['Close'].iloc[-1]
    final_value = cash + shares * final_price
    
    return final_value, portfolio_values, trades

def run_aggressive_ml_trading():
    """Run aggressive ML trading strategy that generates many signals."""
    print("="*80)
    print("🚀 AGGRESSIVE MACHINE LEARNING TRADING STRATEGY")
    print("🎯 More Signals + Active Trading + Reasonable Risk Management")
    print("="*80)
    
    # Load data
    print("\n1. 📊 Loading real AAPL data...")
    data = load_real_aapl_data()
    if data is None:
        return
    print(f"   Loaded {len(data):,} trading days")
    
    # Feature engineering
    print("\n2. 🔧 Engineering aggressive features...")
    features_df = engineer_aggressive_features(data)
    targets_df = create_aggressive_targets(features_df)
    
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                   and not col.startswith('future_')
                   and col != 'target']
    
    print(f"   Created {len(feature_cols)} features")
    
    # Split data
    split_index = int(len(targets_df) * 0.75)
    train_data = targets_df.iloc[:split_index]
    test_data = targets_df.iloc[split_index:]
    
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    print(f"\n3. 📈 Data split:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing:  {len(X_test):,} samples")
    
    # Check target distribution
    train_dist = y_train.value_counts().sort_index()
    test_dist = y_test.value_counts().sort_index()
    print(f"   Training signals: Sell={train_dist.get(-1,0)}, Hold={train_dist.get(0,0)}, Buy={train_dist.get(1,0)}")
    print(f"   Test signals:     Sell={test_dist.get(-1,0)}, Hold={test_dist.get(0,0)}, Buy={test_dist.get(1,0)}")
    
    # Feature selection - keep more features for aggressive trading
    print(f"\n4. 🎯 Selecting features...")
    selector = SelectKBest(score_func=mutual_info_classif, k=min(40, len(feature_cols)))
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    X_test_selected = X_test[selected_features]
    
    print(f"   Selected {len(selected_features)} features")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train models
    print(f"\n5. 🎯 Training aggressive models...")
    models, predictions, probabilities = create_aggressive_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Evaluate models
    print(f"\n6. 📊 Model Performance:")
    print(f"   {'Model':<15} | {'Accuracy':>10} | {'Signals':>15}")
    print(f"   {'-'*15} | {'-'*10} | {'-'*15}")
    
    best_model = None
    best_accuracy = 0
    best_predictions = None
    best_probabilities = None
    
    for model_name, y_pred in predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
            best_predictions = y_pred
            if model_name in probabilities:
                best_probabilities = probabilities[model_name]
        
        signal_str = f"{pred_dist.get(-1,0)}-{pred_dist.get(0,0)}-{pred_dist.get(1,0)}"
        print(f"   {model_name:<15} | {accuracy:>9.1%} | {signal_str:>15}")
    
    print(f"\n🏆 Best model: {best_model} with {best_accuracy:.1%} accuracy")
    
    # Feature importance
    if best_model in models:
        model = models[best_model]
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(selected_features, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
            
            print(f"\n7. 🔍 Top 8 features ({best_model}):")
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i}. {feature:<25}: {importance:.4f}")
    
    # Aggressive backtesting
    print(f"\n8. 💹 Aggressive backtesting...")
    final_value, portfolio_values, trades = backtest_aggressive_strategy(
        test_data, best_predictions, best_probabilities
    )
    
    ml_return = (final_value / 10000 - 1) * 100
    
    # Comparisons
    buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
    
    # MACD comparison
    classical_cash = 10000
    classical_shares = 0
    classical_trades = []
    
    for i, (date, row) in enumerate(test_data.iterrows()):
        if i > 0:
            macd_curr = row['macd']
            macd_signal_curr = row['macd_signal']
            macd_prev = test_data['macd'].iloc[i-1]
            macd_signal_prev = test_data['macd_signal'].iloc[i-1]
            rsi = row['rsi']
            price = row['Close']
            
            if macd_curr > macd_signal_curr and macd_prev <= macd_signal_prev and rsi < 70 and classical_shares == 0:
                classical_shares = classical_cash / price * 0.995
                classical_cash = 0
                classical_trades.append(('BUY', date, price))
            elif macd_curr < macd_signal_curr and macd_prev >= macd_signal_prev and rsi > 30 and classical_shares > 0:
                classical_cash = classical_shares * price * 0.995
                classical_trades.append(('SELL', date, price))
                classical_shares = 0
    
    classical_final_value = classical_cash + classical_shares * test_data['Close'].iloc[-1]
    classical_return = (classical_final_value / 10000 - 1) * 100
    
    # Performance metrics
    if len(portfolio_values) > 1:
        portfolio_series = pd.Series(portfolio_values)
        daily_returns = portfolio_series.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            max_drawdown = ((portfolio_series / portfolio_series.expanding().max()) - 1).min() * 100
            win_rate = (daily_returns > 0).mean() * 100
        else:
            sharpe_ratio = max_drawdown = win_rate = 0
    else:
        sharpe_ratio = max_drawdown = win_rate = 0
    
    # Results
    print(f"\n9. 🏆 AGGRESSIVE PERFORMANCE COMPARISON:")
    print(f"   {'Strategy':<25} | {'Return':>12} | {'Final Value':>12} | {'Trades':>8} | {'Sharpe':>8}")
    print(f"   {'-'*25} | {'-'*12} | {'-'*12} | {'-'*8} | {'-'*8}")
    print(f"   {'🤖 Aggressive ML':<25} | {ml_return:+11.2f}% | ${final_value:11,.0f} | {len(trades):7d} | {sharpe_ratio:7.3f}")
    print(f"   {'📊 Classical MACD':<25} | {classical_return:+11.2f}% | ${classical_final_value:11,.0f} | {len(classical_trades):7d} | {0:7.3f}")
    print(f"   {'📈 Buy & Hold':<25} | {buy_hold_return:+11.2f}% | ${10000 * (1 + buy_hold_return/100):11,.0f} | {0:7d} | {0:7.3f}")
    
    print(f"\n10. 📊 AGGRESSIVE STRATEGY METRICS:")
    print(f"    Model Accuracy:       {best_accuracy:.1%}")
    print(f"    Total Trades:         {len(trades)}")
    print(f"    Sharpe Ratio:         {sharpe_ratio:.3f}")
    print(f"    Max Drawdown:         {max_drawdown:+.2f}%")
    print(f"    Win Rate:             {win_rate:.1f}%")
    
    # Trade breakdown
    trade_types = {}
    for trade in trades:
        trade_type = trade[0]
        trade_types[trade_type] = trade_types.get(trade_type, 0) + 1
    
    print(f"    Trade breakdown:      {dict(trade_types)}")
    
    # Final assessment
    print(f"\n11. 🎉 AGGRESSIVE STRATEGY ASSESSMENT:")
    
    print(f"    🔥 TRADING ACTIVITY: {len(trades)} trades vs 2 in conservative version")
    print(f"    📊 Model accuracy: {best_accuracy:.1%}")
    print(f"    💰 ML Return: {ml_return:+.2f}% vs Buy&Hold: {buy_hold_return:+.2f}%")
    
    if len(trades) > 50:
        print(f"    ✅ SUCCESS: Generated {len(trades)} trades - much more active!")
        
        if ml_return > 0:
            print(f"    💸 Positive returns with active trading")
        
        if ml_return > classical_return:
            print(f"    🥇 Outperformed classical MACD strategy!")
        elif ml_return > buy_hold_return * 0.5:
            print(f"    🥈 Decent performance with much more trading activity")
        else:
            print(f"    📈 More active but needs strategy refinement")
    else:
        print(f"    ⚠️  Still not enough trades - need even more aggressive parameters")
    
    print(f"\n✅ Aggressive ML Trading Analysis Complete!")
    print(f"🎯 Model: {best_model} ({best_accuracy:.1%} accuracy)")
    print(f"📈 Trades: {len(trades)} (vs 2 in conservative version)")
    print(f"💰 Return: {ml_return:+.2f}%")

if __name__ == '__main__':
    run_aggressive_ml_trading() 