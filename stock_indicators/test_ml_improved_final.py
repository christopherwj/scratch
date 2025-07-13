import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
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

def engineer_effective_features(data):
    """Engineer effective features focused on predictive power."""
    df = data.copy()
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
    df['open_close_spread'] = (df['Close'] - df['Open']) / df['Open']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Multi-timeframe moving averages
    periods = [5, 10, 20, 50, 100, 200]
    for period in periods:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        df[f'price_sma_ratio_{period}'] = df['Close'] / df[f'sma_{period}']
        df[f'price_ema_ratio_{period}'] = df['Close'] / df[f'ema_{period}']
        df[f'sma_slope_{period}'] = df[f'sma_{period}'].pct_change(3)
        df[f'ema_slope_{period}'] = df[f'ema_{period}'].pct_change(3)
    
    # RSI with multiple periods
    for period in [14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi_momentum_{period}'] = df[f'rsi_{period}'].diff()
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_momentum'] = df['macd_histogram'].diff()
    
    # Bollinger Bands
    for period in [20, 50]:
        bb_middle = df['Close'].rolling(period).mean()
        bb_std = df['Close'].rolling(period).std()
        df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
        df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
        df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_middle
    
    # Volatility features
    for period in [10, 20, 50]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        df[f'volatility_rank_{period}'] = df[f'volatility_{period}'].rolling(100).rank(pct=True)
    
    # Momentum features
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'].pct_change(period)
        df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
    
    # Support/Resistance
    for period in [20, 50]:
        df[f'highest_{period}'] = df['High'].rolling(period).max()
        df[f'lowest_{period}'] = df['Low'].rolling(period).min()
        df[f'distance_to_high_{period}'] = (df[f'highest_{period}'] - df['Close']) / df['Close']
        df[f'distance_to_low_{period}'] = (df['Close'] - df[f'lowest_{period}']) / df['Close']
    
    # Volume features
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_sma_20'] + 1e-8)
    df['price_volume'] = df['Close'] * df['Volume']
    df['volume_momentum'] = df['Volume'].pct_change()
    
    # Market regime
    df['trend_strength'] = df['Close'].rolling(20).apply(lambda x: np.corrcoef(x, range(len(x)))[0,1] if len(x) > 1 else 0)
    df['mean_reversion'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    
    # Lagged features
    for lag in [1, 2, 3]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        df[f'macd_histogram_lag_{lag}'] = df['macd_histogram'].shift(lag)
        df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)
    
    # Time features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df.dropna()

def create_balanced_targets(data, lookahead_days=5):
    """Create balanced targets with reasonable thresholds."""
    df = data.copy()
    
    # Calculate future returns
    df['future_return'] = df['Close'].shift(-lookahead_days) / df['Close'] - 1
    
    # Use percentile-based thresholds for balanced signals
    returns = df['future_return'].dropna()
    
    # Use 30th and 70th percentiles for more balanced signals
    buy_threshold = returns.quantile(0.70)
    sell_threshold = returns.quantile(0.30)
    
    print(f"   Target thresholds: Sell < {sell_threshold:.3f}, Hold = middle 40%, Buy > {buy_threshold:.3f}")
    
    # Create balanced targets
    df['target'] = 0  # Hold
    df.loc[df['future_return'] > buy_threshold, 'target'] = 1  # Buy
    df.loc[df['future_return'] < sell_threshold, 'target'] = -1  # Sell
    
    return df.dropna()

def create_optimized_models(X_train, y_train, X_test, y_test):
    """Create optimized models with better hyperparameters."""
    print("   🎯 Training optimized models...")
    
    # Convert targets for XGBoost (0, 1, 2 instead of -1, 0, 1)
    y_train_xgb = y_train + 1
    y_test_xgb = y_test + 1
    
    # Calculate class weights for better balance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    models = {}
    predictions = {}
    
    # XGBoost with better hyperparameters
    try:
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 8,
            'learning_rate': 0.08,
            'n_estimators': 400,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_weight': 3,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        # Try GPU first
        try:
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = 0
            models['xgboost'] = xgb.XGBClassifier(**xgb_params)
        except:
            xgb_params['tree_method'] = 'hist'
            models['xgboost'] = xgb.XGBClassifier(**xgb_params)
        
        models['xgboost'].fit(X_train, y_train_xgb)
        predictions['xgboost'] = models['xgboost'].predict(X_test) - 1
        print("   ✅ XGBoost model trained")
        
    except Exception as e:
        print(f"   ❌ XGBoost failed: {e}")
    
    # Random Forest with better parameters
    try:
        rf_params = {
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        models['random_forest'] = RandomForestClassifier(**rf_params)
        models['random_forest'].fit(X_train, y_train)
        predictions['random_forest'] = models['random_forest'].predict(X_test)
        print("   ✅ Random Forest model trained")
        
    except Exception as e:
        print(f"   ❌ Random Forest failed: {e}")
    
    # Ensemble prediction (weighted average based on training accuracy)
    if len(predictions) > 1:
        # Calculate weights based on training accuracy
        weights = {}
        for name, model in models.items():
            if name == 'xgboost':
                train_pred = model.predict(X_train) - 1
            else:
                train_pred = model.predict(X_train)
            weights[name] = accuracy_score(y_train, train_pred)
        
        # Normalize weights
        total_weight = sum(weights.values())
        for name in weights:
            weights[name] /= total_weight
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X_test))
        for name, pred in predictions.items():
            ensemble_pred += pred * weights[name]
        
        ensemble_pred = np.round(ensemble_pred).astype(int)
        predictions['ensemble'] = ensemble_pred
        print("   ✅ Weighted ensemble created")
    
    return models, predictions

def evaluate_model_performance(y_true, y_pred, model_name):
    """Evaluate model performance with detailed metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Class-wise performance
    class_accuracies = {}
    for class_val in [-1, 0, 1]:
        class_mask = (y_true == class_val)
        if class_mask.sum() > 0:
            class_accuracies[class_val] = accuracy_score(y_true[class_mask], y_pred[class_mask])
        else:
            class_accuracies[class_val] = 0.0
    
    # Signal distribution
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    true_dist = pd.Series(y_true).value_counts().sort_index()
    
    return accuracy, class_accuracies, pred_dist, true_dist

def run_improved_ml_trading():
    """Run the improved ML trading strategy."""
    print("="*80)
    print("🚀 IMPROVED MACHINE LEARNING TRADING STRATEGY")
    print("🎯 Balanced Targets + Optimized Models + Better Features")
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
    print("\n2. 🔧 Engineering effective features...")
    features_df = engineer_effective_features(data)
    targets_df = create_balanced_targets(features_df)
    
    # Get feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                   and not col.startswith('future_')
                   and col != 'target']
    
    print(f"   Created {len(feature_cols)} effective features")
    
    # Split data chronologically
    split_index = int(len(targets_df) * 0.75)
    
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print(f"\n4. 🎯 Training optimized models...")
    models, predictions = create_optimized_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Evaluate all models
    print(f"\n5. 📊 Model Performance Evaluation:")
    print(f"   {'Model':<15} | {'Accuracy':>10} | {'Sell Acc':>10} | {'Hold Acc':>10} | {'Buy Acc':>10} | {'Signals':>15}")
    print(f"   {'-'*15} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*15}")
    
    best_model = None
    best_accuracy = 0
    best_predictions = None
    
    for model_name, y_pred in predictions.items():
        accuracy, class_acc, pred_dist, true_dist = evaluate_model_performance(y_test, y_pred, model_name)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
            best_predictions = y_pred
        
        signal_str = f"{pred_dist.get(-1,0)}-{pred_dist.get(0,0)}-{pred_dist.get(1,0)}"
        print(f"   {model_name:<15} | {accuracy:>9.1%} | {class_acc.get(-1, 0):>9.1%} | {class_acc.get(0, 0):>9.1%} | {class_acc.get(1, 0):>9.1%} | {signal_str:>15}")
    
    print(f"\n🏆 Best model: {best_model} with {best_accuracy:.1%} accuracy")
    
    # Feature importance
    if best_model in models:
        model = models[best_model]
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            
            print(f"\n6. 🔍 Top 15 most predictive features ({best_model}):")
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i:2d}. {feature:<30}: {importance:.4f}")
    
    # Backtest strategy
    print(f"\n7. 💹 Backtesting strategy with {best_model}...")
    
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    portfolio_values = []
    trades = []
    
    for i, (date, row) in enumerate(test_data.iterrows()):
        if best_predictions is not None and i < len(best_predictions):
            signal = best_predictions[i]
            price = row['Close']
            
            # Update portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
            
            # Execute trades
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
    
    # Classical MACD comparison
    classical_cash = initial_cash
    classical_shares = 0
    classical_trades = []
    
    for i, (date, row) in enumerate(test_data.iterrows()):
        if i > 0:
            macd_curr = row['macd']
            macd_signal_curr = row['macd_signal']
            macd_prev = test_data['macd'].iloc[i-1]
            macd_signal_prev = test_data['macd_signal'].iloc[i-1]
            rsi = row['rsi_14']
            price = row['Close']
            
            # MACD crossover with RSI filter
            if macd_curr > macd_signal_curr and macd_prev <= macd_signal_prev and rsi < 70 and classical_shares == 0:
                classical_shares = classical_cash / price * 0.995
                classical_cash = 0
                classical_trades.append(('BUY', date, price))
            elif macd_curr < macd_signal_curr and macd_prev >= macd_signal_prev and rsi > 30 and classical_shares > 0:
                classical_cash = classical_shares * price * 0.995
                classical_trades.append(('SELL', date, price))
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
            win_rate = (daily_returns > 0).mean() * 100
        else:
            sharpe_ratio = max_drawdown = win_rate = 0
    else:
        sharpe_ratio = max_drawdown = win_rate = 0
    
    # Results summary
    print(f"\n8. 🏆 PERFORMANCE COMPARISON:")
    print(f"   {'Strategy':<25} | {'Return':>12} | {'Final Value':>12} | {'Trades':>8} | {'Sharpe':>8}")
    print(f"   {'-'*25} | {'-'*12} | {'-'*12} | {'-'*8} | {'-'*8}")
    print(f"   {'🤖 Improved ML':<25} | {ml_return:+11.2f}% | ${final_value:11,.0f} | {len(trades):7d} | {sharpe_ratio:7.3f}")
    print(f"   {'📊 Classical MACD':<25} | {classical_return:+11.2f}% | ${classical_final_value:11,.0f} | {len(classical_trades):7d} | {0:7.3f}")
    print(f"   {'📈 Buy & Hold':<25} | {buy_hold_return:+11.2f}% | ${initial_cash * (1 + buy_hold_return/100):11,.0f} | {0:7d} | {0:7.3f}")
    
    print(f"\n9. 📊 STRATEGY METRICS:")
    print(f"   Model Accuracy:      {best_accuracy:.1%}")
    print(f"   Sharpe Ratio:        {sharpe_ratio:.3f}")
    print(f"   Max Drawdown:        {max_drawdown:+.2f}%")
    print(f"   Win Rate:            {win_rate:.1f}%")
    print(f"   Total Trades:        {len(trades)}")
    
    # Final assessment
    print(f"\n10. 🎉 FINAL ASSESSMENT:")
    if ml_return > max(classical_return, buy_hold_return):
        improvement = ml_return - max(classical_return, buy_hold_return)
        print(f"    🥇 IMPROVED ML STRATEGY WINS!")
        print(f"    🚀 Outperformed best alternative by {improvement:+.2f} percentage points")
        print(f"    📊 Model accuracy: {best_accuracy:.1%}")
        print(f"    💡 Key improvements: balanced targets, optimized models, better features")
        if sharpe_ratio > 0.8:
            print(f"    🎯 Excellent risk-adjusted returns!")
        elif sharpe_ratio > 0.5:
            print(f"    ✅ Good risk-adjusted returns")
        
    elif classical_return > buy_hold_return:
        print(f"    🥈 Classical MACD strategy wins")
        print(f"    📈 ML shows {best_accuracy:.1%} accuracy but needs strategy refinement")
        
    else:
        print(f"    🥉 Buy & Hold wins")
        print(f"    💭 Active strategies need further optimization")
        print(f"    📊 ML model accuracy: {best_accuracy:.1%}")
    
    print(f"\n✅ Improved ML Trading Analysis Complete!")
    print(f"🔧 Features: {len(feature_cols)} effective features")
    print(f"🎯 Best model: {best_model} ({best_accuracy:.1%} accuracy)")
    print(f"⚡ Models trained: {len(models)}")

if __name__ == '__main__':
    run_improved_ml_trading() 