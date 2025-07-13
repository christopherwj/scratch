import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
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

def engineer_optimized_features(data):
    """Engineer optimized features with focus on signal quality."""
    df = data.copy()
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    df['open_close_range'] = (df['Close'] - df['Open']) / df['Open']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['true_range'] = np.maximum(df['High'] - df['Low'], 
                                  np.maximum(np.abs(df['High'] - df['Close'].shift(1)),
                                            np.abs(df['Low'] - df['Close'].shift(1))))
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Moving averages with golden ratio periods
    periods = [8, 13, 21, 34, 55, 89, 144]
    for period in periods:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        df[f'price_sma_ratio_{period}'] = df['Close'] / df[f'sma_{period}']
        df[f'price_ema_ratio_{period}'] = df['Close'] / df[f'ema_{period}']
        df[f'sma_slope_{period}'] = df[f'sma_{period}'].pct_change(3)
        df[f'ema_slope_{period}'] = df[f'ema_{period}'].pct_change(3)
    
    # RSI variations
    for period in [14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi_ma_{period}'] = df[f'rsi_{period}'].rolling(5).mean()
        df[f'rsi_momentum_{period}'] = df[f'rsi_{period}'].diff()
        df[f'rsi_divergence_{period}'] = df[f'rsi_{period}'] - df[f'rsi_{period}'].rolling(10).mean()
    
    # Enhanced MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_momentum'] = df['macd_histogram'].diff()
    df['macd_normalized'] = df['macd'] / df['Close']
    
    # Bollinger Bands with multiple periods
    for period in [20, 50]:
        bb_middle = df['Close'].rolling(period).mean()
        bb_std = df['Close'].rolling(period).std()
        df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
        df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
        df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_middle
        df[f'bb_squeeze_{period}'] = df[f'bb_width_{period}'].rolling(20).rank(pct=True)
    
    # Stochastic Oscillator
    for period in [14, 21]:
        lowest_low = df['Low'].rolling(period).min()
        highest_high = df['High'].rolling(period).max()
        df[f'stoch_k_{period}'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
        df[f'stoch_momentum_{period}'] = df[f'stoch_k_{period}'].diff()
    
    # Williams %R
    for period in [14, 21]:
        highest_high = df['High'].rolling(period).max()
        lowest_low = df['Low'].rolling(period).min()
        df[f'williams_r_{period}'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    
    # Volatility features
    for period in [10, 20, 50]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        df[f'volatility_rank_{period}'] = df[f'volatility_{period}'].rolling(100).rank(pct=True)
        df[f'volatility_ma_{period}'] = df[f'volatility_{period}'].rolling(10).mean()
        df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_ma_{period}']
    
    # Average True Range
    df['atr_14'] = df['true_range'].rolling(14).mean()
    df['atr_ratio'] = df['true_range'] / df['atr_14']
    
    # Momentum features
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'].pct_change(period)
        df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
        df[f'momentum_ma_{period}'] = df[f'momentum_{period}'].rolling(5).mean()
    
    # Support/Resistance levels
    for period in [20, 50]:
        df[f'highest_{period}'] = df['High'].rolling(period).max()
        df[f'lowest_{period}'] = df['Low'].rolling(period).min()
        df[f'distance_to_high_{period}'] = (df[f'highest_{period}'] - df['Close']) / df['Close']
        df[f'distance_to_low_{period}'] = (df['Close'] - df[f'lowest_{period}']) / df['Close']
        df[f'price_channel_{period}'] = (df['Close'] - df[f'lowest_{period}']) / (df[f'highest_{period}'] - df[f'lowest_{period}'])
    
    # Volume features
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma_20'] + 1e-8)
    df['volume_momentum'] = df['Volume'].pct_change()
    df['volume_price_trend'] = df['returns'] * df['volume_ratio']
    df['volume_volatility'] = df['Volume'].rolling(20).std() / df['volume_ma_20']
    
    # On-Balance Volume
    df['obv'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
    df['obv_ma'] = df['obv'].rolling(20).mean()
    df['obv_divergence'] = df['obv'] - df['obv_ma']
    
    # Market structure
    df['trend_strength'] = df['Close'].rolling(20).apply(lambda x: np.corrcoef(x, range(len(x)))[0,1] if len(x) > 1 else 0)
    df['mean_reversion'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['price_efficiency'] = df['Close'].pct_change() / df['high_low_range']
    
    # Cycle indicators
    df['cycle_dominant'] = df['Close'].rolling(20).apply(lambda x: np.fft.fft(x)[1].real if len(x) == 20 else 0)
    
    # Lagged features (most important)
    important_features = ['returns', 'rsi_14', 'macd_histogram', 'volume_ratio', 'bb_position_20', 'stoch_k_14']
    for feature in important_features:
        if feature in df.columns:
            for lag in [1, 2, 3, 5]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    # Cross-feature interactions
    df['rsi_macd_interaction'] = df['rsi_14'] * df['macd_histogram']
    df['volume_momentum_interaction'] = df['volume_ratio'] * df['momentum_5']
    df['volatility_trend_interaction'] = df['volatility_20'] * df['trend_strength']
    
    # Time features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_month'] = df.index.day
    df['is_month_end'] = (df.index.day > 25).astype(int)
    
    # Clean up infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df.dropna()

def create_smart_targets(data, lookahead_days=3):
    """Create smart targets using multiple time horizons and confidence scoring."""
    df = data.copy()
    
    # Calculate multiple horizon returns
    horizons = [1, 3, 5, 10]
    for horizon in horizons:
        df[f'future_return_{horizon}'] = df['Close'].shift(-horizon) / df['Close'] - 1
    
    # Dynamic thresholds based on volatility regime
    volatility_regime = df['volatility_rank_20'].rolling(20).mean()
    
    # Use adaptive percentiles
    returns_3d = df['future_return_3'].dropna()
    
    # Base thresholds
    base_buy_threshold = returns_3d.quantile(0.75)
    base_sell_threshold = returns_3d.quantile(0.25)
    
    # Adjust thresholds based on volatility regime
    vol_adjustment = np.where(volatility_regime > 0.7, 1.3, 
                             np.where(volatility_regime < 0.3, 0.8, 1.0))
    
    buy_threshold = base_buy_threshold * vol_adjustment
    sell_threshold = base_sell_threshold * vol_adjustment
    
    # Create targets with confidence scoring
    df['target'] = 0  # Default: Hold
    
    # Multi-horizon confirmation
    buy_conditions = (
        (df['future_return_3'] > buy_threshold) &
        (df['future_return_5'] > buy_threshold * 0.6) &
        (df['trend_strength'] > -0.2)  # Not in strong downtrend
    )
    
    sell_conditions = (
        (df['future_return_3'] < sell_threshold) &
        (df['future_return_5'] < sell_threshold * 0.6) &
        (df['trend_strength'] < 0.2)  # Not in strong uptrend
    )
    
    df.loc[buy_conditions, 'target'] = 1
    df.loc[sell_conditions, 'target'] = -1
    
    # Calculate signal strength
    df['signal_strength'] = np.where(
        df['target'] == 1, 
        df['future_return_3'] / buy_threshold,
        np.where(
            df['target'] == -1,
            abs(df['future_return_3'] / sell_threshold),
            0
        )
    )
    
    print(f"   Target distribution: Buy={sum(df['target'] == 1)}, Hold={sum(df['target'] == 0)}, Sell={sum(df['target'] == -1)}")
    print(f"   Signal balance: {(sum(df['target'] != 0) / len(df)):.1%} signals, {(sum(df['target'] == 0) / len(df)):.1%} holds")
    
    return df.dropna()

def select_best_features(X, y, n_features=50):
    """Select the best features using mutual information."""
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    print(f"   Selected {len(selected_features)} best features from {len(X.columns)} total features")
    
    return X_selected, selected_features

def create_advanced_ensemble(X_train, y_train, X_test, y_test, feature_names):
    """Create advanced ensemble with multiple models and sophisticated voting."""
    print("   🎯 Training advanced ensemble...")
    
    # Convert targets for XGBoost
    y_train_xgb = y_train + 1
    y_test_xgb = y_test + 1
    
    models = {}
    predictions = {}
    prediction_probabilities = {}
    
    # XGBoost with optimized parameters
    try:
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 7,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_weight': 3,
            'gamma': 0.1,
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
        prediction_probabilities['xgboost'] = models['xgboost'].predict_proba(X_test)
        print("   ✅ XGBoost model trained")
        
    except Exception as e:
        print(f"   ❌ XGBoost failed: {e}")
    
    # Random Forest with optimized parameters
    try:
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
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
    
    # Sophisticated ensemble using probability averaging
    if len(prediction_probabilities) > 1:
        # Calculate model weights based on cross-validation scores
        model_weights = {}
        for name, model in models.items():
            if name == 'xgboost':
                cv_scores = cross_val_score(model, X_train, y_train_xgb, cv=3, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            model_weights[name] = cv_scores.mean()
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        for name in model_weights:
            model_weights[name] /= total_weight
        
        # Weighted probability averaging
        ensemble_proba = np.zeros_like(list(prediction_probabilities.values())[0])
        for name, proba in prediction_probabilities.items():
            ensemble_proba += proba * model_weights[name]
        
        # Convert probabilities to predictions
        ensemble_pred = np.argmax(ensemble_proba, axis=1) - 1
        predictions['ensemble'] = ensemble_pred
        prediction_probabilities['ensemble'] = ensemble_proba
        
        print(f"   ✅ Weighted ensemble created (weights: {model_weights})")
    
    return models, predictions, prediction_probabilities

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
    
    # Calculate precision for buy/sell signals
    buy_precision = 0
    sell_precision = 0
    
    if pred_dist.get(1, 0) > 0:
        buy_precision = sum((y_true == 1) & (y_pred == 1)) / pred_dist.get(1, 1)
    if pred_dist.get(-1, 0) > 0:
        sell_precision = sum((y_true == -1) & (y_pred == -1)) / pred_dist.get(-1, 1)
    
    return accuracy, class_accuracies, pred_dist, true_dist, buy_precision, sell_precision

def backtest_with_risk_management(test_data, predictions, probabilities=None):
    """Backtest with sophisticated risk management."""
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    portfolio_values = []
    trades = []
    
    # Risk management parameters
    max_position_size = 0.95
    min_confidence = 0.4
    stop_loss_pct = 0.05
    take_profit_pct = 0.10
    
    entry_price = 0
    
    for i, (date, row) in enumerate(test_data.iterrows()):
        if i < len(predictions):
            signal = predictions[i]
            price = row['Close']
            
            # Get confidence if available
            confidence = 1.0
            if probabilities is not None:
                confidence = max(probabilities[i])
            
            # Update portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
            
            # Risk management - stop loss and take profit
            if shares > 0 and entry_price > 0:
                # Check stop loss
                if price <= entry_price * (1 - stop_loss_pct):
                    cash = shares * price * 0.995
                    trades.append(('STOP_LOSS', date, price))
                    shares = 0
                    entry_price = 0
                    continue
                
                # Check take profit
                if price >= entry_price * (1 + take_profit_pct):
                    cash = shares * price * 0.995
                    trades.append(('TAKE_PROFIT', date, price))
                    shares = 0
                    entry_price = 0
                    continue
            
            # Position sizing based on confidence
            position_size = max_position_size * confidence
            
            # Execute trades
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

def run_optimized_ml_trading():
    """Run the most optimized ML trading strategy."""
    print("="*80)
    print("🚀 OPTIMIZED MACHINE LEARNING TRADING STRATEGY")
    print("🎯 Advanced Features + Smart Targets + Risk Management + Ensemble")
    print("="*80)
    
    # Load data
    print("\n1. 📊 Loading real AAPL data...")
    data = load_real_aapl_data()
    if data is None:
        print("❌ Failed to load AAPL data. Exiting.")
        return
    print(f"   Loaded {len(data):,} trading days")
    print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Feature engineering
    print("\n2. 🔧 Engineering optimized features...")
    features_df = engineer_optimized_features(data)
    targets_df = create_smart_targets(features_df)
    
    # Get feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                   and not col.startswith('future_')
                   and col not in ['target', 'signal_strength']]
    
    print(f"   Created {len(feature_cols)} optimized features")
    
    # Split data
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
    
    # Feature selection
    print(f"\n4. 🎯 Selecting best features...")
    X_train_selected, selected_features = select_best_features(X_train, y_train, n_features=60)
    X_test_selected = X_test[selected_features]
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train models
    print(f"\n5. 🎯 Training advanced ensemble...")
    models, predictions, probabilities = create_advanced_ensemble(
        X_train_scaled, y_train, X_test_scaled, y_test, selected_features
    )
    
    # Evaluate models
    print(f"\n6. 📊 Model Performance Evaluation:")
    print(f"   {'Model':<15} | {'Accuracy':>10} | {'Buy Prec':>10} | {'Sell Prec':>10} | {'Signals':>15}")
    print(f"   {'-'*15} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*15}")
    
    best_model = None
    best_accuracy = 0
    best_predictions = None
    best_probabilities = None
    
    for model_name, y_pred in predictions.items():
        accuracy, class_acc, pred_dist, true_dist, buy_prec, sell_prec = evaluate_model_performance(y_test, y_pred, model_name)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
            best_predictions = y_pred
            if model_name in probabilities:
                best_probabilities = probabilities[model_name]
        
        signal_str = f"{pred_dist.get(-1,0)}-{pred_dist.get(0,0)}-{pred_dist.get(1,0)}"
        print(f"   {model_name:<15} | {accuracy:>9.1%} | {buy_prec:>9.1%} | {sell_prec:>9.1%} | {signal_str:>15}")
    
    print(f"\n🏆 Best model: {best_model} with {best_accuracy:.1%} accuracy")
    
    # Feature importance
    if best_model in models:
        model = models[best_model]
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(selected_features, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            print(f"\n7. 🔍 Top 10 most predictive features ({best_model}):")
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i:2d}. {feature:<35}: {importance:.4f}")
    
    # Advanced backtesting
    print(f"\n8. 💹 Advanced backtesting with risk management...")
    final_value, portfolio_values, trades = backtest_with_risk_management(
        test_data, best_predictions, best_probabilities
    )
    
    ml_return = (final_value / 10000 - 1) * 100
    
    # Classical comparisons
    buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
    
    # Simple MACD comparison
    classical_cash = 10000
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
    print(f"\n9. 🏆 FINAL PERFORMANCE COMPARISON:")
    print(f"   {'Strategy':<25} | {'Return':>12} | {'Final Value':>12} | {'Trades':>8} | {'Sharpe':>8}")
    print(f"   {'-'*25} | {'-'*12} | {'-'*12} | {'-'*8} | {'-'*8}")
    print(f"   {'🤖 Optimized ML':<25} | {ml_return:+11.2f}% | ${final_value:11,.0f} | {len(trades):7d} | {sharpe_ratio:7.3f}")
    print(f"   {'📊 Classical MACD':<25} | {classical_return:+11.2f}% | ${classical_final_value:11,.0f} | {len(classical_trades):7d} | {0:7.3f}")
    print(f"   {'📈 Buy & Hold':<25} | {buy_hold_return:+11.2f}% | ${10000 * (1 + buy_hold_return/100):11,.0f} | {0:7d} | {0:7.3f}")
    
    print(f"\n10. 📊 STRATEGY METRICS:")
    print(f"    Model Accuracy:       {best_accuracy:.1%}")
    print(f"    Sharpe Ratio:         {sharpe_ratio:.3f}")
    print(f"    Max Drawdown:         {max_drawdown:+.2f}%")
    print(f"    Win Rate:             {win_rate:.1f}%")
    print(f"    Total Trades:         {len(trades)}")
    
    # Trade breakdown
    trade_types = {}
    for trade in trades:
        trade_type = trade[0]
        trade_types[trade_type] = trade_types.get(trade_type, 0) + 1
    
    print(f"    Trade breakdown:      {dict(trade_types)}")
    
    # Final assessment
    print(f"\n11. 🎉 FINAL ASSESSMENT:")
    best_benchmark = max(classical_return, buy_hold_return)
    
    if ml_return > best_benchmark:
        improvement = ml_return - best_benchmark
        print(f"    🥇 OPTIMIZED ML STRATEGY WINS!")
        print(f"    🚀 Outperformed best benchmark by {improvement:+.2f} percentage points")
        print(f"    📊 Model accuracy: {best_accuracy:.1%}")
        print(f"    💡 Key optimizations: feature selection, ensemble methods, risk management")
        
        if sharpe_ratio > 1.0:
            print(f"    🎯 Excellent risk-adjusted returns!")
        elif sharpe_ratio > 0.5:
            print(f"    ✅ Good risk-adjusted returns")
        
    elif classical_return > buy_hold_return:
        print(f"    🥈 Classical MACD strategy wins")
        print(f"    📈 ML shows {best_accuracy:.1%} accuracy - consider different approach")
        
    else:
        print(f"    🥉 Buy & Hold wins")
        print(f"    💭 Market efficiency suggests passive approach")
        print(f"    📊 ML model accuracy: {best_accuracy:.1%}")
    
    print(f"\n✅ Optimized ML Trading Analysis Complete!")
    print(f"🔧 Selected features: {len(selected_features)} from {len(feature_cols)} total")
    print(f"🎯 Best model: {best_model} ({best_accuracy:.1%} accuracy)")
    print(f"⚡ Advanced ensemble: {len(models)} models with risk management")

if __name__ == '__main__':
    run_optimized_ml_trading() 