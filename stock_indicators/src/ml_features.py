import pandas as pd
import numpy as np
from src.indicators import calculate_rsi, calculate_macd
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates comprehensive technical and market features for ML training.
        """
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_change'] = df['Close'] - df['Close'].shift(1)
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['close_open_pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
            df[f'price_ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}']
        
        # RSI with multiple periods
        for period in [14, 21, 28]:
            df[f'rsi_{period}'] = calculate_rsi(df['Close'], period)
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
        
        # MACD variations
        macd_configs = [(12, 26, 9), (5, 35, 5), (19, 39, 9)]
        for fast, slow, signal in macd_configs:
            macd_line, signal_line, histogram = calculate_macd(df['Close'], fast, slow, signal)
            df[f'macd_{fast}_{slow}'] = macd_line
            df[f'macd_signal_{fast}_{slow}'] = signal_line
            df[f'macd_histogram_{fast}_{slow}'] = histogram
            df[f'macd_above_signal_{fast}_{slow}'] = (macd_line > signal_line).astype(int)
            df[f'macd_cross_above_{fast}_{slow}'] = ((macd_line > signal_line) & 
                                                   (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
            df[f'macd_cross_below_{fast}_{slow}'] = ((macd_line < signal_line) & 
                                                   (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_middle = df['Close'].rolling(period).mean()
            bb_std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
            df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_squeeze_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_middle
        
        # Volatility features
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_rank_{period}'] = df[f'volatility_{period}'].rolling(252).rank(pct=True)
        
        # Volume features (if available)
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            df['price_volume'] = df['Close'] * df['Volume']
        
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
        df['trend_strength'] = abs(df['Close'].rolling(20).corr(pd.Series(range(20))))
        df['mean_reversion'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
            df[f'macd_histogram_12_26_lag_{lag}'] = df['macd_histogram_12_26'].shift(lag)
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Store feature names (excluding target and basic OHLCV)
        self.feature_names = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        return df
    
    def create_targets(self, data: pd.DataFrame, lookahead_days: int = 5, threshold: float = 0.02) -> pd.DataFrame:
        """
        Creates target variables for supervised learning.
        """
        df = data.copy()
        
        # Future returns
        df['future_return'] = df['Close'].shift(-lookahead_days) / df['Close'] - 1
        
        # Classification targets
        df['target_buy'] = (df['future_return'] > threshold).astype(int)
        df['target_sell'] = (df['future_return'] < -threshold).astype(int)
        df['target_hold'] = ((df['future_return'] >= -threshold) & (df['future_return'] <= threshold)).astype(int)
        
        # Multi-class target
        df['target_signal'] = 0  # Hold
        df.loc[df['future_return'] > threshold, 'target_signal'] = 1  # Buy
        df.loc[df['future_return'] < -threshold, 'target_signal'] = -1  # Sell
        
        # Regression target (future return)
        df['target_return'] = df['future_return']
        
        return df
    
    def get_feature_importance_names(self):
        """Returns the names of engineered features for importance analysis."""
        return self.feature_names

if __name__ == '__main__':
    # Test feature engineering
    from src.data_loader import fetch_data, load_aapl_split_adjusted
    
    data = load_aapl_split_adjusted()
    if data is not None:
        engineer = FeatureEngineer()
        features_df = engineer.create_features(data)
        targets_df = engineer.create_targets(features_df)
        
        print(f"Original data shape: {data.shape}")
        print(f"Features data shape: {features_df.shape}")
        print(f"With targets shape: {targets_df.shape}")
        print(f"Number of features: {len(engineer.feature_names)}")
        print(f"Feature names: {engineer.feature_names[:10]}...")  # Show first 10
        
        # Check target distribution
        print(f"\nTarget distribution:")
        print(targets_df['target_signal'].value_counts()) 