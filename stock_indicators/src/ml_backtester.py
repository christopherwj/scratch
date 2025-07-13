import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.ml_features import FeatureEngineer
from src.ml_trainer import MLTrader
from src.backtester import Backtester
from src.strategy import Strategy
import warnings
warnings.filterwarnings('ignore')

class MLBacktester:
    def __init__(self, initial_cash=10000.0, transaction_cost=0.001):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.ml_trader = None
        self.feature_engineer = None
        self.portfolio_value = None
        self.trades = None
        
    def train_ml_model(self, training_data: pd.DataFrame, lookahead_days=5, threshold=0.02):
        """
        Trains the ML model on historical data.
        """
        print("="*60)
        print("TRAINING ML MODEL FOR TRADING")
        print("="*60)
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Create features and targets
        features_df = self.feature_engineer.create_features(training_data)
        targets_df = self.feature_engineer.create_targets(features_df, lookahead_days, threshold)
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_importance_names()
        
        print(f"Training data shape: {training_data.shape}")
        print(f"Features created: {len(feature_cols)}")
        print(f"Samples with targets: {len(targets_df.dropna(subset=['target_signal']))}")
        
        # Initialize ML trainer
        self.ml_trader = MLTrader(use_gpu=True)
        
        # Prepare data for training
        X_train, X_test, y_train, y_test = self.ml_trader.prepare_data(
            targets_df, feature_cols, 'target_signal', test_size=0.2
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train all models
        results = self.ml_trader.train_all_models(X_train, y_train, X_test, y_test)
        
        # Print model summary
        self.ml_trader.print_model_summary()
        
        return results
    
    def generate_ml_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals using the trained ML model.
        """
        if self.ml_trader is None or self.feature_engineer is None:
            raise ValueError("ML model must be trained first!")
        
        # Create features for the data
        features_df = self.feature_engineer.create_features(data)
        feature_cols = self.feature_engineer.get_feature_importance_names()
        
        # Get ML predictions
        ml_signals = self.ml_trader.predict(features_df[feature_cols])
        
        # Create signals dataframe
        signals_df = features_df.copy()
        signals_df['ml_signal'] = ml_signals
        signals_df['signal'] = ml_signals  # For compatibility with backtester
        
        return signals_df
    
    def backtest_ml_strategy(self, test_data: pd.DataFrame):
        """
        Backtests the ML strategy on test data.
        """
        print("\n" + "="*60)
        print("BACKTESTING ML STRATEGY")
        print("="*60)
        
        # Generate ML signals
        signals_df = self.generate_ml_signals(test_data)
        
        print(f"Test data period: {signals_df.index.min().date()} to {signals_df.index.max().date()}")
        print(f"Total trading days: {len(signals_df)}")
        
        # Count signals
        buy_signals = (signals_df['signal'] == 1).sum()
        sell_signals = (signals_df['signal'] == -1).sum()
        hold_signals = (signals_df['signal'] == 0).sum()
        
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        print(f"Hold signals: {hold_signals}")
        
        # Run backtest
        portfolio = pd.DataFrame(index=signals_df.index)
        portfolio['holdings'] = 0.0
        portfolio['cash'] = self.initial_cash
        portfolio['total'] = self.initial_cash
        
        trades_list = []
        position = 'OUT'  # Can be 'OUT' or 'IN'
        
        for i, (timestamp, row) in enumerate(signals_df.iterrows()):
            # Carry forward previous day's values
            if i > 0:
                portfolio.loc[timestamp, 'holdings'] = portfolio.iloc[i - 1]['holdings']
                portfolio.loc[timestamp, 'cash'] = portfolio.iloc[i - 1]['cash']
            
            # Update holding value if in position
            if position == 'IN' and i > 0:
                shares_held = portfolio.iloc[i - 1]['holdings'] / signals_df.iloc[i - 1]['Close']
                portfolio.loc[timestamp, 'holdings'] = shares_held * row['Close']
            
            # Check for buy signal
            if position == 'OUT' and row['signal'] == 1:
                shares_to_buy = portfolio.loc[timestamp, 'cash'] / row['Close']
                cost = shares_to_buy * row['Close'] * (1 + self.transaction_cost)
                portfolio.loc[timestamp, 'cash'] -= cost
                portfolio.loc[timestamp, 'holdings'] += shares_to_buy * row['Close']
                position = 'IN'
                trades_list.append({
                    'Date': timestamp, 
                    'Signal': 'BUY', 
                    'Price': row['Close'], 
                    'Shares': shares_to_buy
                })
            
            # Check for sell signal
            elif position == 'IN' and row['signal'] == -1:
                shares_to_sell = portfolio.loc[timestamp, 'holdings'] / signals_df.iloc[i - 1]['Close']
                revenue = shares_to_sell * row['Close'] * (1 - self.transaction_cost)
                portfolio.loc[timestamp, 'cash'] += revenue
                portfolio.loc[timestamp, 'holdings'] = 0.0
                position = 'OUT'
                trades_list.append({
                    'Date': timestamp, 
                    'Signal': 'SELL', 
                    'Price': row['Close'], 
                    'Shares': shares_to_sell
                })
            
            # Update total portfolio value
            portfolio.loc[timestamp, 'total'] = portfolio.loc[timestamp, 'cash'] + portfolio.loc[timestamp, 'holdings']
        
        self.portfolio_value = portfolio['total']
        self.trades = pd.DataFrame(trades_list).set_index('Date') if trades_list else pd.DataFrame()
        
        return self.portfolio_value, self.trades
    
    def calculate_performance_metrics(self, benchmark_data=None):
        """
        Calculates comprehensive performance metrics.
        """
        if self.portfolio_value is None:
            raise ValueError("Backtest must be run first!")
        
        # Basic metrics
        total_return = (self.portfolio_value.iloc[-1] / self.initial_cash) - 1
        daily_returns = self.portfolio_value.pct_change().dropna()
        
        # Risk metrics
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
        max_drawdown = ((self.portfolio_value / self.portfolio_value.expanding().max()) - 1).min()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Win rate
        if not self.trades.empty:
            # Calculate trade returns
            buy_trades = self.trades[self.trades['Signal'] == 'BUY']
            sell_trades = self.trades[self.trades['Signal'] == 'SELL']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                trade_returns = []
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_price = buy_trades.iloc[i]['Price']
                    sell_price = sell_trades.iloc[i]['Price']
                    trade_return = (sell_price - buy_price) / buy_price
                    trade_returns.append(trade_return)
                
                if trade_returns:
                    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
                    avg_trade_return = np.mean(trade_returns)
                else:
                    win_rate = 0
                    avg_trade_return = 0
            else:
                win_rate = 0
                avg_trade_return = 0
        else:
            win_rate = 0
            avg_trade_return = 0
        
        metrics = {
            'Total Return (%)': total_return * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Volatility (%)': volatility * 100,
            'Win Rate (%)': win_rate * 100,
            'Avg Trade Return (%)': avg_trade_return * 100,
            'Total Trades': len(self.trades),
            'Final Portfolio Value ($)': self.portfolio_value.iloc[-1]
        }
        
        return metrics
    
    def compare_with_classical_strategy(self, test_data: pd.DataFrame, classical_params=None):
        """
        Compares ML strategy with classical MACD/RSI strategy.
        """
        print("\n" + "="*60)
        print("COMPARING ML vs CLASSICAL STRATEGY")
        print("="*60)
        
        # Classical strategy parameters
        if classical_params is None:
            classical_params = {
                'rsi_period': 14,
                'macd_fast_period': 12,
                'macd_slow_period': 26,
                'macd_signal_period': 9
            }
        
        # Run classical strategy
        classical_strategy = Strategy(data=test_data, **classical_params)
        classical_backtester = Backtester(classical_strategy, initial_cash=self.initial_cash)
        classical_portfolio, classical_trades = classical_backtester.run()
        
        # Calculate metrics for both strategies
        ml_metrics = self.calculate_performance_metrics()
        
        # Classical metrics
        classical_return = (classical_portfolio.iloc[-1] / self.initial_cash) - 1
        classical_daily_returns = classical_portfolio.pct_change().dropna()
        classical_sharpe = (classical_daily_returns.mean() / classical_daily_returns.std()) * np.sqrt(252) if classical_daily_returns.std() != 0 else 0
        classical_max_dd = ((classical_portfolio / classical_portfolio.expanding().max()) - 1).min()
        
        classical_metrics = {
            'Total Return (%)': classical_return * 100,
            'Sharpe Ratio': classical_sharpe,
            'Max Drawdown (%)': classical_max_dd * 100,
            'Total Trades': len(classical_trades),
            'Final Portfolio Value ($)': classical_portfolio.iloc[-1]
        }
        
        # Print comparison
        print(f"{'Metric':<25} | {'ML Strategy':>15} | {'Classical':>15} | {'Improvement':>15}")
        print("-" * 75)
        
        for metric in ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Total Trades']:
            ml_val = ml_metrics.get(metric, 0)
            classical_val = classical_metrics.get(metric, 0)
            
            if classical_val != 0:
                if metric == 'Max Drawdown (%)':
                    improvement = (classical_val - ml_val) / abs(classical_val) * 100  # Lower is better
                else:
                    improvement = (ml_val - classical_val) / abs(classical_val) * 100
            else:
                improvement = 0
            
            print(f"{metric:<25} | {ml_val:15.2f} | {classical_val:15.2f} | {improvement:+14.1f}%")
        
        return ml_metrics, classical_metrics
    
    def plot_results(self, test_data: pd.DataFrame, save_path='ml_trading_results.png'):
        """
        Plots the ML trading results.
        """
        if self.portfolio_value is None:
            raise ValueError("Backtest must be run first!")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Portfolio value vs Buy & Hold
        ax1.plot(self.portfolio_value.index, self.portfolio_value.values, 
                label='ML Strategy', color='blue', linewidth=2)
        
        # Buy and hold comparison
        buy_hold = (test_data['Close'] / test_data['Close'].iloc[0]) * self.initial_cash
        ax1.plot(buy_hold.index, buy_hold.values, 
                label='Buy & Hold', color='orange', linestyle='--', linewidth=2)
        
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('ML Trading Strategy Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Price with buy/sell signals
        ax2.plot(test_data.index, test_data['Close'], 
                label='Price', color='black', alpha=0.7)
        
        if not self.trades.empty:
            buy_trades = self.trades[self.trades['Signal'] == 'BUY']
            sell_trades = self.trades[self.trades['Signal'] == 'SELL']
            
            if not buy_trades.empty:
                ax2.scatter(buy_trades.index, buy_trades['Price'], 
                          color='green', marker='^', s=100, label='Buy Signal', zorder=5)
            
            if not sell_trades.empty:
                ax2.scatter(sell_trades.index, sell_trades['Price'], 
                          color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax2.set_ylabel('Price ($)')
        ax2.set_xlabel('Date')
        ax2.set_title('Trading Signals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results plot saved to {save_path}")

if __name__ == '__main__':
    # Test the ML backtester
    from src.data_loader import fetch_data, load_aapl_split_adjusted
    
    print("Testing ML Backtester...")
    
    # Load data
    data = load_aapl_split_adjusted()
    if data is not None:
        # Split data for training and testing
        split_date = '2022-01-01'
        train_data = data.loc[:split_date]
        test_data = data.loc[split_date:]
        
        print(f"Training period: {train_data.index.min().date()} to {train_data.index.max().date()}")
        print(f"Testing period: {test_data.index.min().date()} to {test_data.index.max().date()}")
        
        # Initialize backtester
        ml_backtester = MLBacktester(initial_cash=10000)
        
        # Train ML model
        ml_backtester.train_ml_model(train_data)
        
        # Backtest on test data
        portfolio_value, trades = ml_backtester.backtest_ml_strategy(test_data)
        
        # Compare with classical strategy
        ml_metrics, classical_metrics = ml_backtester.compare_with_classical_strategy(test_data)
        
        # Plot results
        ml_backtester.plot_results(test_data) 