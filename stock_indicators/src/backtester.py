import pandas as pd
import numpy as np

from src.strategy import Strategy

class Backtester:
    def __init__(self, strategy: Strategy, initial_cash: float = 10000.0, transaction_cost_pct: float = 0.001):
        """
        Initializes the Backtester.
        Args:
            strategy (Strategy): The strategy object containing the data and signal logic.
            initial_cash (float): The starting cash for the portfolio.
            transaction_cost_pct (float): The percentage cost per transaction.
        """
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost_pct
        self.trades = None
        self.portfolio_value = None

    def run(self):
        """
        Runs the backtest simulation.
        Returns:
            (pd.Series, pd.DataFrame): A tuple of portfolio value series and trades dataframe.
        """
        signals_df = self.strategy.generate_signals()
        
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
                trades_list.append({'Date': timestamp, 'Signal': 'BUY', 'Price': row['Close'], 'Shares': shares_to_buy})

            # Check for sell signal
            elif position == 'IN' and row['signal'] == -1:
                shares_to_sell = portfolio.loc[timestamp, 'holdings'] / signals_df.iloc[i - 1]['Close']
                revenue = shares_to_sell * row['Close'] * (1 - self.transaction_cost)
                portfolio.loc[timestamp, 'cash'] += revenue
                portfolio.loc[timestamp, 'holdings'] = 0.0
                position = 'OUT'
                trades_list.append({'Date': timestamp, 'Signal': 'SELL', 'Price': row['Close'], 'Shares': shares_to_sell})

            # Update total portfolio value
            portfolio.loc[timestamp, 'total'] = portfolio.loc[timestamp, 'cash'] + portfolio.loc[timestamp, 'holdings']

        self.portfolio_value = portfolio['total']
        self.trades = pd.DataFrame(trades_list).set_index('Date') if trades_list else pd.DataFrame()
        
        return self.portfolio_value, self.trades

    def get_performance(self) -> dict:
        """
        Calculates and returns performance metrics of the backtest.
        """
        if self.portfolio_value is None:
            raise ValueError("Backtest has not been run yet. Please call .run() first.")

        if self.portfolio_value.empty or self.portfolio_value.iloc[-1] == self.initial_cash:
            return {'total_return_pct': 0, 'sharpe_ratio': 0}

        total_return = (self.portfolio_value.iloc[-1] / self.initial_cash) - 1
        daily_return = self.portfolio_value.pct_change()
        sharpe_ratio = (daily_return.mean() / daily_return.std()) * np.sqrt(252) if daily_return.std() != 0 else 0

        return {'total_return_pct': total_return * 100, 'sharpe_ratio': sharpe_ratio}

if __name__ == '__main__':
    # Create a dummy dataframe for testing
    dummy_index = pd.to_datetime(pd.date_range(start='2022-01-01', periods=252))
    dummy_data = {'Close': [i + (i*0.1) * (-1)**i for i in range(100, 352)]}
    df = pd.DataFrame(dummy_data, index=dummy_index)
    
    # Generate signals using the Strategy class
    strategy = Strategy(data=df)
    
    # Run the backtester
    backtester = Backtester(strategy=strategy, initial_cash=10000)
    portfolio_value, trades = backtester.run()
    
    # Get performance
    performance = backtester.get_performance()
    
    print("Backtest Performance:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.2f}")

    print("\nFinal Portfolio Value:")
    print(f"${portfolio_value.iloc[-1]:.2f}")
    
    print("\nTrades:")
    print(trades) 