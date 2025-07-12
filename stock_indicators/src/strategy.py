import pandas as pd
from src.indicators import calculate_rsi, calculate_macd

class Strategy:
    def __init__(self, data: pd.DataFrame, rsi_period: int = 14, macd_fast_period: int = 12, macd_slow_period: int = 26, macd_signal_period: int = 9):
        """
        Initializes the Strategy class using crossover logic.
        """
        self.data = data.copy()
        self.rsi_period = int(rsi_period)
        self.macd_fast_period = int(macd_fast_period)
        self.macd_slow_period = int(macd_slow_period)
        self.macd_signal_period = int(macd_signal_period)
        self.indicators = None

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates buy and sell signals based on MACD crossover confirmed by RSI.
        """
        # Calculate Indicators
        self.data['rsi'] = calculate_rsi(self.data['Close'], self.rsi_period)
        macd_line, signal_line, _ = calculate_macd(self.data['Close'], self.macd_fast_period, self.macd_slow_period, self.macd_signal_period)
        self.data['macd_line'] = macd_line
        self.data['macd_signal'] = signal_line

        self.data.dropna(inplace=True)

        # Store indicators for plotting
        self.indicators = self.data[['rsi', 'macd_line', 'macd_signal']].rename(columns={
            'rsi': 'RSI',
            'macd_line': 'MACD',
            'macd_signal': 'MACD_Signal'
        })

        # Generate Signals based on Crossover Logic
        # A buy signal is when MACD crosses above its signal line, and RSI is not overbought.
        buy_conditions = (
            (self.data['macd_line'] > self.data['macd_signal']) &
            (self.data['macd_line'].shift(1) <= self.data['macd_signal'].shift(1)) &
            (self.data['rsi'] < 70)
        )
        
        # A sell signal is when MACD crosses below its signal line, and RSI is not oversold.
        sell_conditions = (
            (self.data['macd_line'] < self.data['macd_signal']) &
            (self.data['macd_line'].shift(1) >= self.data['macd_signal'].shift(1)) &
            (self.data['rsi'] > 30)
        )
        
        # We will use a simple +1 for buy, -1 for sell signal representation
        self.data['signal'] = 0
        self.data.loc[buy_conditions, 'signal'] = 1
        self.data.loc[sell_conditions, 'signal'] = -1
        
        return self.data

if __name__ == '__main__':
    dummy_data = {'Close': [i + (i*0.1) * (-1)**i for i in range(100, 250)]}
    df = pd.DataFrame(dummy_data)
    
    strategy = Strategy(data=df)
    signal_df = strategy.generate_signals()
    
    print("Strategy Crossover Signal Results:")
    print(signal_df[['Close', 'macd_line', 'macd_signal', 'rsi', 'signal']].tail(15))
    print("\nTrades Generated:")
    print(signal_df[signal_df['signal'] != 0]) 