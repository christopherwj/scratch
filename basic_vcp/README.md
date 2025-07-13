# Volatility Contraction Pattern (VCP) Detection System

An automated system for identifying Volatility Contraction Patterns (VCP) in stocks, detecting swing points, and generating breakout signals for trading opportunities.

## Features

- **Real-time Stock Data**: Fetches live and historical data using Yahoo Finance API
- **VCP Pattern Detection**: Identifies volatility contraction patterns using advanced algorithms
- **Swing Point Analysis**: Detects key support and resistance levels
- **Breakout Signal Generation**: Identifies optimal entry and exit points
- **Backtesting Framework**: Tests strategy performance on historical data
- **Web Dashboard**: Interactive visualization of patterns and signals
- **Automated Scanning**: Continuous monitoring of multiple stocks

## System Components

1. **Data Layer** (`data/`)
   - Stock data fetching and preprocessing
   - Historical data management

2. **Analysis Engine** (`analysis/`)
   - VCP pattern detection algorithms
   - Swing point identification
   - Breakout level calculation

3. **Signal Generator** (`signals/`)
   - Entry/exit signal generation
   - Risk management rules

4. **Backtesting** (`backtesting/`)
   - Strategy performance testing
   - Performance metrics calculation

5. **Dashboard** (`dashboard/`)
   - Web-based visualization interface
   - Real-time monitoring

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Setup Configuration**:
   ```bash
   python setup.py
   ```

2. **Run VCP Scanner**:
   ```bash
   python main.py
   ```

3. **Access Dashboard**:
   ```bash
   python dashboard/app.py
   ```

## VCP Pattern Recognition

The system identifies VCP patterns by analyzing:
- Price consolidation over time
- Decreasing volatility (Bollinger Band contraction)
- Volume patterns during consolidation
- Relative strength vs market

## Trading Signals

- **Entry**: Breakout above resistance with volume confirmation
- **Exit**: Stop loss below support or profit target
- **Risk Management**: Position sizing based on volatility

## License

MIT License 