import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

from src.data_loader import fetch_data
from src.strategy import Strategy
from src.backtester import Backtester
from src.rolling_optimizer import RollingOptimizer

# Global ticker for visualization labels
TICKER = ''

def visualize_performance_and_indicators(portfolio_value, price_data, trades, indicators, benchmark_data, title):
    """
    Visualizes the backtesting results, including portfolio performance, buy/sell signals, and indicators.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20), sharex=True, gridspec_kw={'height_ratios': [3, 3, 2, 2]})
    fig.suptitle(title, fontsize=16)

    # --- Plot 1: Portfolio Value vs. Buy & Hold vs. Benchmark ---
    ax1.plot(portfolio_value, label='Strategy Portfolio Value', color='blue')
    buy_and_hold_value = (price_data / price_data.iloc[0]) * portfolio_value.iloc[0]
    ax1.plot(buy_and_hold_value, label=f'Buy and Hold {TICKER}', color='orange', linestyle='--')
    if benchmark_data is not None and not benchmark_data.empty:
        benchmark_normalized = (benchmark_data / benchmark_data.iloc[0]) * portfolio_value.iloc[0]
        ax1.plot(benchmark_normalized, label=f'Buy and Hold S&P 500 (SPY)', color='green', linestyle='--')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # --- Plot 2: Price with Buy/Sell Signals ---
    ax2.plot(price_data, label='Close Price', color='black', alpha=0.8)
    
    buy_signals = trades[trades['Signal'] == 'BUY']
    sell_signals = trades[trades['Signal'] == 'SELL']

    if not buy_signals.empty:
        ax2.plot(buy_signals.index, price_data.loc[buy_signals.index], '^', markersize=10, color='green', label='Buy Signal', linestyle='None')
    if not sell_signals.empty:
        ax2.plot(sell_signals.index, price_data.loc[sell_signals.index], 'v', markersize=10, color='red', label='Sell Signal', linestyle='None')
    
    ax2.set_ylabel('Price ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # --- Plot 3: MACD ---
    ax3.plot(indicators['MACD'], label='MACD', color='blue')
    ax3.plot(indicators['MACD_Signal'], label='Signal Line', color='red')
    ax3.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax3.set_ylabel('MACD')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    # --- Plot 4: RSI ---
    ax4.plot(indicators['RSI'], label='RSI', color='purple')
    ax4.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax4.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax4.set_ylabel('RSI')
    ax4.set_xlabel('Date')
    ax4.legend(loc='upper left')
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the figure
    plt.savefig('trading_analysis_results.png', dpi=300)
    print("\nChart saved to trading_analysis_results.png")
    plt.show()


def save_results_summary(portfolio_value, trades, params_history, stock_data, benchmark_data, initial_cash, ticker, benchmark_ticker):
    """
    Saves a human-readable summary of the analysis results and a detailed trade log.
    """
    # 1. Save trades log to CSV
    trades_path = 'trades_log.csv'
    trades.to_csv(trades_path)
    print(f"\nDetailed trade log saved to {trades_path}")

    # 2. Calculate performance metrics
    # Strategy
    strategy_return = (portfolio_value.iloc[-1] / initial_cash) - 1
    strategy_daily_returns = portfolio_value.pct_change()
    strategy_sharpe = (strategy_daily_returns.mean() / strategy_daily_returns.std()) * np.sqrt(252) if strategy_daily_returns.std() != 0 else 0

    # Buy and Hold Ticker
    stock_buy_hold = (stock_data / stock_data.iloc[0]) * initial_cash
    stock_return = (stock_buy_hold.iloc[-1] / initial_cash) - 1
    stock_daily_returns = stock_buy_hold.pct_change()
    stock_sharpe = (stock_daily_returns.mean() / stock_daily_returns.std()) * np.sqrt(252) if stock_daily_returns.std() != 0 else 0

    # Buy and Hold Benchmark
    if benchmark_data is not None and not benchmark_data.empty:
        benchmark_buy_hold = (benchmark_data / benchmark_data.iloc[0]) * initial_cash
        benchmark_return = (benchmark_buy_hold.iloc[-1] / initial_cash) - 1
        benchmark_daily_returns = benchmark_buy_hold.pct_change()
        benchmark_sharpe = (benchmark_daily_returns.mean() / benchmark_daily_returns.std()) * np.sqrt(252) if benchmark_daily_returns.std() != 0 else 0
    else:
        benchmark_return = 0.0
        benchmark_sharpe = 0.0
        benchmark_buy_hold = pd.Series([initial_cash], index=[stock_data.index[0]])


    # 3. Create summary string
    summary = []
    summary.append("="*80)
    summary.append("ADAPTIVE STRATEGY PERFORMANCE ANALYSIS")
    summary.append("="*80)
    summary.append(f"\n--- Final Performance Metrics ({portfolio_value.index.min().date()} to {portfolio_value.index.max().date()}) ---")
    summary.append(f"\n{'Metric':<25} | {'Strategy':>15} | {'Buy & Hold ' + ticker:>15} | {'Buy & Hold ' + benchmark_ticker:>20}")
    summary.append("-"*80)
    summary.append(f"{'Total Return (%)':<25} | {strategy_return*100:15.2f} | {stock_return*100:15.2f} | {benchmark_return*100:20.2f}")
    summary.append(f"{'Sharpe Ratio':<25} | {strategy_sharpe:15.2f} | {stock_sharpe:15.2f} | {benchmark_sharpe:20.2f}")
    summary.append(f"{'Final Portfolio Value ($)':<25} | {portfolio_value.iloc[-1]:15.2f} | {stock_buy_hold.iloc[-1]:15.2f} | {benchmark_buy_hold.iloc[-1]:20.2f}")

    summary.append("\n\n--- Best Parameters Per Optimization Window ---")
    summary.append(f"\n{'Window Start Date':<20} | {'RSI Period':>12} | {'MACD Fast':>12} | {'MACD Slow':>12} | {'MACD Signal':>12}")
    summary.append("-"*80)
    for item in params_history:
        params = item['params']
        summary.append(
            f"{str(item['start'].date()):<20} | {params['rsi_period']:12.0f} | {params['macd_fast_period']:12.0f} | {params['macd_slow_period']:12.0f} | {params['macd_signal_period']:12.0f}"
        )

    # 4. Write to file
    summary_path = 'results_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary))
    print(f"Human-readable results summary saved to {summary_path}")


def run_rolling_analysis(ticker, spy_ticker, start_date, end_date, train_window, test_window, num_cores):
    """
    Main function to run the rolling optimization analysis and visualize the results.
    """
    global TICKER
    TICKER = ticker
    
    # --- 1. Fetch Data ---
    print(f"Fetching full data for {ticker}...")
    full_data = fetch_data(ticker, start_date, end_date)
    if full_data is None:
        print("Could not fetch main ticker data. Exiting.")
        return
        
    # --- 2. Define Parameter Grid ---
    # Generate large, unique integer values for the grid
    rsi_p = np.unique(np.linspace(5, 25, 10, dtype=int))
    macd_f = np.unique(np.linspace(5, 40, 12, dtype=int))
    macd_s = np.unique(np.linspace(20, 70, 12, dtype=int))
    macd_sig = np.unique(np.linspace(5, 25, 12, dtype=int))
    
    param_grid = {
        'rsi_period': rsi_p,
        'macd_fast_period': macd_f,
        'macd_slow_period': macd_s,
        'macd_signal_period': macd_sig
    }
    
    # Calculate the number of valid combinations where slow > fast
    valid_combos = sum(1 for f in macd_f for s in macd_s if s > f) * len(rsi_p) * len(macd_sig)
    print(f"\nStarting optimization with a grid of {valid_combos} valid combinations using {num_cores} cores.")
    
    # --- 3. Run Rolling Optimizer ---
    rolling_optimizer = RollingOptimizer(
        data=full_data,
        train_window=train_window,
        test_window=test_window,
        param_grid=param_grid,
        num_cores=num_cores
    )
    all_trades, final_portfolio_value, best_params_history = rolling_optimizer.run()

    if all_trades.empty:
        print("Rolling optimization resulted in no trades. Cannot visualize.")
        return
        
    # --- 4. Run a final backtest with the last best parameters for visualization ---
    print("\nRunning final backtest for visualization...")
    # We need to get the data for the whole period for the final backtest plot
    analysis_start_date = final_portfolio_value.index.min()
    analysis_end_date = final_portfolio_value.index.max()
    backtest_data = full_data.loc[analysis_start_date:analysis_end_date]
    
    final_params = best_params_history[-1]['params']
    final_strategy = Strategy(
        data=backtest_data, # Use data only from the analysis period
        rsi_period=final_params['rsi_period'],
        macd_fast_period=final_params['macd_fast_period'],
        macd_slow_period=final_params['macd_slow_period'],
        macd_signal_period=final_params['macd_signal_period']
    )
    final_backtester = Backtester(final_strategy)
    _, final_trades_for_plot = final_backtester.run()

    # --- 5. Fetch Benchmark Data for the Same Period ---
    print(f"Fetching benchmark data for {spy_ticker} from {analysis_start_date.strftime('%Y-%m-%d')} to {analysis_end_date.strftime('%Y-%m-%d')}...")
    benchmark_data = fetch_data(spy_ticker, analysis_start_date.strftime('%Y-%m-%d'), analysis_end_date.strftime('%Y-%m-%d'))
    
    # --- 6. Save Results Summary ---
    save_results_summary(
        portfolio_value=final_portfolio_value,
        trades=all_trades,
        params_history=best_params_history,
        stock_data=full_data['Close'].loc[analysis_start_date:analysis_end_date],
        benchmark_data=benchmark_data['Close'] if benchmark_data is not None else None,
        initial_cash=rolling_optimizer.initial_cash,
        ticker=ticker,
        benchmark_ticker=spy_ticker
    )
    
    # --- 7. Visualize Results ---
    visualize_performance_and_indicators(
        portfolio_value=final_portfolio_value,
        price_data=backtest_data['Close'],
        trades=final_trades_for_plot,
        indicators=final_strategy.indicators,
        benchmark_data=benchmark_data['Close'] if benchmark_data is not None else None,
        title=f"Adaptive Strategy Analysis for {ticker}"
    )


if __name__ == '__main__':
    # --- Parameters ---
    TICKER_SYMBOL = 'AAPL'
    BENCHMARK_TICKER = 'SPY'
    START_DATE = '2018-01-01'
    END_DATE = '2023-12-31'
    NUM_CORES = 12
    
    # Define the rolling window parameters
    TRAIN_WINDOW = relativedelta(years=2)
    TEST_WINDOW = relativedelta(months=6)

    run_rolling_analysis(
        ticker=TICKER_SYMBOL,
        spy_ticker=BENCHMARK_TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        num_cores=NUM_CORES
    ) 