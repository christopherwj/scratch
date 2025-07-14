import yfinance as yf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# --- Settings ---
ticker = 'TSLA'
period = 'max'
interval = '1d'
min_trades_required = 1  # accept results with at least 1 trade
log_file = f"{ticker}_vcp_param_log.csv"

# --- Download or load cached data ---
def download_or_cache_data(ticker, period='10y', interval='1d'):
    filename = f"{ticker}_{period}_{interval}.csv"
    if os.path.exists(filename):
        print(f"📁 Loading cached data: {filename}")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        print(f"🌐 Downloading {ticker} from yFinance...")
        df = yf.download(ticker, period=period, interval=interval)
        df.dropna(inplace=True)
        df.to_csv(filename)

    # Ensure numeric columns are correct dtype
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)
    return df

# --- Calculate ATR ---
def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def atr_percentile(atr_series):
    return atr_series.rank(pct=True) * 100

# --- Detect VCP trades ---
def detect_vcp_trades(df, max_contraction_pct, atr_percentile_thresh, breakout_volume_ratio):
    df = df.copy()
    df['ATR'] = calculate_atr(df)
    df['ATR_pct'] = atr_percentile(df['ATR'])

    trades = []
    position = False
    entry_price = 0
    stop_price = 0
    window = 20

    for i in range(window, len(df) - 1):
        atr_pct_val = df['ATR_pct'].iloc[i]
        if pd.isna(atr_pct_val):
            continue
        atr_low = atr_pct_val < atr_percentile_thresh

        price_now = df['Close'].iloc[i].item()
        price_prev = df['Close'].iloc[i - window:i]
        vol_now = df['Volume'].iloc[i].item()
        vol_prev = df['Volume'].iloc[i - window:i]

        max_price_prev = price_prev.max().item()
        min_price_prev = price_prev.min().item()
        avg_vol = vol_prev.mean().item()

        pullback_pct = (max_price_prev - price_now) / max_price_prev * 100
        breakout = (price_now > max_price_prev) and (vol_now > avg_vol * breakout_volume_ratio)

        if not position:
            if atr_low and (pullback_pct <= max_contraction_pct) and breakout:
                position = True
                entry_price = price_now
                stop_price = min_price_prev
        else:
            next_price = df['Close'].iloc[i + 1].item()
            if next_price <= stop_price:
                ret = (next_price - entry_price) / entry_price
                ret = np.clip(ret, -0.9, 5.0)  # cap big outliers
                trades.append(ret)
                position = False
            elif i == len(df) - 2:
                ret = (next_price - entry_price) / entry_price
                ret = np.clip(ret, -0.9, 5.0)
                trades.append(ret)
                position = False

    return np.array(trades)

# --- Monte Carlo Simulation ---
def monte_carlo_sim(trade_returns, num_trials=1000, path_length=100):
    simulations = []
    for _ in range(num_trials):
        sample = np.random.choice(trade_returns, size=path_length, replace=True)
        cum_returns = np.cumprod(1 + sample) - 1
        drawdowns = np.maximum.accumulate(cum_returns) - cum_returns
        simulations.append((cum_returns[-1], np.max(drawdowns)))
    return simulations

# --- Parameter sweep space ---
param_space = {
    'max_contraction_pct': [20, 25, 30, 35],
    'atr_percentile_thresh': [30, 35, 40, 45],
    'breakout_volume_ratio': [1.0, 1.2, 1.5],
}

# --- Evaluate parameters and log all results ---
def evaluate_params(df):
    all_results = []

    for max_contraction_pct, atr_thresh, vol_ratio in product(
        param_space['max_contraction_pct'],
        param_space['atr_percentile_thresh'],
        param_space['breakout_volume_ratio']):

        trades = detect_vcp_trades(df, max_contraction_pct, atr_thresh, vol_ratio)
        n = len(trades)

        if n == 0:
            all_results.append({
                'max_contraction_pct': max_contraction_pct,
                'atr_percentile_thresh': atr_thresh,
                'breakout_volume_ratio': vol_ratio,
                'num_trades': n,
                'median_return': None,
                'avg_return': None,
                'win_rate': None,
                'worst_drawdown': None
            })
            continue
        elif n == 1:
            single_ret = trades[0]
            all_results.append({
                'max_contraction_pct': max_contraction_pct,
                'atr_percentile_thresh': atr_thresh,
                'breakout_volume_ratio': vol_ratio,
                'num_trades': n,
                'median_return': single_ret,
                'avg_return': single_ret,
                'win_rate': 1.0 if single_ret > 0 else 0.0,
                'worst_drawdown': 0.0
            })
            continue

        sims = monte_carlo_sim(trades)
        cum_rets = [s[0] for s in sims]
        max_dds = [s[1] for s in sims]

        all_results.append({
            'max_contraction_pct': max_contraction_pct,
            'atr_percentile_thresh': atr_thresh,
            'breakout_volume_ratio': vol_ratio,
            'num_trades': n,
            'median_return': np.median(cum_rets),
            'avg_return': np.mean(cum_rets),
            'win_rate': np.mean(np.array(cum_rets) > 0),
            'worst_drawdown': np.percentile(max_dds, 95)
        })

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(log_file, index=False)
    print(f"📊 All parameter results saved to: {log_file}")

    return df_all[df_all['num_trades'] >= min_trades_required].copy()

# --- Main ---
if __name__ == '__main__':
    df = download_or_cache_data(ticker, period, interval)

    print("🔍 Evaluating parameter combinations...")
    df_results = evaluate_params(df)

    if df_results.empty:
        raise ValueError("❌ No parameter sets produced enough trades. Try relaxing filters or increasing the data period.")

    df_results = df_results.sort_values(['median_return', 'worst_drawdown'], ascending=[False, True])
    print("✅ Top Parameter Sets:")
    print(df_results.head(5))

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_results,
                    x='worst_drawdown', y='median_return',
                    size='win_rate', hue='max_contraction_pct',
                    palette='coolwarm', sizes=(50, 200), alpha=0.8)
    plt.title(f'VCP Monte Carlo Results: {ticker}')
    plt.xlabel('Worst 95th Percentile Drawdown')
    plt.ylabel('Median Cumulative Return')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
