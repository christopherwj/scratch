import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your results CSV (adjust filename if needed)
log_file = "TSLA_vcp_param_log.csv"
df_results = pd.read_csv(log_file)

# Filter for top parameter sets with high win rate (e.g. >= 0.98)
top_results = df_results[df_results['win_rate'] >= 0.98].copy()

# Assume average loss % on losing trades (e.g. -20%)
avg_loss = -0.20

starting_capital = 1000
num_trades = 100  # simulate 100 trades

def simulate_growth(win_rate, avg_return, avg_loss, starting_capital, num_trades):
    capital = starting_capital
    capital_over_time = [capital]

    expected_return_per_trade = win_rate * avg_return + (1 - win_rate) * avg_loss

    for _ in range(num_trades):
        capital = capital * (1 + expected_return_per_trade)
        capital_over_time.append(capital)
    return capital_over_time

plt.figure(figsize=(12, 7))

for idx, row in top_results.iterrows():
    win_rate = row['win_rate']
    avg_return = row['avg_return']  # e.g. 43.3 means +4330%
    # Convert avg_return from absolute return to fraction for calculation:
    avg_return_frac = avg_return  # Assuming avg_return is already fractional (e.g. 0.433)
    # If avg_return looks like large numbers (e.g. 43.3), divide by 100:
    if avg_return > 10:
        avg_return_frac = avg_return / 100.0

    growth = simulate_growth(win_rate, avg_return_frac, avg_loss, starting_capital, num_trades)
    plt.plot(growth, label=f"Contraction {row['max_contraction_pct']}%, ATR thresh {row['atr_percentile_thresh']}, Vol ratio {row['breakout_volume_ratio']}")

plt.title("Simulated Capital Growth Over 100 Trades\nAssuming 20% loss on losing trades")
plt.xlabel("Number of Trades")
plt.ylabel("Capital ($)")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()
