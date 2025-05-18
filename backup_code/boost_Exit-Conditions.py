import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
symbol = 'TSLA'
start_date = '2022-01-01'
end_date = '2024-12-31'
breakout_window = 20

# Download data
data = yf.download(symbol, start=start_date, end=end_date)

# Flatten columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# Calculate 20-day high
data['20d_high'] = data['High'].rolling(window=breakout_window).max()

# Drop NA
data.dropna(subset=['20d_high'], inplace=True)

# ENTRY: Breakout condition (slightly relaxed)
data['position'] = 0
data.loc[data['Close'] >= data['20d_high'] * 0.995, 'position'] = 1
data['position'] = data['position'].shift(1)  # Enter next day

# EXIT: Price drops below 10-day low (tight stop-loss)
data['10d_low'] = data['Low'].rolling(window=10).min()
data['exit_position'] = 0
data.loc[data['Close'] < data['10d_low'], 'exit_position'] = -1
data['exit_position'] = data['exit_position'].shift(1)

# Combine entry and exit
data['final_position'] = 0
position_flag = False

for i in range(1, len(data)):
    if data['position'].iloc[i] == 1 and not position_flag:
        position_flag = True
    elif data['exit_position'].iloc[i] == -1 and position_flag:
        position_flag = False
    data['final_position'].iloc[i] = 1 if position_flag else 0

# Calculate returns
data['daily_return'] = data['Close'].pct_change()
data['strategy_return'] = data['daily_return'] * data['final_position']

# Metrics
total_return = data['strategy_return'].add(1).prod() - 1
annualized_return = (1 + total_return) ** (252 / len(data)) - 1
max_drawdown = ((data['strategy_return'].cumsum() - data['strategy_return'].cumsum().cummax()).min())
sharpe_ratio = data['strategy_return'].mean() / data['strategy_return'].std()

# Print metrics
print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Number of breakout signals: {data['position'].sum()}")
print(f"Number of exit signals: {data['exit_position'].sum()}")

# Plot
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['20d_high'], label='20-Day High', linestyle='--', alpha=0.7)
plt.scatter(data.index[data['position'] == 1], data['Close'][data['position'] == 1], marker='^', color='green', label='Buy Signal')
plt.scatter(data.index[data['exit_position'] == -1], data['Close'][data['exit_position'] == -1], marker='v', color='red', label='Sell Signal')
plt.title(f'{symbol} Price with Breakout and Exit Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()
