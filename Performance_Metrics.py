import yfinance as yf
import pandas as pd
import numpy as np

# Parameters
symbol = 'TSLA'  # Try a more volatile stock
start_date = '2022-01-01'
end_date = '2024-12-31'
breakout_window = 10  # You can try 20 again later

# Download data
data = yf.download(symbol, start=start_date, end=end_date)

# Flatten multi-index columns if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]
print(f"Flattened columns: {list(data.columns)}")

# Calculate 10-day high
data['20d_high'] = data['High'].rolling(window=breakout_window).max()

# Drop rows with NaNs
data.dropna(subset=['20d_high'], inplace=True)

# Generate signals
data['position'] = 0
data.loc[data['Close'] > data['20d_high'], 'position'] = 1
data['position'] = data['position'].shift(1)  # Enter at next day open

# Debug: See if signals are being triggered
print(f"Number of breakout signals: {data['position'].sum()}")
print(data[['Close', '20d_high', 'position']].tail(10))

# Calculate returns
data['daily_return'] = data['Close'].pct_change()
data['strategy_return'] = data['daily_return'] * data['position']

# Performance metrics
total_return = data['strategy_return'].add(1).prod() - 1
annualized_return = (1 + total_return) ** (252 / len(data)) - 1
max_drawdown = ((data['strategy_return'].cumsum() - data['strategy_return'].cumsum().cummax()).min())
sharpe_ratio = data['strategy_return'].mean() / data['strategy_return'].std()

# Display performance
print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
