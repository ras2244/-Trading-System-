import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download historical data
ticker = "AAPL"
data = yf.download(ticker, start="2022-01-01", end="2023-01-01")

# If MultiIndex columns, flatten them
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Calculate 20-day high and 10-day low
data['20d_high'] = data['High'].rolling(window=20).max()
data['10d_low'] = data['Low'].rolling(window=10).min()

# Generate signals
data['position'] = 0
data.loc[data['Close'] > data['20d_high'], 'position'] = 1
data.loc[data['Close'] < data['10d_low'], 'position'] = 0
data['position'] = data['position'].ffill().fillna(0)

# Calculate returns
data['market_return'] = data['Close'].pct_change()
data['strategy_return'] = data['position'].shift(1) * data['market_return']

# Output performance
total_return = (data['strategy_return'] + 1).prod() - 1
print(f"Total Strategy Return: {total_return:.2%}")

# Plot the price and signals
data[['Close', '20d_high', '10d_low']].plot(figsize=(12, 6), title="Breakout Strategy: Price with 20d High & 10d Low")
plt.show()
