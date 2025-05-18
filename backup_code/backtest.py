import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters ---
ticker = "AAPL"
start_date = "2020-01-01"
short_window = 20
long_window = 50

# --- Download Data ---
print(f"Downloading data for {ticker}...")
data = yf.download(ticker, start=start_date)
data = data[['Close']]  # Use only Close prices

# --- Calculate Moving Averages ---
data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
data['SMA_long'] = data['Close'].rolling(window=long_window).mean()

# --- Generate Signals ---
data['position'] = 0
data.loc[data['SMA_short'] > data['SMA_long'], 'position'] = 1
data.loc[data['SMA_short'] < data['SMA_long'], 'position'] = -1

# Shift to avoid lookahead bias (trade next day)
data['signal'] = data['position'].shift(1)

# --- Backtest Logic ---
data['daily_return'] = data['Close'].pct_change()
data['strategy_return'] = data['daily_return'] * data['signal']

# --- Performance Metrics ---
cumulative_market = (1 + data['daily_return']).cumprod()
cumulative_strategy = (1 + data['strategy_return']).cumprod()

# --- Plot Results ---
plt.figure(figsize=(14, 7))
plt.plot(cumulative_market, label='Market Return (Buy & Hold)')
plt.plot(cumulative_strategy, label='Strategy Return (MA Crossover)', linestyle='--')
plt.title(f"{ticker} - MA Crossover Strategy vs Market")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



