import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Choose your stock symbol (e.g., Apple)
symbol = "AAPL"

# 2. Download data
ticker = yf.Ticker(symbol)
data = ticker.history(period="6mo")  # e.g., last 6 months

# 3. Display basic info
print(f"Stock info for {symbol}:")
print(ticker.info['longName'])
print(data.head())

# 4. Plot closing price
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Close'], label="Close Price")
plt.title(f"{symbol} Closing Price - Last 6 Months")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
