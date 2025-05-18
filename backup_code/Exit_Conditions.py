import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
symbols = ['TSLA', 'AAPL', 'NVDA']  # Add more if you'd like
start_date = '2022-01-01'
end_date = '2024-12-31'
breakout_window = 20
trailing_stop_pct = 0.05
slippage = 0.001  # 0.1%
capital = 100000
position_size_pct = 0.1  # 10% per trade

# Store metrics for each symbol
results = {}

for symbol in symbols:
    print(f"\nBacktesting {symbol}...")

    # Download data
    data = yf.download(symbol, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Calculate breakout high
    data['20d_high'] = data['High'].rolling(window=breakout_window).max()

    # Generate signals
    data['entry_signal'] = (data['Close'] > data['20d_high']).astype(int)
    data['entry_signal'] = data['entry_signal'].shift(1)  # Enter next day

    # Track trailing stop and position
    data['in_position'] = False
    data['stop_price'] = np.nan
    data['buy_price'] = np.nan
    data['sell_price'] = np.nan
    data['position_value'] = 0.0
    trade_log = []

    in_position = False
    stop_price = 0
    shares = 0
    buy_price = 0

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i - 1]

        if not in_position and row['entry_signal'] == 1:
            buy_price = row['Open'] * (1 + slippage)
            shares = (capital * position_size_pct) // buy_price
            cost = shares * buy_price
            stop_price = row['Close'] * (1 - trailing_stop_pct)
            data.at[data.index[i], 'buy_price'] = buy_price
            in_position = True
            data.at[data.index[i], 'in_position'] = True

            trade_log.append({
                'Date': data.index[i],
                'Type': 'Buy',
                'Price': buy_price,
                'Shares': shares,
                'Cost': cost
            })

        elif in_position:
            # Update trailing stop
            if row['Close'] > prev_row['Close']:
                stop_price = max(stop_price, row['Close'] * (1 - trailing_stop_pct))

            if row['Close'] < stop_price:
                sell_price = row['Open'] * (1 - slippage)
                proceeds = shares * sell_price
                data.at[data.index[i], 'sell_price'] = sell_price
                data.at[data.index[i], 'in_position'] = False
                in_position = False
                shares = 0

                trade_log.append({
                    'Date': data.index[i],
                    'Type': 'Sell',
                    'Price': sell_price,
                    'Shares': shares,
                    'Proceeds': proceeds
                })

            else:
                data.at[data.index[i], 'in_position'] = True

    # Calculate performance
    data['daily_return'] = data['Close'].pct_change()
    data['strategy_return'] = data['daily_return'] * data['in_position'].shift(1).fillna(False)
    total_return = data['strategy_return'].add(1).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(data)) - 1
    max_drawdown = (data['strategy_return'].cumsum() - data['strategy_return'].cumsum().cummax()).min()
    sharpe_ratio = data['strategy_return'].mean() / data['strategy_return'].std()

    results[symbol] = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Trade Log': pd.DataFrame(trade_log)
    }

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.scatter(data.index, data['buy_price'], marker='^', color='green', label='Buy', s=100)
    plt.scatter(data.index, data['sell_price'], marker='v', color='red', label='Sell', s=100)
    plt.title(f'{symbol} - Breakout Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print("Trade Log:")
    print(results[symbol]['Trade Log'])

# You now have a dictionary `results` with metrics and trade logs per symbol
