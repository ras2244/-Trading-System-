import yfinance as yf
import pandas as pd
import numpy as np

# --- Parameters ---
symbol = 'TSLA'
start_date = '2022-01-01'
end_date = '2024-12-31'
breakout_window = 20
trailing_stop_pct = 0.05
capital = 10000
position_fraction = 0.1  # Use 10% of capital per trade
slippage_pct = 0.001  # 0.1% slippage

# --- Data ---
data = yf.download(symbol, start=start_date, end=end_date)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

data['20d_high'] = data['High'].rolling(window=breakout_window).max()
data.dropna(subset=['20d_high'], inplace=True)

# --- Signals ---
data['entry_signal'] = data['Close'] > data['20d_high']
data['position'] = 0
in_position = False
entry_price = 0
stop_price = 0
shares_held = 0
cash = capital
trade_log = []

for i in range(1, len(data)):
    today = data.index[i]
    yesterday = data.index[i - 1]

    if not in_position and data.loc[yesterday, 'entry_signal']:
        entry_price = data.loc[today, 'Open'] * (1 + slippage_pct)
        stop_price = entry_price * (1 - trailing_stop_pct)
        shares_held = int((capital * position_fraction) / entry_price)
        cost = shares_held * entry_price
        cash -= cost
        in_position = True

        trade_log.append({
            'Entry Date': today,
            'Entry Price': entry_price,
            'Shares': shares_held,
            'Exit Date': None,
            'Exit Price': None,
            'Profit': None
        })

    elif in_position:
        high_price = data.loc[today, 'High']
        low_price = data.loc[today, 'Low']
        stop_price = max(stop_price, high_price * (1 - trailing_stop_pct))

        if low_price < stop_price:
            exit_price = stop_price * (1 - slippage_pct)
            proceeds = shares_held * exit_price
            cash += proceeds
            in_position = False

            trade_log[-1]['Exit Date'] = today
            trade_log[-1]['Exit Price'] = exit_price
            trade_log[-1]['Profit'] = proceeds - (shares_held * entry_price)
            shares_held = 0

    data.loc[today, 'position'] = shares_held

# --- Performance ---
data['daily_return'] = data['Close'].pct_change()
data['strategy_value'] = shares_held * data['Close'] + cash
data['strategy_return'] = data['strategy_value'].pct_change()
data['buy_hold_return'] = data['daily_return'].cumsum()
data['capital_curve'] = data['strategy_return'].cumsum()

total_return = data['strategy_value'].iloc[-1] / capital - 1
buy_hold = data['Close'].iloc[-1] / data['Close'].iloc[0] - 1
max_drawdown = (data['strategy_value'].cummax() - data['strategy_value']).max()
sharpe_ratio = data['strategy_return'].mean() / data['strategy_return'].std()

# --- Results ---
print(f"Total Strategy Return: {total_return:.2%}")
print(f"Buy-and-Hold Return: {buy_hold:.2%}")
print(f"Max Drawdown: {max_drawdown:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Trade Count: {len([t for t in trade_log if t['Exit Date'] is not None])}")

trade_log_df = pd.DataFrame(trade_log)
print("\nTrade Log:")
print(trade_log_df)

# Optional: Save trade log
trade_log_df.to_csv("trade_log.csv", index=False)
print("\nTrade log exported to 'trade_log.csv'")
