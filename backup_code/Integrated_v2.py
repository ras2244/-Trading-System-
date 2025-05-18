import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product

# Parameters
symbol = 'TSLA'  # Try a more volatile stock
start_date = '2022-01-01'
end_date = '2024-12-31'
breakout_window = 10

# Download data
data = yf.download(symbol, start=start_date, end=end_date)

# Flatten multi-index columns if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# Calculate 10-day high
data['20d_high'] = data['High'].rolling(window=breakout_window).max()

# Drop rows with NaNs
data.dropna(subset=['20d_high'], inplace=True)

# Grid search parameters
risk_percentages = [0.01, 0.02, 0.05]  # 1%, 2%, 5%
atr_factors = [1.0, 1.5, 2.0]  # Trailing stop ATR factor
reward_risk_ratios = [1.5, 2.0, 3.0]  # Risk-to-reward ratios

# Results list
optimization_results = []

# Test each combination of parameters
for risk_percentage, atr_factor, reward_risk_ratio in product(risk_percentages, atr_factors, reward_risk_ratios):
    # Reset capital and position for each run
    capital = 100000  # Starting capital
    position = 0
    entry_price = 0
    entry_date = None
    capital_curve = [capital]
    trade_log = []

    # Calculate ATR for stop-loss and trailing stop
    data['ATR'] = data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()

    # Generate signals
    data['position'] = 0
    data.loc[data['Close'] > data['20d_high'], 'position'] = 1
    data['position'] = data['position'].shift(1)  # Enter at next day open

    # Calculate stop-loss and take-profit levels
    data['stop_loss'] = data['Close'] - (data['ATR'] * atr_factor)
    data['take_profit'] = data['Close'] + (data['ATR'] * reward_risk_ratio)

    # Backtest loop (one bar per day)
    for i in range(1, len(data)):
        if data['position'].iloc[i] == 1 and position == 0:  # Buy signal
            # Calculate position size based on risk management
            position_size = capital * risk_percentage / data['ATR'].iloc[i]
            entry_price = data['Open'].iloc[i]  # Buy at next day's open price
            position = position_size
            stop_loss = data['stop_loss'].iloc[i]  # Stop-Loss
            take_profit = data['take_profit'].iloc[i]  # Take-Profit
            entry_date = data.index[i]
            trade_log.append({
                'action': 'BUY',
                'date': entry_date,
                'price': entry_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
        
        elif position > 0:
            # Check if stop-loss or take-profit is hit
            if data['Low'].iloc[i] <= stop_loss:  # Stop-loss hit
                capital += position * (stop_loss - entry_price)  # Close position at stop-loss
                trade_log.append({
                    'action': 'SELL (STOP)',
                    'date': data.index[i],
                    'price': stop_loss,
                    'size': position,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
                position = 0
            elif data['High'].iloc[i] >= take_profit:  # Take-profit hit
                capital += position * (take_profit - entry_price)  # Close position at take-profit
                trade_log.append({
                    'action': 'SELL (PROFIT)',
                    'date': data.index[i],
                    'price': take_profit,
                    'size': position,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
                position = 0
            elif data['Close'].iloc[i] > entry_price:  # Trailing stop condition
                stop_loss = max(stop_loss, data['Close'].iloc[i] - (data['ATR'].iloc[i] * atr_factor))
        
        # Update capital curve and position
        capital_curve.append(capital + position * (data['Close'].iloc[i] - entry_price) if position > 0 else capital)

    # Create a DataFrame to analyze the capital curve
    capital_df = pd.DataFrame(capital_curve, index=data.index, columns=['Capital'])
    capital_df['Returns'] = capital_df['Capital'].pct_change()

    # Performance metrics
    total_return = (capital_df['Capital'].iloc[-1] - capital_df['Capital'].iloc[0]) / capital_df['Capital'].iloc[0]
    annualized_return = (1 + total_return) ** (252 / len(capital_df)) - 1
    max_drawdown = (capital_df['Capital'].min() - capital_df['Capital'].iloc[0]) / capital_df['Capital'].iloc[0]
    sharpe_ratio = capital_df['Returns'].mean() / capital_df['Returns'].std() * np.sqrt(252)

    # Store the results
    optimization_results.append({
        'risk_percentage': risk_percentage,
        'atr_factor': atr_factor,
        'reward_risk_ratio': reward_risk_ratio,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    })

# Convert the results to a DataFrame for easy analysis
optimization_results_df = pd.DataFrame(optimization_results)

# Display the best results based on Sharpe Ratio
best_results = optimization_results_df.sort_values(by='sharpe_ratio', ascending=False).iloc[0]
print("Best Results:")
print(best_results)

