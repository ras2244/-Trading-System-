import yfinance as yf
import pandas as pd
import numpy as np

# Parameters
symbols = ['TSLA', 'AAPL', 'NVDA']
start_date = '2022-01-01'
end_date = '2024-12-31'
breakout_window = 10
train_size = 252  # 1 year
test_size = 21    # 1 month
initial_capital = 100000
risk_pct = 0.01

# Download data for all symbols
all_data = {}
for symbol in symbols:
    df = yf.download(symbol, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df['20d_high'] = df['High'].rolling(window=breakout_window).max()
    df.dropna(subset=['20d_high'], inplace=True)
    all_data[symbol] = df

# Align on common dates
date_index = set.intersection(*(set(df.index) for df in all_data.values()))
date_index = sorted(list(date_index))

# Portfolio-level walk-forward testing
walk_forward_results = []

for start_idx in range(len(date_index) - train_size - test_size):
    window_dates = date_index[start_idx:start_idx + train_size + test_size]
    train_dates = window_dates[:train_size]
    test_dates = window_dates[train_size:]

    portfolio_curve = []
    capital = initial_capital
    daily_returns = pd.DataFrame(index=test_dates, columns=symbols)
    trades_total = 0

    for symbol in symbols:
        df = all_data[symbol].copy()
        df = df.loc[window_dates].copy()
        df['position'] = 0
        df.loc[df['Close'] > df['20d_high'], 'position'] = 1
        df['position'] = df['position'].shift(1)
        df['ATR'] = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
        df['stop_loss'] = df['Close'] - (df['ATR'] * 1.5)
        df['take_profit'] = df['Close'] + (df['ATR'] * 2.0)

        cap = capital / len(symbols)
        pos = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        returns = []

        for i in range(train_size, len(df)):
            idx = df.index[i]
            if df['position'].iloc[i] == 1 and pos == 0:
                pos = cap * risk_pct / df['ATR'].iloc[i]
                entry_price = df['Open'].iloc[i]
                stop_loss = df['stop_loss'].iloc[i]
                take_profit = df['take_profit'].iloc[i]
                trades_total += 1
            elif pos > 0:
                if df['Low'].iloc[i] <= stop_loss:
                    ret = (stop_loss - entry_price) / entry_price
                    returns.append(ret)
                    pos = 0
                elif df['High'].iloc[i] >= take_profit:
                    ret = (take_profit - entry_price) / entry_price
                    returns.append(ret)
                    pos = 0
                elif df['Close'].iloc[i] > entry_price:
                    stop_loss = max(stop_loss, df['Close'].iloc[i] - df['ATR'].iloc[i] * 1.5)
                else:
                    returns.append(0)
            else:
                returns.append(0)

        returns_series = pd.Series(returns, index=test_dates[:len(returns)])
        daily_returns[symbol] = returns_series

    daily_returns.fillna(0, inplace=True)
    portfolio_daily_returns = daily_returns.mean(axis=1)
    capital_curve = (portfolio_daily_returns + 1).cumprod() * capital
    total_return = capital_curve.iloc[-1] / capital - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_daily_returns)) - 1
    max_drawdown = (capital_curve - capital_curve.cummax()).min() / capital

    if portfolio_daily_returns.std() == 0 or portfolio_daily_returns.isna().all():
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = portfolio_daily_returns.mean() / portfolio_daily_returns.std() * np.sqrt(252)

    walk_forward_results.append({
        'start_date': test_dates[0],
        'end_date': test_dates[-1],
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'trades_executed': trades_total
    })

walk_forward_results_df = pd.DataFrame(walk_forward_results)
print(walk_forward_results_df)
