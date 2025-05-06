import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import winsound
    SOUND_ENABLED = True
except ImportError:
    SOUND_ENABLED = False

# === CONFIGURATION ===
symbols = ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'JPM', 'V', 'UNH', 'DIS', 'INTC', 'NFLX', 'AMD', 'BA']
start_date = '2022-01-01'
end_date = '2024-12-31'
allocation_mode = 'volatility'  # 'equal' or 'volatility'
initial_capital = 100000
breakout_window = 20
risk_pct = 0.01
atr_factor = 1.5

# === FUNCTION ma_crossover_strategy ===
def run_ma_crossover_strategy(data_dict, symbols, initial_capital, short_window=10, long_window=30, allocation_mode='equal'):
    capital_per_symbol = {}
    ma_equity_curves = {}
    position = {}
    
    # === Allocation Calculation ===
    if allocation_mode == 'equal':
        alloc = {s: 1 / len(symbols) for s in symbols}
    elif allocation_mode == 'volatility':
        atrs = {s: df['High'].rolling(14).max().iloc[-1] - df['Low'].rolling(14).min().iloc[-1] for s, df in data_dict.items()}
        inv_vol = {s: 1 / v for s, v in atrs.items() if v > 0}
        total = sum(inv_vol.values())
        alloc = {s: w / total for s, w in inv_vol.items()}
    else:
        raise ValueError("Invalid allocation mode.")

    for symbol in symbols:
        df = data_dict[symbol].copy()
        if len(df) < long_window:
            continue

        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        df['Signal'] = 0
        df.loc[df.index[short_window:], 'Signal'] = np.where(
            df['SMA_Short'].iloc[short_window:] > df['SMA_Long'].iloc[short_window:], 1, 0
            )
        df['Signal'] = df['Signal'].shift(1)  # Shift to avoid lookahead bias
        df['Position'] = df['Signal'].diff()

        cash = initial_capital * alloc.get(symbol, 0)
        shares = 0
        capital_curve = []

        for date, row in df.iterrows():
            price = row['Open']
            if row['Position'] == 1:  # Buy
                shares = cash / price
                cash = 0
            elif row['Position'] == -1 and shares > 0:  # Sell
                cash = shares * price
                shares = 0
            total_value = cash + shares * price
            capital_curve.append(total_value)

        ma_equity_curves[symbol] = capital_curve

    # === Combine equity curves ===
    aligned_curves = pd.DataFrame(ma_equity_curves)
    aligned_curves['Total'] = aligned_curves.sum(axis=1)
    ma_portfolio_series = aligned_curves['Total']
    ma_portfolio_series.index = data_dict[symbols[0]].index[-len(ma_portfolio_series):]

    return ma_portfolio_series


# === INITIALIZATION ===
data_dict = {}
trade_log = []
highest_price = {}
symbol_results = {}

# === DOWNLOAD AND PREP DATA ===
for symbol in symbols:
    print(f"Downloading {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    if len(df) >= max(breakout_window, 14):
        df['20d_high'] = df['High'].rolling(window=breakout_window).max()
        df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        df.dropna(subset=['20d_high', 'ATR'], inplace=True)
        if not df.empty:
            data_dict[symbol] = df.copy()

# === VALIDATE SYMBOLS ===
symbols = list(data_dict.keys())
if not symbols:
    raise ValueError("No valid symbols with sufficient data.")

# === ALLOCATION ===
if allocation_mode == 'equal':
    alloc = {s: 1 / len(symbols) for s in symbols}
elif allocation_mode == 'volatility':
    atrs = {s: df['ATR'].iloc[-1] for s, df in data_dict.items()}
    inv_vol = {s: 1 / v for s, v in atrs.items() if v > 0}
    total = sum(inv_vol.values())
    alloc = {s: w / total for s, w in inv_vol.items()}
else:
    raise ValueError("Invalid allocation mode.")

print("\nCapital Allocation:")
print(pd.Series(alloc))

# === BACKTEST LOOP ===
dates = data_dict[symbols[0]].index
portfolio_value = []
capital_per_symbol = {s: initial_capital * alloc[s] for s in symbols}
position = {s: 0 for s in symbols}
entry_price = {s: 0 for s in symbols}
highest_price = {s: 0 for s in symbols}
symbol_capital_curves = {}

for date in dates:
    total_equity = 0
    for symbol in symbols:
        df = data_dict[symbol]
        if date not in df.index:
            continue
        row = df.loc[date]
        cap = capital_per_symbol[symbol]

        # Entry signal
        if position[symbol] == 0 and row['Close'] > row['20d_high']:
            entry_price[symbol] = row['Open']
            position[symbol] = (cap * risk_pct) / row['ATR']
            highest_price[symbol] = entry_price[symbol]
            trade_log.append({'symbol': symbol, 'date': date, 'action': 'BUY', 'price': entry_price[symbol], 'capital_allocated': cap})

        # Exit logic: Trailing stop
        elif position[symbol] > 0:
            highest_price[symbol] = max(highest_price[symbol], row['High'])
            trailing_stop = highest_price[symbol] - row['ATR'] * atr_factor

            if row['Low'] <= trailing_stop:
                exit_price = trailing_stop
                cap += position[symbol] * (exit_price - entry_price[symbol])
                position[symbol] = 0
                trade_log.append({'symbol': symbol, 'date': date, 'action': 'SELL (TRAIL STOP)', 'price': exit_price, 'capital_after_trade': cap})
            else:
                cap += position[symbol] * (row['Close'] - entry_price[symbol])

        capital_per_symbol[symbol] = cap
        total_equity += cap

        # Track capital curve per symbol
        symbol_capital_curves.setdefault(symbol, []).append(cap)

    portfolio_value.append(total_equity)

# === PERFORMANCE METRICS ===
portfolio_series = pd.Series(portfolio_value, index=dates)
returns = portfolio_series.pct_change().dropna()
total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
max_dd = (portfolio_series.cummax() - portfolio_series).max() / portfolio_series.cummax().max()
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

print(f"\nFinal Portfolio Value: ${portfolio_series.iloc[-1]:,.2f}")
print(f"Total Return: {total_return:.2%}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# === PER-SYMBOL PERFORMANCE ===
for symbol in symbols:
    curve = symbol_capital_curves.get(symbol, [])
    if len(curve) < 2:
        continue
    s_returns = pd.Series(curve).pct_change().dropna()
    f_cap = curve[-1]
    t_ret = (f_cap / curve[0]) - 1 if curve[0] > 0 else 0
    mdd = (pd.Series(curve).cummax() - curve).max() / pd.Series(curve).cummax().max()
    s_sharpe = s_returns.mean() / s_returns.std() * np.sqrt(252) if s_returns.std() > 0 else 0
    symbol_results[symbol] = {
        'Final Capital': f_cap,
        'Total Return': t_ret,
        'Sharpe Ratio': s_sharpe,
        'Max Drawdown': mdd
    }

perf_df = pd.DataFrame(symbol_results).T.sort_values('Total Return', ascending=False)
print("\nPer-Symbol Performance:")
print(perf_df)

# === EXPORT TO CSV ===
portfolio_series.to_csv("portfolio_equity_curve.csv", header=["Portfolio Value"])
pd.DataFrame(trade_log).to_csv("trade_log.csv", index=False)
perf_df.to_csv("symbol_performance.csv")
print("\nExported: portfolio_equity_curve.csv, trade_log.csv, symbol_performance.csv")

# === PLOT ===
portfolio_series.plot(figsize=(12, 6), title='Portfolio Equity Curve')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Run Moving Average Crossover Strategy ===
ma_portfolio_series = run_ma_crossover_strategy(
    data_dict=data_dict,
    symbols=symbols,
    initial_capital=initial_capital,
    short_window=10,
    long_window=30,
    allocation_mode=allocation_mode
)

# === Calculate MA strategy metrics ===
ma_returns = ma_portfolio_series.pct_change().dropna()
ma_total_return = (ma_portfolio_series.iloc[-1] / ma_portfolio_series.iloc[0]) - 1
ma_max_dd = (ma_portfolio_series.cummax() - ma_portfolio_series).max() / ma_portfolio_series.cummax().max()
ma_sharpe = ma_returns.mean() / ma_returns.std() * np.sqrt(252) if ma_returns.std() > 0 else 0

# === Strategy Comparison Summary ===

results_summary = [
    {
        'Strategy': 'Breakout',
        'Final Capital': portfolio_series.iloc[-1],
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_dd
    },
    {
        'Strategy': 'MA Crossover',
        'Final Capital': ma_portfolio_series.iloc[-1],
        'Total Return': ma_total_return,
        'Sharpe Ratio': ma_sharpe,
        'Max Drawdown': ma_max_dd
    }
]

# === Display & Save ===
comparison_df = pd.DataFrame(results_summary)
print("\nStrategy Comparison Summary:")
print(comparison_df)

# Save results
comparison_df.to_csv("strategy_comparison.csv", index=False)

# === Plot Capital Curves ===
plt.figure(figsize=(12, 6))
portfolio_series.plot(label='Breakout Strategy')
ma_portfolio_series.plot(label='MA Crossover Strategy')
plt.title("Strategy Capital Curve Comparison")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
