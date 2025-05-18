import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import glob

all_trades = []
os.makedirs("exports/equity_curves", exist_ok=True)
os.makedirs("exports/individual_logs/breakout", exist_ok=True)
os.makedirs("exports/individual_logs/ma_crossover", exist_ok=True)
os.makedirs("exports/individual_logs/buy_hold", exist_ok=True)

# === Configuration ===
start_date = '2023-01-01'
end_date = '2024-04-30'
initial_capital = 100000

# === Load best breakout parameters ===
df_summary = pd.read_csv("symbol_results/summary_best_params.csv")
capital_per_symbol = initial_capital / len(df_summary)

# === Breakout Strategy ===
def breakout_strategy(symbol, bw, atr_factor):
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty:
        print(f"WARNING: Skipping {symbol} — no data.")
        return None

    try:
        df['20d_high'] = df['High'].rolling(window=bw).max()
        df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
    except Exception as e:
        print(f"WARNING: Skipping {symbol} — error during rolling calc: {e}")
        return None

    required_cols = ['20d_high', 'ATR']
    if not all(col in df.columns for col in required_cols):
        print(f"WARNING: Skipping {symbol} — required columns not found.")
        return None
    if df[required_cols].isna().all(axis=0).any():
        print(f"WARNING: Skipping {symbol} — one or more indicators are entirely NaN.")
        return None

    df = df[df[required_cols].notna().all(axis=1)]
    if df.empty:
        print(f"WARNING: Skipping {symbol} — empty after filtering NaNs.")
        return None

    capital = capital_per_symbol
    position = 0
    entry_price = 0
    entry_date = None
    peak = 0
    equity = []

    for date, row in df.iterrows():
        try:
            close_val = row['Close']
            high_val = row['20d_high']
            atr_val = row['ATR']
            high_price = row['High']
            low_price = row['Low']
            open_price = row['Open']

            if position == 0 and close_val > high_val:
                entry_price = open_price
                entry_date = date
                position = (capital * 0.01) / atr_val
                peak = high_price

            elif position > 0:
                peak = max(peak, high_price)
                stop = peak - atr_val * atr_factor
                if low_price <= stop:
                    exit_price = stop
                    exit_date = date
                    pnl = (exit_price - entry_price) * position
                    capital += pnl
                    ret_pct = (exit_price - entry_price) / entry_price * 100
                    all_trades.append({
                        'Symbol': symbol,
                        'Strategy': 'Breakout',
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'Return %': round(ret_pct, 2)
                    })
                    position = 0
                else:
                    capital += position * (close_val - entry_price)

            equity.append(capital)

        except Exception as e:
            print(f"WARNING: Skipping row due to error: {e}")
            equity.append(capital)

    return pd.Series(equity, index=df.index)


# === MA Crossover Strategy ===
def ma_crossover_strategy(symbol, short=20, long=50):
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty:
        print(f"WARNING: Skipping {symbol} — no data.")
        return None

    df['SMA_Short'] = df['Close'].rolling(window=short).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long).mean()
    df.dropna(inplace=True)
    if df.empty:
        print(f"WARNING: Skipping {symbol} — empty after SMA dropna.")
        return None

    position = 0
    capital = capital_per_symbol
    entry_price = 0
    entry_date = None
    equity = []

    for date, row in df.iterrows():
        try:
            short_val = row['SMA_Short']
            long_val = row['SMA_Long']
            close = row['Close']
            open_price = row['Open']

            if position == 0 and short_val > long_val:
                entry_price = open_price
                entry_date = date
                position = capital / entry_price
            elif position > 0 and short_val < long_val:
                exit_price = open_price
                exit_date = date
                pnl = (exit_price - entry_price) * position
                capital = pnl + (position * exit_price)
                ret_pct = (exit_price - entry_price) / entry_price * 100
                all_trades.append({
                    'Symbol': symbol,
                    'Strategy': 'MA_Crossover',
                    'Entry Date': entry_date,
                    'Entry Price': entry_price,
                    'Exit Date': exit_date,
                    'Exit Price': exit_price,
                    'Return %': round(ret_pct, 2)
                })
                position = 0

            equity.append(capital + (position * close if position > 0 else 0))

        except Exception as e:
            print(f"WARNING: Skipping row in MA strategy: {e}")
            equity.append(capital)

    return pd.Series(equity, index=df.index)


# === Buy & Hold Strategy ===
def buy_and_hold_strategy(symbol):
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty or df.shape[0] < 2:
        print(f"WARNING: Skipping {symbol} — no data.")
        return None

    try:
        first_open = df.iloc[0]['Open']
        shares = capital_per_symbol / first_open
        equity = df['Close'] * shares
        return equity
    except Exception as e:
        print(f"WARNING: Error in Buy & Hold for {symbol}: {e}")
        return None

# === Portfolio Containers ===
portfolio_breakout = pd.DataFrame()
portfolio_ma = pd.DataFrame()
portfolio_hold = pd.DataFrame()

# === Run strategies per symbol ===
for _, row in df_summary.iterrows():
    symbol = row['Symbol']
    try:
        bw = int(row['Best Return (BW, ATR)'].split(',')[0].strip('('))
        atr = float(row['Best Return (BW, ATR)'].split(',')[1].strip(')'))
    except Exception as e:
        print(f"ERROR: Couldn't parse best params for {symbol}: {e}")
        continue

    bo = breakout_strategy(symbol, bw, atr)
    ma = ma_crossover_strategy(symbol)
    bh = buy_and_hold_strategy(symbol)

    if bo is not None:
        portfolio_breakout[symbol] = bo
    if ma is not None:
        portfolio_ma[symbol] = ma
    if bh is not None:
        portfolio_hold[symbol] = bh
        
    if bo is not None:
        portfolio_breakout[symbol] = bo
        bo.to_csv(f"exports/individual_logs/breakout/{symbol}_breakout.csv")

    if ma is not None:
        portfolio_ma[symbol] = ma
        ma.to_csv(f"exports/individual_logs/ma_crossover/{symbol}_ma_crossover.csv")

    if bh is not None:
        portfolio_hold[symbol] = bh
        bh.to_csv(f"exports/individual_logs/buy_hold/{symbol}_buy_hold.csv")
    

# === Combine portfolios ===
portfolio_breakout['Total'] = portfolio_breakout.sum(axis=1)
portfolio_ma['Total'] = portfolio_ma.sum(axis=1)
portfolio_hold['Total'] = portfolio_hold.sum(axis=1)

df = pd.concat([
    portfolio_breakout['Total'],
    portfolio_ma['Total'],
    portfolio_hold['Total']
], axis=1)
df.columns = ['Breakout', 'MA_Crossover', 'BuyHold']
df.dropna(inplace=True)

# === Performance Statistics ===
def strategy_stats(series):
    ret = (series.iloc[-1] / series.iloc[0]) - 1
    r = series.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    max_dd = (series.cummax() - series).max() / series.cummax().max()
    return ret, sharpe, max_dd

# === Output and Plot ===
if df.empty or df.shape[0] < 2:
    print("No valid data available to compute performance metrics.")
else:
    br_ret, br_sharpe, br_dd = strategy_stats(df['Breakout'])
    ma_ret, ma_sharpe, ma_dd = strategy_stats(df['MA_Crossover'])
    bh_ret, bh_sharpe, bh_dd = strategy_stats(df['BuyHold'])

    print("\n=== Strategy Comparison ===")
    print(f"Breakout     | Return: {br_ret:.2%} | Sharpe: {br_sharpe:.2f} | Max Drawdown: {br_dd:.2%}")
    print(f"MA Crossover | Return: {ma_ret:.2%} | Sharpe: {ma_sharpe:.2f} | Max Drawdown: {ma_dd:.2%}")
    print(f"Buy & Hold   | Return: {bh_ret:.2%} | Sharpe: {bh_sharpe:.2f} | Max Drawdown: {bh_dd:.2%}")

    df.plot(figsize=(12, 6), title="Portfolio Equity Curve Comparison")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Save Combined Portfolio Equity Curves ===
df.to_csv("exports/equity_curves/portfolio_equity_curves.csv")
print("Saved portfolio equity curves to CSV.")

# === Export trade log ===

if all_trades:
    df_trades = pd.DataFrame(all_trades)
    os.makedirs("exports/trades", exist_ok=True)

    # Split by strategy and export
    for strategy_name in df_trades['Strategy'].unique():
        strategy_df = df_trades[df_trades['Strategy'] == strategy_name]
        strategy_df.to_csv(f"exports/trades/trade_log_{strategy_name.lower()}.csv", index=False)

    print("✅ Trade logs saved to exports/trades/")
    
# === Zip Exported Results ===
def zip_exported_results(zip_name="exports_results.zip"):
    export_dir = "exports"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, export_dir)
                zipf.write(file_path, arcname=os.path.join("exports", arcname))
    print(f"Zipped all export results to: {zip_name}")

# Call the function
zip_exported_results()

# === Analyze Trade Log Statistics ===
def analyze_trades(trades_df):
    summary = []

    for strategy in trades_df['Strategy'].unique():
        df = trades_df[trades_df['Strategy'] == strategy]
        total = len(df)
        wins = df[df['Return %'] > 0]
        losses = df[df['Return %'] <= 0]
        win_rate = len(wins) / total * 100 if total else 0
        avg_win = wins['Return %'].mean() if not wins.empty else 0
        avg_loss = losses['Return %'].mean() if not losses.empty else 0
        max_gain = df['Return %'].max()
        max_loss = df['Return %'].min()
        median_ret = df['Return %'].median()

        summary.append({
            'Strategy': strategy,
            'Total Trades': total,
            'Win Rate %': round(win_rate, 2),
            'Avg Win %': round(avg_win, 2),
            'Avg Loss %': round(avg_loss, 2),
            'Max Gain %': round(max_gain, 2),
            'Max Loss %': round(max_loss, 2),
            'Median Return %': round(median_ret, 2)
        })

    return pd.DataFrame(summary)

# Load trade log if not already in memory
if 'df_trades' not in globals():
    trade_files = glob.glob("exports/trades/trade_log_*.csv")

if not trade_files:
    print("No trade logs found to analyze.")
else:
    dfs = []
    for file in trade_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"ERROR reading {file}: {e}")

    if not dfs:
        print("Trade log files exist but are all empty.")
    else:
        df_trades = pd.concat(dfs, ignore_index=True)

        # === Analyze and Save Summary ===
        def analyze_trades(trades_df):
            summary = []
            for strategy in trades_df['Strategy'].unique():
                df = trades_df[trades_df['Strategy'] == strategy]
                total = len(df)
                wins = df[df['Return %'] > 0]
                losses = df[df['Return %'] <= 0]
                win_rate = len(wins) / total * 100 if total else 0
                avg_win = wins['Return %'].mean() if not wins.empty else 0
                avg_loss = losses['Return %'].mean() if not losses.empty else 0
                max_gain = df['Return %'].max()
                max_loss = df['Return %'].min()
                median_ret = df['Return %'].median()
                summary.append({
                    'Strategy': strategy,
                    'Total Trades': total,
                    'Win Rate %': round(win_rate, 2),
                    'Avg Win %': round(avg_win, 2),
                    'Avg Loss %': round(avg_loss, 2),
                    'Max Gain %': round(max_gain, 2),
                    'Max Loss %': round(max_loss, 2),
                    'Median Return %': round(median_ret, 2)
                })
            return pd.DataFrame(summary)

        df_stats = analyze_trades(df_trades)
        print("\nTrade Log Summary:\n")
        print(df_stats.to_string(index=False))
        df_stats.to_csv("exports/trades/trade_log_summary.csv", index=False)
        print("Saved trade summary to: exports/trades/trade_log_summary.csv")

# Run analysis
df_stats = analyze_trades(df_trades)

df_ranked = rank_symbols_strategies(df_trades)

# === Filter poor-performing symbol-strategy combinations ===
filtered = df_ranked[
    (df_ranked['Avg Return %'] < 0) | 
    (df_ranked['Total Trades'] > 3) & (df_ranked['Avg Return %'] < 1) |
    (df_ranked['Avg Return %'].isna()) |
    (df_ranked['Avg Return %'] < 0.01) |
    (df_ranked['Total Trades'] >= 2) & (df_ranked['Total Return %'] < 0)
]

if not filtered.empty:
    print("\nWeak Strategies (low win rate or avg return):\n")
    print(filtered.to_string(index=False))
    filtered.to_csv("exports/trades/flagged_weak_strategies.csv", index=False)
    print("Saved to: exports/trades/flagged_weak_strategies.csv")
else:
    print("No weak strategies detected under current filters.")


# Show and export
print("\nRanked Symbols + Strategies:\n")
print(df_ranked.to_string(index=False))
df_ranked.to_csv("exports/trades/ranked_symbol_strategy_summary.csv", index=False)
print("Saved ranked summary to: exports/trades/ranked_symbol_strategy_summary.csv")


# Output stats
print("\nTrade Log Summary:\n")
print(df_stats.to_string(index=False))

# Export summary to CSV
df_stats.to_csv("exports/trades/trade_log_summary.csv", index=False)
print("Saved trade summary to: exports/trades/trade_log_summary.csv")

def rank_symbols_strategies(trades_df):
    rankings = trades_df.copy()
    rankings['Total Return %'] = (
        (rankings['Exit Price'] - rankings['Entry Price']) / rankings['Entry Price'] * 100
    )

    grouped = rankings.groupby(['Symbol', 'Strategy']).agg({
        'Return %': ['count', 'mean'],
        'Total Return %': 'sum'
    })

    grouped.columns = ['Total Trades', 'Avg Return %', 'Total Return %']
    grouped = grouped.reset_index()
    grouped.sort_values(by='Total Return %', ascending=False, inplace=True)

    return grouped

# === Load and display latest live alerts ===
if latest_alerts := sorted(glob.glob("exports/live/alerts_*.csv"))[-1:]:
    df_live = pd.read_csv(latest_alerts[0])
    print("\nLive Alerts from Today:\n")
    print(df_live.to_string(index=False))

# === Load and display top strategies ===   
valid_strategies = pd.read_csv("exports/trades/ranked_symbol_strategy_summary.csv")
top_strategies = valid_strategies[valid_strategies['Avg Return %'] > 2]['Symbol'].unique().tolist()

# === Filter top strategies ===
symbols = top_strategies
