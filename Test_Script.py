import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# === Backtest Function ===
def run_breakout_backtest(symbol, start_date, end_date, initial_capital, breakout_window, atr_factor, risk_pct=0.01):
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty or len(df) < max(breakout_window, 14) + 5:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df['20d_high'] = df['High'].rolling(window=breakout_window).max()
    df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()

    if '20d_high' not in df.columns or 'ATR' not in df.columns:
        return None

    df = df.dropna(subset=['20d_high', 'ATR'])
    if df.empty:
        return None

    capital = initial_capital
    position = 0
    entry_price = 0
    highest_price = 0
    equity_curve = []

    for _, row in df.iterrows():
        price = row['Open']

        if position == 0 and row['Close'] > row['20d_high']:
            entry_price = price
            position = (capital * risk_pct) / row['ATR']
            highest_price = price

        elif position > 0:
            highest_price = max(highest_price, row['High'])
            trailing_stop = highest_price - atr_factor * row['ATR']

            if row['Low'] <= trailing_stop:
                capital += position * (trailing_stop - entry_price)
                position = 0
            else:
                capital += position * (row['Close'] - entry_price)

        equity_curve.append(capital)

    if len(equity_curve) < 2:
        return None

    series = pd.Series(equity_curve)
    returns = series.pct_change().dropna()
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    return total_return, sharpe_ratio

# === Parameters ===
symbols = ['TSLA', 'AAPL', 'MSFT']
start_date = '2022-01-01'
end_date = '2024-12-31'
initial_capital = 100000
breakout_windows = [10, 20, 30]
atr_factors = [1.0, 1.5, 2.0]

# === Run Optimization for All Symbols ===
results = []

for symbol in symbols:
    for window in breakout_windows:
        for atr in atr_factors:
            result = run_breakout_backtest(symbol, start_date, end_date, initial_capital, window, atr)
            if result:
                total_return, sharpe = result
                results.append({
                    'symbol': symbol,
                    'breakout_window': window,
                    'atr_factor': atr,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe
                })

# === DataFrame and Aggregation ===
df_all = pd.DataFrame(results)
df_all.to_csv("multi_symbol_optimization.csv", index=False)

df_agg = df_all.groupby(['breakout_window', 'atr_factor']).agg({
    'total_return': 'mean',
    'sharpe_ratio': 'mean'
}).reset_index()

# === Total Return Heatmap with Contours ===
pivot_return = df_agg.pivot(index='breakout_window', columns='atr_factor', values='total_return')
best_ret = df_agg.loc[df_agg['total_return'].idxmax()]

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_return, annot=True, fmt=".2%", cmap="YlOrBr", linewidths=.5, cbar_kws={'label': 'Avg Total Return'})
X, Y = np.meshgrid(pivot_return.columns, pivot_return.index)
Z = pivot_return.values
CS = plt.contour(X, Y, Z, levels=5, colors='black', linewidths=1)
plt.clabel(CS, inline=True, fontsize=8, fmt=lambda x: f"{x:.2%}")
plt.plot(best_ret['atr_factor'], best_ret['breakout_window'], marker='o', color='lime', markersize=12, markeredgecolor='black', label='Best Avg Return')
plt.title("Avg Total Return Heatmap (All Symbols)")
plt.xlabel("ATR Factor")
plt.ylabel("Breakout Window")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# === Sharpe Ratio Heatmap with Contours ===
pivot_sharpe = df_agg.pivot(index='breakout_window', columns='atr_factor', values='sharpe_ratio')
best_sharpe = df_agg.loc[df_agg['sharpe_ratio'].idxmax()]

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_sharpe, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Avg Sharpe Ratio'})
X, Y = np.meshgrid(pivot_sharpe.columns, pivot_sharpe.index)
Z = pivot_sharpe.values
CS = plt.contour(X, Y, Z, levels=5, colors='black', linewidths=1)
plt.clabel(CS, inline=True, fontsize=8, fmt=lambda x: f"{x:.2f}")
plt.plot(best_sharpe['atr_factor'], best_sharpe['breakout_window'], marker='*', color='red', markersize=14, markeredgecolor='black', label='Best Avg Sharpe')
plt.title("Avg Sharpe Ratio Heatmap (All Symbols)")
plt.xlabel("ATR Factor")
plt.ylabel("Breakout Window")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# === Create folder for outputs
os.makedirs("symbol_results", exist_ok=True)

# === Save CSV per symbol
for symbol in df_all['symbol'].unique():
    df_symbol = df_all[df_all['symbol'] == symbol]
    csv_path = f"symbol_results/{symbol}_optimization.csv"
    df_symbol.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    
# === Per-symbol heatmaps ===
os.makedirs("symbol_heatmaps", exist_ok=True)

for symbol in df_all['symbol'].unique():
    df_symbol = df_all[df_all['symbol'] == symbol]

    # Pivot for return
    pivot_return = df_symbol.pivot(index='breakout_window', columns='atr_factor', values='total_return')
    pivot_sharpe = df_symbol.pivot(index='breakout_window', columns='atr_factor', values='sharpe_ratio')

    # === Total Return Heatmap ===
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot_return, annot=True, fmt=".2%", cmap="YlOrBr", linewidths=.5, cbar_kws={'label': 'Total Return'})
    plt.title(f"{symbol} – Total Return")
    plt.xlabel("ATR Factor")
    plt.ylabel("Breakout Window")
    plt.tight_layout()
    plt.savefig(f"symbol_heatmaps/{symbol}_total_return.png")
    plt.close()

    # === Sharpe Ratio Heatmap ===
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot_sharpe, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Sharpe Ratio'})
    plt.title(f"{symbol} – Sharpe Ratio")
    plt.xlabel("ATR Factor")
    plt.ylabel("Breakout Window")
    plt.tight_layout()
    plt.savefig(f"symbol_heatmaps/{symbol}_sharpe_ratio.png")
    plt.close()

    print(f"Heatmaps saved for {symbol}")
    
    # === Summary table of best parameters per symbol ===
summary_rows = []

for symbol in df_all['symbol'].unique():
    df_symbol = df_all[df_all['symbol'] == symbol]

    best_return_row = df_symbol.loc[df_symbol['total_return'].idxmax()]
    best_sharpe_row = df_symbol.loc[df_symbol['sharpe_ratio'].idxmax()]

    summary_rows.append({
        'Symbol': symbol,
        'Best Return (BW, ATR)': f"({int(best_return_row['breakout_window'])}, {best_return_row['atr_factor']})",
        'Return %': f"{best_return_row['total_return']:.2%}",
        'Best Sharpe (BW, ATR)': f"({int(best_sharpe_row['breakout_window'])}, {best_sharpe_row['atr_factor']})",
        'Sharpe': f"{best_sharpe_row['sharpe_ratio']:.2f}"
    })

df_summary = pd.DataFrame(summary_rows)
print("\nBest Parameters Summary Per Symbol:\n")
print(df_summary.to_string(index=False))

# Save to CSV
df_summary.to_csv("symbol_results/summary_best_params.csv", index=False)
print("\nSaved: symbol_results/summary_best_params.csv")
