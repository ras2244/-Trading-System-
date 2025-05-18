import pandas as pd
def backtest_breakout_strategy(df, lookback=20, hold_period=5):
    trades = []
    cumulative_pnl = 0
    exit_pending_until = None
    last_entry_price = None

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)

    for i in range(lookback, len(df) - hold_period):
        now = df.index[i]
        if exit_pending_until and now < exit_pending_until:
            continue

        window = df.iloc[i - lookback:i]
        breakout_level = window["high"].max()
        current_price = df.iloc[i]["close"]

        if current_price > breakout_level:
            entry_date = df.index[i]
            entry_price = df.iloc[i]["close"]
            exit_date = df.index[i + hold_period]
            exit_price = df.iloc[i + hold_period]["close"]
            pnl = exit_price - entry_price
            cumulative_pnl += pnl

            trades.append({
                "Entry Date": entry_date,
                "Entry Price": entry_price,
                "Exit Date": exit_date,
                "Exit Price": exit_price,
                "PnL": pnl,
                "Strategy": "Breakout"
            })

            exit_pending_until = df.index[i + hold_period]
            last_entry_price = entry_price

    df_result = pd.DataFrame(trades)
    if not df_result.empty:
        df_result["Cumulative PnL"] = df_result["PnL"].cumsum()

    return df_result, df_result[["Exit Date", "Cumulative PnL"]].set_index("Exit Date") if not df_result.empty else pd.DataFrame()


def backtest_ma_crossover_strategy(df, fast_window=5, slow_window=20, hold_period=5):
    trades = []
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    df["MA_fast"] = df["close"].rolling(window=fast_window).mean()
    df["MA_slow"] = df["close"].rolling(window=slow_window).mean()

    exit_pending_until = None

    for i in range(slow_window, len(df) - hold_period):
        now = df.index[i]
        if exit_pending_until and now < exit_pending_until:
            continue

        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if prev["MA_fast"] < prev["MA_slow"] and curr["MA_fast"] >= curr["MA_slow"]:
            entry_date = df.index[i]
            entry_price = df.iloc[i]["close"]
            exit_date = df.index[i + hold_period]
            exit_price = df.iloc[i + hold_period]["close"]
            pnl = exit_price - entry_price

            trades.append({
                "Entry Date": entry_date,
                "Entry Price": entry_price,
                "Exit Date": exit_date,
                "Exit Price": exit_price,
                "PnL": pnl,
                "Strategy": "MA Crossover"
            })

            exit_pending_until = df.index[i + hold_period]

    df_result = pd.DataFrame(trades)
    if not df_result.empty:
        df_result["Cumulative PnL"] = df_result["PnL"].cumsum()

    return df_result, df_result[["Exit Date", "Cumulative PnL"]].set_index("Exit Date") if not df_result.empty else pd.DataFrame()


def backtest_rsi_strategy(df, rsi_window=14, hold_period=5):
    trades = []
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exit_pending_until = None

    for i in range(rsi_window, len(df) - hold_period):
        now = df.index[i]
        if exit_pending_until and now < exit_pending_until:
            continue

        if df.iloc[i]["RSI"] < 30:
            entry_date = df.index[i]
            entry_price = df.iloc[i]["close"]
            exit_date = df.index[i + hold_period]
            exit_price = df.iloc[i + hold_period]["close"]
            pnl = exit_price - entry_price

            trades.append({
                "Entry Date": entry_date,
                "Entry Price": entry_price,
                "Exit Date": exit_date,
                "Exit Price": exit_price,
                "PnL": pnl,
                "Strategy": "RSI"
            })

            exit_pending_until = df.index[i + hold_period]

    df_result = pd.DataFrame(trades)
    if not df_result.empty:
        df_result["Cumulative PnL"] = df_result["PnL"].cumsum()

    return df_result, df_result[["Exit Date", "Cumulative PnL"]].set_index("Exit Date") if not df_result.empty else pd.DataFrame()
