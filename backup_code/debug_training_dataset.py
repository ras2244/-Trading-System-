import os
import pandas as pd
from datetime import datetime

trade_log_path = "exports/trades/trade_log_breakout.csv"
candle_dir = "exports/candles"

df_trades = pd.read_csv(trade_log_path)
print(f"📘 Total trades in log: {len(df_trades)}\n")

for i, row in df_trades.iterrows():
    symbol = row["Symbol"]
    entry_time = pd.to_datetime(row["Entry Date"], utc=True)
    exit_time = pd.to_datetime(row["Exit Date"], utc=True) if pd.notna(row["Exit Date"]) and row["Exit Date"] else None
    pnl = float(row["PnL"]) if "PnL" in row and row["PnL"] != "" else 0
    label = int(pnl > 0)

    candle_file = os.path.join(candle_dir, f"{symbol}_candles.csv")

    if not os.path.exists(candle_file):
        print(f"❌ Missing candle file for {symbol}")
        continue

    df_candles = pd.read_csv(candle_file)
    df_candles["datetime"] = pd.to_datetime(df_candles["start"], utc=True)
    df_candles.set_index("datetime", inplace=True)

    if entry_time not in df_candles.index:
        nearest_idx = df_candles.index.get_indexer([entry_time], method="nearest")[0]
        if nearest_idx == -1:
            print(f"⚠️ No nearby candle for {symbol} at {entry_time}")
            continue

    try:
        window = df_candles.loc[:entry_time].tail(20)
        if window.empty or len(window) < 5:
            print(f"⚠️ Not enough candle data for {symbol} before entry")
            continue

        features = {
            "avg_close": window["close"].mean(),
            "volatility": window["close"].std(),
            "momentum": window["close"].iloc[-1] - window["close"].iloc[0],
            "volume_avg": window["volume"].mean(),
        }
        print(f"✅ {symbol} | Features extracted successfully")
    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
