import os
import pandas as pd

def generate_training_dataset(trade_log_path, candle_dir, output_path="exports/model_training_data.csv"):
    if not os.path.exists(trade_log_path):
        print("❌ Trade log not found.")
        return

    df_log = pd.read_csv(trade_log_path)
    df_log["Entry Date"] = pd.to_datetime(df_log["Entry Date"])

    rows = []

    for _, row in df_log.iterrows():
        symbol = row["Symbol"]
        pnl = row.get("PnL")
        if pd.isna(pnl): continue
        label = 1 if pnl > 0 else 0
        entry_date = pd.to_datetime(row["Entry Date"])

        candle_file = os.path.join(candle_dir, f"{symbol}_candles.csv")
        if not os.path.exists(candle_file):
            print(f"⚠️ Missing candles for {symbol}")
            continue

        df_candles = pd.read_csv(candle_file)
        df_candles["datetime"] = pd.to_datetime(df_candles["datetime"])
        df_candles.set_index("datetime", inplace=True)
        df_candles.sort_index(inplace=True)

        lookback = df_candles.loc[df_candles.index < entry_date].tail(5)
        if len(lookback) < 5:
            print(f"⏳ Not enough lookback for {symbol} at {entry_date}")
            continue

        features = {
            "symbol": symbol,
            "entry_date": entry_date,
            "label": label,
            "avg_volume": lookback["volume"].mean(),
            "avg_close": lookback["close"].mean(),
            "max_high": lookback["high"].max(),
            "min_low": lookback["low"].min(),
            "close_change_pct": (lookback["close"].iloc[-1] - lookback["close"].iloc[0]) / lookback["close"].iloc[0]
        }

        rows.append(features)

    if not rows:
        print("⚠️ No data collected. Check your trade log or candle files.")
        return

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    #print(f"✅ Training dataset saved to {output_path}")
