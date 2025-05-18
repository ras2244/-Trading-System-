import pandas as pd
import os
from utils.signal_dataset_builder import generate_training_dataset
from questrade_api import search_symbol, get_candles

def save_recent_candles_from_trades(trade_log_path, days=90, output_dir="exports/candles"):
    df = pd.read_csv(trade_log_path)
    symbols = df["Symbol"].dropna().unique().tolist()

    os.makedirs(output_dir, exist_ok=True)

    for symbol in symbols:
        try:
            result = search_symbol(symbol)
            symbol_id = result["symbols"][0]["symbolId"]
            candles_data = get_candles(symbol_id, days=days)
            candles = candles_data.get("candles", [])

            if candles:
                df_candles = pd.DataFrame(candles)
                df_candles.to_csv(os.path.join(output_dir, f"{symbol}_candles.csv"), index=False)
                print(f"✅ Saved: {symbol}")
            else:
                print(f"⚠️ No candles for {symbol}")
        except Exception as e:
            print(f"❌ Failed for {symbol}: {e}")

# Example usage:
save_recent_candles_from_trades("exports/trades/trade_log_breakout.csv")


generate_training_dataset(
    trade_log_path="exports/trades/trade_log_breakout.csv",
    candle_dir="exports/candles"
)

df = pd.read_csv("exports/trades/trade_log_breakout.csv")
print(df.head())
