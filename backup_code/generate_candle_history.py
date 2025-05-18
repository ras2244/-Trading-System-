import os
import pandas as pd
from questrade_api import search_symbol, get_candles

def save_recent_candles(symbols, days=30, output_dir="exports/candles"):
    os.makedirs(output_dir, exist_ok=True)

    for symbol in symbols:
        try:
            result = search_symbol(symbol)
            symbol_id = result["symbols"][0]["symbolId"]
            candles_data = get_candles(symbol_id, days=days)
            candles = candles_data.get("candles", [])

            if candles:
                df = pd.DataFrame(candles)
                df.to_csv(os.path.join(output_dir, f"{symbol}_candles.csv"), index=False)
                print(f"✅ Saved: {symbol}")
            else:
                print(f"⚠️ No candles found for {symbol}")

        except Exception as e:
            print(f"❌ Failed for {symbol}: {e}")

# Example usage
# save_recent_candles(["AAPL", "TSLA", "MSFT"])
