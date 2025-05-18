# utils/get_candles.py

import os
import pandas as pd
from questrade_api import search_symbol, get_candles

def ensure_candle_file(symbol, days=90):
    path = f"exports/candles/{symbol}_candles.csv"
    if os.path.exists(path):
        print(f"✅ Candle file already exists for {symbol}")
        return

    try:
        os.makedirs("exports/candles", exist_ok=True)

        result = search_symbol(symbol)
        symbol_data = result.get("symbols", [])
        if not symbol_data:
            print(f"❌ No symbols returned for {symbol}")
            return

        symbol_id = symbol_data[0].get("symbolId")
        if not symbol_id:
            print(f"❌ Symbol ID missing for {symbol}")
            return

        print(f"🔍 Fetching candles for {symbol} using ID {symbol_id}")
        candle_data = get_candles(symbol_id, days=days)
        candles = candle_data.get("candles", [])
        if not candles:
            print(f"⚠️ No candles returned for {symbol}")
            return

        df = pd.DataFrame(candles)
        df.to_csv(path, index=False)
        print(f"✅ Saved candles for {symbol} to {path}")
    except Exception as e:
        print(f"❌ Failed to fetch candles for {symbol}: {e}")

