# screening_pipeline.py

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from questrade_api import get_all_symbols_with_cache, get_price_data

def screen_symbols(min_price=5, min_volume=100000, min_volatility=0.02, limit=100):
    """
    Load symbols from Questrade and apply basic filters.
    """
    symbols = get_all_symbols_with_cache()
    print(f"🔎 Screening {len(symbols)} symbols...")

    selected = []
    for symbol in symbols:
        try:
            df = get_price_data(symbol)
            if len(df) < 5:
                continue

            close_prices = df["close"]
            avg_price = close_prices[-5:].mean()
            avg_volume = df["volume"][-5:].mean()
            volatility = close_prices.pct_change().std()

            if avg_price >= min_price and avg_volume >= min_volume and volatility >= min_volatility:
                selected.append({
                    "symbol": symbol,
                    "price": avg_price,
                    "volume": avg_volume,
                    "volatility": volatility
                })

            if len(selected) >= limit:
                break
        except Exception as e:
            print(f"⚠️ Skipping {symbol}: {e}")

    print(f"✅ Selected {len(selected)} symbols after screening.")
    return selected
