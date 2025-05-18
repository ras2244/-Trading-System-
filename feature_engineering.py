# data_pipeline/get_price_data.py

import pandas as pd
from datetime import datetime, timedelta
from questrade_api import search_symbol, get_candles

def get_price_data(symbol, days=90, interval="OneDay"):
    """
    Fetch historical price data for a given symbol and interval from Questrade.

    Parameters:
        symbol (str): Ticker symbol (e.g., "AAPL")
        days (int): Number of calendar days of history to retrieve
        interval (str): Questrade candle interval ("OneDay", "OneMinute", etc.)

    Returns:
        pd.DataFrame or None: DataFrame with OHLCV candles or None on failure
    """
    try:
        search_result = search_symbol(symbol)
        symbol_data = search_result.get("symbols", [])
        if not symbol_data:
            print(f"❌ Symbol not found: {symbol}")
            return None

        symbol_id = symbol_data[0].get("symbolId")
        if not symbol_id:
            print(f"❌ No symbol ID for {symbol}")
            return None

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        candles = get_candles(symbol_id, start_time=start_time, end_time=end_time, interval=interval)
        candle_data = candles.get("candles", [])
        if not candle_data:
            print(f"⚠️ No candle data for {symbol}")
            return None

        df = pd.DataFrame(candle_data)
        df["datetime"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
        return df

    except Exception as e:
        print(f"❌ Failed to fetch price data for {symbol}: {e}")
        return None
