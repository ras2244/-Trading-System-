#model_utils.py

import os
import pandas as pd
import joblib
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from utils.price_data import get_price_data
from questrade_api import get_all_symbols_with_cache
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


# Define feature columns used in model training and prediction
FEATURE_COLUMNS = [
    "MA20", "STD20", "UpperBB", "LowerBB", "bb_width",
    "EMA12", "EMA26", "MACD", "Signal", "macd_hist",
    "atr", "rsi", "vwap", "volatility"
]

def save_model(model, filename="stock_selector_model.pkl"):
    os.makedirs("exports", exist_ok=True)
    path = os.path.join("exports", filename)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def load_model(filename="stock_selector_model.pkl"):
    path = os.path.join("exports", filename)
    if os.path.exists(path):
        print(f"✅ Model loaded from {path}")
        return joblib.load(path)
    print("⚠️ Model file not found.")
    return None

def get_recent_daily_data(symbol_id, days=90):
    end = datetime.now()
    start = end - timedelta(days=days)
    return get_price_data(symbol_id, days=days, interval="OneDay")

def get_symbol_id(symbol, all_symbols=None):
    if all_symbols is None:
        all_symbols = get_all_symbols_with_cache()
    for entry in all_symbols:
        if entry.get("symbol", "").upper() == symbol.upper():
            return entry.get("symbolId")
    print(f"⚠️ Symbol ID not found for ticker: {symbol}")
    return None

def add_technical_indicators(df):
    if "datetime" not in df.columns and "start" in df.columns:
        df["datetime"] = pd.to_datetime(df["start"], errors="coerce", utc=True)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        if df["datetime"].notna().sum() > 0:
            df = df.set_index("datetime")
            df = df.sort_index()
        else:
            print("⚠️ datetime column exists but is all NaT, skipping index set.")
    else:
        print("⚠️ 'datetime' column missing from DataFrame before technical indicators.")

    df["MA20"] = df["close"].rolling(window=20).mean()
    df["STD20"] = df["close"].rolling(window=20).std()
    df["UpperBB"] = df["MA20"] + 2 * df["STD20"]
    df["LowerBB"] = df["MA20"] - 2 * df["STD20"]
    df["bb_width"] = df["UpperBB"] - df["LowerBB"]

    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["MACD"] - df["Signal"]

    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=14).mean()

    try:
        df["rsi"] = ta.rsi(df["close"], length=14)
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(pd.to_datetime(df.index, utc=True))
        df["vwap"] = ta.vwap(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"])
    except Exception as e:
        print(f"❌ VWAP/RSI computation failed: {e}")

    try:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_return"].rolling(window=20).std()
    except Exception as e:
        print(f"❌ Volatility calculation failed: {e}")

    return df


def generate_training_data(df, forward_days=4, return_threshold=0.01):
    df = add_technical_indicators(df)
    try:
        df["future_return"] = df["close"].shift(-forward_days) / df["close"] - 1
        df["label"] = (df["future_return"] > return_threshold).astype(int)
    except Exception as e:
        print(f"❌ Label generation failed: {e}")
    return df.dropna().copy()

def score_live_data(df, model_path="exports/stock_selector_model.pkl", return_full_row=False):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    df = add_technical_indicators(df)
    df = df.dropna()

    if df.empty:
        print("⚠️ No valid data after feature extraction.")
        return None

    try:
        X_live = df[FEATURE_COLUMNS]
    except KeyError as e:
        print(f"❌ Missing expected features: {e}")
        return None

    try:
        probs = model.predict_proba(X_live)[:, 1]  # Probability of class 1
        df["score"] = probs
        if return_full_row:
            return df.reset_index().tail(1).copy()
        return df[["score"]].tail(1)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def extract_features(df):
    try:
        if isinstance(df, list):
            df = pd.DataFrame(df)
        if df is None or df.empty or len(df) < 10:
            raise ValueError(f"Not enough data to extract features: got {len(df)} rows")
        
        df = add_technical_indicators(df)

        df["avg_close"] = df["close"].rolling(window=10).mean()
        df["volatility"] = df["close"].rolling(window=10).std()
        df["momentum"] = df["close"] - df["close"].shift(10)
        df["volume_avg"] = df["volume"].rolling(window=10).mean()
        df["delta_1d"] = df["close"].pct_change(periods=1)
        df["delta_5d"] = df["close"].pct_change(periods=5)

        # Debug preview
        # print(f"🧮 Feature input preview: close={df['close'].tail(10).tolist()}")

        return df

    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None

def get_sp500_tickers(csv_path="data/sp500_constituents.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please download the S&P 500 constituents file.")
    
    df = pd.read_csv(csv_path)
    if "Symbol" not in df.columns:
        raise ValueError("CSV must contain a 'Symbol' column.")

    tickers = df["Symbol"].dropna().unique().tolist()
    return sorted(tickers)