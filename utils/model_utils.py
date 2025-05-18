import os
import pandas as pd
import joblib
from datetime import datetime, timedelta
from utils.price_data import get_price_data
from questrade_api import get_all_symbols_with_cache

# Define feature columns used in model training and prediction
FEATURE_COLUMNS = [
    "avg_close", "volatility", "momentum",
    "volume_avg", "delta_1d", "delta_5d",
    "bb_width", "macd_hist", "atr"
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
    return df

def extract_features(df):
    try:
        if isinstance(df, list):
            df = pd.DataFrame(df)
        if df is None or df.empty or len(df) < 10:
            raise ValueError(f"Not enough data to extract features: got {len(df)} rows")
        df = add_technical_indicators(df)
        close = df["close"]
        print(f"🧮 Feature input preview: close={close[-10:].tolist()}")
        return {
            "avg_close": close[-10:].mean(),
            "volatility": close[-10:].std(),
            "momentum": close.iloc[-1] - close.iloc[-10] if len(close) >= 10 else 0,
            "volume_avg": df["volume"].iloc[-10:].mean() if len(df) >= 10 else 0,
            "delta_1d": (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) >= 2 else 0,
            "delta_5d": (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0,
            "bb_width": df["bb_width"].iloc[-1],
            "macd_hist": df["macd_hist"].iloc[-1],
            "atr": df["atr"].iloc[-1]
        }
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None

def generate_training_data(tickers, days=90):
    data = []
    for symbol_obj in tickers:
        try:
            symbol_id = symbol_obj["symbolId"]
            ticker = symbol_obj["symbol"]
            df = get_price_data(symbol_id, days=days)
            print("Label distribution:")
            print(df["label"].value_counts())
            print(df["label"].value_counts())
            if df is None or len(df) < 20:
                print(f"⚠️ Not enough data for {ticker}")
                continue
            features = extract_features(df)
            if features is None:
                continue
            future_return = (df["close"].iloc[-1] - df["close"].iloc[-4]) / df["close"].iloc[-4]
            label = 1 if future_return > 0.02 else 0
            data.append(features)           
        except Exception as e:
            print(f"⚠️ Skipped {ticker}: {e}")
    return pd.DataFrame(data)
