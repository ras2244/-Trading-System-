import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime

MODEL_PATH = "exports/signal_scoring_model.pkl"

def score_trade_signal(df_candles, model_path="exports/signal_scoring_model.pkl", context="entry"):
    model = joblib.load(model_path)
    print("Model expects:", model.feature_names_in_)
    window = df_candles.tail(10)
    if len(window) < 5:
        return None

    features = {
        "avg_close": window["close"].mean(),
        "volatility": window["close"].std(),
        "momentum": window["close"].iloc[-1] - window["close"].iloc[0],
        "volume_avg": window["volume"].mean(),
    }

    if context == "exit" and hasattr(model, "feature_names_in_") and "exit_avg_close" in model.feature_names_in_:
        features.update({
            "exit_avg_close": features["avg_close"],
            "exit_volatility": features["volatility"],
            "exit_momentum": features["momentum"],
            "exit_volume_avg": features["volume_avg"]
        })

    X = pd.DataFrame([features])
    expected_cols = list(model.feature_names_in_)
    X = X.reindex(columns=expected_cols)

    # Ensure all feature values are plain floats (not np.float64)
    X = X.astype(np.float64)

    proba = model.predict_proba(X)[0]

    #print("X.columns:", X.columns.tolist())
    #print("X.head():", X.head())

    if hasattr(model, "classes_"):
        if 1 in model.classes_:
            idx = list(model.classes_).index(1)
            return proba[idx]
        else:
            return None
    else:
        return None

def auto_exit_trades(trade_log_path="exports/trades/trade_log_breakout.csv", candle_dir="exports/candles", ai_threshold=0.3):
    df_trades = pd.read_csv(trade_log_path)
    df_trades["Entry Date"] = pd.to_datetime(df_trades["Entry Date"], format='mixed', errors='coerce')
    df_trades["Exit Date"] = pd.to_datetime(df_trades["Exit Date"], format='mixed', errors='coerce')


    open_trades = df_trades[df_trades["Exit Date"].isna()]

    for idx, row in open_trades.iterrows():
        symbol = row["Symbol"]
        candle_file = os.path.join(candle_dir, f"{symbol}_candles.csv")
        if not os.path.exists(candle_file):
            continue

        df_candles = pd.read_csv(candle_file)
        df_candles["datetime"] = pd.to_datetime(df_candles["start"], utc=True)
        df_candles.set_index("datetime", inplace=True)

        score = score_trade_signal(df_candles, context="exit")

        if score is not None and score < ai_threshold:
            exit_price = df_candles["close"].iloc[-1]
            df_trades.at[idx, "Exit Date"] = datetime.now().isoformat()
            df_trades.at[idx, "Exit Price"] = exit_price
            df_trades.at[idx, "PnL"] = round(exit_price - float(row["Entry Price"]), 2)
            print(f"\U0001f6aa Auto-exit triggered for {symbol} | Score: {score:.2f}")

    df_trades.to_csv(trade_log_path, index=False)
    #print("\u2705 Auto-exit logic executed and trade log updated.")




def score_features_live(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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

    window = df.tail(10)
    if len(window) < 5:
        return None

    try:
        close = window["close"]
        features = {
            "avg_close": float(close.mean()),
            "volatility": float(close.std()),
            "momentum": float(close.iloc[-1] - close.iloc[0]),
            "volume_avg": float(window["volume"].mean()),
            "delta_1d": float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]),
            "delta_5d": float((close.iloc[-1] - close.iloc[0]) / close.iloc[0]),
            "bb_width": float(window["bb_width"].iloc[-1]),
            "macd_hist": float(window["macd_hist"].iloc[-1]),
            "atr": float(window["atr"].iloc[-1])
        }
        return pd.DataFrame([features])
    except Exception:
        return None

def compute_technical_indicators(df):
    df = df.copy()
    df["returns"] = df["close"].pct_change()

    # Bollinger Band Width
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["std20"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["ma20"] + 2 * df["std20"]
    df["bb_lower"] = df["ma20"] - 2 * df["std20"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma20"]

    # MACD Histogram
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["signal"]

    # ATR (Average True Range)
    df["h-l"] = df["high"] - df["low"]
    df["h-c"] = abs(df["high"] - df["close"].shift())
    df["l-c"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["h-l", "h-c", "l-c"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=14).mean()

    # Momentum (n-day change)
    df["momentum"] = df["close"] - df["close"].shift(10)

    # Volatility
    df["volatility"] = df["returns"].rolling(window=10).std()

    # Delta (close diff from previous)
    df["delta"] = df["close"].diff()

    return df

# Placeholder: list of all tickers from Questrade or a local watchlist