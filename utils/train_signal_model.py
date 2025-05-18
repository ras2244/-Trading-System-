import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def generate_training_dataset(
    trade_log_path="exports/trades/trade_log_breakout.csv",
    candle_dir="exports/candles",
    output_path="exports/model_training_data.csv"
):
    df_trades = pd.read_csv(trade_log_path)
    rows = []

    for _, row in df_trades.iterrows():
        symbol = row["Symbol"]
        entry_time = pd.to_datetime(row["Entry Date"], utc=True)
        pnl = float(row["PnL"]) if "PnL" in row and row["PnL"] != "" else 0
        label = int(pnl > 0)

        candle_file = os.path.join(candle_dir, f"{symbol}_candles.csv")
        if not os.path.exists(candle_file):
            continue

        df_candles = pd.read_csv(candle_file)
        df_candles["datetime"] = pd.to_datetime(df_candles["start"], utc=True)
        df_candles.set_index("datetime", inplace=True)

        # Add indicators
        df_candles["MA20"] = df_candles["close"].rolling(window=20).mean()
        df_candles["STD20"] = df_candles["close"].rolling(window=20).std()
        df_candles["UpperBB"] = df_candles["MA20"] + 2 * df_candles["STD20"]
        df_candles["LowerBB"] = df_candles["MA20"] - 2 * df_candles["STD20"]
        df_candles["bb_width"] = df_candles["UpperBB"] - df_candles["LowerBB"]

        df_candles["EMA12"] = df_candles["close"].ewm(span=12, adjust=False).mean()
        df_candles["EMA26"] = df_candles["close"].ewm(span=26, adjust=False).mean()
        df_candles["MACD"] = df_candles["EMA12"] - df_candles["EMA26"]
        df_candles["Signal"] = df_candles["MACD"].ewm(span=9, adjust=False).mean()
        df_candles["macd_hist"] = df_candles["MACD"] - df_candles["Signal"]

        df_candles["high_low"] = df_candles["high"] - df_candles["low"]
        df_candles["high_close"] = abs(df_candles["high"] - df_candles["close"].shift())
        df_candles["low_close"] = abs(df_candles["low"] - df_candles["close"].shift())
        df_candles["tr"] = df_candles[["high_low", "high_close", "low_close"]].max(axis=1)
        df_candles["atr"] = df_candles["tr"].rolling(window=14).mean()

        try:
            nearest_idx = df_candles.index.get_indexer([entry_time], method="nearest")[0]
            window = df_candles.iloc[max(0, nearest_idx - 10):nearest_idx]
        except Exception:
            continue

        if len(window) < 5:
            continue

        close = window["close"]
        features = {
            "avg_close": close.mean(),
            "volatility": close.std(),
            "momentum": close.iloc[-1] - close.iloc[0],
            "volume_avg": window["volume"].mean(),
            "delta_1d": ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) if len(close) > 1 else 0,
            "delta_5d": ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) if len(close) > 4 else 0,
            "bb_width": window["bb_width"].iloc[-1],
            "macd_hist": window["macd_hist"].iloc[-1],
            "atr": window["atr"].iloc[-1],
            "label": label
        }

        rows.append(features)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)

    #print(f"✅ Training dataset saved to: {output_path}")
    #print(f"📦 Total rows generated: {len(df_out)}")
    #if not df_out.empty:
        #print("📊 Sample output:")
        #print(df_out.head(5))


def train_and_save_model(input_path="exports/model_training_data.csv", model_path="exports/signal_scoring_model.pkl"):
    df = pd.read_csv(input_path)
    df = df.dropna()

    if df.empty:
        print("⚠️ No data available after cleaning. Training aborted.")
        return

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    #print(f"✅ Model trained and saved to: {model_path}")

    y_pred = model.predict(X_test)
    #print("📊 Classification Report:")
    #print(classification_report(y_test, y_pred))


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs.iloc[-1])) if not rs.isna().all() else np.nan


def compute_technical_indicators(df):
    df = df.copy()
    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["std"] = df["close"].rolling(window=20).std()

    df["bollinger_bw"] = (2 * df["std"]) / df["close"]

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = abs(df["high"] - df["close"].shift(1))
    df["L-PC"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=14).mean()

    df["ROC"] = df["close"].pct_change(periods=5)

    return df


if __name__ == "__main__":
    generate_training_dataset()
