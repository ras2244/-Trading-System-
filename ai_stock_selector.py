# ai_stock_selector.py

"""
This module selects stocks daily for day trading using AI feature generation
and journal analysis, learning over time using backtest evaluation.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import time
from questrade_api import get_symbol_id_by_ticker, get_access_token
#from questrade_api import get_all_symbols_with_cache
#from utils.get_candles import ensure_candle_file
from utils.model_utils import generate_training_data, save_model, load_model, FEATURE_COLUMNS, extract_features
from utils.price_data import get_price_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# Configurable Parameters
NUM_AI_STOCKS = 4
NUM_JOURNAL_STOCKS = 1
MODEL_PATH = "exports/stock_selector_model.pkl"
TRADE_LOG_PATH = "exports/trades/trade_log_breakout.csv"
CANDLE_DIR = "exports/candles"


# Load list of watchlist hardcoded symbols
def load_watchlist():
    access_token, api_server = get_access_token()
    tickers = ["AAPL", "MSFT", "GOOG", "NVDA", "TSLA"]
    watchlist = []
    for ticker in tickers:
        symbol_id = get_symbol_id_by_ticker(ticker, access_token, api_server)
        if symbol_id:
            watchlist.append({"symbol": ticker, "symbolId": symbol_id})
        else:
            print(f"❌ Could not find symbolId for {ticker}")
    return watchlist

# Placeholder journal targets
def score_journal_targets():
    return [{"symbol": "NFLX", "score": 0.5}]

# Select top-ranked stocks for today
def select_stocks_for_today():
    #model_path = "exports/stock_selector_model.pkl"
    watchlist = load_watchlist()
    model = load_model(filename="stock_selector_model.pkl")

    need_training = False

    if model is None:
        print("⚠️ Model file not found. Will train a new one.")
        need_training = True
    else:
        try:
            model.predict([[0]*len(FEATURE_COLUMNS)])
        except NotFittedError:
            print("⚠️ Model is not fitted. Will train a new one.")
            need_training = True
        except Exception as e:
            print(f"⚠️ Error checking model: {e}. Will retrain.")
            need_training = True

    if need_training:
        df_train = generate_training_data(watchlist, days=90)
        if df_train["label"].nunique() < 2:
            print("❌ Not enough class variety in training data. Only one class present.")
            print(df_train["label"].value_counts())
            return []
        if df_train.empty:
            print("❌ No training data available. Model training skipped.")
            return []
        # ✅ Defensive fix: avoid model training if only one label
        if "label" not in df_train.columns or df_train["label"].nunique() < 2:
            print("❌ Not enough variety in training labels. Found:")
            print(df_train["label"].value_counts())
            return []
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df_train[FEATURE_COLUMNS], df_train["label"])
        save_model(model, filename="stock_selector_model.pkl")
        print("✅ Model trained and saved.")

    candidates = []
    for symbol in watchlist:
        result = realtime_score_symbol(symbol, model)
        if result["score"] is not None:
            candidates.append(result)

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:NUM_AI_STOCKS]

# Score a symbol live using current candle data
def realtime_score_symbol(symbol_obj, model):
    try:
        ticker = symbol_obj["symbol"]
        symbol_id = symbol_obj["symbolId"]
        df = get_price_data(symbol_id, days=90)
        if df is None:
            print(f"❌ {ticker}: get_price_data() returned None")
            return {"symbol": ticker, "score": None}

        print(f"✅ {ticker}: Received {len(df)} rows")
        print(df.head(3).to_string(index=False))  # Show first few rows
        print(df.tail(3).to_string(index=False))  # Show last few rows

        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 10:
            print(f"⚠️ Not enough data for {ticker} (received {len(df)} rows)")
            return {"symbol": ticker, "score": None}

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

             
        row = extract_features(df)
        if row is None:
            print(f"❌ Skipping {ticker} - feature extraction returned None")
            return {"symbol": ticker, "score": None}

        X = pd.DataFrame([row])
        print(f"🔬 Model input: {X}")
        X = pd.DataFrame([row])[FEATURE_COLUMNS]  # this will fail if any column missing
        missing_cols = [col for col in FEATURE_COLUMNS if col not in row]
        if missing_cols:
            print(f"⚠️ Missing columns for {ticker}: {missing_cols}")
            return {"symbol": ticker, "score": None}

        try:
            proba_array = model.predict_proba(X)[0]
            if len(proba_array) == 2:
                proba = proba_array[1]  # Use P(label=1)
                return {"symbol": ticker, "score": proba}
            else:
                print(f"⚠️ Only one class in model. Probabilities: {proba_array}")
                return {"symbol": ticker, "score": proba_array[0]}  # Only class available
        except Exception as e:
            print(f"❌ Prediction failed for {ticker}: {e}")
            return {"symbol": ticker, "score": None}
    except Exception as e:
        print(f"⚠️ Realtime scoring failed for {symbol_obj}: {e}")
        return {"symbol": symbol_obj["symbol"], "score": None}

# Chart rendering for Streamlit
def plot_stock_chart(df, symbol):
    try:
        if not isinstance(df, pd.DataFrame):
            print(f"{symbol}: df is not a DataFrame")
            return None

        if "datetime" not in df.columns:
            if "start" in df.columns:
                df["datetime"] = pd.to_datetime(df["start"], utc=True)
            else:
                print(f"{symbol}: No 'start' column to create datetime")
                return None

        if "close" not in df.columns:
            print(f"{symbol}: No 'close' column")
            return None

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["datetime", "close"])

        if df.empty:
            print(f"{symbol}: DataFrame is empty after cleaning")
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["datetime"], df["close"], label="Close Price")
        ax.set_title(f"{symbol} - Close Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Plot failed for {symbol}: {e}")
        return None



# Display selected stocks in Streamlit
def show_selected_stock_charts():
    selected = select_stocks_for_today()
    st.header("📊 Selected Stock Charts")
    for stock in selected:
        symbol = stock["symbol"]
        st.subheader(f"{symbol} (AI Score: {stock['score']:.2f})")
        fig = plot_stock_chart(symbol)
        if fig:
            st.pyplot(fig)
        else:
            st.warning(f"Chart unavailable for {symbol}.")

# Score live list in Streamlit
def realtime_monitor(symbols):
    st.header("⏱️ Real-Time AI Signal Scores")
    model = load_model(MODEL_PATH)
    if model is None:
        st.error("Model not available. Please train first.")
        return

    for symbol in symbols:
        result = realtime_score_symbol(symbol, model)
        if result["score"] is not None:
            st.metric(label=f"{symbol}", value=f"{result['score']:.2f}")
        else:
            st.warning(f"{symbol} - score unavailable")

# Standalone CLI test
if __name__ == "__main__":
    today = select_stocks_for_today()
    print("Selected stocks:", today)
    for stock in today:
        fig = plot_stock_chart(stock["symbol"])
        if fig:
            fig.show()

