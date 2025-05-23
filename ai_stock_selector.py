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
from questrade_api import get_symbol_id_by_ticker, get_access_token, get_quote_data
#from questrade_api import get_all_symbols_with_cache
#from utils.get_candles import ensure_candle_file
#from utils.model_utils import generate_training_data, save_model, load_model, FEATURE_COLUMNS, extract_features,get_sp500_tickers
from utils.model_utils import generate_training_data, save_model, load_model, FEATURE_COLUMNS, extract_features, add_technical_indicators

from utils.price_data import get_price_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
#from premarket_scanner import scan_premarket_gainers
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurable Parameters
NUM_AI_STOCKS = 4
NUM_JOURNAL_STOCKS = 1
MODEL_PATH = "exports/stock_selector_model.pkl"
TRADE_LOG_PATH = "exports/trades/trade_log_breakout.csv"
CANDLE_DIR = "exports/candles"
csv_path = "data/sp500_constituents.csv"

def load_sp500_from_csv(csv_path="data/sp500_constituents.csv"):
    df = pd.read_csv(csv_path)
    tickers = df["Symbol"].dropna().unique().tolist()
    return [ticker.strip().upper() for ticker in tickers]

def load_watchlist_top_liquid(limit=100, min_volume=1_000_000, min_price=5.0):
    access_token, api_server = get_access_token()
    tickers = load_sp500_from_csv()

    print(f"🔍 Checking quotes for {len(tickers)} tickers...")
    liquid = []
    for ticker in tickers:
        try:
            quote = get_quote_data(ticker, access_token, api_server)
            volume = quote.get("volume")
            price = quote.get("lastTradePrice")
            if volume is not None and price is not None and volume >= min_volume and price >= min_price:
                symbol_id = get_symbol_id_by_ticker(ticker, access_token, api_server)
                if symbol_id:
                    liquid.append({"symbol": ticker, "symbolId": symbol_id})
        except Exception as e:
            print(f"⚠️ Skipping {ticker}: {e}")

    print(f"✅ Final watchlist contains {len(liquid)} symbols.")
    return liquid[:limit]


# Placeholder journal targets
def score_journal_targets():
    return [{"symbol": "NFLX", "score": 0.5}]


def select_stocks_for_today(return_watchlist=False):
    watchlist = load_watchlist_top_liquid()
    model = load_model(filename="stock_selector_model.pkl")
    need_training = False

    if model is None or not hasattr(model, "predict"):
        need_training = True
    else:
        try:
            model.predict([[0]*len(FEATURE_COLUMNS)])
        except NotFittedError:
            need_training = True

    if need_training:
        df_train = generate_training_data(watchlist, days=90)
        if df_train.empty or df_train["label"].nunique() < 2:
            return []
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df_train[FEATURE_COLUMNS], df_train["label"])
        save_model(model, filename="stock_selector_model.pkl")

    candidates = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {executor.submit(realtime_score_symbol, s, model): s for s in watchlist}
        for future in as_completed(future_to_symbol):
            result = future.result()
            if result and result["score"] is not None:
                candidates.append(result)

    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:NUM_AI_STOCKS]
    return (sorted_candidates, watchlist) if return_watchlist else sorted_candidates


# Score a symbol live using current candle data
def realtime_score_symbol(symbol_obj, model):
    try:
        ticker = symbol_obj["symbol"]
        symbol_id = symbol_obj.get("symbolId", None)

        df = get_price_data(symbol_id or ticker, days=90)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 30:
            print(f"⚠️ Not enough data for {ticker}")
            return {"symbol": ticker, "score": None}

        # Add technical indicators
        
        df = add_technical_indicators(df)

        # Ensure all required columns exist (for models expecting these)
        required_cols = [
            'MA20', 'STD20', 'UpperBB', 'LowerBB', 'EMA12', 'EMA26',
            'MACD', 'Signal', 'rsi', 'vwap'
        ]
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️ {ticker} missing column: {col}")
                df[col] = np.nan  # fallback to NaN

        row = extract_features(df)
        if row is None:
            print(f"❌ Skipping {ticker} - feature extraction returned None")
            return {"symbol": ticker, "score": None}

        # Check for missing model features
        X = pd.DataFrame([row])
        missing_cols = [col for col in FEATURE_COLUMNS if col not in X.columns]
        if missing_cols:
            print(f"⚠️ Missing model input columns for {ticker}: {missing_cols}")
            return {"symbol": ticker, "score": None}

        # Predict
        try:
            X = X[FEATURE_COLUMNS]
            print(f"🔬 Model input: {X}")
            proba_array = model.predict_proba(X)[0]
            proba = proba_array[1] if len(proba_array) > 1 else proba_array[0]
            return {
                "symbol": ticker,
                "score": proba,
                "features": row  # optional, useful for logging or analysis
            }
        except Exception as e:
            print(f"❌ Prediction failed for {ticker}: {e}")
            return {"symbol": ticker, "score": None}
    except Exception as e:
        print(f"⚠️ Realtime scoring failed for {symbol_obj}: {e}")
        return {"symbol": symbol_obj['symbol'], "score": None}

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
#def show_selected_stock_charts():
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




