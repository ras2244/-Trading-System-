import streamlit as st
import pandas as pd
from datetime import datetime
from utils.train_signal_model import generate_training_dataset, train_and_save_model
from utils.signal_scorer import score_features_live
from questrade_api import search_symbol, get_candles, get_account_info, get_positions
from ai_stock_selector import select_stocks_for_today

st.set_page_config(page_title="AI Bot Manager", layout="wide")
st.title("🤖 AI Strategy & Bot Control")

# Questrade sidebar integration
st.sidebar.header("📡 Questrade Account Snapshot")
try:
    account_number = get_account_info()["accounts"][0]["number"]
    positions = get_positions(account_number)["positions"]
    df_account = pd.DataFrame(positions)[["symbol", "openQuantity", "currentMarketValue"]]
    df_account.columns = ["Symbol", "Quantity", "Market Value"]
    st.sidebar.dataframe(df_account)
except Exception as e:
    st.sidebar.warning(f"Could not load Questrade positions: {e}")

# Strategy assignment (placeholder, can be expanded)
st.sidebar.header("📊 Strategy Assignment")
strategy_options = ["Breakout", "MA Crossover", "RSI"]
strategy_map = {}

# Load selected stocks using AI selector module
selected_stocks = select_stocks_for_today()
st.subheader("📈 Today's AI-Selected Stocks")
for stock in selected_stocks:
    st.write(f"✅ {stock['symbol']} (score: {stock['score']:.2f})")

symbols = [s["symbol"] for s in selected_stocks]

for symbol in symbols:
    strategy = st.sidebar.selectbox(f"Strategy for {symbol}", strategy_options, key=f"strategy_{symbol}")
    strategy_map[symbol] = strategy

# Strategy parameters
st.sidebar.header("🔧 Strategy Parameters")
ma_fast = st.sidebar.slider("Fast MA", 2, 20, 5)
ma_slow = st.sidebar.slider("Slow MA", 10, 50, 20)
rsi_window = st.sidebar.slider("RSI Window", 2, 50, 14)

# Live signal monitoring + scoring (preview only)
st.header("📡 Live Signal Monitor with AI Score")
for symbol in symbols:
    try:
        symbol_id = search_symbol(symbol)["symbols"][0]["symbolId"]
        candle_data = get_candles(symbol_id)
        df = score_features_live(candle_data.get("candles", []))

        st.subheader(symbol)
        if df is not None and not df.empty:
            st.write(df.tail(1))
        else:
            st.warning("No recent features to score.")
    except Exception as e:
        st.error(f"Failed for {symbol}: {e}")

# Model Training Section
st.sidebar.header("🧠 AI Model Controls")
if st.sidebar.button("📊 Generate Training Dataset"):
    try:
        generate_training_dataset()
        st.sidebar.success("Training dataset generated.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if st.sidebar.button("🧠 Train Signal Model"):
    try:
        train_and_save_model()
        st.sidebar.success("Model trained and saved.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

