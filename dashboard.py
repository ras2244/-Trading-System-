import csv
import streamlit as st
import pandas as pd
import glob
from joblib import load
from datetime import datetime
import matplotlib.pyplot as plt
import os
import altair as alt
import warnings
from streamlit_autorefresh import st_autorefresh



from utils.backtest import (
    backtest_breakout_strategy,
    backtest_ma_crossover_strategy,
    backtest_rsi_strategy
)

from utils.signal_scorer import score_trade_signal, auto_exit_trades, score_features_live


from utils.auto_exit_rules import (
    rsi_exit,
    ma_cross_down_exit,
    pnl_exit
)

from utils.train_signal_model import (
    generate_training_dataset,
    train_and_save_model
)

from questrade_api import (
    get_quote,
    search_symbol,
    get_candles,
    place_order,
    get_account_info,
    get_positions
)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
st.set_page_config(page_title="Trading Dashboard", layout="wide")

# Auto-refresh every 60 seconds (60000 ms)
st_autorefresh(interval=60000, key="refresh")

st.title("💹 AI-Enhanced Trading Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Ensure required folders and files exist
os.makedirs("exports/trades", exist_ok=True)
os.makedirs("exports/live", exist_ok=True)

# Create empty trade log if missing
trade_log_path = "exports/trades/trade_log_breakout.csv"
if not os.path.exists(trade_log_path):
    pd.DataFrame(columns=["Symbol", "Entry Date", "Exit Date", "PnL", "Strategy"]).to_csv(trade_log_path, index=False)

# Create empty alert log if missing
alert_path = "exports/live/alerts_2025-05-06.csv"
if not os.path.exists(alert_path):
    pd.DataFrame(columns=["Symbol", "Strategy", "Timestamp", "Signal"]).to_csv(alert_path, index=False)

# Create empty ranking file if missing
ranked_path = "exports/trades/ranked_symbol_strategy_summary.csv"
if not os.path.exists(ranked_path):
    pd.DataFrame(columns=["Symbol", "Strategy", "Total Return %", "Sharpe Ratio", "Max Drawdown"]).to_csv(ranked_path, index=False)

st.subheader("Real-Time Breakout Monitor")


if "trade_log" not in st.session_state:
    st.session_state.trade_log = []

# === Add sidebar controls ===
st.sidebar.markdown("## ⚙️ Trading Mode")
live_mode = st.sidebar.toggle("Live Trading", value=False, help="Turn ON to place real orders")

if live_mode:
    st.sidebar.success("LIVE MODE ENABLED")
else:
    st.sidebar.warning("TEST MODE (No orders will be sent)")

# === Auto-Exit Logic ===
st.sidebar.markdown("## 🚩 Testing")
force_trade = st.sidebar.button("🚀 Force Test Trade (Current Symbol)")

if live_mode:
    st.sidebar.success("LIVE MODE ENABLED")
else:
    st.sidebar.warning("TEST MODE (No orders will be sent)")
    
if st.sidebar.button("🚪 Run Auto-Exit Logic"):
    auto_exit_trades()

# === AI Model Training ===
st.sidebar.markdown("## 🤖 AI Model Training")

if st.sidebar.button("📊 Generate Training Dataset"):
    try:
        generate_training_dataset()
        st.sidebar.success("✅ Training dataset generated!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to generate dataset: {e}")

#
if st.sidebar.button("🧠 Train AI Model"):
    try:
        train_and_save_model()
        st.sidebar.success("✅ Model trained and saved!")
    except Exception as e:
        st.sidebar.error(f"❌ Model training failed: {e}")
        
# === Dashboard Header ===
st.title("📈 Live Trading Dashboard")
st.markdown(
    f"### 🚦 Current Mode: {'💥 Live Trading' if live_mode else '🧪 Simulation'}",
    unsafe_allow_html=True
)



# === Logging functions ===
def log_trade(symbol, price, strategy="Breakout"):
    filename = "exports/live_trades.csv"
    os.makedirs("exports", exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "price", "strategy"])
        writer.writerow([datetime.now().isoformat(), symbol, price, strategy])

def log_simulated_trade(symbol, price, strategy="Breakout"):
    filename = "exports/simulated_trades.csv"
    os.makedirs("exports", exist_ok=True)
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "price", "strategy"])
        writer.writerow([datetime.now().isoformat(), symbol, price, strategy])
def append_to_trade_log(symbol, price, strategy):
    log_path = "exports/trades/trade_log_breakout.csv"
    os.makedirs("exports/trades", exist_ok=True)

    df = pd.DataFrame([{
        "Symbol": symbol,
        "Entry Date": datetime.now().isoformat(),
        "Entry Price": price,
        "Exit Date": "",
        "Exit Price": "",
        "PnL": "",
        "Strategy": strategy
    }])

    # Append to the file
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, mode='w', header=True, index=False)
def append_to_trade_log(symbol, price, strategy):
    log_path = "exports/trades/trade_log_breakout.csv"
    os.makedirs("exports/trades", exist_ok=True)

    df = pd.DataFrame([{
        "Symbol": symbol,
        "Entry Date": datetime.now().isoformat(),
        "Entry Price": price,
        "Exit Date": "",
        "Exit Price": "",
        "PnL": "",
        "Strategy": strategy
    }])

    # Append to the file
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, mode='w', header=True, index=False)
def append_to_trade_log(symbol, price, strategy):
    log_path = "exports/trades/trade_log_breakout.csv"
    os.makedirs("exports/trades", exist_ok=True)

    df = pd.DataFrame([{
        "Symbol": symbol,
        "Entry Date": datetime.now().isoformat(),
        "Entry Price": price,
        "Exit Date": "",
        "Exit Price": "",
        "PnL": "",
        "Strategy": strategy
    }])

    # Append to the file
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, mode='w', header=True, index=False)
        
                                
symbols = ["AAPL", "TSLA", "MSFT"]

# === Strategy Assignment ===
st.sidebar.markdown("## 📊 Strategy Assignment")
strategy_options = ["Breakout", "MA Crossover", "RSI"]

st.sidebar.markdown("## 🔧 Strategy Parameters")
ma_fast = st.sidebar.number_input("MA Crossover - Fast MA (e.g. MA5)", min_value=2, max_value=50, value=5, key="ma_fast")
ma_slow = st.sidebar.number_input("MA Crossover - Slow MA (e.g. MA20)", min_value=5, max_value=100, value=20, key="ma_slow")
rsi_window = st.sidebar.number_input("RSI Window", min_value=2, max_value=50, value=14, key="rsi_window")
strategy_map = {}
for symbol in symbols:
    strategy = st.sidebar.selectbox(f"Strategy for {symbol}", strategy_options, key=f"strategy_{symbol}")
    strategy_map[symbol] = strategy
for symbol in symbols:
    try:
        search_result = search_symbol(symbol)
        symbol_id = search_result["symbols"][0]["symbolId"]

        # Quote
        quote_data = get_quote(symbol_id)["quotes"][0]

        #✅ FIX: Get candle data
        candle_data = get_candles(symbol_id)
        candles = candle_data.get("candles", [])
        last_price = quote_data["lastTradePrice"]

        # Candles for breakout
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["start"], utc=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        
        # Train AI model 
        model = load("exports/signal_scoring_model.pkl")

        # In your loop:
        X = score_features_live(df)
        ai_score = None

        if X is not None and all(col in X.columns for col in model.feature_names_in_):
            try:
                proba = model.predict_proba(X)[0]
                ai_score = proba[list(model.classes_).index(1)]
            except Exception:
                pass  # Let it fall back to "score unavailable"

        # Display
        st.markdown(f"## {symbol} - ${last_price:.2f}")
        if ai_score is not None:
            st.metric("🤖 AI Signal Score", f"{ai_score:.2f}")
            if ai_score >= 0.8:
                st.success("✅ Strong Signal")
            elif ai_score >= 0.5:
                st.warning("⚠️ Moderate Signal")
            else:
                st.error("❌ Weak Signal")
        else:
            st.warning("⚠️ AI score unavailable (not enough data or model issue)")

        strategy_used = strategy_map.get(symbol, "Breakout")

        if strategy_used == "Breakout":
           #high20 = df["high"].iloc[:-1].max()  # Exclude current day
                        
            if strategy_used == "Breakout":
                high20 = df["high"].iloc[:-1].max()
                signal_triggered = last_price > high20
                st.write(f"Breakout Threshold (20-day High): ${high20:.2f}")
                high_chart = alt.Chart(df.reset_index().iloc[-50:]).mark_line().encode(
                    x='datetime:T',
                    y=alt.Y('high:Q', title='20-Day High')
                ).properties(height=200)
                st.altair_chart(high_chart, use_container_width=True)
                signal_triggered = last_price > high20

                # Load AI model for scoring
                model = load("exports/signal_scoring_model.pkl")
                expected_features = list(model.feature_names_in_)

                ai_score = None
                features_df = score_features_live(df)
                if features_df is not None and not features_df.isnull().values.any():
                    try:
                        features_df = features_df[expected_features]
                        ai_score = model.predict_proba(features_df)[0][1]
                        st.write(f"🤖 AI Score: {ai_score:.2f}")
                    except Exception as e:
                        st.warning(f"⚠️ AI scoring failed: {e}")
                else:
                    st.warning("⚠️ AI Score unavailable (not enough data or model issue)")
                
                # Only act on strong signals (e.g., score ≥ 0.7)
                should_trade = signal_triggered and (ai_score is None or ai_score >= 0.7)
                
                if should_trade or force_trade:
                    st.success(f"{strategy_used} Signal Triggered!")
            else:
                signal_triggered = False
        exit_signal = False

        if strategy_used == "RSI":
            exit_signal = rsi_exit(df)
        elif strategy_used == "MA Crossover":
            exit_signal = ma_cross_down_exit(df, fast=ma_fast, slow=ma_slow)
        elif strategy_used == "Breakout":
            # Optional: add price-based or PnL-based exit
            # Load open trades to get last entry price
            df_trades = pd.read_csv("exports/trades/trade_log_breakout.csv")
            df_open = df_trades[(df_trades["Symbol"] == symbol) & (df_trades["Exit Date"].isna() | (df_trades["Exit Date"] == ""))]

            # Fallback default if no entry found
            last_entry_price = None
            if not df_open.empty:
                last_entry_price = float(df_open.iloc[-1]["Entry Price"])
            if strategy_used == "Breakout" and last_entry_price is not None:
                exit_signal = pnl_exit(current_price=last_price, entry_price=last_entry_price)

        elif strategy_used == "MA Crossover":
            df["MA5"] = df["close"].rolling(window=ma_fast).mean()
            df["MA20"] = df["close"].rolling(window=ma_slow).mean()
            st.write(f"MA5: {df['MA5'].iloc[-1]:.2f}, MA20: {df['MA20'].iloc[-1]:.2f}")
            ma_chart = alt.Chart(df.reset_index().iloc[-50:]).transform_fold(
                ["MA5", "MA20"],
                as_=["MA", "value"]
            ).mark_line().encode(
                x='datetime:T',
                y=alt.Y('value:Q', title='Moving Averages'),
                color='MA:N'
            ).properties(height=200)
            st.altair_chart(ma_chart, use_container_width=True)
            signal_triggered = df["MA5"].iloc[-2] < df["MA20"].iloc[-2] and df["MA5"].iloc[-1] > df["MA20"].iloc[-1]
        elif strategy_used == "RSI":
            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=rsi_window).mean()
            avg_loss = loss.rolling(window=rsi_window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            df["RSI"] = rsi
            st.write(f"RSI: {df['RSI'].iloc[-1]:.2f}")
            rsi_chart = alt.Chart(df.reset_index().iloc[-50:]).mark_line().encode(
                x='datetime:T',
                y=alt.Y('RSI:Q', title='RSI')
            ).properties(height=200)
            st.altair_chart(rsi_chart, use_container_width=True)
            signal_triggered = df["RSI"].iloc[-1] < 30
        else:
            signal_triggered = False
        df["datetime"] = pd.to_datetime(df["start"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        high20 = df["high"].iloc[:-1].max()  # Exclude current day
        breakout = last_price > high20

        st.markdown(f"### {symbol}")
        st.write(f"**Last Price:** ${last_price:.2f}")
        st.write(f"**20-Day High:** ${high20:.2f}")

        if signal_triggered or force_trade:
            if signal_triggered:
                st.success(f"{strategy_used} Signal Triggered!")
            elif force_trade:
                st.warning("⚠️ Forced trade triggered manually.")

            strategy_used = strategy_map.get(symbol, "Breakout")
            log_trade(symbol, last_price, strategy=strategy_used)
            log_simulated_trade(symbol, last_price, strategy=strategy_used)
            append_to_trade_log(symbol, last_price, strategy_used)
            order_type = st.selectbox(f"Order Type for {symbol}", ["Market", "Limit", "Stop"], key=f"order_type_{symbol}")

            if order_type == "Limit":
                limit_price = st.number_input(f"Limit Price for {symbol}", value=round(last_price * 0.99, 2), key=f"limit_{symbol}")
            elif order_type == "Stop":
                stop_price = st.number_input(f"Stop Price for {symbol}", value=round(last_price * 1.01, 2), key=f"stop_{symbol}")

            order_params = {
                "account_number": get_account_info()["accounts"][0]["number"],
                "symbol_id": symbol_id,
                "quantity": 1,
                "action": "Buy",
                "order_type": order_type
            }

            if order_type == "Limit":
                order_params["limitPrice"] = limit_price
            elif order_type == "Stop":
                order_params["stopPrice"] = stop_price

            if live_mode:
                try:
                    order_response = place_order(**order_params)
                    st.success("✅ LIVE order submitted successfully!")
                    st.write(order_response)
                except Exception as e:
                    st.error(f"❌ Order failed: {e}")
            else:
                st.info(f"🧪 Simulated order: {symbol}, {order_type}, Price: {last_price}")
                log_simulated_trade(symbol, last_price, strategy=strategy_used)
                
            if exit_signal:
                st.warning(f"🚪 Exit signal triggered for {symbol} ({strategy_used})")

            # Auto-close logic (update trade log)
            df_open = pd.read_csv("exports/trades/trade_log_breakout.csv")
            open_trade_idx = df_open[(df_open["Symbol"] == symbol) & (df_open["Exit Date"].isna())].index

            if not open_trade_idx.empty:
                idx = open_trade_idx[0]
                df_open.at[idx, "Exit Date"] = datetime.now().isoformat()
                df_open.at[idx, "Exit Price"] = last_price
                df_open.at[idx, "PnL"] = round(last_price - float(df_open.at[idx, "Entry Price"]), 2)
                df_open.to_csv("exports/trades/trade_log_breakout.csv", index=False)
                st.success(f"✅ Auto-exit recorded for {symbol}")

        else:
            st.info("No breakout currently.")

    except Exception as e:
        st.error(f"Error loading {symbol}: {e}")

# === Manual Exit Input Section ===
st.header("📤 Auto/Manual Close for Open Trades")

if os.path.exists(trade_log_path):
    df_trades = pd.read_csv(trade_log_path)
    df_open = df_trades[df_trades["Exit Date"].isna() | (df_trades["Exit Date"] == "")]

    for idx, row in df_open.iterrows():
        st.subheader(f"{row['Symbol']} - {row['Strategy']}")
        st.write(f"Entry Date: {row['Entry Date']} | Entry Price: {row['Entry Price']}")

        # Load recent candles for exit scoring
        candle_file = f"exports/candles/{row['Symbol']}_candles.csv"
        ai_exit_score = None
        if os.path.exists(candle_file):
            df_candles = pd.read_csv(candle_file)
            df_candles["datetime"] = pd.to_datetime(df_candles["start"])
            df_candles.set_index("datetime", inplace=True)
            df_candles.sort_index(inplace=True)

            try:
                ai_exit_score = score_trade_signal(df_candles)
                if ai_exit_score is not None:
                    st.write(f"🤖 AI Exit Score: {ai_exit_score:.2f}")
                else:
                    st.write("🤖 AI Exit Score: N/A (not enough data)")
            except Exception as e:
                st.warning(f"AI exit scoring failed: {e}")
        else:
            st.warning("No candle file found for AI scoring.")

        # Allow user to submit exit manually OR simulate auto exit
        exit_price = st.number_input(f"Exit Price for {row['Symbol']}", key=f"exit_price_{idx}")
        auto_exit_threshold = 0.7  # adjustable if needed

        if ai_exit_score is not None and ai_exit_score < auto_exit_threshold:
            if st.button(f"🚨 Auto Exit {row['Symbol']} (AI score {ai_exit_score:.2f})", key=f"auto_exit_{idx}"):
                df_trades.at[idx, "Exit Date"] = datetime.now().isoformat()
                df_trades.at[idx, "Exit Price"] = exit_price
                df_trades.at[idx, "PnL"] = round(exit_price - float(row["Entry Price"]), 2)
                df_trades.to_csv(trade_log_path, index=False)
                st.success(f"Auto-exit recorded for {row['Symbol']}")
        
        if st.button(f"✅ Manual Exit for {row['Symbol']}", key=f"manual_exit_{idx}"):
            df_trades.at[idx, "Exit Date"] = datetime.now().isoformat()
            df_trades.at[idx, "Exit Price"] = exit_price
            df_trades.at[idx, "PnL"] = round(exit_price - float(row["Entry Price"]), 2)
            df_trades.to_csv(trade_log_path, index=False)
            st.success(f"Manual exit recorded for {row['Symbol']}")
            
# === Trade History & Performance Dashboard ===
if all(col in df_trades.columns for col in ["Exit Price", "PnL"]):
    df_closed = df_trades[df_trades["Exit Price"].notna() & (df_trades["Exit Price"] != "")]
    df_closed["PnL"] = pd.to_numeric(df_closed["PnL"], errors='coerce')
else:
    df_closed = pd.DataFrame()

st.header("📘 Closed Trade History")
if not df_closed.empty:
    st.subheader("Filters")
    filter_symbols = st.multiselect("Filter by Symbol", sorted(df_closed['Symbol'].unique()), key="closed_filter_symbol")
    filter_strategies = st.multiselect("Filter by Strategy", sorted(df_closed['Strategy'].unique()), key="closed_filter_strategy")
    filter_dates = st.multiselect(
        "Filter by Exit Date",
        sorted(pd.to_datetime(df_closed['Exit Date'], format='ISO8601').dt.date.unique()),
        key="closed_filter_date"
    )

    df_closed['Exit Date'] = pd.to_datetime(df_closed['Exit Date'], format='mixed', errors='coerce')
    if filter_symbols:
        df_closed = df_closed[df_closed['Symbol'].isin(filter_symbols)]
    if filter_strategies:
        df_closed = df_closed[df_closed['Strategy'].isin(filter_strategies)]
    if filter_dates:
        df_closed['Date'] = df_closed['Exit Date'].dt.date
        df_closed = df_closed[df_closed['Date'].isin(filter_dates)]

    #st.dataframe(
    #df_closed.drop(columns=['Date']) if 'Date' in df_closed.columns else df_closed.sort_values(by="Exit Date", ascending=False),
    #use_container_width=True
    #)
    st.dataframe(df_closed, use_container_width=True)

    st.subheader("📈 Cumulative PnL Over Time")
    # Normalize Exit Date column with UTC handling
    # Normalize Exit Date column with UTC handling
    df_closed.loc[:, "Exit Date"] = pd.to_datetime(df_closed["Exit Date"], errors="coerce")
    df_closed.loc[:, "Exit Date"] = df_closed["Exit Date"].dt.tz_localize(None)


    # Drop extra 'Date' column if present
    if "Date" in df_closed.columns:
        df_closed = df_closed.drop(columns=["Date"])

    # Sort correctly by Exit Date
    df_closed = df_closed.sort_values(by="Exit Date", ascending=False)
    
    df_closed["Cumulative PnL"] = df_closed["PnL"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_closed["Exit Date"], df_closed["Cumulative PnL"], marker='o')
    ax.set_title("Cumulative PnL")
    ax.set_xlabel("Exit Date")
    ax.set_ylabel("PnL")
    st.pyplot(fig)

    st.subheader("📊 Trade Win/Loss Summary")

    st.subheader("📌 Performance Metrics")
    avg_pnl = df_closed['PnL'].mean()
    best_trade = df_closed['PnL'].max()
    worst_trade = df_closed['PnL'].min()
    st.metric("Average PnL", f"${avg_pnl:.2f}")
    st.metric("Best Trade", f"${best_trade:.2f}")
    st.metric("Worst Trade", f"${worst_trade:.2f}")

    st.subheader("Grouped by Symbol")
    summary_by_symbol = df_closed.groupby("Symbol")["PnL"].agg(["count", "mean", "sum"]).reset_index()
    summary_by_symbol.columns = ["Symbol", "Trades", "Avg PnL", "Total PnL"]
    st.dataframe(summary_by_symbol, use_container_width=True)

    st.subheader("📊 Total PnL by Symbol")
    chart_symbol = alt.Chart(summary_by_symbol).mark_bar().encode(
    x=alt.X('Symbol:N', title="Symbol"),
    y=alt.Y('Total PnL:Q', title="Total PnL"),
    color=alt.condition(
        alt.datum['Total PnL'] > 0,
        alt.value("green"),
        alt.value("red")
    ),
    tooltip=['Symbol', 'Trades', 'Avg PnL', 'Total PnL']
    ).properties(height=300)
    st.altair_chart(chart_symbol, use_container_width=True)

    st.subheader("Grouped by Strategy")
    summary_by_strategy = df_closed.groupby("Strategy")["PnL"].agg(["count", "mean", "sum"]).reset_index()
    summary_by_strategy.columns = ["Strategy", "Trades", "Avg PnL", "Total PnL"]
    st.dataframe(summary_by_strategy, use_container_width=True)

    st.subheader("📊 Total PnL by Strategy")
    chart_strategy = alt.Chart(summary_by_strategy).mark_bar().encode(
        x=alt.X('Strategy:N', title="Strategy"),
        y=alt.Y('Total PnL:Q', title="Total PnL"),
        color=alt.condition(
            alt.datum['Total PnL'] > 0,
            alt.value("green"),
            alt.value("red")
        ),
        tooltip=['Strategy', 'Trades', 'Avg PnL', 'Total PnL']
    ).properties(height=300)
    st.altair_chart(chart_strategy, use_container_width=True)
    st.header("📊 Trade Win/Loss Summary")
    if all(col in df_trades.columns for col in ["Exit Price", "PnL"]):
        df_closed = df_trades[df_trades["Exit Price"].notna() & (df_trades["Exit Price"] != "")]
        df_closed["PnL"] = pd.to_numeric(df_closed["PnL"], errors='coerce')
        wins = df_closed[df_closed["PnL"] > 0].shape[0]
        losses = df_closed[df_closed["PnL"] <= 0].shape[0]
        total = wins + losses
        win_rate = round((wins / total) * 100, 2) if total > 0 else 0

        st.metric("Total Trades", total)
        st.metric("Winning Trades", wins)
        st.metric("Losing Trades", losses)
        st.metric("Win Rate (%)", win_rate)
    else:
        st.info("Trade log missing required columns for win/loss analysis.")
else:
    st.info("No trade log found yet.")

# === Clear Test Data Button ===
if st.sidebar.button("🧹 Clear Trade Logs", key="clear_logs"):
    with open(trade_log_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Symbol", "Entry Date", "Entry Price", "Exit Date", "Exit Price", "PnL", "Strategy"])
    st.sidebar.success("Trade logs cleared.")

# === Data Utilities (Candles files) ===    
st.sidebar.markdown("## 📦 Data Utilities")

if st.sidebar.button("📥 Generate Candle Files from Trade Log"):
    trade_log_path = "exports/trades/trade_log_breakout.csv"
    output_dir = "exports/candles"

    if not os.path.exists(trade_log_path):
        st.sidebar.error("Trade log not found.")
    else:
        df = pd.read_csv(trade_log_path)
        symbols = df["Symbol"].dropna().unique().tolist()
        os.makedirs(output_dir, exist_ok=True)

        for symbol in symbols:
            try:
                st.sidebar.write(f"Fetching candles for {symbol}...")
                result = search_symbol(symbol)
                symbol_id = result["symbols"][0]["symbolId"]
                candle_data = get_candles(symbol_id, days=90)
                candles = candle_data.get("candles", [])

                if candles:
                    df_candles = pd.DataFrame(candles)
                    df_candles["datetime"] = pd.to_datetime(df_candles["start"])
                    df_candles.to_csv(f"{output_dir}/{symbol}_candles.csv", index=False)
                    st.sidebar.success(f"Saved {symbol} candles.")
                else:
                    st.sidebar.warning(f"No candle data for {symbol}.")
            except Exception as e:
                st.sidebar.error(f"Failed {symbol}: {e}")


# === Reinitialize Trade Log Button ===
if st.sidebar.button("🔄 Reinitialize Trade Log", key="reinit_trade_log"):
    os.makedirs("exports/trades", exist_ok=True)
    with open(trade_log_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Symbol", "Entry Date", "Entry Price", "Exit Date", "Exit Price", "PnL", "Strategy"])
    st.sidebar.success("Trade log reinitialized with headers.")

# === Reinitialize Simulated Trades Log Button ===
sim_log_path = "exports/simulated_trades.csv"
if st.sidebar.button("🔄 Reinitialize Simulated Log", key="reinit_sim_log"):
    os.makedirs("exports", exist_ok=True)
    with open(sim_log_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "symbol", "price", "strategy"])
    st.sidebar.success("Simulated trade log reinitialized with headers.")

# === Questrade Portfolio Sync ===
st.header("🔄 Questrade Portfolio Sync")
try:
    account_number = get_account_info()["accounts"][0]["number"]
    positions = get_positions(account_number)["positions"]
    df_questrade = pd.DataFrame(positions)
    df_questrade = df_questrade[["symbol", "openQuantity", "currentMarketValue"]]
    df_questrade.columns = ["Symbol", "Questrade Quantity", "Questrade Market Value"]

    df_local = pd.read_csv("exports/trades/trade_log_breakout.csv")
    df_open = df_local[df_local["Exit Date"].isna() | (df_local["Exit Date"] == "")]
    df_local_summary = df_open.groupby("Symbol").size().reset_index(name="Local Quantity")

    df_compare = pd.merge(df_questrade, df_local_summary, on="Symbol", how="outer").fillna(0)
    df_compare["Local Quantity"] = df_compare["Local Quantity"].astype(int)
    df_compare["Mismatch"] = df_compare["Questrade Quantity"] != df_compare["Local Quantity"]

    st.dataframe(df_compare, use_container_width=True)
except Exception as e:
    st.warning(f"Unable to sync Questrade portfolio: {e}")

# === Simulated Trades Viewer ===
st.header("🧪 Simulated Trades Log")
if os.path.exists(sim_log_path):
    df_sim = pd.read_csv(sim_log_path)

    st.subheader("Filters")
    sim_symbols = st.multiselect("Filter by Symbol", sorted(df_sim['symbol'].unique()), key="sim_filter_symbol")
    sim_dates = st.multiselect("Filter by Date", sorted(pd.to_datetime(df_sim['timestamp']).dt.date.unique()), key="sim_filter_date")


    if sim_symbols:
        df_sim = df_sim[df_sim['symbol'].isin(sim_symbols)]
    if sim_dates:
        df_sim['date'] = pd.to_datetime(df_sim['timestamp']).dt.date
        df_sim = df_sim[df_sim['date'].isin(sim_dates)]

    st.dataframe(df_sim.drop(columns=['date']) if 'date' in df_sim.columns else df_sim.sort_values(by="timestamp", ascending=False), use_container_width=True)
    st.download_button("📥 Download Simulated Log", df_sim.to_csv(index=False), file_name="simulated_trades.csv")

    st.subheader("📊 Simulated Trade Summary")
    st.metric("Total Simulated Trades", len(df_sim))
    st.metric("Average Simulated Price", round(df_sim['price'].mean(), 2) if not df_sim.empty else 0)

    grouped = df_sim.groupby(['symbol', 'strategy'])['price'].agg(['count', 'mean']).reset_index()
    grouped.columns = ['Symbol', 'Strategy', 'Total Trades', 'Average Price']
    st.dataframe(grouped, use_container_width=True)
else:
    st.info("No simulated trades logged yet.")
    


# === Auto-refresh every 60 seconds ===
st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

st.title("Trading Strategy Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === Section 1: Live Alerts ===
st.header("Live Signal Alerts")

alert_files = sorted(glob.glob("exports/live/alerts_*.csv"))

if alert_files:
    df_alerts = pd.concat([pd.read_csv(f) for f in alert_files[-7:]], ignore_index=True)

    st.subheader("Filters")
    symbols = st.multiselect("Filter by Symbol", sorted(df_alerts['Symbol'].unique()))
    strategies = st.multiselect("Filter by Strategy", sorted(df_alerts['Strategy'].unique()))
    dates = st.multiselect("Filter by Date", sorted(pd.to_datetime(df_alerts['Timestamp']).dt.date.unique()))

    # Apply filters
    if symbols:
        df_alerts = df_alerts[df_alerts['Symbol'].isin(symbols)]
    if strategies:
        df_alerts = df_alerts[df_alerts['Strategy'].isin(strategies)]
    if dates:
        df_alerts['Date'] = pd.to_datetime(df_alerts['Timestamp']).dt.date
        df_alerts = df_alerts[df_alerts['Date'].isin(dates)]

    st.dataframe(df_alerts.drop(columns=['Date']) if 'Date' in df_alerts.columns else df_alerts)
else:
    st.info("No live alerts found.")

# === Section 2: Trade Logs ===
st.header("Trade Logs")

trade_files = glob.glob("exports/trades/trade_log_*.csv")
if trade_files:
    selected_file = st.selectbox("Choose a trade log", trade_files)
    df_trades = pd.read_csv(selected_file)

    st.subheader("Trade Filters")
    trade_symbols = st.multiselect("Filter Trade Log by Symbol", sorted(df_trades['Symbol'].unique()))
    if 'Strategy' in df_trades.columns:
        trade_strategies = st.multiselect("Filter Trade Log by Strategy", sorted(df_trades['Strategy'].unique()))
    else:
        trade_strategies = []

    if trade_symbols:
        df_trades = df_trades[df_trades['Symbol'].isin(trade_symbols)]
    if trade_strategies:
        df_trades = df_trades[df_trades['Strategy'].isin(trade_strategies)]

    st.dataframe(df_trades)
    
    # === Plot cumulative return ===
    if 'Exit Date' in df_trades.columns and 'PnL' in df_trades.columns:
        st.subheader("Strategy Cumulative Return")

        df_trades['Exit Date'] = pd.to_datetime(df_trades['Exit Date'], errors='coerce', utc=True)
        df_trades['Exit Date'] = df_trades['Exit Date'].dt.tz_localize(None)

        df_trades.sort_values('Exit Date', inplace=True)
        df_trades['Cumulative Return'] = df_trades['PnL'].cumsum()

        fig, ax = plt.subplots(figsize=(10, 4))
        for sym in df_trades['Symbol'].unique():
            sym_data = df_trades[df_trades['Symbol'] == sym]
            ax.plot(sym_data['Exit Date'], sym_data['Cumulative Return'], label=sym)
        ax.set_title("Cumulative Return by Symbol")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL")
        ax.legend()
        st.pyplot(fig)
else:
    st.warning("No trade logs found.")

# === Section 3: Ranked Strategy Summary ===
st.header("Ranked Strategy Performance")

try:
    df_ranked = pd.read_csv("exports/trades/ranked_symbol_strategy_summary.csv")

    st.subheader("Ranking Filters")
    ranked_symbols = st.multiselect("Filter Rankings by Symbol", sorted(df_ranked['Symbol'].unique()))
    ranked_strategies = st.multiselect("Filter Rankings by Strategy", sorted(df_ranked['Strategy'].unique()))

    if ranked_symbols:
        df_ranked = df_ranked[df_ranked['Symbol'].isin(ranked_symbols)]
    if ranked_strategies:
        df_ranked = df_ranked[df_ranked['Strategy'].isin(ranked_strategies)]

    st.dataframe(df_ranked)

    st.subheader("Top Strategy Returns")
    top_n = st.slider("Top N Strategies", 5, 30, 10)
    df_plot = df_ranked.sort_values('Total Return %', ascending=False).head(top_n)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.barh(df_plot['Symbol'] + " - " + df_plot['Strategy'], df_plot['Total Return %'])
    ax2.set_xlabel("Total Return (%)")
    ax2.set_title("Top Performing Strategies")
    st.pyplot(fig2)
except FileNotFoundError:
    st.warning("Ranked strategy summary file not found.")


# === Section 5: Recent Trade Log ===
st.header("🧾 Recent Trade Log")

log_file = "exports/live_trades.csv"

if os.path.exists(log_file):
    df_trades = pd.read_csv(log_file)

    # Optional filters
    symbols = df_trades["symbol"].unique().tolist()
    selected_symbol = st.selectbox("🔎 Filter by Symbol", ["All"] + symbols)

    if selected_symbol != "All":
        df_trades = df_trades[df_trades["symbol"] == selected_symbol]

    st.dataframe(df_trades.sort_values(by="timestamp", ascending=False), use_container_width=True)

else:
    st.info("No trades logged yet.")



# === Section 6: Live Trade Log ===
st.markdown("---")
st.header("Live Trade Log")

trade_log_path = "live_trades.csv"

if os.path.exists(trade_log_path):
    trade_log_df = pd.read_csv(trade_log_path)
    st.dataframe(trade_log_df.tail(10), use_container_width=True)
else:
    trade_log_df = pd.DataFrame(columns=["Datetime", "Symbol", "Signal", "Price", "OrderType"])
    st.info("No live trades logged yet.")       

# === Section 7: Live Trade Summary ===       
st.markdown("### Live Trade Summary")

if not trade_log_df.empty:
    wins = trade_log_df[trade_log_df['Signal'].str.contains("Buy|Breakout")].shape[0]
    last_signal_time = trade_log_df["Datetime"].max()
    unique_symbols = trade_log_df["Symbol"].nunique()

    st.metric("Total Signals", len(trade_log_df))
    st.metric("Breakouts Detected", wins)
    st.metric("Unique Symbols", unique_symbols)
    st.metric("Last Signal", last_signal_time)

# === Section 8: Signal Timeline ===
chart = alt.Chart(trade_log_df).mark_bar().encode(
    x='Datetime:T',
    y='Symbol',
    color='OrderType'
).properties(
    width='container',
    height=300,
    title='Signal Timeline'
)

st.altair_chart(chart, use_container_width=True)


# === Section X: Simulated Trades Log ===
st.markdown("---")
st.header("📋 Simulated Trades Log")

sim_file = "exports/simulated_trades.csv"
if os.path.exists(sim_file):
    df_sim = pd.read_csv(sim_file)
    st.dataframe(df_sim.sort_values(by="timestamp", ascending=False), use_container_width=True)
else:
    st.info("No simulated trades logged yet.")

# === Backtest: Strategy Simulation === 
st.header("🧪 Backtest a Strategy")

bt_symbol = st.selectbox("Select Symbol", ["AAPL", "TSLA", "MSFT"], key="bt_symbol")
bt_strategy = st.selectbox("Strategy", ["Breakout", "MA Crossover", "RSI"], key="bt_strategy")

lookback = st.slider("Lookback Window", 5, 30, 20, key="bt_lookback")
hold_period = st.slider("Hold Period After Entry", 1, 15, 5, key="bt_hold")

if bt_strategy == "MA Crossover":
    fast_ma = st.slider("Fast MA", 2, 20, 5, key="bt_fast_ma")
    slow_ma = st.slider("Slow MA", 10, 50, 20, key="bt_slow_ma")

if bt_strategy == "RSI":
    rsi_window = st.slider("RSI Window", 2, 50, 14, key="bt_rsi_window")

if st.button("Run Backtest", key="run_bt"):
    try:
        search_result = search_symbol(bt_symbol)
        symbol_id = search_result["symbols"][0]["symbolId"]
        candle_data = get_candles(symbol_id, days=90)
        candles = candle_data.get("candles", [])
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["start"])

        # Strategy Routing
        if bt_strategy == "Breakout":
            trades, equity = backtest_breakout_strategy(df, lookback, hold_period)
        elif bt_strategy == "MA Crossover":
            trades, equity = backtest_ma_crossover_strategy(df, fast_window=fast_ma, slow_window=slow_ma, hold_period=hold_period)
        elif bt_strategy == "RSI":
            trades, equity = backtest_rsi_strategy(df, rsi_window=rsi_window, hold_period=hold_period)

        st.subheader("📄 Simulated Trades")
        st.dataframe(trades)

        if not trades.empty:
            st.download_button("📥 Download Backtest Results", trades.to_csv(index=False), file_name=f"{bt_symbol}_{bt_strategy}_backtest.csv")

            st.subheader("📌 Backtest Summary Metrics")
            wins = trades[trades["PnL"] > 0].shape[0]
            losses = trades[trades["PnL"] <= 0].shape[0]
            total = wins + losses
            win_rate = round((wins / total) * 100, 2) if total > 0 else 0
            avg_pnl = trades["PnL"].mean()
            best = trades["PnL"].max()
            worst = trades["PnL"].min()

            st.metric("Total Trades", total)
            st.metric("Win Rate", f"{win_rate:.2f}%")
            st.metric("Average PnL", f"${avg_pnl:.2f}")
            st.metric("Best Trade", f"${best:.2f}")
            st.metric("Worst Trade", f"${worst:.2f}")

        st.subheader("📈 Cumulative Return")
        if not equity.empty:
            chart = alt.Chart(equity.reset_index()).mark_line(point=True).encode(
                x="Exit Date:T",
                y="Cumulative PnL:Q"
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No trades were triggered by the strategy.")

    except Exception as e:
        st.error(f"Backtest failed: {e}")

