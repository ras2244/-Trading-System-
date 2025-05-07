import csv
import streamlit as st
import pandas as pd
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import os
import altair as alt
from streamlit_autorefresh import st_autorefresh
from questrade_api import get_quote, search_symbol, get_candles
from questrade_api import place_order, get_account_info

st.set_page_config(page_title="Trading Dashboard", layout="wide")

# Auto-refresh every 60 seconds (60000 ms)
st_autorefresh(interval=60000, key="refresh")


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


st.sidebar.markdown("## 🚩 Testing")
force_trade = st.sidebar.button("🚀 Force Test Trade (Current Symbol)")

if live_mode:
    st.sidebar.success("LIVE MODE ENABLED")
else:
    st.sidebar.warning("TEST MODE (No orders will be sent)")

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
        
symbols = ["AAPL", "TSLA", "MSFT"]
for symbol in symbols:
    try:
        search_result = search_symbol(symbol)
        symbol_id = search_result["symbols"][0]["symbolId"]

        # Quote
        quote_data = get_quote(symbol_id)["quotes"][0]
        last_price = quote_data["lastTradePrice"]

        # Candles for breakout
        candle_data = get_candles(symbol_id)
        candles = candle_data.get("candles", [])

        if len(candles) < 20:
            st.warning(f"Not enough candle data for {symbol}")
            continue

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["start"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        high20 = df["high"].iloc[:-1].max()  # Exclude current day
        breakout = last_price > high20

        st.markdown(f"### {symbol}")
        st.write(f"**Last Price:** ${last_price:.2f}")
        st.write(f"**20-Day High:** ${high20:.2f}")

        if breakout or force_trade:
            if breakout:
                st.success("20-day Breakout Signal Triggered!")
            elif force_trade:
                st.warning("⚠️ Forced trade triggered manually.")

            log_trade(symbol, last_price, strategy="Breakout")

            # 👇 Show order type options BEFORE placing order
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
                log_simulated_trade(symbol, last_price, strategy="Breakout")
        else:
            st.info("No breakout currently.")

    except Exception as e:
        st.error(f"Error loading {symbol}: {e}")

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

        df_trades['Exit Date'] = pd.to_datetime(df_trades['Exit Date'])
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
