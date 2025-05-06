import streamlit as st
import pandas as pd
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import os

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


st.set_page_config(page_title="Trading Dashboard", layout="wide")

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

