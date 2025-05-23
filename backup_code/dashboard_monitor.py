# dashboard_monitor.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import os
import threading
import time
import json
from datetime import datetime, timedelta, timezone
import warnings

from ai_stock_selector import plot_stock_chart, get_symbol_id_by_ticker
from questrade_api import (
    get_account_info, get_positions, fetch_candles, get_access_token,
    get_quote_data, ping_api
)
from utils.health_check import check_api_health
from utils.model_utils import score_live_data

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
PING_INTERVAL = 180  # 3 minutes
JSON_PATH = "exports/daily_ai_selection_scores.json"
NUM_AI_STOCKS = 4

# Background token keep-alive loop
def keep_alive_loop(token, api_server, interval_seconds=PING_INTERVAL):
    
    def loop():
        while True:
            ping_api(token, api_server)
            time.sleep(interval_seconds)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

# Streamlit setup
st.set_page_config(page_title="Trade Monitor", layout="wide")
st.title("📊 Trade Monitoring Dashboard")

# Health status
is_healthy, msg = check_api_health()
st.markdown(f"### 🔌 API Connection Status: <span style='color:{'green' if is_healthy else 'red'}'>{msg}</span>", unsafe_allow_html=True)

# =============================
# 🧠 AI Strategy: Load from JSON
# =============================
st.header("🧠 AI Strategy Stock Selection (Preloaded JSON)")
token, api_server = get_access_token()

if 'last_ping' not in st.session_state:
    st.session_state.last_ping = time.time()
now = time.time()
last_ping = st.session_state.get("last_ping", 0)

with st.sidebar:
    st.markdown("### 🔁 Questrade API Ping")
    if 'keep_alive_started' not in st.session_state:
        keep_alive_loop(token, api_server)
        st.session_state.keep_alive_started = True
        remaining = int(PING_INTERVAL - (now - last_ping))
        st.info(f"⏳ Next ping in {remaining} sec")
    else:
        st.info("⏳ Ping has stopped")

# =============================
# 📈 Load Daily JSON Scores
# =============================
today_str = datetime.now().strftime("%Y-%m-%d")
selected_stocks = []

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    selected_stocks = data.get(today_str, [])[:NUM_AI_STOCKS]
else:
    st.warning(f"No preloaded JSON data found at {JSON_PATH}. Run prepare_daily_data.py first.")

if selected_stocks:
    summary_rows = []
    for stock in selected_stocks:
        summary_rows.append({
            "Symbol": stock["symbol"],
            "Score": round(stock["score"], 4),
            "Gap %": stock.get("gap_pct"),
            "Price": stock.get("price"),
            "Volume": stock.get("volume"),
            "RSI": stock.get("features", {}).get("rsi"),
            "VWAP": stock.get("features", {}).get("vwap"),
            "Volatility": stock.get("features", {}).get("volatility"),
            "🔥 Alert": "🔥" if stock["score"] >= 0.85 else ""
        })

    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary.sort_values("Score", ascending=False), use_container_width=True)

    export_path = f"exports/daily_ai_selection_{today_str}.csv"
    df_summary.to_csv(export_path, index=False)
    st.success(f"📁 Exported AI picks to {export_path}")

# =============================
# 📋 Full Watchlist Overview
# =============================
st.subheader("📋 Full Watchlist Overview (Today)")

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    full_today = data.get(today_str, [])
    if full_today:
        watchlist_df = pd.DataFrame([
            {
                "Symbol": s["symbol"],
                "Score": round(s["score"], 4),
                "Gap %": s.get("gap_pct"),
                "Price": s.get("price"),
                "Volume": s.get("volume"),
                "RSI": s.get("features", {}).get("rsi"),
                "VWAP": s.get("features", {}).get("vwap"),
                "Volatility": s.get("features", {}).get("volatility")
            }
            for s in full_today
        ])
        st.dataframe(watchlist_df.sort_values("Score", ascending=False), use_container_width=True)
        st.download_button(
            "📥 Download Full Watchlist",
            watchlist_df.to_csv(index=False),
            file_name=f"watchlist_full_{today_str}.csv"
        )
    else:
        st.info("Watchlist JSON exists but is empty for today.")
else:
    st.warning("No full watchlist data found.")

# =============================
# 📁 CSV Summary from Prepare Daily Data
# =============================
csv_summary_path = "exports/daily_ai_selection_summary.csv"
if os.path.exists(csv_summary_path):
    st.subheader("📁 CSV Summary: Full Scored Watchlist")
    try:
        df_csv_summary = pd.read_csv(csv_summary_path)
        df_csv_summary = df_csv_summary.rename(columns={
            "symbol": "Symbol",
            "score": "Score",
            "price": "Price",
            "volume": "Volume",
            "gap_pct": "Gap %"
        })
        st.dataframe(df_csv_summary.sort_values("Score", ascending=False), use_container_width=True)
        st.download_button(
            label="📥 Download CSV Summary",
            data=df_csv_summary.to_csv(index=False),
            file_name=f"watchlist_summary_{today_str}.csv"
        )
    except Exception as e:
        st.error(f"❌ Failed to load summary CSV: {e}")
else:
    st.info("CSV summary not available yet. Run prepare_daily_data.py first.")
    # Show individual charts
    for stock in selected_stocks:
        symbol = stock["symbol"]
        score = stock["score"]
        st.subheader(f"{symbol} (Score: {score:.2f}) {'🔥' if score >= 0.85 else ''}")
        try:
            symbol_id = stock.get("symbolId") or get_symbol_id_by_ticker(symbol, token, api_server)
            df = fetch_candles(symbol_id, days=90, start_time=None, end_time=None, token=token, api_server=api_server)
            fig = plot_stock_chart(df, symbol)
            if fig:
                st.pyplot(fig)
            else:
                st.warning(f"Chart failed for {symbol}")
        except Exception as e:
            st.error(f"❌ Error plotting {symbol}: {e}")
st.info("No AI picks to display today.")
# =============================
# Trade Logs & Reconciliation
# =============================
log_path = "exports/trades/trade_log_breakout.csv"
if os.path.exists(log_path):
    df_trades = pd.read_csv(log_path)
    df_trades["Exit Date"] = pd.to_datetime(df_trades["Exit Date"], errors="coerce")
    df_trades["PnL"] = pd.to_numeric(df_trades["PnL"], errors="coerce")
    df_trades = df_trades.dropna(subset=["Exit Date", "PnL"])

    st.header("🧾 Closed Trade Summary")

    st.subheader("🔎 Filters")
    symbols = st.multiselect("Filter by Symbol", sorted(df_trades["Symbol"].dropna().unique()))
    strategies = st.multiselect("Filter by Strategy", sorted(df_trades["Strategy"].dropna().unique()))
    dates = st.multiselect("Filter by Exit Date", sorted(df_trades["Exit Date"].dt.date.unique()))

    if symbols:
        df_trades = df_trades[df_trades["Symbol"].isin(symbols)]
    if strategies:
        df_trades = df_trades[df_trades["Strategy"].isin(strategies)]
    if dates:
        df_trades["Exit Date Date"] = df_trades["Exit Date"].dt.date
        df_trades = df_trades[df_trades["Exit Date Date"].isin(dates)]

    st.dataframe(df_trades.sort_values("Exit Date", ascending=False), use_container_width=True)
    st.download_button("📥 Download Filtered Trades", df_trades.to_csv(index=False), file_name="filtered_trades.csv")

    # Metrics
    st.subheader("📌 Performance Metrics")
    st.metric("Average PnL", f"${df_trades['PnL'].mean():.2f}")
    st.metric("Win Rate", f"{(df_trades['PnL'] > 0).mean() * 100:.2f}%")
    st.metric("Best Trade", f"${df_trades['PnL'].max():.2f}")
    st.metric("Worst Trade", f"${df_trades['PnL'].min():.2f}")

    # PnL Curve
    st.subheader("📈 Cumulative PnL Over Time")
    df_trades["Cumulative PnL"] = df_trades["PnL"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_trades["Exit Date"], df_trades["Cumulative PnL"], marker="o")
    st.pyplot(fig)

    # Strategy/Symbol Charts
    def plot_bar(summary, title):
        st.subheader(title)
        st.altair_chart(
            alt.Chart(summary).mark_bar().encode(
                x=summary.columns[0] + ":N", y="Total PnL:Q",
                color=alt.condition(alt.datum["Total PnL"] > 0, alt.value("green"), alt.value("red")),
                tooltip=list(summary.columns)
            ),
            use_container_width=True
        )

    strat_summary = df_trades.groupby("Strategy")["PnL"].agg(["count", "mean", "sum"]).reset_index()
    strat_summary.columns = ["Strategy", "Trades", "Avg PnL", "Total PnL"]
    st.dataframe(strat_summary)
    plot_bar(strat_summary, "📊 Strategy Breakdown")

    symbol_summary = df_trades.groupby("Symbol")["PnL"].agg(["count", "mean", "sum"]).reset_index()
    symbol_summary.columns = ["Symbol", "Trades", "Avg PnL", "Total PnL"]
    st.dataframe(symbol_summary)
    plot_bar(symbol_summary, "📊 Symbol Breakdown")

    # Summary Overview
    st.subheader("📋 Trade Summary Overview")
    st.markdown(f"**Total Closed Trades:** {len(df_trades)}")
    st.markdown(f"**Symbols Traded:** {df_trades['Symbol'].nunique()}")
    st.markdown(f"**Strategies Used:** {df_trades['Strategy'].nunique()}")
    st.markdown(f"**Date Range:** {df_trades['Exit Date'].min().date()} → {df_trades['Exit Date'].max().date()}")

    # Reconciliation with Questrade
    st.header("🔄 Questrade Account Reconciliation")
    try:
        acct_num = get_account_info()["accounts"][0]["number"]
        positions = get_positions(acct_num)["positions"]
        df_qt = pd.DataFrame(positions)[["symbol", "openQuantity", "currentMarketValue"]]
        df_qt.columns = ["Symbol", "Questrade Quantity", "Questrade Market Value"]

        df_open = df_trades[df_trades["Exit Date"].isna()]
        df_local_summary = df_open.groupby("Symbol").size().reset_index(name="Local Quantity")
        df_merge = pd.merge(df_qt, df_local_summary, on="Symbol", how="outer").fillna(0)
        df_merge["Local Quantity"] = df_merge["Local Quantity"].astype(int)
        df_merge["Mismatch"] = df_merge["Questrade Quantity"] != df_merge["Local Quantity"]
        st.dataframe(df_merge, use_container_width=True)
    except Exception as e:
        st.warning(f"Questrade reconciliation failed: {e}")
else:
    st.warning("Trade log not found.")
