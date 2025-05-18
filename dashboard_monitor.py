import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import os
from datetime import datetime, timedelta, timezone
from ai_stock_selector import select_stocks_for_today, plot_stock_chart
from questrade_api import get_account_info, get_positions, fetch_candles,get_access_token, get_symbol_id_by_ticker
from utils.health_check import check_api_health


st.set_page_config(page_title="Trade Monitor", layout="wide")
st.title("📊 Trade Monitoring Dashboard")
# Health status section
is_healthy, msg = check_api_health()
color = "green" if is_healthy else "red"
st.markdown(f"### 🔌 API Connection Status: <span style='color:{color}'>{msg}</span>", unsafe_allow_html=True)

# =============================
# AI Strategy: Live Stock Selection
# =============================
st.header("🧠 AI Strategy Stock Selection (Live from Questrade Symbols)")
# Define your time window
days=90
end_time = datetime.now()
start_time = end_time - timedelta(days=90)
if not start_time or not end_time:
        end_time = datetime.now(timezone.utc).replace(microsecond=0)
        start_time = end_time - timedelta(days=days)

# Format as ISO strings
start_time_str = start_time.astimezone().isoformat(),
end_time_str = end_time.astimezone().isoformat(),

# You must already have this from your token load
token, api_server = get_access_token()  # adjust based on your implemen
# Fetch today's selected stocks from AI model
selected_stocks = select_stocks_for_today()

# Validate selection
if selected_stocks:
    for stock in selected_stocks:
        symbol = stock["symbol"]
        score = stock["score"]
        st.subheader(f"{symbol} (AI Score: {score:.2f})")

        try:
            symbol_id = get_symbol_id_by_ticker(symbol, token, api_server)
            if not symbol_id:
                st.warning(f"{symbol}: Could not find symbolId.")
                continue

            df = fetch_candles(symbol_id, start_time, end_time, token, api_server)
            if df.empty:
                st.warning(f"{symbol}: No candle data received.")
                continue

            fig = plot_stock_chart(df, symbol)
            if fig:
                st.pyplot(fig)
            else:
                st.warning(f"{symbol}: Plotting failed.")

        except Exception as e:
            st.error(f"Error fetching or plotting data for {symbol}: {e}")
else:
    st.info("No stocks selected for today by the AI model.")
# =============================
# Trade Logs
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

    # Cumulative PnL
    st.subheader("📈 Cumulative PnL Over Time")
    df_trades = df_trades.sort_values("Exit Date")
    df_trades["Cumulative PnL"] = df_trades["PnL"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_trades["Exit Date"], df_trades["Cumulative PnL"], marker='o')
    st.pyplot(fig)

    # Strategy breakdown
    st.subheader("📊 Strategy Breakdown")
    strat_summary = df_trades.groupby("Strategy")["PnL"].agg(["count", "mean", "sum"]).reset_index()
    strat_summary.columns = ["Strategy", "Trades", "Avg PnL", "Total PnL"]
    st.dataframe(strat_summary)
    st.altair_chart(
        alt.Chart(strat_summary).mark_bar().encode(
            x="Strategy:N", y="Total PnL:Q",
            color=alt.condition(alt.datum["Total PnL"] > 0, alt.value("green"), alt.value("red")),
            tooltip=["Strategy", "Trades", "Avg PnL", "Total PnL"]
        ),
        use_container_width=True
    )

    # Symbol breakdown
    st.subheader("📊 Symbol Breakdown")
    symbol_summary = df_trades.groupby("Symbol")["PnL"].agg(["count", "mean", "sum"]).reset_index()
    symbol_summary.columns = ["Symbol", "Trades", "Avg PnL", "Total PnL"]
    st.dataframe(symbol_summary)
    st.altair_chart(
        alt.Chart(symbol_summary).mark_bar().encode(
            x="Symbol:N", y="Total PnL:Q",
            color=alt.condition(alt.datum["Total PnL"] > 0, alt.value("green"), alt.value("red")),
            tooltip=["Symbol", "Trades", "Avg PnL", "Total PnL"]
        ),
        use_container_width=True
    )

    # Summary counts
    st.subheader("📋 Trade Summary Overview")
    st.markdown(f"**Total Closed Trades:** {len(df_trades)}")
    st.markdown(f"**Symbols Traded:** {df_trades['Symbol'].nunique()}")
    st.markdown(f"**Strategies Used:** {df_trades['Strategy'].nunique()}")
    st.markdown(f"**Date Range:** {df_trades['Exit Date'].min().date()} → {df_trades['Exit Date'].max().date()}")

    # Questrade reconciliation
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
