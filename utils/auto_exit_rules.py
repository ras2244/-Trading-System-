# utils/auto_exit_rules.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
import streamlit as st

def rsi_exit(df, rsi_window=14, overbought_level=70):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] > overbought_level

def ma_cross_down_exit(df, fast=5, slow=20):
    df["ma_fast"] = df["close"].rolling(window=fast).mean()
    df["ma_slow"] = df["close"].rolling(window=slow).mean()
    return df["ma_fast"].iloc[-2] > df["ma_slow"].iloc[-2] and df["ma_fast"].iloc[-1] < df["ma_slow"].iloc[-1]

def pnl_exit(entry_price, current_price, gain_threshold=0.05, loss_threshold=-0.03):
    pnl_pct = (current_price - entry_price) / entry_price
    return pnl_pct >= gain_threshold or pnl_pct <= loss_threshold

def auto_exit_trades(trade_log_path="exports/trades/trade_log_breakout.csv", candle_dir="exports/candles", ai_threshold=0.3):
    df_trades = pd.read_csv(trade_log_path)
    df_trades["Entry Date"] = pd.to_datetime(df_trades["Entry Date"], utc=True)
    df_trades["Exit Date"] = pd.to_datetime(df_trades["Exit Date"], errors="coerce", utc=True)

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
            print(f"🚪 Auto-exit triggered for {symbol} | Score: {score:.2f}")
            st.warning(f"🚪 Auto-exit triggered for {symbol} | Score: {score:.2f}")

    df_trades.to_csv(trade_log_path, index=False)
    #print("✅ Auto-exit logic executed and trade log updated.")
    #st.success("✅ Auto-exit logic executed and trade log updated.")