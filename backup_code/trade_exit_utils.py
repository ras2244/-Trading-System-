import os
import pandas as pd
import streamlit as st


def auto_close_open_trades(df_price, hold_period=5, log_path="exports/trades/trade_log_breakout.csv"):
    """
    Auto-close open trades after a fixed hold period using daily close prices.
    df_price: must include 'symbol', 'datetime', and 'close' columns
    """
    if not os.path.exists(log_path):
        return

    df_log = pd.read_csv(log_path)
    if "Exit Date" not in df_log.columns or "Exit Price" not in df_log.columns:
        return

    updated = False

    for i, row in df_log.iterrows():
        if pd.isna(row["Exit Date"]) or row["Exit Date"] == "":
            entry_date = pd.to_datetime(row["Entry Date"])
            symbol = row["Symbol"]

            try:
                df_symbol = df_price[df_price["symbol"] == symbol].copy()
                df_symbol["datetime"] = pd.to_datetime(df_symbol["datetime"], utc=True)
                df_symbol.set_index("datetime", inplace=True)
                df_symbol.sort_index(inplace=True)

                if entry_date in df_symbol.index:
                    entry_idx = df_symbol.index.get_loc(entry_date)
                else:
                    entry_idx = df_symbol.index.get_indexer([entry_date], method="nearest")[0]

                exit_idx = entry_idx + hold_period
                if exit_idx < len(df_symbol):
                    exit_row = df_symbol.iloc[exit_idx]
                    df_log.at[i, "Exit Date"] = exit_row.name.isoformat()
                    df_log.at[i, "Exit Price"] = exit_row["close"]
                    df_log.at[i, "PnL"] = round(exit_row["close"] - float(row["Entry Price"]), 2)
                    updated = True
            except Exception as e:
                st.warning(f"Auto-close failed for {symbol}: {e}")

    if updated:
        df_log.to_csv(log_path, index=False)
        st.success("Auto-close applied to eligible open trades.")
