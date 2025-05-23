import pandas as pd
import os

def get_sp500_tickers(csv_path="data/sp500_constituents.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please download the S&P 500 constituents file.")
    
    df = pd.read_csv(csv_path)
    if "Symbol" not in df.columns:
        raise ValueError("CSV must contain a 'Symbol' column.")

    tickers = df["Symbol"].dropna().unique().tolist()
    return sorted(tickers)