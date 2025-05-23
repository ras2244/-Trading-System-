import pandas as pd
import requests
import time

API_KEY = "da0dOE0XqdBsvXAoFk2HfF14xTkU1epe"
CSV_PATH = "data/sp500_constituents.csv"

def main():
    # Load tickers
    sp500_df = pd.read_csv(CSV_PATH)
    tickers = sp500_df["Symbol"].dropna().str.strip().tolist()
    print(f"📌 Loaded {len(tickers)} tickers from CSV.")

    # Batch fetch from FMP
    def fetch_fmp_data(symbols):
        url = f"https://financialmodelingprep.com/api/v3/quote/{','.join(symbols)}?apikey={API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching data for batch: {symbols} -> {e}")
            return []

    all_data = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"🔄 Fetching batch {i // batch_size + 1}...")
        all_data.extend(fetch_fmp_data(batch))
        time.sleep(1)

    df = pd.DataFrame(all_data)
    required_cols = {"symbol", "name", "marketCap", "volume"}
    if not required_cols.issubset(df.columns):
        print("❌ One or more required fields missing from API response.")
        return

    df = df[["symbol", "name", "marketCap", "volume"]].dropna()
    df["score"] = df["marketCap"] * 0.7 + df["volume"] * 0.3
    top_100 = df.sort_values("score", ascending=False).head(100)
    top_100.to_csv("top_100_sp500.csv", index=False)
    print("✅ Saved top 100 stocks to top_100_sp500.csv")

if __name__ == "__main__":
    main()

