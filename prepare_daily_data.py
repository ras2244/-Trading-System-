#premarket_scanner.py
import os
import json
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from ai_stock_selector import load_watchlist_top_liquid
from utils.model_utils import load_model, FEATURE_COLUMNS, extract_features
from questrade_api import get_quote_data, get_access_token, get_symbol_id
from utils.price_data import get_price_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


# Configuration
JSON_OUTPUT_PATH = "exports/daily_ai_selection_scores.json"
CSV_OUTPUT_PATH = "exports/daily_ai_selection_summary.csv"
RAW_DATA_PATH = "exports/premarket_raw_data.json"
NUM_AI_STOCKS = 4

def scan_finviz_premarket_gainers(min_gap_pct=2.0):
    url = "https://finviz.com/screener.ashx?v=111&s=ta_premarket"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch Finviz screener page: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="table-light")
    if not table:
        all_tables = soup.find_all("table")
        for t in all_tables:
            if "Ticker" in t.text and "Change" in t.text:
                table = t
                break

    if not table:
        print("❌ Screener table not found on Finviz page.")
        return []

    gainers = []
    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 10:
            continue
        try:
            ticker = cols[1].text.strip()
            name = cols[2].text.strip()
            price_str = cols[8].text.strip().replace("$", "").replace(",", "")
            change_str = cols[6].text.strip().replace("%", "").replace("+", "").strip()
            volume_str = cols[9].text.strip().replace(",", "")

            if any(x in ticker.lower() for x in ["ticker", "etf", "custom"]) or not change_str or not price_str:
                continue

            try:
                price = float(price_str)
                gap_pct = float(change_str)
            except ValueError:
                continue

            if gap_pct >= min_gap_pct:
                gainers.append({
                    "symbol": ticker,
                    "name": name,
                    "price": price,
                    "gap_pct": gap_pct,
                    "volume": volume_str
                })
        except Exception as e:
            print(f"⚠️ Skipping row due to error: {e}")

    print(f"✅ Found {len(gainers)} premarket gainers with gap > {min_gap_pct}%")
    return gainers

def fetch_symbol_snapshot(row, gainers, access_token, api_server):
    symbol = row["symbol"]
    try:
        symbol_id = get_symbol_id(symbol, access_token, api_server)
        quote = get_quote_data(symbol, access_token, api_server)
        price = quote.get("lastTradePrice")
        volume = quote.get("volume")
        gap_info = next((g for g in gainers if g["symbol"] == symbol), {})
        gap_pct = gap_info.get("gap_pct") or quote.get("percentChange")

        return {
            "symbol": symbol,
            "symbolId": symbol_id,
            "price": price,
            "volume": volume,
            "gap_pct": gap_pct
        }
    except Exception as e:
        print(f"⚠️ Quote failed for {symbol}: {e}")
        return None

def gather_premarket_data():
    access_token, api_server = get_access_token()
    today_str = datetime.now().strftime("%Y-%m-%d")

    try:
        df_top = pd.read_csv("top_100_sp500.csv")
    except Exception as e:
        print(f"❌ Failed to load top_100_sp500.csv: {e}")
        return

    gainers = scan_finviz_premarket_gainers(min_gap_pct=2.0)

    snapshot = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_symbol_snapshot, row, gainers, access_token, api_server): row["symbol"]
            for _, row in df_top.iterrows()
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                snapshot.append(result)

    os.makedirs("exports", exist_ok=True)
    with open(RAW_DATA_PATH, "w") as f:
        json.dump({today_str: snapshot}, f, indent=2)
    print(f"📁 Premarket snapshot saved to {RAW_DATA_PATH}")

def prepare_daily_data():
    today_str = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Raw premarket data file not found: {RAW_DATA_PATH}")
        return

    with open(RAW_DATA_PATH, "r") as f:
        raw_data = json.load(f).get(today_str, [])

    model = load_model("stock_selector_model.pkl")
    if model is None:
        print("❌ Model not found. Please train it first.")
        return

    print(f"✅ Scoring {len(raw_data)} symbols from premarket snapshot")
    candidates = []

    def score_candidate(item):
        try:
            candles = get_price_data(item["symbolId"], days=20)

            if candles is None or candles.empty:
                raise ValueError("No candle data")

            features = extract_features(candles)
            last_row = features.iloc[-1]

            if not all(col in last_row for col in FEATURE_COLUMNS):
                missing = [col for col in FEATURE_COLUMNS if col not in last_row]
                print(f"⚠️ Missing model input columns for {item['symbol']}: {missing}")
                return None

            X = pd.DataFrame([last_row[FEATURE_COLUMNS]])
            score = model.predict_proba(X)[0][1]

            return {
                "symbol": item["symbol"],
                "symbolId": item["symbolId"],
                "score": round(score, 4),
                "gap_pct": item.get("gap_pct"),
                "price": item.get("price"),
                "volume": item.get("volume"),
                "features": last_row.to_dict()
            }
        except Exception as e:
            print(f"⚠️ Error scoring {item['symbol']}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {
            executor.submit(score_candidate, item): item for item in raw_data
        }
        for future in as_completed(future_to_symbol):
            result = future.result()
            if result and result.get("score") is not None:
                candidates.append(result)

    existing_data = {}
    if os.path.exists(JSON_OUTPUT_PATH):
        with open(JSON_OUTPUT_PATH, "r") as f:
            existing_data = json.load(f)

    existing_data[today_str] = sorted(candidates, key=lambda x: x["score"], reverse=True)

    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(existing_data, f, indent=2)

    df_summary = pd.DataFrame(candidates)
    if not df_summary.empty:
        df_summary_simple = df_summary[["symbol", "score", "price", "volume", "gap_pct"]].copy()
        df_summary_simple.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"📁 Exported CSV summary to {CSV_OUTPUT_PATH}")

    print(f"✅ Saved {len(candidates)} scored stocks to {JSON_OUTPUT_PATH}")

if __name__ == "__main__":
    gather_premarket_data()
    prepare_daily_data()

