#premarket_scanner.py

from yahoo_fin import stock_info as si

def get_premarket_gap(ticker):
    try:
        data = si.get_quote_table(ticker, dict_result=True)
        pm_price = data.get("Pre-Market Price") or data.get("Ask")  # fallback
        reg_price = data.get("Previous Close")

        if pm_price and reg_price:
            gap_pct = (pm_price - reg_price) / reg_price * 100
            return round(gap_pct, 2)
    except Exception as e:
        print(f"⚠️ {ticker} failed: {e}")
    return None

def scan_premarket_gainers(ticker_list, threshold=2.0):
    results = []
    for ticker in ticker_list:
        gap = get_premarket_gap(ticker)
        if gap and abs(gap) >= threshold:
            results.append({"symbol": ticker, "gap_pct": gap})
    return results
