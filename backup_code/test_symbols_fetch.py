from questrade_api import get_all_symbols_with_cache

symbols = get_all_symbols_with_cache()
print(f"✅ Loaded {len(symbols)} symbols from API or cache.")
print("Sample symbols:", symbols[:10])

def load_watchlist(min_volume=100_000, min_price=1):
    access_token, api_server = get_access_token()

    # Step 1: Load index-based symbols (S&P 500 etc. from cache)
    all_tickers = get_all_symbols_with_cache()
    watchlist = []

    for ticker in all_tickers:
        try:
            symbol_id = get_symbol_id_by_ticker(ticker, access_token, api_server)
            if not symbol_id:
                continue

            quote = get_quote(symbol_id)
            quote_data = quote.get("quotes", [])[0]

            price = quote_data.get("lastTradePrice")
            volume = quote_data.get("volume")

            # Only evaluate if both values are valid numbers
            if price is not None and volume is not None:
                if price >= min_price and volume >= min_volume:
                    watchlist.append({"symbol": ticker, "symbolId": symbol_id})
            else:
                print(f"⚠️ Skipping {ticker}: Missing price or volume (price={price}, volume={volume})")


            # Optional: sleep briefly to avoid rate limits
            time.sleep(0.05)

        except Exception as e:
            print(f"⚠️ Skipped {ticker}: {e}")

    print(f"✅ Final watchlist contains {len(watchlist)} symbols")
    print([s['symbol'] for s in watchlist])
    return watchlist

def select_stocks_for_today(return_watchlist=False, gap_threshold=2.0):
    access_token, api_server = get_access_token()

    # Step 1: Load S&P 500 tickers
    tickers = get_sp500_tickers()

    # Step 2: Liquidity + price filtering
    filtered = []
    for ticker in tickers:
        try:
            symbol_id = get_symbol_id_by_ticker(ticker, access_token, api_server)
            if not symbol_id:
                continue
            df = get_price_data(symbol_id, days=10)
            if df is None or len(df) < 5:
                continue

            avg_volume = df["volume"].tail(5).mean()
            last_close = df["close"].iloc[-1]

            if avg_volume >= 1_000_000 and last_close >= 5:
                filtered.append({"symbol": ticker, "symbolId": symbol_id})
        except Exception as e:
            print(f"⚠️ Skipping {ticker}: {e}")

    print(f"✅ Filtered liquid tickers: {len(filtered)}")

    # Step 3: Premarket scan
    gap_results = scan_premarket_gainers([s["symbol"] for s in filtered], threshold=gap_threshold)
    gap_tickers = {g["symbol"] for g in gap_results}
    print(f"📈 Premarket movers (±{gap_threshold}%): {gap_tickers}")

    final_watchlist = [s for s in filtered if s["symbol"] in gap_tickers]
    print(f"✅ Final watchlist contains {len(final_watchlist)} symbols")

    # Step 4: Load or train model
    model = load_model("stock_selector_model.pkl")
    need_training = False

    if model is None:
        print("⚠️ No model found. Will train a new one.")
        need_training = True
    else:
        try:
            model.predict([[0] * len(FEATURE_COLUMNS)])
        except NotFittedError:
            print("⚠️ Model not fitted.")
            need_training = True
        except Exception as e:
            print(f"⚠️ Model check error: {e}")
            need_training = True

    if need_training:
        df_train = generate_training_data(final_watchlist, days=90)
        if df_train.empty or df_train["label"].nunique() < 2:
            print("❌ Insufficient training data.")
            return []
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df_train[FEATURE_COLUMNS], df_train["label"])
        save_model(model, "stock_selector_model.pkl")
        print("✅ Model trained.")

    # Step 5: Score selected symbols
    results = []
    for symbol_obj in final_watchlist:
        result = realtime_score_symbol(symbol_obj, model)
        if result["score"] is not None:
            results.append(result)

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    if return_watchlist:
        return sorted_results[:NUM_AI_STOCKS], final_watchlist
    return sorted_results[:NUM_AI_STOCKS]