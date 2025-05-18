from questrade_api import get_all_symbols_with_cache

symbols = get_all_symbols_with_cache()
print(f"✅ Loaded {len(symbols)} symbols from API or cache.")
print("Sample symbols:", symbols[:10])

