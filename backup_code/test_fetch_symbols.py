import os
import sys
import json
import requests
import logging
import time
from questrade_api import get_access_token


def fetch_and_cache_symbols():
    cache_path = "exports/symbol_list.json"
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    os.makedirs("exports", exist_ok=True)
    all_symbols = []

    for ch in "ABC":  # Limit for test
        try:
            response = requests.get(
                f"{api_server}/v1/symbols/search",
                headers=headers,
                params={"prefix": ch}
            )
            response.raise_for_status()
            symbols = response.json().get("symbols", [])
            print(f"✅ {ch} - Retrieved {len(symbols)} symbols.")
            all_symbols.extend(symbols)
        except Exception as e:
            print(f"❌ Failed {ch}: {e}")

    stock_symbols = [s for s in all_symbols if s.get("securityType") == "Stock" and s.get("isTradable")]
    symbol_names = list({s["symbol"] for s in stock_symbols})

    with open(cache_path, "w") as f:
        json.dump(symbol_names, f)
    print(f"✅ Cached {len(symbol_names)} tradable stock symbols.")

fetch_and_cache_symbols()

