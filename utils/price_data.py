# Here's the updated get_price_data() function using Option 1, which returns a dictionary
# structured like Questrade's native API format.

import os
import requests
import json
from urllib.parse import quote
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
from questrade_api import get_access_token, fetch_candles


def get_price_data(symbol_id, days=90, interval="OneDay", start_time=None, end_time=None, max_retries=3):
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    # Round to nearest second to avoid microseconds (which break Questrade API)
    if not start_time or not end_time:
        end_time = datetime.now(timezone.utc).replace(microsecond=0)
        start_time = end_time - timedelta(days=days)

    # Format timestamps correctly
    start_iso = start_time.astimezone().isoformat(),
    end_iso = end_time.astimezone().isoformat(),

    params = {
        "startTime": start_iso,
        "endTime": end_iso,
        "interval": interval
    }

    url = f"{api_server}/v1/markets/candles/{symbol_id}"
    print(f"📡 Requesting candles: {url}")
    print(f"🧾 Params: {params}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            candles = response.json().get("candles", [])
            if not candles:
                print(f"⚠️ Empty candle list for symbol_id {symbol_id}")
                return None
            return pd.DataFrame(candles)
        except requests.exceptions.HTTPError as http_err:
            if response.status_code in {400, 404}:
                print(f"❌ Symbol ID {symbol_id} returned 400/404: likely invalid or unsupported interval.")
                break
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed with error: {e}")

        time.sleep(2 ** attempt)

    print(f"❌ Exhausted retries for symbol_id {symbol_id}")
    return None