#questrade_api.py
import requests
import json
import time
import os
import csv
from datetime import datetime, timedelta, timezone
import pytz
import string
import logging
import threading
import pandas as pd
import streamlit as st

logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more verbosity
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("token_debug.log")  # You could add FileHandler() here if needed
    ]
)

# Use the correct token file name
TOKEN_PATH = "questrade_token.json"
REFRESH_URL = "https://login.questrade.com/oauth2/token"

def ping_api(token, api_server):
    try:
        url = f"{api_server}/v1/accounts"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print("✅ Ping successful - API connection alive.")
        st.session_state.last_ping = time.time()
    except Exception as e:
        print(f"❌ Ping failed: {e}")

def save_token(data):
    logging.debug("Saving token to file...")
    with open(TOKEN_PATH, "w") as f:
        json.dump(data, f)
    logging.info("Token saved successfully.")

def load_token():
    logging.debug("Loading token from file...")
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "r") as f:
            data = json.load(f)
        logging.info(f"Token loaded. Expiry: {data.get('expiry')}")
        return data
    logging.warning("Token file does not exist.")
    return None

def get_access_token():
    logging.info("Retrieving access token...")
    token_data = load_token()

    if not token_data:
        logging.error("Token file not found.")
        raise FileNotFoundError("Token file not found. Please authorize manually.")

    if "refresh_token" not in token_data:
        logging.error("Refresh token missing.")
        raise ValueError("Refresh token missing. Please re-authenticate.")

    # Check if the token is still valid
    if "expiry" in token_data:
        expiry = datetime.fromisoformat(token_data["expiry"])
        logging.debug(f"Token expiry time: {expiry.isoformat()}")
        if datetime.now() < expiry:
            logging.info("Access token is still valid.")
            return token_data["access_token"], token_data["api_server"].rstrip("/")

    logging.info("Access token expired or missing. Attempting to refresh...")

    # Refresh access token
    response = requests.get(
        REFRESH_URL,
        params={
            "grant_type": "refresh_token",
            "refresh_token": token_data["refresh_token"]
        }
    )

    logging.debug(f"Refresh response status: {response.status_code}")
    if response.status_code != 200:
        logging.error(f"Failed to refresh token: {response.text}")
        raise Exception(f"Failed to refresh token: {response.text}")

    new_data = response.json()
    new_data["expiry"] = (datetime.now() + timedelta(seconds=new_data["expires_in"])).isoformat()
    new_data["api_server"] = new_data["api_server"].rstrip("/")

    logging.info(f"New token acquired. Expires at: {new_data['expiry']}")
    save_token(new_data)

    return new_data["access_token"], new_data["api_server"]


def get_headers():
    access_token, _ = get_access_token()
    return {
        "Authorization": f"Bearer {access_token}"
    }

def get_account_info():
    access_token, api_server = get_access_token()
    url = f"{api_server}/v1/accounts"
    response = requests.get(url, headers=get_headers())
    return response.json()

def get_positions(account_number):
    access_token, api_server = get_access_token()
    url = f"{api_server}/v1/accounts/{account_number}/positions"
    response = requests.get(url, headers=get_headers())
    return response.json()

def search_symbol(symbol):
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{api_server}/v1/symbols/search"
    params = {"prefix": symbol}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def get_quote_data(ticker, token, api_server):
    symbol_id = get_symbol_id_by_ticker(ticker, token, api_server)
    if not symbol_id:
        raise ValueError(f"Symbol ID not found for {ticker}")
    
    url = f"{api_server}/v1/markets/quotes/{symbol_id}"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get("quotes", [{}])[0]  # Assuming quotes is a list


def get_candles(symbol_id, days=10):
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    response = requests.get(
        f"{api_server}/v1/markets/candles/{symbol_id}",
        headers=headers,
        params={
            "interval": "OneDay",
            "startTime": start_time.isoformat(),
            "endTime": end_time.isoformat()
        }
    )
    response.raise_for_status()
    return response.json().get("candles", [])

def fetch_candles(symbol_id, start_time, end_time, token, api_server, days=90):
    url = f"{api_server}/v1/markets/candles/{symbol_id}"
    headers = {"Authorization": f"Bearer {token}"}
    if not start_time or not end_time:
        end_time = datetime.now(timezone.utc).replace(microsecond=0)
        #days = 90
        start_time = end_time - timedelta(days=days)

    # Format timestamps correctly
    start_iso = start_time.astimezone().isoformat()
    end_iso = end_time.astimezone().isoformat()
    interval = "OneDay"

    params = {
        "startTime": start_iso,
        "endTime": end_iso,
        "interval": interval
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        #st.write("Raw API response for", symbol, ":", response.json())
        if "candles" not in data:
            raise ValueError(f"'candles' key not in response: {data}")

        candles = data["candles"]
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["start"], utc=True)
        return df

    except Exception as e:
        raise RuntimeError(f"Error fetching candles for {symbol_id}: {e}")
    
def place_order(account_number, symbol_id, quantity, action, order_type="Market", **kwargs):
    access_token, api_server = get_access_token()
    url = f"{api_server}/v1/accounts/{account_number}/orders"
    order = {
        "symbolId": symbol_id,
        "quantity": quantity,
        "action": action,
        "orderType": order_type,
        "timeInForce": "GoodTillCancelled"
    }
    order.update(kwargs)
    response = requests.post(url, headers=get_headers(), json=order)
    return response.json()

def get_all_symbols_with_cache(cache_path="exports/symbol_list.json", cache_expiry_days=1, verbose=False):
    # Check cache freshness
    if os.path.exists(cache_path):
        modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - modified < timedelta(days=cache_expiry_days):
            if verbose:
                print("✅ Loaded symbols from recent cache.")
            with open(cache_path, "r") as f:
                return json.load(f)

    # Fetch from API
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    all_symbols = []

    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        try:
            response = requests.get(
                f"{api_server}/v1/symbols/search",
                headers=headers,
                params={"prefix": ch}
            )
            response.raise_for_status()
            batch = response.json().get("symbols", [])
            if verbose:
                print(f"✅ {ch} - Retrieved {len(batch)} symbols.")
            all_symbols.extend(batch)
        except Exception as e:
            print(f"❌ Failed {ch}: {e}")

    # Filter
    stock_symbols = [s for s in all_symbols if s.get("securityType") == "Stock" and s.get("isTradable")]
    symbol_list = sorted({s["symbol"] for s in stock_symbols})

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(symbol_list, f, indent=2)

    if verbose:
        print(f"✅ Cached {len(symbol_list)} tradable stock symbols.")

    return symbol_list

def get_symbol_id_by_ticker(ticker, access_token, api_server):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{api_server}/v1/symbols/search"
    params = {"prefix": ticker}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json().get("symbols", [])
        for item in results:
            if item["symbol"] == ticker:
                return item["symbolId"]
    except Exception as e:
        print(f"⚠️ Failed to fetch symbolId for {ticker}: {e}")

    return None

def get_symbol_id(ticker, access_token, api_server):
    url = f"{api_server}/v1/symbols/search?prefix={ticker}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if matches := [
        item for item in data.get("symbols", []) if item["symbol"] == ticker
    ]:
        return matches[0]["symbolId"]
    else:
        raise ValueError(f"Symbol ID not found for {ticker}")
