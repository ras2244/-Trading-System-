import requests
import json
import time
import os
import csv
from datetime import datetime, timedelta, timezone
import pytz

# 🔁 Store this in a safe place or load from a secret manager
# REFRESH_TOKEN = "bk7xYmnYB16f9zeBoZwM03dRLBwcpsYO0"
# TOKEN_URL = "https://login.questrade.com/oauth2/token"

# Store cached token so we don't refresh too often
_token_cache = {}

TOKEN_FILE = "questrade_tokens.json"
APP_KEY = "1ynygCQMoEluN1yfxfjMYlP0QC0H9UKX0"  # Replace with your Questrade App Key

def save_tokens(data):
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f)

def load_tokens():
    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError("Token file not found. Please authorize manually.")
    with open(TOKEN_FILE, "r") as f:
        return json.load(f)

def refresh_access_token(refresh_token):
    url = f"https://login.questrade.com/oauth2/token?grant_type=refresh_token&refresh_token={refresh_token}"
    r = requests.get(url)
    r.raise_for_status()
    tokens = r.json()
    tokens['timestamp'] = datetime.now().isoformat()
    save_tokens(tokens)
    return tokens

def get_access_token():
    tokens = load_tokens()
    timestamp = datetime.fromisoformat(tokens.get("timestamp", datetime.now().isoformat()))
    expires_in = tokens.get("expires_in", 1800)  # seconds
    expires_at = timestamp + timedelta(seconds=expires_in)

    if datetime.now() >= expires_at:
        tokens = refresh_access_token(tokens["refresh_token"])
    
    return tokens["access_token"], tokens["api_server"]

def get_account_info():
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{api_server}v1/accounts"

    r = requests.get(url, headers=headers)
    return r.json()


def get_quote(symbol_id):
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{api_server}v1/markets/quotes/{symbol_id}"

    r = requests.get(url, headers=headers)
    return r.json()


def search_symbol(keyword):
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{api_server}v1/symbols/search?prefix={keyword}"

    r = requests.get(url, headers=headers)
    return r.json()

def get_candles(symbol_id, interval='OneDay', days=21):
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{api_server}v1/markets/candles/{symbol_id}"

    eastern = pytz.timezone("US/Eastern")
    end = datetime.now(eastern)
    start = end - timedelta(days=days * 2)  # buffer for weekends/holidays

    params = {
        "interval": interval,
        "startTime": start.isoformat(),  # includes offset
        "endTime": end.isoformat()
    }

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    return r.json()

def place_order(account_number, symbol_id, quantity=1, action="Buy", order_type="Market", limitPrice=None, stopPrice=None):
    access_token, api_server = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    order = {
        "accountNumber": account_number,
        "symbolId": symbol_id,
        "quantity": quantity,
        "action": action,
        "orderType": order_type,
        "timeInForce": "Day",
        "isAllOrNone": False,
        "isAnonymous": False,
        "isLimitOffsetInDollar": False
    }

    if limitPrice is not None:
        order["limitPrice"] = limitPrice
    if stopPrice is not None:
        order["stopPrice"] = stopPrice

    url = f"{api_server}v1/accounts/{account_number}/orders"
    r = requests.post(url, headers=headers, json=order)
    r.raise_for_status()
    return r.json()

def log_live_trade(symbol, signal, price, order_type="Signal"):
    file_exists = os.path.isfile(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Datetime", "Symbol", "Signal", "Price", "OrderType"])
        writer.writerow([datetime.now().isoformat(), symbol, signal, price, order_type])

