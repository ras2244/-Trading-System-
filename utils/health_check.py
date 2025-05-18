# utils/health_check.py

import requests
from questrade_api import get_access_token

def check_api_health():
    try:
        access_token, api_server = get_access_token()
        print("🔍 Token (first 10):", access_token[:10])
        print("🔗 API Server:", api_server)

        api_server = api_server.rstrip("/")
        url = f"{api_server}/v1/accounts"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(url, headers=headers)

        print("🔁 Response:", response.status_code, response.text)

        if response.status_code == 200:
            return True, "✅ API and token are valid"
        else:
            return False, f"❌ Token rejected: {response.status_code}"
    except Exception as e:
        return False, f"❌ Exception: {e}"
