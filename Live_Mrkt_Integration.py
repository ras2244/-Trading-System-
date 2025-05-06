import yfinance as yf
import pandas as pd
from datetime import datetime

# === Parameters ===
symbols = ['AAPL', 'TSLA', 'MSFT']
breakout_window = 20
short_window = 20
long_window = 50

# === Breakout Signal Function ===
def check_breakout(symbol):
    data = yf.download(symbol, period='60d', interval='1d', progress=False)
    if data.empty or 'High' not in data.columns:
        return False

    data['20d_high'] = data['High'].rolling(window=breakout_window).max()
    if data[['Close', '20d_high']].isna().sum().sum() > 0:
        return False

    latest = data.iloc[-1]
    yesterday = data.iloc[-2]

    try:
        breakout = float(latest['Close']) > float(yesterday['20d_high'])
    except:
        breakout = False
    return breakout

# === Moving Average Crossover Signal Function ===
def check_ma_crossover(symbol):
    data = yf.download(symbol, period='90d', interval='1d', progress=False)
    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()
    data = data.dropna()
    if len(data) < 2:
        return False

    prev = data.iloc[-2]
    curr = data.iloc[-1]

    try:
        crossover = (
            float(prev['SMA_Short']) < float(prev['SMA_Long']) and
            float(curr['SMA_Short']) > float(curr['SMA_Long'])
        )
    except:
        crossover = False
    return crossover

# === Run Signal Checks ===
alerts = []
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

for symbol in symbols:
    bo_signal = check_breakout(symbol)
    ma_signal = check_ma_crossover(symbol)

    if bo_signal:
        alerts.append({'Timestamp': now, 'Symbol': symbol, 'Strategy': 'Breakout'})
    if ma_signal:
        alerts.append({'Timestamp': now, 'Symbol': symbol, 'Strategy': 'MA_Crossover'})

# === Display Alerts ===
print(f"\n{now} — Live Signal Alerts:")
if alerts:
    for alert in alerts:
        print(f"{alert['Symbol']} triggered {alert['Strategy']} signal")
else:
    print("No signals triggered.")

# === Optional: Save to CSV log ===
if alerts:
    df_alerts = pd.DataFrame(alerts)
    df_alerts.to_csv("exports/live_alerts.csv", index=False, mode='a', header=not pd.io.common.file_exists("exports/live_alerts.csv"))
    print("Alerts saved to: exports/live_alerts.csv")

