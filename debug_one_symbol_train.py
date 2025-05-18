import pandas as pd
from utils.get_candles import ensure_candle_file
from utils.model_utils import save_model, debug_print_matching_entry, generate_training_data
from sklearn.ensemble import RandomForestClassifier

symbol = "AAPL"
ensure_candle_file(symbol)

# Generate training data using your trade log and candle file
df_train = generate_training_data(
    tickers=[symbol],
    trade_log_path="exports/trades/trade_log_breakout.csv",
    candle_dir="exports/candles"
)

# Print sample output for debug
if df_train.empty:
    print("❌ No training rows built from trades.")
else:
    print(f"✅ Training data rows for {symbol}: {len(df_train)}")
    print(df_train.head())

    # Train a quick model for debugging
    try:
        FEATURE_COLUMNS = [
            "avg_close", "volatility", "momentum", "volume_avg",
            "delta_1d", "delta_5d", "bb_width", "macd_hist", "atr"
        ]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(df_train[FEATURE_COLUMNS], df_train["label"])
        save_model(model, filename="debug_model.pkl")
    except Exception as e:
        print(f"❌ Failed during model training: {e}")
