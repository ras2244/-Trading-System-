from ai_stock_selector import load_watchlist_top_liquid
from utils.model_utils import generate_training_data, save_model, FEATURE_COLUMNS
from utils.price_data import get_price_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 🔍 Load symbols
watchlist = load_watchlist_top_liquid()

print(f"🔍 Checking quotes for {len(watchlist)} tickers...")

all_data = []

# 📥 Collect price data and generate training rows per stock
for item in watchlist:
    symbol = item["symbol"]
    symbol_id = item["symbolId"]
    try:
        df = get_price_data(symbol_id, days=90)
        df["symbol"] = symbol  # Track symbol per row
        df = generate_training_data(df)
        all_data.append(df)
    except Exception as e:
        print(f"⚠️ Skipped {symbol}: {e}")

# 🧠 Merge and train if we have data
if all_data:
    df_train = pd.concat(all_data, ignore_index=True)
    print("📊 Label distribution:\n", df_train["label"].value_counts())

    # 🎯 Features & labels
    X = df_train[FEATURE_COLUMNS]
    y = df_train["label"]

    # 🔪 Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🌳 Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 🧪 Evaluate
    y_pred = model.predict(X_test)
    print("📈 Classification Report:\n", classification_report(y_test, y_pred))

    # 💾 Save
    save_model(model)
else:
    print("❌ No training data was generated.")