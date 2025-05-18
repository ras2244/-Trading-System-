import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(input_path="exports/model_training_data.csv", model_path="exports/signal_scoring_model.pkl"):
    df = pd.read_csv(input_path)

    features = ["avg_close", "volatility", "momentum", "volume_avg"]
    if "exit_avg_close" in df.columns:
        features += ["exit_avg_close", "exit_volatility", "exit_momentum", "exit_volume_avg"]

    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    #print("✅ Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    #print(f"✅ Model saved to: {model_path}")
