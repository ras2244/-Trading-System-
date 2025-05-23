from utils.get_candles import ensure_candle_file
from sklearn.ensemble import RandomForestClassifier
from utils.model_utils import save_model, load_model

ensure_candle_file("AAPL")
# Create a dummy model
dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)

# Save the model
save_success = save_model(dummy_model, filename="stock_selector_model.pkl")

# Attempt to load the model back
if save_success:
    loaded_model = load_model(filename="stock_selector_model.pkl")