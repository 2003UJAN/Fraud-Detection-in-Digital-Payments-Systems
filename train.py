import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)

    data = pd.DataFrame({
        "amount": np.random.uniform(1, 20000, n_samples),  # transaction amount
        "time": np.random.randint(0, 24, n_samples),       # hour of day
        "location": np.random.choice(["US", "EU", "ASIA", "AFRICA"], n_samples),
        "merchant_category": np.random.choice(["electronics", "fashion", "grocery", "gaming", "others"], n_samples),
        "device_type": np.random.choice(["mobile", "desktop", "tablet"], n_samples),
        "previous_transactions": np.random.randint(0, 1000, n_samples),
    })

    # Encode categorical manually
    data["loc_encoded"] = data["location"].astype("category").cat.codes
    data["merchant_encoded"] = data["merchant_category"].astype("category").cat.codes
    data["device_encoded"] = data["device_type"].astype("category").cat.codes

    features = data[["amount", "time", "loc_encoded", "merchant_encoded", "device_encoded", "previous_transactions"]]

    # Label fraud based on rules + noise
    data["is_fraud"] = (
        (data["amount"] > 10000) |
        (data["time"].isin([1, 2, 3])) |
        ((data["location"] == "ASIA") & (data["amount"] > 7000)) |
        (data["device_type"] == "tablet")
    ).astype(int)

    return features, data["is_fraud"]

def train_model():
    X, y = generate_synthetic_data()
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
