import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json
import os

MODEL_PATH = "fraud_model.pkl"
SCHEMA_PATH = "feature_schema.json"

def generate_training_data(n_samples=5000):
    np.random.seed(42)

    data = pd.DataFrame({
        "amount": np.random.uniform(1, 20000, n_samples),
        "time": np.random.randint(0, 24, n_samples),
        "loc_encoded": np.random.choice([0, 1, 2, 3], n_samples),
        "merchant_encoded": np.random.choice([0, 1, 2, 3, 4], n_samples),
        "device_encoded": np.random.choice([0, 1, 2], n_samples),
        "previous_transactions": np.random.randint(0, 1000, n_samples)
    })

    # Rule-based fraud simulation
    data["is_fraud"] = (
        (data["amount"] > 15000) |
        ((data["time"] < 6) & (data["amount"] > 5000)) |
        ((data["loc_encoded"] == 2) & (data["amount"] > 10000))
    ).astype(int)

    return data

def train_model():
    data = generate_training_data()

    X = data.drop("is_fraud", axis=1)
    y = data["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # Save feature schema
    schema = {"features": list(X.columns)}
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=4)
    print(f"✅ Feature schema saved to {SCHEMA_PATH}")

if __name__ == "__main__":
    train_model()
