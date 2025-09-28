import joblib
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")

def train_dummy_model():
    """Train a quick dummy model if no saved model exists"""
    print("⚠️ No trained model found. Training a new dummy model...")

    # Small dummy dataset (replace with real dataset if available)
    data = {
        "amount": [100, 500, 200, 300, 10000, 50],
        "oldbalanceOrg": [1000, 1500, 300, 400, 20000, 60],
        "newbalanceOrig": [900, 1000, 100, 100, 15000, 10],
        "oldbalanceDest": [0, 500, 0, 200, 5000, 0],
        "newbalanceDest": [100, 1000, 200, 500, 15000, 50],
        "fraud": [0, 0, 0, 0, 1, 0],
    }

    df = pd.DataFrame(data)
    X = df.drop("fraud", axis=1)
    y = df["fraud"]

    # Train a Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Dummy model trained and saved at {MODEL_PATH}")
    return model

def load_model():
    """Load model if exists, else auto-train a dummy model"""
    if not os.path.exists(MODEL_PATH):
        return train_dummy_model()
    return joblib.load(MODEL_PATH)
