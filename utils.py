import joblib
import os
import json

MODEL_PATH = "fraud_model.pkl"
SCHEMA_PATH = "feature_schema.json"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)

def load_schema():
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"❌ Schema not found at {SCHEMA_PATH}. Run train.py first.")
    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)["features"]
