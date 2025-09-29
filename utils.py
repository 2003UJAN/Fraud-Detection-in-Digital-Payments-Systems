import joblib
import os
import json
from train import train_model

MODEL_PATH = "fraud_model.pkl"
SCHEMA_PATH = "feature_schema.json"

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCHEMA_PATH):
        print("⚠️ Model or schema not found. Training a new model...")
        train_model()
    return joblib.load(MODEL_PATH)

def load_schema():
    if not os.path.exists(SCHEMA_PATH):
        print("⚠️ Schema not found. Training a new model...")
        train_model()
    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)
