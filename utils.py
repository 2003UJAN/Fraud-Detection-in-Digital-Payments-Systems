import os
import joblib
import pandas as pd
from train import train_model, load_schema

MODEL_PATH = "model.pkl"

def load_model():
    """Load trained model, auto-train if missing."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not found. Auto-training a new one...")
        train_model()
    return joblib.load(MODEL_PATH)

def load_schema_safe():
    """Load schema.json safely via train.py helper."""
    return load_schema()

def preprocess_for_inference(input_data, schema_data):
    """
    Preprocess input (dict or DataFrame) into model-ready numeric features.
    - Maps categorical values using schema.json
    - Returns DataFrame with proper columns
    """

    # Load mappings
    loc_map = schema_data.get("mappings", {}).get("LOC_MAP", {})
    type_map = schema_data.get("mappings", {}).get("TYPE_MAP", {})

    # Convert dict → DataFrame if needed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Apply mappings
    if "location" in df.columns:
        df["location"] = df["location"].map(loc_map).fillna(-1).astype(int)
    if "transaction_type" in df.columns:
        df["transaction_type"] = df["transaction_type"].map(type_map).fillna(-1).astype(int)

    # Ensure all features exist
    expected_features = schema_data.get("features", [])
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # default fallback

    return df[expected_features]
