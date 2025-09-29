import os
import joblib
from train import train_model, load_schema, preprocess

MODEL_PATH = "model.pkl"

def load_model():
    """Load trained model, auto-train if missing."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not found. Auto-training a new one...")
        train_model()
    return joblib.load(MODEL_PATH)

def load_schema_safe():
    """Load schema.json safely."""
    return load_schema()

def preprocess_for_inference(df):
    """Use same preprocessing as training for inference."""
    return preprocess(df)
