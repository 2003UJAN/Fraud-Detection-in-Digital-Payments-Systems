import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)
