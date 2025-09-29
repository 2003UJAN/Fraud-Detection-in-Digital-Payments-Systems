import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from train import train_model

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not found, training a new one...")
        train_model()
    return joblib.load(MODEL_PATH)
