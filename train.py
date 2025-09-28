import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")

# Example training dataset (replace with real one)
data = {
    "amount": [100, 500, 200, 300, 10000, 50, 700, 1200],
    "oldbalanceOrg": [1000, 1500, 300, 400, 20000, 60, 1500, 3000],
    "newbalanceOrig": [900, 1000, 100, 100, 15000, 10, 800, 2500],
    "oldbalanceDest": [0, 500, 0, 200, 5000, 0, 600, 2000],
    "newbalanceDest": [100, 1000, 200, 500, 15000, 50, 1000, 3500],
    "fraud": [0, 0, 0, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)
X = df.drop("fraud", axis=1)
y = df["fraud"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model trained and saved at {MODEL_PATH}")
