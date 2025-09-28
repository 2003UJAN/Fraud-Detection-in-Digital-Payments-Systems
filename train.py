import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")

# Example dummy dataset (replace with real dataset)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model trained and saved at {MODEL_PATH}")
