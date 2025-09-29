import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import os

MODEL_PATH = "model.pkl"
SCHEMA_PATH = "schema.json"
DATA_PATH = "training_data.csv"

def load_schema():
    """Safely load schema.json when needed."""
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"❌ {SCHEMA_PATH} not found. Please add schema.json in project root.")
    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)

def generate_dataset(n_samples=5000):
    """Generate synthetic fraud detection dataset."""
    schema_data = load_schema()
    LOC_MAP = schema_data["mappings"]["LOC_MAP"]
    TYPE_MAP = schema_data["mappings"]["TYPE_MAP"]

    np.random.seed(42)

    amounts = np.random.uniform(10, 5000, n_samples).round(2)
    transaction_types = np.random.choice(list(TYPE_MAP.keys()), n_samples)
    locations = np.random.choice(list(LOC_MAP.keys()), n_samples)
    times = np.random.randint(0, 24, n_samples)
    devices = np.random.randint(1000, 1100, n_samples)

    fraud_prob = (
        (amounts > 3000).astype(int) * 0.3
        + (times > 22).astype(int) * 0.2
        + (np.isin(transaction_types, ["Online", "Transfer"]).astype(int) * 0.3)
        + np.random.uniform(0, 0.2, n_samples)
    )

    labels = (fraud_prob > 0.5).astype(int)

    df = pd.DataFrame({
        "amount": amounts,
        "transaction_type": transaction_types,
        "location": locations,
        "time": times,
        "device_id": devices,
        "label": labels
    })

    return df

def preprocess(df):
    """Convert categorical values to numerical using mappings."""
    schema_data = load_schema()
    LOC_MAP = schema_data["mappings"]["LOC_MAP"]
    TYPE_MAP = schema_data["mappings"]["TYPE_MAP"]

    df["transaction_type"] = df["transaction_type"].map(TYPE_MAP).fillna(-1)
    df["location"] = df["location"].map(LOC_MAP).fillna(-1)
    return df

def train_model():
    print("📊 Generating synthetic dataset...")
    df = generate_dataset()

    df.to_csv(DATA_PATH, index=False)
    print(f"💾 Training dataset saved at {DATA_PATH}")

    print("🔄 Preprocessing dataset...")
    df_processed = preprocess(df)

    X = df_processed[["amount", "transaction_type", "location", "time", "device_id"]]
    y = df_processed["label"]

    print("✂️ Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🌲 Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"✅ Training completed with accuracy: {acc:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
