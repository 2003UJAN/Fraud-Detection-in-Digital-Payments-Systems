# utils.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Gemini client import (optional)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Local fallback synthetic generator (fast, deterministic)
def generate_local_synthetic_transactions(n=20000, n_merchants=200, seed=42):
    np.random.seed(seed)
    user_ids = np.random.randint(1, 5000, size=n)
    amounts = np.round(np.random.exponential(scale=80, size=n) + np.random.normal(0, 5, n), 2)
    amounts = np.clip(amounts, 0.5, None)
    merchant_ids = [f"m_{i}" for i in np.random.randint(1, n_merchants+1, n)]
    device_types = np.random.choice(["mobile","desktop","pos","tablet"], size=n, p=[0.6,0.2,0.15,0.05])
    countries = np.random.choice(["IN","US","GB","CN","DE","FR","BR","NG","AU"], size=n)
    hours = np.random.randint(0,24,size=n)
    days = np.random.randint(0,7,size=n)

    txn_interval = np.random.exponential(scale=1000, size=n)
    num_prev_txn_1h = np.random.poisson(0.3, size=n)
    is_new_device = np.random.binomial(1, 0.05, size=n)

    base_prob = (amounts > 500).astype(float) * 0.05
    base_prob += (device_types == "desktop").astype(float) * 0.01
    base_prob += (txn_interval < 60).astype(float) * 0.1
    base_prob += (num_prev_txn_1h > 5).astype(float) * 0.15
    base_prob += (is_new_device == 1).astype(float) * 0.08

    probs = np.clip(base_prob + np.random.normal(0, 0.03, size=n), 0, 1)
    labels = np.random.binomial(1, probs)

    df = pd.DataFrame({
        'user_id': user_ids,
        'amount': amounts,
        'merchant_id': merchant_ids,
        'device_type': device_types,
        'country': countries,
        'hour': hours,
        'day_of_week': days,
        'txn_interval_sec': txn_interval,
        'num_prev_txn_1h': num_prev_txn_1h,
        'is_new_device': is_new_device,
        'label': labels
    })

    # add many features
    for i in range(1, 11):
        df[f'num_feature_{i}'] = np.round(np.random.normal(loc=0, scale=1+i*0.2, size=n), 3)
    for i in range(1,6):
        df[f'cat_feature_{i}'] = np.random.choice([f'c{i}_a', f'c{i}_b', f'c{i}_c', f'c{i}_d'], size=n)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

# Gemini-based synthetic generator (prompts Gemini to produce JSON-like rows)
def generate_gemini_synthetic_transactions(n=1000, batch=50, model="gemini-2.5-flash"):
    """
    Use Gemini to produce synthetic transaction rows in JSON format.
    Fallback to local generator if Gemini isn't available.
    Note: Keep batch small to avoid token/cost issues; adapt to your quotas.
    """
    if not GEMINI_AVAILABLE:
        print("Gemini SDK not available — using local generator fallback.")
        return generate_local_synthetic_transactions(n=n)

    client = genai.Client()
    records = []
    batches = max(1, n // batch)
    for b in range(batches):
        ask = (
            f"Generate {min(batch, n - b*batch)} realistic fraudulent and normal payment transactions "
            "as a JSON array. Each object must contain keys: user_id (int), amount (float), merchant_id (string), "
            "device_type (mobile|desktop|pos|tablet), country (ISO2), hour (0-23), day_of_week (0-6), txn_interval_sec (float), "
            "num_prev_txn_1h (int), is_new_device (0|1), label (0=normal/1=fraud), plus a few numeric features num_feature_1..num_feature_5. "
            "Return *only* valid JSON array and keep numbers reasonable (amounts up to a few thousands)."
        )
        resp = client.models.generate_content(model=model, contents=ask)
        text = getattr(resp, "text", None) or (resp.rendered_output if hasattr(resp, "rendered_output") else str(resp))
        # Try to parse JSON from text
        try:
            arr = json.loads(text)
            if isinstance(arr, dict):
                # some models return a single object; wrap it
                arr = [arr]
            records.extend(arr)
        except Exception:
            # Some times text contains explanations + JSON; try to extract using simple heuristics
            import re
            match = re.search(r"(\[\\?.*\\?\])", text, flags=re.S)
            if match:
                try:
                    arr = json.loads(match.group(1))
                    records.extend(arr)
                except Exception:
                    pass
            # fallback to local generation for this batch
            fall = generate_local_synthetic_transactions(n=min(batch, n - b*batch))
            records.extend(fall.to_dict(orient="records"))

    # truncate/convert to DataFrame
    df = pd.DataFrame(records)[:n]
    # fill missing columns with defaults
    required_cols = ['user_id','amount','merchant_id','device_type','country','hour','day_of_week','txn_interval_sec','num_prev_txn_1h','is_new_device','label']
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0
    # add extra numeric/cat features if missing
    for i in range(1, 11):
        col = f'num_feature_{i}'
        if col not in df.columns:
            df[col] = 0.0
    for i in range(1,6):
        col = f'cat_feature_{i}'
        if col not in df.columns:
            df[col] = f'c{i}_a'
    return df.reset_index(drop=True)

# Simple preprocessing and train
def build_pipeline(numeric_cols, categorical_cols):
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    preproc = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)], remainder="drop")
    return preproc

def train_models_quick(df, save_dir=MODELS_DIR):
    features = [c for c in df.columns if c not in ["label"]]
    categorical_cols = [c for c in features if df[c].dtype == "object" or c.startswith("cat_feature")]
    numeric_cols = [c for c in features if c not in categorical_cols]

    X = df[features]
    y = df["label"].astype(int)

    preproc = build_pipeline(numeric_cols, categorical_cols)
    rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", preproc), ("rf", rf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:,1]
    metrics = {"accuracy": float(accuracy_score(y_test, preds)), "roc_auc": float(roc_auc_score(y_test, probs))}

    joblib.dump(pipe, os.path.join(save_dir, "rf_pipeline.joblib"))
    return metrics

def build_features_for_prediction(df):
    df_copy = df.copy()
    # ensure necessary columns exist
    for i in range(1, 11):
        col = f"num_feature_{i}"
        if col not in df_copy.columns:
            df_copy[col] = 0.0
    for i in range(1,6):
        col = f"cat_feature_{i}"
        if col not in df_copy.columns:
            df_copy[col] = f"c{i}_a"
    if "merchant_id" not in df_copy.columns:
        df_copy["merchant_id"] = "m_1"
    # keep ordering stable
    return df_copy
