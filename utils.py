import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

def train_model(df, label_col="label"):
    features = [c for c in df.columns if c != label_col]
    cat_cols = [c for c in features if df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]

    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
    ])

    pipe = Pipeline([
        ("pre", preproc),
        ("rf", RandomForestClassifier(n_estimators=150,
                                      class_weight="balanced",
                                      random_state=42,
                                      n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

    return pipe, metrics


def save_model(model, path="rf_pipeline.joblib"):
    joblib.dump(model, path)
    return path


def load_model(path="rf_pipeline.joblib"):
    return joblib.load(path)
