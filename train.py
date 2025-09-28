import pandas as pd
from utils import train_model, save_model

def main():
    # Load synthetic data (Gemini-generated or local fallback)
    df = pd.read_csv("synthetic_transactions.csv")

    model, metrics = train_model(df, label_col="label")
    path = save_model(model, "rf_pipeline.joblib")

    print(f"✅ Model saved to {path}")
    print(f"📊 Metrics: Accuracy={metrics['accuracy']:.3f}, ROC AUC={metrics['roc_auc']:.3f}")

if __name__ == "__main__":
    main()
