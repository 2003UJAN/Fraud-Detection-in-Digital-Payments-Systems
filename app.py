import streamlit as st
import pandas as pd
import os
import numpy as np
from utils import load_model, load_schema
from explainer import explain_prediction

# Load trained model + schema
model = load_model()
schema_data = load_schema()
schema = schema_data["features"]
LOC_MAP = schema_data["mappings"]["LOC_MAP"]
MERCHANT_MAP = schema_data["mappings"]["MERCHANT_MAP"]
DEVICE_MAP = schema_data["mappings"]["DEVICE_MAP"]

# Reverse maps for human-readable CSV
LOC_MAP_REV = {v: k for k, v in LOC_MAP.items()}
MERCHANT_MAP_REV = {v: k for k, v in MERCHANT_MAP.items()}
DEVICE_MAP_REV = {v: k for k, v in DEVICE_MAP.items()}

SAMPLE_CSV = "sample_data.csv"
SAMPLE_CSV_HUMAN = "sample_data_human.csv"

def generate_sample_csvs(n_samples=20):
    np.random.seed(42)
    df = pd.DataFrame({
        "amount": np.random.uniform(1, 20000, n_samples),
        "time": np.random.randint(0, 24, n_samples),
        "loc_encoded": np.random.choice(list(LOC_MAP.values()), n_samples),
        "merchant_encoded": np.random.choice(list(MERCHANT_MAP.values()), n_samples),
        "device_encoded": np.random.choice(list(DEVICE_MAP.values()), n_samples),
        "previous_transactions": np.random.randint(0, 1000, n_samples)
    })

    # Save encoded CSV
    df.to_csv(SAMPLE_CSV, index=False)

    # Create human-readable version
    df_human = df.copy()
    df_human["Location"] = df_human["loc_encoded"].map(LOC_MAP_REV)
    df_human["Merchant"] = df_human["merchant_encoded"].map(MERCHANT_MAP_REV)
    df_human["Device"] = df_human["device_encoded"].map(DEVICE_MAP_REV)
    df_human = df_human.drop(columns=["loc_encoded", "merchant_encoded", "device_encoded"])

    df_human.to_csv(SAMPLE_CSV_HUMAN, index=False)

    print(f"✅ Sample CSVs generated: {SAMPLE_CSV}, {SAMPLE_CSV_HUMAN}")

# Auto-generate CSVs if missing
if not os.path.exists(SAMPLE_CSV) or not os.path.exists(SAMPLE_CSV_HUMAN):
    generate_sample_csvs()

# -------------------------------------------------
# Streamlit UI
st.title("🔍 Fraud Detection in Digital Payment Systems")
st.write("Real-time and batch fraud detection with explainable AI (SHAP).")

tab1, tab2 = st.tabs(["🧍 Single Transaction", "📂 Batch Detection (CSV)"])

# (Single transaction UI unchanged, already human-readable)

# ---------------- Batch Detection ----------------
with tab2:
    st.subheader("Upload CSV for Batch Fraud Detection")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.info("No file uploaded. Using sample_data_human.csv for demo.")
        data = pd.read_csv(SAMPLE_CSV_HUMAN)

    # Handle human-readable CSVs
    if "Location" in data.columns:
        data["loc_encoded"] = data["Location"].map(LOC_MAP)
        data.drop(columns=["Location"], inplace=True)

    if "Merchant" in data.columns:
        data["merchant_encoded"] = data["Merchant"].map(MERCHANT_MAP)
        data.drop(columns=["Merchant"], inplace=True)

    if "Device" in data.columns:
        data["device_encoded"] = data["Device"].map(DEVICE_MAP)
        data.drop(columns=["Device"], inplace=True)

    # Validate schema
    if list(data.columns) != schema:
        st.error(f"❌ Invalid CSV schema. Expected columns: {schema}")
    else:
        st.write("### 📄 Input Data Preview")
        st.dataframe(data.head())

        if st.button("🚀 Run Batch Detection"):
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1]

            data["Fraud_Prediction"] = predictions
            data["Fraud_Probability"] = probabilities

            st.write("### ✅ Results")
            st.dataframe(data.head())

            csv_download = data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv_download, "fraud_results.csv", "text/csv")
