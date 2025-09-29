import streamlit as st
import pandas as pd
import os
import numpy as np
from utils import load_model
from explainer import explain_prediction

# Load trained model
model = load_model()

# Auto-generate sample CSV if not present
SAMPLE_CSV = "sample_data.csv"
if not os.path.exists(SAMPLE_CSV):
    def generate_sample_csv(filename=SAMPLE_CSV, n_samples=20):
        np.random.seed(42)
        data = pd.DataFrame({
            "amount": np.random.uniform(1, 20000, n_samples),
            "time": np.random.randint(0, 24, n_samples),
            "loc_encoded": np.random.choice([0, 1, 2, 3], n_samples),
            "merchant_encoded": np.random.choice([0, 1, 2, 3, 4], n_samples),
            "device_encoded": np.random.choice([0, 1, 2], n_samples),
            "previous_transactions": np.random.randint(0, 1000, n_samples)
        })
        data.to_csv(filename, index=False)

    generate_sample_csv()

# Streamlit UI
st.title("🔍 Fraud Detection in Digital Payment Systems")
st.write("Real-time and batch fraud detection with explainable AI (SHAP).")

tab1, tab2 = st.tabs(["🧍 Single Transaction", "📂 Batch Detection (CSV)"])

# ---------------- Single Transaction ----------------
with tab1:
    st.subheader("Enter Transaction Details")

    amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=100.0)
    time = st.slider("Transaction Time (0–23 hrs)", 0, 23, 12)
    loc_encoded = st.selectbox("Location", [0, 1, 2, 3])  # US=0, EU=1, ASIA=2, AFRICA=3
    merchant_encoded = st.selectbox("Merchant Category", [0, 1, 2, 3, 4])  # electronics=0 ... others=4
    device_encoded = st.selectbox("Device Type", [0, 1, 2])  # mobile=0, desktop=1, tablet=2
    previous_transactions = st.number_input("Previous Transactions", min_value=0, max_value=1000, value=10)

    if st.button("🔎 Predict Fraud"):
        features = [[amount, time, loc_encoded, merchant_encoded, device_encoded, previous_transactions]]
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        if pred == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected (Confidence: {proba:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {1-proba:.2f})")

        # Explain prediction
        explanation = explain_prediction(model, features)
        st.write("### 🔍 Explanation (SHAP Values)")
        st.json(explanation)

# ---------------- Batch Detection ----------------
with tab2:
    st.subheader("Upload CSV for Batch Fraud Detection")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    schema = load_schema()

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.info("No file uploaded. Using sample_data.csv for demo.")
        data = pd.read_csv(SAMPLE_CSV)

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
