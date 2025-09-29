import streamlit as st
import pandas as pd
from utils import load_model, load_schema_safe, preprocess_for_inference
from explainer import explain_prediction

# Load model + schema
model = load_model()
schema_data = load_schema_safe()

st.set_page_config(page_title="Fraud Detection in Digital Payments", layout="wide")

st.title("💳 Fraud Detection in Digital Payment Systems")
st.markdown("Upload a CSV or enter transaction details manually to detect fraud in real-time.")

# --- Sidebar: Mode selection ---
mode = st.sidebar.radio("Choose Input Mode:", ["Manual Entry", "CSV Upload"])

# --- Manual Entry ---
if mode == "Manual Entry":
    st.subheader("📌 Enter Transaction Details")

    amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0)
    transaction_type = st.selectbox("Transaction Type", list(schema_data["mappings"]["TYPE_MAP"].keys()))
    location = st.selectbox("Transaction Location", list(schema_data["mappings"]["LOC_MAP"].keys()))
    time = st.slider("Transaction Hour of Day", 0, 23, 12)
    device_id = st.number_input("Device ID", min_value=1, max_value=9999, value=123)

    if st.button("🚀 Predict Fraud"):
        input_dict = {
            "amount": amount,
            "transaction_type": transaction_type,
            "location": location,
            "time": time,
            "device_id": device_id,
        }

        # Preprocess
        features = preprocess_for_inference(input_dict, schema_data)

        # Predict
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        # Show results
        st.markdown(f"### 🔍 Prediction: {'⚠️ Fraudulent' if pred == 1 else '✅ Legitimate'}")
        st.markdown(f"**Fraud Probability:** {proba:.2%}")

        # Explainability
        explanation = explain_prediction(model, features, prob=proba)
        st.markdown("### 🧾 Explanation")
        st.json(explanation)

# --- CSV Upload ---
elif mode == "CSV Upload":
    st.subheader("📂 Upload a Batch of Transactions (CSV)")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Preprocess
        processed_df = preprocess_for_inference(df, schema_data)

        # Predict
        preds = model.predict(processed_df)
        probs = model.predict_proba(processed_df)[:, 1]

        # Add results
        df["Fraud_Prediction"] = ["⚠️ Fraudulent" if p == 1 else "✅ Legitimate" for p in preds]
        df["Fraud_Probability"] = [f"{p:.2%}" for p in probs]

        st.markdown("### 📊 Batch Prediction Results")
        st.dataframe(df)

        # Downloadable results
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv_out, "fraud_predictions.csv", "text/csv")
