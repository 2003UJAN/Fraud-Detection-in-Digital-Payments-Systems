import streamlit as st
import pandas as pd
from utils import load_model, load_schema_safe, preprocess_for_inference
from explainer import explain_prediction

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Load model + schema
model = load_model()
schema_data = load_schema_safe()

LOC_MAP = schema_data.get("mappings", {}).get("LOC_MAP", {})
TYPE_MAP = schema_data.get("mappings", {}).get("TYPE_MAP", {})

st.title("💳 Fraud Detection in Digital Payments")

st.sidebar.header("Navigation")
mode = st.sidebar.radio("Choose Mode:", ["Single Transaction", "Batch Upload (CSV)"])

# ------------------- Single Transaction -------------------
if mode == "Single Transaction":
    st.subheader("🔍 Check a Single Transaction")

    amount = st.number_input("Amount ($)", min_value=1.0, max_value=10000.0, value=100.0)
    transaction_type = st.selectbox("Transaction Type", list(TYPE_MAP.keys()))
    location = st.selectbox("Location", list(LOC_MAP.keys()))
    time = st.slider("Hour of Day", 0, 23, 12)
    device_id = st.number_input("Device ID", min_value=1000, max_value=1100, value=1005)

    if st.button("Predict Fraud"):
        df = pd.DataFrame([{
            "amount": amount,
            "transaction_type": transaction_type,
            "location": location,
            "time": time,
            "device_id": device_id
        }])

        features = preprocess_for_inference(df)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        st.write(f"**Fraud Prediction:** {'🚨 Fraudulent' if pred else '✅ Legitimate'}")
        st.write(f"**Fraud Probability:** {prob:.2f}")

        # GenAI explanation
        explanation = explain_prediction(df.iloc[0].to_dict(), prob)
        st.info(explanation)

# ------------------- Batch Mode -------------------
else:
    st.subheader("📂 Batch Fraud Detection via CSV")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:", df.head())

        features = preprocess_for_inference(df)
        preds = model.predict(features)
        probs = model.predict_proba(features)[:, 1]

        df["fraud_prediction"] = preds
        df["fraud_probability"] = probs

        st.write("✅ Predictions Completed:")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "fraud_predictions.csv", "text/csv")
