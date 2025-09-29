import os
import streamlit as st
import pandas as pd
from utils import load_model
from explainer import explain_prediction
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

# Load model
model = load_model()

st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Fraud Detection in Digital Payments")

# Tabs
tab1, tab2 = st.tabs(["🔹 Single Transaction", "📂 Batch Detection (CSV)"])

# ---------------------------
# 🔹 Single Transaction Tab
# ---------------------------
with tab1:
    st.subheader("Enter Transaction Details")
    amount = st.number_input("Transaction Amount", 1, 20000)
    time = st.slider("Transaction Hour (0-23)", 0, 23, 12)
    location = st.selectbox("Location", ["US", "EU", "ASIA", "AFRICA"])
    merchant = st.selectbox("Merchant Category", ["electronics", "fashion", "grocery", "gaming", "others"])
    device = st.selectbox("Device Type", ["mobile", "desktop", "tablet"])
    prev_txns = st.number_input("Previous Transactions", 0, 2000)

    # Encode categorical variables (must match training)
    loc_encoded = ["US", "EU", "ASIA", "AFRICA"].index(location)
    merchant_encoded = ["electronics", "fashion", "grocery", "gaming", "others"].index(merchant)
    device_encoded = ["mobile", "desktop", "tablet"].index(device)

    # Create DataFrame with correct feature order
    features_df = pd.DataFrame([{
        "amount": amount,
        "time": time,
        "loc_encoded": loc_encoded,
        "merchant_encoded": merchant_encoded,
        "device_encoded": device_encoded,
        "previous_transactions": prev_txns
    }])

    if st.button("🚀 Predict Fraud"):
        pred = model.predict(features_df)[0]
        st.write("🔒 Prediction:", "Fraudulent ❌" if pred == 1 else "Legitimate ✅")

        shap_path = explain_prediction(
            features_df,
            ["amount", "time", "loc_encoded", "merchant_encoded", "device_encoded", "previous_transactions"]
        )
        if shap_path:
            st.image(shap_path, caption="SHAP Explanation")

        # Gemini explanation
        if API_KEY:
            fraud_text = (
                f"Transaction details: amount={amount}, time={time}, location={location}, "
                f"merchant={merchant}, device={device}, prev_txns={prev_txns}. Prediction={pred}"
            )
            model_g = genai.GenerativeModel("gemini-1.5-flash")
            response = model_g.generate_content(
                f"Explain in simple language why this transaction might be fraudulent or legitimate: {fraud_text}"
            )
            st.info(response.text)

# ---------------------------
# 📂 Batch Detection Tab
# ---------------------------
with tab2:
    st.subheader("Upload CSV for Batch Fraud Detection")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = ["amount", "time", "loc_encoded", "merchant_encoded", "device_encoded", "previous_transactions"]

        if all(col in df.columns for col in required_cols):
            preds = model.predict(df[required_cols])
            df["Prediction"] = ["Fraudulent ❌" if p == 1 else "Legitimate ✅" for p in preds]
            st.dataframe(df)

            csv_download = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv_download, "fraud_predictions.csv", "text/csv")
        else:
            st.error(f"CSV must contain columns: {required_cols}")
