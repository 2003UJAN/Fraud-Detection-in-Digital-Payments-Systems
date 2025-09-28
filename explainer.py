import streamlit as st
import pandas as pd
import numpy as np
from utils import load_model
from explainer import explain_transaction

# Load model
model = load_model()

st.title("🔎 Fraud Detection with Gemini-Powered Explanations")

# Sidebar: user input
st.sidebar.header("Transaction Input")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=120.0)
device_type = st.sidebar.selectbox("Device Type", ["mobile", "desktop", "pos", "tablet"])
country = st.sidebar.selectbox("Country", ["IN", "US", "GB", "CN", "DE", "FR", "BR", "NG", "AU"])
hour = st.sidebar.slider("Hour of Day", 0, 23, 14)
day_of_week = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
txn_interval_sec = st.sidebar.number_input("Txn Interval (sec)", min_value=0.0, value=300.0)
num_prev_txn_1h = st.sidebar.number_input("Num Previous Txns (1h)", min_value=0, value=2)
is_new_device = st.sidebar.selectbox("Is New Device?", [0, 1])
num_feature_1 = st.sidebar.number_input("num_feature_1", value=0.5)
num_feature_2 = st.sidebar.number_input("num_feature_2", value=-0.3)
num_feature_3 = st.sidebar.number_input("num_feature_3", value=1.1)
num_feature_4 = st.sidebar.number_input("num_feature_4", value=0.0)
num_feature_5 = st.sidebar.number_input("num_feature_5", value=2.2)

# Convert to DataFrame
tx = pd.DataFrame([{
    "user_id": 1234,
    "amount": amount,
    "merchant_id": "m_test",
    "device_type": device_type,
    "country": country,
    "hour": hour,
    "day_of_week": day_of_week,
    "txn_interval_sec": txn_interval_sec,
    "num_prev_txn_1h": num_prev_txn_1h,
    "is_new_device": is_new_device,
    "num_feature_1": num_feature_1,
    "num_feature_2": num_feature_2,
    "num_feature_3": num_feature_3,
    "num_feature_4": num_feature_4,
    "num_feature_5": num_feature_5
}])

if st.button("🔍 Predict Fraud"):
    prob = model.predict_proba(tx)[0, 1]
    st.metric("Fraud Probability", f"{prob:.2%}")

    with st.spinner("Explaining with Gemini..."):
        explanation = explain_transaction(tx.to_dict(orient="records")[0], prob)
    st.subheader("Investigator Note")
    st.write(explanation)
