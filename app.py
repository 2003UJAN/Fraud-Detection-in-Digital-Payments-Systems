import sys
import os
import streamlit as st
import pandas as pd

# Ensure utils.py can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model
from explainer import explain_prediction

# Load model (auto-trains if missing)
model = load_model()

st.title("💳 Fraud Detection in Digital Payment Systems (GenAI Powered)")

st.write("Enter transaction details to check if it's fraudulent or not:")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, format="%.2f")

if st.button("Predict Fraud"):
    features = [[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]]
    feature_names = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.subheader("🔍 Explanation")
    shap_result, note = explain_prediction(features, feature_names)

    if isinstance(shap_result, str) and shap_result.endswith(".png"):
        st.image(shap_result, caption="SHAP Explanation", use_column_width=True)
    else:
        st.info(shap_result)

    st.markdown("### 📝 GenAI Fraud Investigation Note")
    st.write(note)
