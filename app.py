import sys
import os
import streamlit as st
import pandas as pd

# Ensure utils.py can be imported no matter where this script runs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model
from explainer import explain_prediction

# Load model (auto-trains if not found)
model = load_model()

# Streamlit app UI
st.title("Fraud Detection in Digital Payment Systems")

st.write("Enter transaction details to check if it's fraudulent or not:")

# Example input fields (adjust to match your dataset)
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, format="%.2f")

if st.button("Predict Fraud"):
    features = [[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]]
    feature_names = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

    # Model prediction
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    # Generate explanation
    st.subheader("Explanation")
    explanation_result = explain_prediction(features, feature_names)

    if isinstance(explanation_result, str) and explanation_result.endswith(".png"):
        st.image(explanation_result, caption="SHAP Explanation", use_column_width=True)
    else:
        st.info(explanation_result)
