import sys
import os
import streamlit as st

# Ensure utils.py is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model

# Load model
model = load_model()

st.title("💳 Fraud Detection in Digital Payment Systems")
st.write("Enter transaction details to check if it's fraudulent or not:")

# Example input fields (modify to match dataset features)
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, format="%.2f")

if st.button("Predict Fraud"):
    features = [[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]]
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

