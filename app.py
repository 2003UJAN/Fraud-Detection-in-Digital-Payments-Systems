# app.py
import os
import time
import joblib
import streamlit as st
import pandas as pd

from utils import (
    generate_local_synthetic_transactions,
    train_models_quick,
    build_features_for_prediction,
    generate_gemini_synthetic_transactions
)
from explainer import gemini_explain_transaction

st.set_page_config(page_title="Fraud Detection (Gemini-enhanced)", layout="wide")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

st.title("⚡ Fraud Detection — Gemini-enhanced Streamlit Demo")

st.sidebar.header("Controls")
mode = st.sidebar.selectbox("Mode", ["Demo (synthetic)", "Upload CSV", "Manual Entry"])
st.sidebar.markdown("**Gemini settings**")
use_gemini = st.sidebar.checkbox("Use Gemini for synthetic data & explanations", value=True)

# Buttons for training / generation
if st.sidebar.button("Generate synthetic data (Gemini or local) + Train (full)"):
    with st.spinner("Generating data and training models..."):
        if use_gemini:
            # use Gemini to create synthetic data (may need API key & model access)
            df = generate_gemini_synthetic_transactions(n=2000)
        else:
            df = generate_local_synthetic_transactions(n=20000)
        metrics = train_models_quick(df, save_dir=MODELS_DIR)
    st.sidebar.success("Models trained")
    st.sidebar.write(metrics)

# Demo mode
if mode == "Demo (synthetic)":
    st.subheader("Demo dataset (sample)")
    sample_size = st.slider("Demo dataset size", 500, 10000, 2000, step=500)
    if use_gemini:
        df = generate_gemini_synthetic_transactions(n=sample_size//4)  # gemini generation is costlier; sample smaller
    else:
        df = generate_local_synthetic_transactions(n=sample_size)
    st.dataframe(df.head(8))

    if st.button("Retrain quick on current demo data"):
        with st.spinner("Retraining..."):
            metrics = train_models_quick(df, save_dir=MODELS_DIR)
        st.success("Quick retrain finished")
        st.write(metrics)

    st.subheader("Real-time Transaction Simulator")
    start_stream = st.button("Start Streaming (demo 100 txns)")
    if start_stream:
        if not os.path.exists(os.path.join(MODELS_DIR, "rf_pipeline.joblib")):
            st.warning("Models not found — training quick baseline...")
            train_models_quick(df.sample(min(2000, len(df))), save_dir=MODELS_DIR)
        pipe = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.joblib"))

        stream_container = st.empty()
        for idx, row in df.sample(min(200, len(df))).reset_index(drop=True).iterrows():
            tx = row.to_frame().T.drop(columns=["label"], errors="ignore")
            X = build_features_for_prediction(tx)
            prob = pipe.predict_proba(X)[0,1]
            pred = int(prob > 0.5)

            stream_container.metric(f"Txn #{idx+1} — Fraud prob", f"{prob:.3f}")
            st.write(X.T)
            # Gemini explanation (if available)
            note = gemini_explain_transaction(X, float(prob), use_gemini=use_gemini)
            st.text_area("Investigation note (Gemini)", value=note, height=140)
            if pred:
                st.warning(f"Flagged as FRAUD (prob {prob:.2f})")
            else:
                st.success(f"Looks normal (prob {prob:.2f})")
            time.sleep(0.25)

# Upload CSV mode
elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (transaction rows). Must include typical cols: amount, device_type, country, hour, day_of_week, txn_interval_sec, num_prev_txn_1h, is_new_device", type=["csv"])
    if uploaded is not None:
        df_uploaded = pd.read_csv(uploaded)
        st.write("Preview")
        st.dataframe(df_uploaded.head())

    if st.button("Predict on uploaded file") and uploaded is not None:
        if not os.path.exists(os.path.join(MODELS_DIR, "rf_pipeline.joblib")):
            st.warning("Models not found — training quick baseline using local synthetic data...")
            train_models_quick(generate_local_synthetic_transactions(5000), save_dir=MODELS_DIR)
        pipe = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.joblib"))
        X = build_features_for_prediction(df_uploaded)
        preds = pipe.predict(X)
        probs = pipe.predict_proba(X)[:,1]
        df_uploaded["pred_label"] = preds
        df_uploaded["pred_prob"] = probs
        st.dataframe(df_uploaded.head(200))
        st.download_button("Download predictions", df_uploaded.to_csv(index=False).encode("utf-8"), "predictions.csv")

# Manual entry
else:
    st.subheader("Manual transaction entry")
    with st.form("manual"):
        user_id = st.number_input("user_id", min_value=1, value=100)
        amount = st.number_input("amount", min_value=0.0, value=50.0)
        device_type = st.selectbox("device_type", ["mobile","desktop","pos","tablet"])
        country = st.selectbox("country", ["IN","US","GB","CN","DE","FR","BR","NG","AU"])
        hour = st.slider("hour", 0, 23, 14)
        day_of_week = st.slider("day_of_week", 0, 6, 2)
        txn_interval_sec = st.number_input("txn_interval_sec", value=300.0)
        num_prev_txn_1h = st.number_input("num_prev_txn_1h", value=0)
        is_new_device = st.checkbox("is_new_device")
        submitted = st.form_submit_button("Check")

    if submitted:
        row = {
            "user_id": user_id,
            "amount": amount,
            "merchant_id": "m_1",
            "device_type": device_type,
            "country": country,
            "hour": hour,
            "day_of_week": day_of_week,
            "txn_interval_sec": txn_interval_sec,
            "num_prev_txn_1h": num_prev_txn_1h,
            "is_new_device": int(is_new_device)
        }
        df_row = pd.DataFrame([row])
        if not os.path.exists(os.path.join(MODELS_DIR, "rf_pipeline.joblib")):
            st.warning("Models not found — training quick baseline...")
            train_models_quick(generate_local_synthetic_transactions(5000), save_dir=MODELS_DIR)
        pipe = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.joblib"))
        X_row = build_features_for_prediction(df_row)
        prob = pipe.predict_proba(X_row)[0,1]
        st.write(f"Fraud probability: {prob:.3f}")
        note = gemini_explain_transaction(X_row, float(prob), use_gemini=use_gemini)
        st.text_area("Investigation note (Gemini)", value=note, height=180)
        if prob > 0.5:
            st.error("Transaction flagged as FRAUD")
        else:
            st.success("Transaction appears normal")

st.markdown("---")
st.markdown("**Notes:** The demo uses a RandomForest baseline for fast inference. Gemini is used optionally for higher-quality synthetic examples and explanations. For production, use a model serving layer and secure the Gemini API key.")
