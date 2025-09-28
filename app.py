import os
import time
import joblib
import streamlit as st
import pandas as pd

from utils import generate_synthetic_transactions, train_models_quick, build_features_for_prediction
from gan import augment_with_gan
from explainer import genai_explainer

st.set_page_config(page_title="Real-time Fraud Detection (GenAI)", layout="wide")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

st.title("⚡ Real-time Fraud Detection — Streamlit Demo (GenAI)")

st.sidebar.header("Controls")
mode = st.sidebar.selectbox("Mode", ["Demo (synthetic)", "Upload CSV", "Manual Entry"])
if st.sidebar.button("Generate data + Train (full)"):
    with st.spinner("Generating synthetic data and training (may take a while)..."):
        df = generate_synthetic_transactions(n=20000)
        metrics = train_models_quick(df, save_dir=MODELS_DIR)  # quick wrapper in utils
    st.sidebar.success("Models trained")
    st.sidebar.write(metrics)

# Demo mode
if mode == "Demo (synthetic)":
    st.sidebar.write("Using synthetic dataset (sample)")
    df = generate_synthetic_transactions(n=5000)  # smaller for interactive demo
    st.subheader("Dataset sample")
    st.dataframe(df.sample(8))

    if st.button("Augment frauds with GAN + Retrain (quick)"):
        with st.spinner("Augmenting dataset using GAN and retraining..."):
            df_aug = augment_with_gan(df, target_label_col="label", n_fake=2000)
            metrics = train_models_quick(df_aug, save_dir=MODELS_DIR)
        st.success("Retrained with GAN-augmented data")
        st.write(metrics)

    st.subheader("Real-time Transaction Simulator")
    run_stream = st.button("Start streaming simulation (demo 100 txns)")
    if run_stream:
        if not os.path.exists(os.path.join(MODELS_DIR, "rf_pipeline.joblib")):
            st.warning("Models not found — training quick baseline...")
            train_models_quick(df.sample(2000), save_dir=MODELS_DIR)
        pipe = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.joblib"))

        stream_container = st.empty()
        sample = df.sample(200).reset_index(drop=True)
        for idx, row in sample.iterrows():
            tx = row.to_frame().T.drop(columns=["label"])
            prob = pipe.predict_proba(tx)[0, 1]
            pred = pipe.predict(tx)[0]

            stream_container.metric(f"Txn #{idx+1} — Fraud prob", f"{prob:.3f}")
            st.write(tx)
            expl = genai_explainer(tx, prob)
            st.text_area("Investigation note (GenAI)", value=expl, height=120)
            if prob > 0.5:
                st.warning(f"Flagged as FRAUD (prob {prob:.2f})")
            else:
                st.success(f"Looks normal (prob {prob:.2f})")
            time.sleep(0.25)

# Upload CSV
elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (transaction rows). Required cols: amount, device_type, country, hour, day_of_week, txn_interval_sec, num_prev_txn_1h, is_new_device", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df.head())

    if st.button("Predict on uploaded file") and uploaded is not None:
        if not os.path.exists(os.path.join(MODELS_DIR, "rf_pipeline.joblib")):
            st.warning("Models not found — training quick baseline using synthetic data...")
            train_models_quick(generate_synthetic_transactions(5000), save_dir=MODELS_DIR)
        pipe = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.joblib"))
        X = build_features_for_prediction(df)  # align columns / fill missing features
        preds = pipe.predict(X)
        probs = pipe.predict_proba(X)[:, 1]
        df["pred_label"] = preds
        df["pred_prob"] = probs
        st.dataframe(df.head(200))
        st.download_button("Download predictions", df.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")

# Manual Entry
else:
    st.subheader("Manual transaction entry")
    with st.form("manual"):
        user_id = st.number_input("user_id", min_value=1, value=100)
        amount = st.number_input("amount", min_value=0.0, value=50.0)
        device_type = st.selectbox("device_type", ["mobile", "desktop", "pos", "tablet"])
        country = st.selectbox("country", ["IN", "US", "GB", "CN", "DE", "FR", "BR", "NG", "AU"])
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
            train_models_quick(generate_synthetic_transactions(5000), save_dir=MODELS_DIR)
        pipe = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.joblib"))
        X_row = build_features_for_prediction(df_row)
        prob = pipe.predict_proba(X_row)[0, 1]
        pred = pipe.predict(X_row)[0]
        st.write(f"Fraud probability: {prob:.3f}")
        st.text_area("Investigation note (GenAI)", value=genai_explainer(df_row, prob), height=140)
        if prob > 0.5:
            st.error("Transaction flagged as FRAUD")
        else:
            st.success("Transaction appears normal")

st.markdown("---")
st.markdown("**Notes:** This demo uses synthetic data. For production, swap in real streaming inference (FastAPI/Kafka), a feature store, proper model monitoring, and secure LLM usage (audit logs, redaction).")
