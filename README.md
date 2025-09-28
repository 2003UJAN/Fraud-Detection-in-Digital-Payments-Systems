# Fraud Detection in Digital Payment Systems (GenAI-Powered)

This project demonstrates a **real-time fraud detection system** with:
- **RandomForest ML Model** (auto-trains if not available)
- **SHAP Explainability** (visual insights into predictions)
- **GenAI (Gemini API)** for natural-language **fraud investigation reports**
- **Streamlit UI** for manual input and CSV upload

---

## 🚀 Run Locally

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt

# Create .env file and add your Gemini API Key:
# GOOGLE_API_KEY=your_api_key_here
# MODEL_PATH=models/fraud_model.pkl

streamlit run app.py
