# Fraud Detection in Digital Payments — Streamlit + GenAI

## 🚀 Features
- Hybrid Fraud Detection (RandomForest + Synthetic Data)
- Auto-training if model missing
- Batch fraud detection via CSV upload
- Real-time single transaction check
- GenAI-powered explanations using **Gemini API**
- Synthetic dataset export (`training_data.csv`)

---

## 🛠️ Setup

```bash
git clone <your_repo_url>
cd fraud-detection-in-digital-payments-systems

python -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt

Set up .env with your Gemini API key:

GEMINI_API_KEY=your_gemini_api_key_here
