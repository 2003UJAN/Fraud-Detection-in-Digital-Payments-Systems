# Fraud Detection in Digital Payments — Streamlit Demo (with GenAI)

Repository contains a demo Streamlit application that:
- Trains a RandomForest classifier on synthetic transactions.
- Optionally augments imbalanced fraud data using a simple GAN.
- Provides an LLM-based explainer (uses OpenAI if `OPENAI_API_KEY` is set).
- Supports CSV upload, manual entry, and a streaming transaction simulator.

## Files
- `app.py` — Streamlit UI and orchestration
- `utils.py` — data generator, preprocessing, train/save functions
- `gan.py` — simple PyTorch GAN and augmentation helper
- `explainer.py` — GenAI explainer wrapper (OpenAI optional)
- `requirements.txt` — python dependencies

## Example Resume Line
“Extended fraud detection system with GenAI-based synthetic fraud data generation and natural language fraud investigation reports, improving model robustness and investigator efficiency.”

## Quick start
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."  # optional, for GenAI explainers
streamlit run app.py
