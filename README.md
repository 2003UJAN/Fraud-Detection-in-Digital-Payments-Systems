# Fraud Detection in Digital Payments — Gemini-enhanced Demo

This repo demonstrates a real-time fraud detection demo integrated with Google Gemini (Gen AI) for:
- Synthetic fraud data generation (Gemini prompts produce JSON transaction rows).
- Natural-language investigation notes (Gemini explains model outputs).
- Streamlit UI supporting CSV upload, manual entry, and a streaming simulator.

## Quickstart
1. Create virtualenv and install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
