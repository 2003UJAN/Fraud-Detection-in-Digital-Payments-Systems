import os
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def explain_prediction(transaction, prob):
    """Generate a natural language fraud investigation note."""
    prompt = f"""
    You are a fraud analyst. Explain why the following transaction may be fraudulent.

    Transaction: {transaction}
    Fraud Probability: {prob:.2f}

    Provide a clear, concise investigation-style note.
    """

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"(⚠️ Explanation unavailable: {str(e)})"
