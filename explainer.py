import os
from google import genai

# Initialize Gemini client
def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in environment variables")
    return genai.Client(api_key=api_key)


def explain_transaction(transaction_dict, fraud_prob, model="gemini-2.5-flash"):
    client = get_client()

    prompt = (
        f"You are a concise fraud investigator assistant. "
        f"Given this transaction and its model fraud probability, "
        f"write 3 short bullet points explaining why it might be flagged and next steps.\n\n"
        f"Transaction: {transaction_dict}\n"
        f"Model fraud probability: {fraud_prob:.3f}\n"
        "Return only 3 bullet points."
    )

    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text if hasattr(resp, "text") else str(resp)
