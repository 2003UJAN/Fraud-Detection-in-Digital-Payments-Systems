
import os
import json

# attempt to import Gemini client
try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# Choose the Gemini model you have access to
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def gemini_explain_transaction(tx_df, prob, use_gemini=True):
    """
    Return a short investigation note for a single-row DataFrame tx_df and probability float prob.
    If Gemini is available and use_gemini=True, call the model; otherwise return a templated note.
    """
    tx = tx_df.iloc[0].to_dict()
    template = (
        f"Fraud probability: {prob:.3f}\n"
        f"Transaction: user={tx.get('user_id')}, amount={tx.get('amount')}, merchant={tx.get('merchant_id')}, device={tx.get('device_type')}, country={tx.get('country')}, hour={tx.get('hour')}.\n\n"
        "Investigation notes:\n"
        "- Model flagged this transaction because of anomalous transaction features relative to typical patterns.\n"
        "- Recommended actions: verify user identity, review user's recent activity, contact merchant as necessary, consider temporary block if multiple alerts.\n"
    )

    if use_gemini and GEMINI_AVAILABLE:
        client = genai.Client()
        prompt = (
            "You are a concise fraud analyst assistant. Given a payment transaction and model probability, "
            "write a short 3-bullet investigation note with reasons and recommended next steps. "
            f"Transaction: {json.dumps(tx)}\nModel fraud probability: {prob:.3f}\n"
            "Return bullet points only."
        )
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = getattr(resp, "text", None) or str(resp)
            # strip/clean
            return text.strip()
        except Exception as e:
            return template + f"\n(GenAI call failed: {e})"
    else:
        return template
