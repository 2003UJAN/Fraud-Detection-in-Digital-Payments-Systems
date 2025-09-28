import shap
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_model
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def explain_prediction(features, feature_names=None):
    """SHAP explanation + GenAI fraud note"""
    model = load_model()

    # Convert features into DataFrame
    if feature_names:
        X = pd.DataFrame(features, columns=feature_names)
    else:
        X = pd.DataFrame(features)

    try:
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig("shap_summary.png", bbox_inches="tight")
        plt.close()

        shap_result = "shap_summary.png"
    except Exception as e:
        shap_result = f"Explanation not available (SHAP failed: {e})"

    # GenAI fraud note
    try:
        prompt = f"""
        A fraud detection model flagged this transaction.
        Transaction details: {X.to_dict(orient='records')[0]}
        Write a short fraud investigation note in simple language.
        """
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        note = response.text.strip()
    except Exception as e:
        note = f"GenAI explanation not available ({e})"

    return shap_result, note
