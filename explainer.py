import shap
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_model

def explain_prediction(features, feature_names=None):
    """
    Try to generate SHAP explanation. If fails, return a simple text.
    features: list of lists [[...]] for prediction
    feature_names: optional list of feature names
    """
    model = load_model()

    try:
        # Convert features into DataFrame for SHAP
        if feature_names:
            X = pd.DataFrame(features, columns=feature_names)
        else:
            X = pd.DataFrame(features)

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Save SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig("shap_summary.png", bbox_inches="tight")
        plt.close()

        return "shap_summary.png"

    except Exception as e:
        # Fallback simple explanation
        print(f"⚠️ SHAP explanation failed: {e}")
        return "Explanation not available (SHAP failed)."
