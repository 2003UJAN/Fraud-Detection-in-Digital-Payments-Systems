import shap
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_model

def explain_prediction(features, feature_names=None):
    model = load_model()

    try:
        if feature_names:
            X = pd.DataFrame(features, columns=feature_names)
        else:
            X = pd.DataFrame(features)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig("shap_summary.png", bbox_inches="tight")
        plt.close()

        return "shap_summary.png"
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")
        return None
