import shap
import matplotlib.pyplot as plt
from utils import load_model

def explain_prediction(features):
    model = load_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    shap.summary_plot(shap_values, features, show=False)
    plt.savefig("shap_summary.png")
    return "shap_summary.png"
