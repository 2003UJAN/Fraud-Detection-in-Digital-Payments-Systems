import shap
import pandas as pd
import numpy as np

def explain_prediction(model, features, prob=None):
    """
    Generate human-readable explanation for fraud prediction.
    Works for tree-based classifiers (RandomForest, XGBoost, etc.).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Handle classifiers (list of arrays) vs regressors (array)
    if isinstance(shap_values, list):
        # Take fraud class = 1
        shap_importance = shap_values[1][0]
    else:
        shap_importance = shap_values[0]

    # Convert to numpy array for safety
    shap_importance = np.array(shap_importance)

    explanation = {
        "Top Features": {}
    }

    # Align features with importance
    feature_names = features.columns
    top_features = sorted(
        zip(feature_names, shap_importance),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    explanation["Top Features"] = {f: float(val) for f, val in top_features}

    # Add fraud probability if provided
    if prob is not None:
        explanation["Fraud Probability"] = f"{prob:.2%}"

    return explanation
