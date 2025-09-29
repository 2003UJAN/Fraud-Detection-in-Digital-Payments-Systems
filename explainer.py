import shap
import pandas as pd

def explain_prediction(model, features, prob=None):
    """
    Generate human-readable explanation for fraud prediction.
    Uses SHAP values + optional fraud probability.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    explanation = {
        "Top Features": {},
    }

    # Take feature importance for the first prediction
    feature_names = features.columns
    shap_importance = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    # Store top features
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
