import shap
import numpy as np

def explain_prediction(model, features, prob=None):
    """
    Generate human-readable explanation for fraud prediction.
    Uses SHAP if available, otherwise falls back to feature_importances_.
    """
    explanation = {"Top Features": {}}

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        if isinstance(shap_values, list):  # classifier
            shap_importance = shap_values[1][0]
        else:  # regressor
            shap_importance = shap_values[0]

        shap_importance = np.array(shap_importance)

        # --- Align lengths ---
        feature_names = list(features.columns)
        if len(shap_importance) != len(feature_names):
            raise ValueError("SHAP mismatch with feature count")

        top_features = sorted(
            zip(feature_names, shap_importance),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        explanation["Top Features"] = {f: float(val) for f, val in top_features}

    except Exception as e:
        # --- Fallback to model.feature_importances_ ---
        if hasattr(model, "feature_importances_"):
            feature_names = list(features.columns)
            importances = model.feature_importances_

            # Align lengths just in case
            min_len = min(len(feature_names), len(importances))
            top_features = sorted(
                zip(feature_names[:min_len], importances[:min_len]),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            explanation["Top Features"] = {f: float(val) for f, val in top_features}
        else:
            explanation["Top Features"] = {"info": "No SHAP or feature_importances available"}

    # Add fraud probability if passed
    if prob is not None:
        explanation["Fraud Probability"] = f"{prob:.2%}"

    return explanation
