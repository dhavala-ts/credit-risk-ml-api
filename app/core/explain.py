import shap
import numpy as np

from app.core.model import model, preprocessor


# Lazy-loaded explainer
_explainer = None


def get_explainer():
    global _explainer
    if _explainer is None:
        rf_model = model.named_steps["model"]
        _explainer = shap.TreeExplainer(rf_model)
    return _explainer


def explain_prediction(df):
    explainer = get_explainer()

    X_transformed = preprocessor.transform(df)
    shap_values = explainer.shap_values(X_transformed)

    feature_names = preprocessor.get_feature_names_out()

    # Handle all SHAP output formats safely
    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    else:
        shap_array = shap_values

    shap_array = np.array(shap_array)[0]

    if shap_array.ndim == 2:
        shap_array = shap_array.flatten()

    # Aggregate one-hot encoded features
    grouped = {}

    for name, value in zip(feature_names, shap_array):
        original_feature = name.split("__")[-1].split("_")[0]
        grouped.setdefault(original_feature, 0.0)
        grouped[original_feature] += float(value)

    # Top 5 most impactful features
    top_features = dict(
        sorted(
            grouped.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
    )

    return top_features
