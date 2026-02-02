import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class CreditRiskExplainer:
    def __init__(self, pipeline):
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(self.model)

        self.output_dir = "outputs/shap"
        os.makedirs(self.output_dir, exist_ok=True)

    def explain(self, data: dict, top_k: int = 5, save_plot: bool = True):
        df = pd.DataFrame([data])

        X = self.preprocessor.transform(df)
        feature_names = self.preprocessor.get_feature_names_out()

        shap_values = self.explainer.shap_values(X)

        # ---- SAFE SHAP HANDLING (binary classifier) ----
        if isinstance(shap_values, list):
            shap_array = shap_values[1][0]
        else:
            shap_array = shap_values[0]

        shap_array = np.asarray(shap_array).flatten()

        # ---- AGGREGATE ONE-HOT FEATURES ----
        grouped = {}
        for name, value in zip(feature_names, shap_array):
            base_feature = name.split("__")[-1].split("_")[0]
            grouped.setdefault(base_feature, 0.0)
            grouped[base_feature] += float(value)

        # ---- TOP-K FEATURES ----
        top_features = dict(
            sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        )

        image_name = None
        if save_plot:
            image_name = self._save_bar_plot(top_features)

        return top_features, image_name

    def _save_bar_plot(self, top_features: dict) -> str:
        features = list(top_features.keys())
        values = list(top_features.values())

        plt.figure(figsize=(8, 4))
        colors = ["red" if v > 0 else "green" for v in values]

        plt.barh(features, values, color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("SHAP Contribution (→ higher risk | ← lower risk)")
        plt.title("Top Drivers of Credit Risk Prediction")
        plt.tight_layout()

        filename = f"shap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)

        plt.savefig(filepath, dpi=150)
        plt.close()

        return filename


# ---- SINGLETON WRAPPER ----
_explainer_instance = None


def explain_prediction(pipeline, data: dict, top_k: int = 5):
    global _explainer_instance

    if _explainer_instance is None:
        _explainer_instance = CreditRiskExplainer(pipeline)

    return _explainer_instance.explain(data, top_k=top_k)
