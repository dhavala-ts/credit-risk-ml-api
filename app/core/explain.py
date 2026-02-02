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

        # TreeExplainer works for RandomForest / XGBoost / ExtraTrees
        self.explainer = shap.TreeExplainer(self.model)

        self.output_dir = "outputs/shap"
        os.makedirs(self.output_dir, exist_ok=True)

    def explain(self, data: dict, top_k: int = 5, save_plot: bool = True):
        # ---- INPUT ----
        df = pd.DataFrame([data])

        # ---- TRANSFORM ----
        X = self.preprocessor.transform(df)
        feature_names = self.preprocessor.get_feature_names_out()

        # ---- SHAP VALUES ----
        shap_values = self.explainer.shap_values(X)

        # Binary classifier safe handling
        if isinstance(shap_values, list):
            shap_array = shap_values[1]   # positive class
        else:
            shap_array = shap_values

        shap_array = np.asarray(shap_array)
        shap_array = np.squeeze(shap_array)

        # Ensure shape = (n_features,)
        if shap_array.ndim > 1:
            shap_array = shap_array.reshape(-1)

        # ---- AGGREGATE ONE-HOT FEATURES (ROBUST) ----
        grouped = {}

        for name, value in zip(feature_names, shap_array):
            base_feature = name.split("__")[-1].split("_")[0]

            # CRITICAL FIX: always collapse to scalar
            scalar_value = float(np.sum(value))

            grouped.setdefault(base_feature, 0.0)
            grouped[base_feature] += scalar_value

        # ---- TOP-K FEATURES ----
        top_features = dict(
            sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        )

        image_path = None
        if save_plot:
            image_path = self._save_bar_plot(top_features)

        return top_features, image_path

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

        filename = f"{self.output_dir}/shap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

        return filename


# ---- SINGLETON EXPLAINER (IMPORTANT FOR API PERFORMANCE) ----
_explainer_instance = None


def explain_prediction(pipeline, data: dict, top_k: int = 5):
    global _explainer_instance

    if _explainer_instance is None:
        _explainer_instance = CreditRiskExplainer(pipeline)

    return _explainer_instance.explain(data, top_k=top_k)
