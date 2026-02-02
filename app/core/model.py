import joblib
import pandas as pd

# Load pipeline ONCE at import time
PIPELINE_PATH = "app/model.pkl"

try:
    pipeline = joblib.load(PIPELINE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model pipeline: {e}")


def predict_credit(df: pd.DataFrame):
    """
    Returns (probability, prediction)
    """
    probability = pipeline.predict_proba(df)[0][1]
    prediction = 1 if probability >= 0.5 else 0
    return probability, prediction
