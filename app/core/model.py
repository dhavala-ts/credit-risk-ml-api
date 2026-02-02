import joblib
import pandas as pd
import os

MODEL_PATH = "app/model.pkl"
THRESHOLD = float(os.getenv("CREDIT_THRESHOLD", 0.5))


# Load model once
model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocess"]


def predict_credit(df: pd.DataFrame):
    """
    Returns probability and binary prediction
    """
    probability = model.predict_proba(df)[0][1]
    prediction = 1 if probability >= THRESHOLD else 0
    return probability, prediction
