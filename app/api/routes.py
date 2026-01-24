import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from app.schemas.schema import CreditRequest, CreditResponse

router = APIRouter()

try:
    model = joblib.load("app/model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@router.post("/predict", response_model=CreditResponse)
def predict_credit_risk(request: CreditRequest):
    try:
        df = pd.DataFrame([request.data])
        probability = model.predict_proba(df)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        return CreditResponse(
            prediction=prediction,
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
