import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.schema import (
    CreditRequest,
    CreditResponse,
    ExplainResponse
)
from app.core.model import predict_credit, pipeline
from app.core.explain import explain_prediction

router = APIRouter()


@router.post("/predict", response_model=CreditResponse)
def predict_credit_risk(request: CreditRequest):
    try:
        df = pd.DataFrame([request.data])
        probability, prediction = predict_credit(df)

        return CreditResponse(
            prediction=prediction,
            probability=probability
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/explain", response_model=ExplainResponse)
def explain_credit_risk(request: CreditRequest):
    try:
        df = pd.DataFrame([request.data])

        probability, prediction = predict_credit(df)

        top_features, image_path = explain_prediction(
            pipeline=pipeline,
            data=request.data
        )

        return ExplainResponse(
            prediction=prediction,
            probability=probability,
            top_features=top_features,
            image_path=image_path
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
