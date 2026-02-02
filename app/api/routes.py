import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

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
        top_features, image_name = explain_prediction(pipeline, request.data)

        image_url = f"/explain/image/{image_name}" if image_name else None

        return ExplainResponse(
            prediction=prediction,
            probability=probability,
            top_features=top_features,
            image_path=image_url
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/explain/image/{image_name}")
def get_explain_image(image_name: str):
    image_path = os.path.join("outputs/shap", image_name)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        image_path,
        media_type="image/png",
        filename=image_name
    )
