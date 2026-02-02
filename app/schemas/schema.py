from typing import Dict, Optional
from pydantic import BaseModel


class CreditRequest(BaseModel):
    data: dict


class CreditResponse(BaseModel):
    prediction: int
    probability: float


class ExplainResponse(BaseModel):
    prediction: int
    probability: float
    top_features: Dict[str, float]
    image_path: Optional[str]
