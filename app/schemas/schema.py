from pydantic import BaseModel
from typing import Dict, Any

class CreditRequest(BaseModel):
    data: Dict[str, Any]

class CreditResponse(BaseModel):
    prediction: int
    probability: float
