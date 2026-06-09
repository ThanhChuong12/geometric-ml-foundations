from pydantic import BaseModel
from typing import Optional

class PredictionResult(BaseModel):
    digit: int
    confidence: float

class PredictionResponse(BaseModel):
    baseline: PredictionResult
    augmentation: PredictionResult
    averaging: PredictionResult

class PredictRequest(BaseModel):
    image: str # base64 encoded image
