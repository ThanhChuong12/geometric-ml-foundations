from fastapi import APIRouter
from models.schemas import PredictRequest, PredictionResponse
from services.model_service import generate_prediction
import asyncio

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_digit(request: PredictRequest):
    # Dùng hàm generate_prediction (sẽ chạy AI thật nếu có model, hoặc mock nếu chưa train)
    response = generate_prediction(request.image)
    return response
