from fastapi import APIRouter
from app.schemas import TextInput, PredictionOutput
from app.sentiment import analyze_sentiment

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
def predict_sentiment(data: TextInput):
    result = analyze_sentiment(data.text)
    return result
