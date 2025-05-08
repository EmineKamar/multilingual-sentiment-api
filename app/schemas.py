from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    language: str
    sentiment: str
    score: float
