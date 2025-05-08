from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Multilingual Sentiment API")
app.include_router(router)
