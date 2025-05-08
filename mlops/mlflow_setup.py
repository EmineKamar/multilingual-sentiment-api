import mlflow
import os

def init_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Multilingual Sentiment Analysis")

def log_model(model, model_name="sentiment_model"):
    with mlflow.start_run():
        mlflow.pytorch.log_model(model, model_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_artifact("requirements.txt")
        print("Model başarıyla MLflow'a kaydedildi.")
