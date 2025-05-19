# src/train.py

import pandas as pd
import joblib
import json
import os
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Veriyi oku
data = pd.read_csv("data/raw/train.csv")

X = data["text"]
y = data["label"]

# Vektörleştir
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# MLflow deneyini başlat
mlflow.set_experiment("Multilingual Sentiment Analysis")

with mlflow.start_run():
    # Modeli eğit
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    # Metrikleri hesapla
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Metrikleri logla
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Modeli ve vectorizer'ı MLflow'a logla
    mlflow.sklearn.log_model(model, "model")

    # Model ve vektörleştiriciyi kaydet
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    # Metrikleri dosyaya kaydet
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

