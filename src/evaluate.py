# src/evaluate.py

import os
import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Veriyi oku
data = pd.read_csv("data/raw/train.csv")
X = data["text"]
y = data["label"]

# Vektörleştiriciyi ve modeli yükle
vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/model.pkl")

# Veriyi vektörleştir
X_vec = vectorizer.transform(X)

# Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Tahmin
y_pred = model.predict(X_test)

# Metrikleri hesapla
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
    "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
}

# Klasör oluştur
os.makedirs("metrics", exist_ok=True)

# JSON olarak kaydet
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# TXT olarak da kaydet (DVC için)
with open("metrics/evaluation.txt", "w") as f:
    for key, value in metrics.items():
        f.write(f"{key.capitalize()}: {value:.4f}\n")

# Ekrana yaz (GitHub Actions loglarında görünsün)
print("Evaluation metrics saved to metrics/metrics.json:")
print(json.dumps(metrics, indent=4))
