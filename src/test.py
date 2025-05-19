# src/test_model.py

import os
import json
import pandas as pd
import joblib
import sys

# ✅ METRİKLERİ KONTROL ET
METRICS_PATH = "metrics/metrics.json"

# Eşikler (CI/CD kriteri)
THRESHOLDS = {
    "accuracy": 0.40,
    "f1_score": 0.30
}

# Metrikleri oku
if not os.path.exists(METRICS_PATH):
    print(f"❌ Metrik dosyası bulunamadı: {METRICS_PATH}")
    sys.exit(1)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# Eşik testleri
for metric, threshold in THRESHOLDS.items():
    value = metrics.get(metric)
    if value is None:
        print(f"❌ '{metric}' metriği bulunamadı.")
        sys.exit(1)
    if value < threshold:
        print(f"❌ {metric} = {value:.4f} < threshold = {threshold:.2f}")
        sys.exit(1)
    else:
        print(f"✅ {metric} = {value:.4f} (≥ {threshold:.2f})")

print("✅ Tüm metrik eşikleri geçti. Model test edildi.")

# 🔄 Test verisinden tahmin üret
test_data = pd.read_csv("data/raw/test.csv")
X_test = test_data["text"]

# Model ve vectorizer'ı yükle
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Vektörleştir ve tahmin yap
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

# Tahminleri kaydet
results = test_data.copy()
results["prediction"] = y_pred

# 📁 test-results klasörü oluştur
os.makedirs("test-results", exist_ok=True)
results.to_csv("test-results/predictions.csv", index=False)

print("📄 Tahminler 'test-results/predictions.csv' dosyasına kaydedildi.")
