import os
import pandas as pd
import joblib

# Test verisini oku
test_data = pd.read_csv("data/raw/test.csv")
X_test = test_data["text"]

# Model ve vectorizer'ı yükle
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Test verisini vektörleştir
X_test_vec = vectorizer.transform(X_test)

# Tahmin yap
y_pred = model.predict(X_test_vec)

# Sonuçları yazdır ve kaydet
results = test_data.copy()
results["prediction"] = y_pred

# 📁 test-results klasörü oluştur
os.makedirs("test-results", exist_ok=True)

# 📄 tahminleri dosyaya yaz
results.to_csv("test-results/predictions.csv", index=False)
