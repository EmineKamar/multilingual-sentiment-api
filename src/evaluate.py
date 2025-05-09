import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
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

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Tahmin ve değerlendirme
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 📁 Klasörü oluştur
os.makedirs("metrics", exist_ok=True)

# 📄 Sonucu yaz
with open("metrics/evaluation.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.4f}")
