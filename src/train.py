# src/train.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Veriyi oku
data = pd.read_csv("C:/Users/emina/OneDrive/Masaüstü/vscode/multilingual-sentiment-api/data/raw/train.csv")

X = data["text"]
y = data["label"]

# Vektörleştir
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Modeli eğit
model = LogisticRegression()
model.fit(X_train, y_train)

# Model ve vektörleştiriciyi kaydet
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

