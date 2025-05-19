# src/predict.py
import pandas as pd
import joblib
import os

# Yollar
test_path = "data/raw/test.csv"
model_path = "models/model.pkl"
vectorizer_path = "models/vectorizer.pkl"
output_path = "test-results/predictions.csv"

# Veri ve model
df = pd.read_csv(test_path)
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Tahmin
X_test = vectorizer.transform(df["text"])
df["prediction"] = model.predict(X_test)

# Sonuçları kaydet (label varsa onu da dahil et)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

columns_to_save = ["text", "prediction"]
if "label" in df.columns:
    columns_to_save.insert(1, "label")  # prediction'dan önce koy

df[columns_to_save].to_csv(output_path, index=False)
