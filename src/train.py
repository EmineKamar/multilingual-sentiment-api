# src/train.py

from src.preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Veriyi yükle ve işle
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/raw/sample.csv')

# Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modeli değerlendir
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model doğruluğu:", accuracy)
