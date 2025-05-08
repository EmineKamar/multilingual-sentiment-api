# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # CSV dosyasını yükle
    df = pd.read_csv(file_path)
    print("Verinin ilk 5 satırı:")
    print(df.head())  # Verinin ilk 5 satırını yazdır
    
    # Metin ve etiket sütunlarını al
    X = df['text']
    y = df['label']
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Veriyi sayısal verilere dönüştür (Örnek olarak TfidfVectorizer kullanılabilir)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    # Özellikleri standartlaştır
    scaler = StandardScaler(with_mean=False)  # Sparse verilerde mean kaldırılır
    X_train = scaler.fit_transform(X_train.toarray())
    X_test = scaler.transform(X_test.toarray())
    
    return X_train, X_test, y_train, y_test

