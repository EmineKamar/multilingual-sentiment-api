from flask import Flask, request, jsonify
import joblib
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# Model ve vectorizer'ı yükle
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "Text verisi gönderilmedi"}), 400

    # Vektörleştirme ve tahmin
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    # MLflow ile tahmin loglama
    with mlflow.start_run(nested=True):
        mlflow.log_param("input_text", text)
        mlflow.log_param("prediction", prediction)

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
