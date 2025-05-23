# 🌍 Multilingual Sentiment Analysis API

A production-ready, multilingual sentiment analysis system with full **MLOps pipeline**, **CI/CD integration**, **Docker support**, and **RESTful API** powered by **Flask**. This project enables fast experimentation, model versioning, testing, and deployment — all in one scalable framework.

---

## 🚀 Features

* 🧠 Sentiment Analysis (English + Multilingual)
* 📦 Model training, evaluation, versioning with **MLflow** and **DVC**
* 🔁 Automated workflows using **GitHub Actions**
* 📈 Auto-generated metrics (e.g., accuracy, F1) saved in `metrics.json`
* 🐳 Docker containerization
* 🤪 Unit and integration tests
* 🌐 REST API (Flask)

---

## 📁 Project Structure

```
.
├── src/                  # Core modules: training, evaluation, API
├── tests/                # Unit & integration tests
├── models/               # Trained model artifacts
├── data/                 # Input data
├── mlruns/               # MLflow tracking logs
├── metrics.json          # Evaluation results
├── Dockerfile            # Docker configuration
├── dvc.yaml              # DVC pipeline configuration
├── .github/workflows/    # GitHub Actions workflow
```

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multilingual-sentiment-api.git
cd multilingual-sentiment-api

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
# Run model training
python src/train.py
```

Track experiments automatically with **MLflow**:

```bash
mlflow ui
```

---

## 📊 Evaluate the Model

```bash
python src/evaluate.py
# Results saved to metrics.json
```

---

## 🤪 Run Tests

```bash
pytest tests/
```

---

## 🌐 Run the REST API (Flask)

```bash
python src/api.py
```

Access at: `http://localhost:5000/predict`

**Example request:**

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d "{\"text\": \"I love this product!\"}"
```

---

## 🐳 Docker Support

Build and run the app with Docker:

```bash
docker build -t sentiment-flask-api .
docker run -p 5010:5003 sentiment-flask-api
```

Then, send a request:

```bash
curl -X POST http://localhost:5010/predict \
-H "Content-Type: application/json" \
-d "{\"text\": \"I love this product!\"}"
```

---

## ⚙️ DVC Pipeline

Use [DVC](https://dvc.org/) for model reproducibility and data versioning:

```bash
dvc repro
```

---

## 🤖 CI/CD with GitHub Actions

The project runs tests, builds, and evaluations on every push via `.github/workflows/main.yml`.

Example snippet:

```yaml
- name: Run Tests
  run: pytest tests/
```

---

## 📈 Metrics

Model evaluation results are written to `metrics.json`:

```json
{
  "accuracy": 0.91,
  "f1": 0.89
}
```

---

## 📬 Endpoints

| Method | Endpoint   | Description       |
| ------ | ---------- | ----------------- |
| POST   | `/predict` | Predict sentiment |


---

## 📄 License

This project is licensed under the MIT License.



