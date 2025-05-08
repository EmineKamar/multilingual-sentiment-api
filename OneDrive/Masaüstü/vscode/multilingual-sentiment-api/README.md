# Multilingual Sentiment Analysis API (MLOps)

Orta-ileri seviye bir MLOps projesidir. Proje NLP modeli içerir ve model dağıtımı, versiyonlama, izleme ve CI/CD süreçlerini kapsar.

## Bileşenler
- FastAPI (REST API)
- Hugging Face Transformers (Model)
- MLflow (Model Tracking)
- DVC (Data Versioning)
- Docker (Deployment)
- GitHub Actions (Opsiyonel CI/CD)

## Kullanım
```bash
pip install -r requirements.txt
dvc pull
uvicorn app.main:app --reload
