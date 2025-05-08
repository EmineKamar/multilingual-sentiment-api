from models.load_model import get_model
from utils.language_detect import detect_language

def analyze_sentiment(text: str):
    lang = detect_language(text)
    model, tokenizer = get_model(lang)
    # tokenizer + model inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1).detach().numpy()[0]
    sentiment = probs.argmax()
    return {
        "language": lang,
        "sentiment": ["negative", "neutral", "positive"][sentiment],
        "score": float(probs[sentiment])
    }
