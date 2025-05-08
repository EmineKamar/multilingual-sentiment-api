from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Basit cache mantığı
_models = {}

def get_model(lang="en"):
    if lang.startswith("tr"):
        model_name = "savasy/bert-base-turkish-sentiment-cased"
    else:
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    if model_name not in _models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _models[model_name] = (model, tokenizer)
    return _models[model_name]
