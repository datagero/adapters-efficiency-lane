from transformers import RobertaModel, RobertaTokenizer
from pathlib import Path

models = [
    "allenai/cs_roberta_base",
    "allenai/biomed_roberta_base",
    "allenai/reviews_roberta_base",
    "allenai/news_roberta_base",
    "roberta-base"  # Base RoBERTa model added
]

for model_name in models:
    print(f"Downloading model: {model_name}")
    # Download and save the tokenizer and model

    # Create a directory for the model if it doesn't exist
    model_dir = Path(f"./pretrained_models/{model_name.replace('/', '_')}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the model and the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=model_dir)
    model = RobertaModel.from_pretrained(model_name, cache_dir=model_dir)

    print(f"Model saved in {model_dir}\n")
